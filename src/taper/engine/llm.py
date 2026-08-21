"""Layer 3: the model. Proposes only; arithmetic disposes.

Two jobs, both chosen because a model is genuinely better at them than code:

  1. **Narration parsing.** Free-text bank narrations where the strict regex at
     layer 1 refused to guess.
  2. **Exception classification.** Residual items with more than one plausible
     explanation - is a shortfall an unrecorded adjustment, or the first leg of
     a split whose sibling has not landed?

Everything else stays deterministic. The model never does arithmetic, never
decides a match on its own authority, and never writes a rule directly into the
store. Every proposal it makes is re-verified numerically in ``verify_proposal``
before it is allowed to become a finding. If the maths does not confirm it, the
item stays on the exception list.

That gate exists because of a specific failure during development: the model
confidently proposed a batch match whose credits were off by ~Rs.4,000, and the
pipeline accepted it because the response *looked* well-formed. Structural
validity is not numerical truth. Now nothing is accepted on plausibility alone.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from ..models import AMOUNT_TOLERANCE, DefectClass, Money
from .results import Exception_, Finding, Layer

MODEL = "claude-sonnet-5"

# Largest share of a batch that a deduction may plausibly be. Bank charges are
# small next to the payout; anything bigger is a missing credit wearing an
# adjustment's label. Set generously - a real charge is well under 1% - so it
# only ever rejects the absurd.
MAX_ADJUSTMENT_SHARE = Decimal("0.25")

SYSTEM_PROMPT = """You are the exception-resolution layer of a settlement reconciliation \
system for an Indian payment gateway.

You are shown ONE unresolved item that deterministic matching could not settle. \
Deterministic layers have already handled every unambiguous case, so assume the \
easy explanation has been ruled out.

Return STRICT JSON, no prose, with these keys:
  defect_class   MUST be exactly one of these six strings, copied verbatim:
                   "unrecorded_adjustment"  the bank deducted something the report
                                            does not show (a processing or service
                                            charge). Use this together with
                                            claimed_adjustment below.
                   "split_settlement"       one payout arrived as several credits
                   "narration_drift"        the narration carries no usable reference
                   "missing_utr"            the settlement report has no reference
                   "timing_shift"           the money landed later than expected
                   "unknown"                you cannot tell
                 Do NOT invent a value, and do NOT put the name of any other field
                 here - "claimed_adjustment" is a separate numeric field, never a
                 defect_class.
  bank_txn_ids   list of bank credit ids you believe pay this batch (may be empty)
  reasoning      one short sentence
  confidence     0.0-1.0, calibrated: use <0.7 when genuinely unsure
  claimed_adjustment
                 null, or a positive amount you believe the bank deducted and the
                 settlement report does not show. When you set this, defect_class
                 must be "unrecorded_adjustment". Use it when the credits you named
                 are short of the expected payout by a consistent amount. The system
                 verifies that the sums close exactly with your number, so a guess
                 that is merely close will be rejected - omit it rather than
                 approximate.
  proposed_rule  null, or an object {kind, params, confidence} where kind is one of
                 bank_timing {bank, offset_days}
                 narration_alias {pattern, utr}
                 fee_variant {method, rate}
                 adjustment_pattern {keyword, category}

Rules for you specifically:
- Do NOT perform arithmetic. State which credits you believe belong together and
  the system will verify the sums itself.
- Answering "unknown" with low confidence is a correct and useful answer. An
  item left on the exception list costs one human review. A wrong confident
  answer corrupts the close.
- Before naming credits, sanity-check the magnitudes. If the credits you would
  name are not clearly close to the expected payout - within a small charge, or
  summing to roughly it - then you have not found the answer. Say "unknown" with
  low confidence instead of naming the nearest candidate.
- Never invent a claimed_adjustment to close a large gap. A processing or
  service charge is a small round amount next to the payout. If the shortfall is
  a large fraction of the batch, the missing money is another credit you have
  not been shown, not a charge - answer "unknown".
- Only propose a rule when the pattern would plausibly recur every month."""


class LLMClient(Protocol):
    def classify(self, exc: Exception_, context: dict[str, Any]) -> dict[str, Any]: ...
    @property
    def name(self) -> str: ...


@dataclass
class AnthropicClient:
    """The real thing. Reported metrics must come from this path."""

    model: str = MODEL
    max_tokens: int = 700
    _calls: int = 0

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def classify(self, exc: Exception_, context: dict[str, Any]) -> dict[str, Any]:
        from anthropic import Anthropic  # imported lazily so the mock path needs no dep

        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        payload = {
            "kind": exc.kind,
            "subject_id": exc.subject_id,
            "reason_deterministic_failed": exc.reason,
            "batch": exc.context,
            "candidate_credits": context.get("candidates", []),
        }
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
        )
        self._calls += 1
        text = "".join(b.text for b in resp.content if b.type == "text")
        return _parse_json(text)


@dataclass
class OpenAICompatibleClient:
    """Layer 3 against any OpenAI-shaped endpoint: Ollama, Groq, OpenRouter.

    Layer 3 is a swappable component, and this is the proof rather than the
    claim. The same exception queue, the same prompt, and - critically - the
    same ``verify_proposal`` gate on the way out, so the provider changes
    nothing about what is allowed to become a finding.

    That is what makes a small local model a reasonable choice here rather than
    a compromise. A 14B model on a laptop cannot produce a false reconciliation
    for the same reason a compromised model cannot: it proposes, and arithmetic
    disposes. The worst a weak model does is decline more often and leave more
    on the exception list, which is the failure mode this system is built to
    absorb.

    Uses only the standard library. Adding an HTTP client for one POST would
    contradict the rest of the project.
    """

    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5:14b"
    api_key: str | None = None      # unused by Ollama; required by hosted ones
    timeout: float = 120.0          # local models on CPU are not fast
    _calls: int = 0

    @property
    def name(self) -> str:
        host = self.base_url.split("//")[-1].split("/")[0]
        return f"{host}:{self.model}"

    def classify(self, exc: Exception_, context: dict[str, Any]) -> dict[str, Any]:
        import urllib.error
        import urllib.request

        payload = {
            "kind": exc.kind,
            "subject_id": exc.subject_id,
            "reason_deterministic_failed": exc.reason,
            "batch": exc.context,
            "candidate_credits": context.get("candidates", []),
        }
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, indent=2)},
                ],
                # Deterministic decoding: this is a classification with one
                # right answer, not a generation task, and a close that changes
                # between reruns cannot be signed off.
                "temperature": 0,
                "max_tokens": 700,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(  # noqa: S310 - operator-configured host
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        self._calls += 1
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc_err:
            # A provider that is down or slow must not fail the close. The item
            # stays on the exception list, which is where it already was.
            return {
                "defect_class": "unknown",
                "bank_txn_ids": [],
                "confidence": 0.0,
                "reasoning": f"provider unreachable: {type(exc_err).__name__}",
                "proposed_rule": None,
            }

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return {
                "defect_class": "unknown", "bank_txn_ids": [], "confidence": 0.0,
                "reasoning": "unexpected response shape", "proposed_rule": None,
            }
        return _parse_json(text)


@dataclass
class MockClient:
    """Offline stand-in so the repo runs end-to-end with no API key.

    IMPORTANT: this is a heuristic, not a model. Any ablation run against it is
    measuring heuristic-vs-heuristic and is NOT evidence about LLM value. It
    exists for CI and for reviewers without a key. Every metrics artifact
    records which client produced it, and the CLI prints a warning banner when
    this one is in use.
    """

    _calls: int = 0

    @property
    def name(self) -> str:
        return "mock:offline-heuristic"

    def classify(self, exc: Exception_, context: dict[str, Any]) -> dict[str, Any]:
        self._calls += 1
        cands = context.get("candidates", [])
        if exc.kind == "unmatched_batch" and cands:
            # Prefer the pair that could form a split; else call it a shortfall.
            if len(cands) >= 2:
                return {
                    "defect_class": "split_settlement",
                    "bank_txn_ids": [c["bank_txn_id"] for c in cands[:2]],
                    "reasoning": "two candidate credits in window may form a split",
                    "confidence": 0.72,
                    "proposed_rule": None,
                }
            return {
                "defect_class": "unrecorded_adjustment",
                "bank_txn_ids": [cands[0]["bank_txn_id"]],
                "reasoning": "single credit short of expected net",
                "confidence": 0.68,
                "proposed_rule": None,
            }
        return {
            "defect_class": "unknown",
            "bank_txn_ids": [],
            "reasoning": "no candidates in window",
            "confidence": 0.2,
            "proposed_rule": None,
        }


def _parse_json(text: str) -> dict[str, Any]:
    """Tolerate fenced or chatty responses; refuse to guess at broken ones."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.S)
        if brace:
            text = brace.group(0)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"defect_class": "unknown", "bank_txn_ids": [], "confidence": 0.0,
                "reasoning": "unparseable response", "proposed_rule": None}
    return parsed if isinstance(parsed, dict) else {}


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

@dataclass
class Verification:
    ok: bool
    reason: str
    credited: Money = Money("0.00")
    delta: Money = Money("0.00")


def verify_proposal(
    proposal: dict[str, Any],
    expected_net: Money,
    credit_amounts: dict[str, Money],
) -> Verification:
    """Re-derive the model's claim from the numbers. This is the whole safety story.

    The model says "these credits pay this batch". We add them up ourselves. If
    the sum does not reconcile - exactly, or as a shortfall consistent with the
    adjustment it claims - the proposal is refused and the item stays an
    exception. A confident model and a wrong number lose to the number.
    """
    ids = proposal.get("bank_txn_ids") or []
    if not isinstance(ids, list) or not ids:
        return Verification(False, "proposal named no credits")

    unknown = [i for i in ids if i not in credit_amounts]
    if unknown:
        return Verification(False, f"proposal named unknown credits: {unknown[:3]}")

    try:
        credited = sum((credit_amounts[i] for i in ids), Money("0.00"))
    except (InvalidOperation, TypeError):
        return Verification(False, "credit amounts not summable")

    delta = expected_net - credited

    claimed = str(proposal.get("defect_class", "")).lower()

    # A batch can be split across credits *and* short by a charge at the same
    # time. Layer 1 resolves those deterministically when the charge is already
    # known; when it is not, the model may name the shortfall it believes is
    # there. The claim is only accepted if the arithmetic closes exactly with
    # that number - the model supplies a candidate, it never supplies a fudge.
    claimed_adj = proposal.get("claimed_adjustment")
    if claimed_adj is not None:
        try:
            adjustment = Money(str(claimed_adj))
        except (InvalidOperation, ValueError, TypeError):
            return Verification(False, f"claimed_adjustment {claimed_adj!r} is not a number")
        if adjustment <= 0:
            return Verification(False, "claimed adjustment must be a positive deduction")
        if adjustment > credited:
            # A deduction larger than the payout is not an explanation, it is a
            # free parameter big enough to reconcile anything.
            return Verification(
                False, f"claimed adjustment {adjustment} exceeds the credited amount"
            )
        residual = delta - adjustment
        if abs(residual) <= AMOUNT_TOLERANCE:
            return Verification(
                True,
                f"credits sum to expected net less a claimed {adjustment} deduction",
                credited,
                delta,
            )
        return Verification(
            False,
            f"claimed adjustment {adjustment} leaves {residual} unexplained",
            credited,
            delta,
        )

    if claimed == "split_settlement":
        # A split must actually reconcile to the batch total. No tolerance for
        # "close enough" - that is how a wrong pairing gets waved through.
        if abs(delta) <= AMOUNT_TOLERANCE:
            return Verification(True, "credits sum to expected net", credited, delta)
        return Verification(False, f"claimed split is off by {delta}", credited, delta)

    if claimed == "unrecorded_adjustment":
        # A shortfall is only an adjustment if money is actually *missing*.
        # An overpayment is a different problem and must not be relabelled.
        if delta <= AMOUNT_TOLERANCE:
            return Verification(False, f"no shortfall to explain (delta {delta})", credited, delta)

        # And it has to be adjustment-*sized*. A bank charge is small next to
        # the payout it comes out of; a "shortfall" that is a large fraction of
        # the batch is a missing credit, not a deduction.
        #
        # Without this bound, "unrecorded adjustment" accepts a gap of any size
        # and becomes a free parameter that explains everything - which is
        # exactly how it slipped through: the mock claimed a Rs.64,051
        # adjustment on a Rs.64,051 credit, i.e. the batch's missing half.
        if expected_net > 0 and delta > expected_net * MAX_ADJUSTMENT_SHARE:
            share = delta / expected_net
            return Verification(
                False,
                f"claimed adjustment is {share:.0%} of the batch - too large to be "
                f"a charge, more likely a missing credit",
                credited,
                delta,
            )
        return Verification(True, f"shortfall of {delta} confirmed", credited, delta)

    if claimed in {"narration_drift", "missing_utr", "timing_shift"}:
        if abs(delta) <= AMOUNT_TOLERANCE:
            return Verification(True, "amounts reconcile; defect is metadata-only",
                                credited, delta)
        return Verification(False, f"metadata defect claimed but amounts differ by {delta}",
                            credited, delta)

    return Verification(False, f"unrecognised defect_class {claimed!r}")


def to_finding(
    proposal: dict[str, Any], exc: Exception_, verification: Verification
) -> Finding | None:
    """Only reachable for proposals that already survived verification."""
    try:
        dc = DefectClass(str(proposal["defect_class"]).lower())
    except (KeyError, ValueError):
        return None
    conf = float(proposal.get("confidence", 0.5))
    return Finding(
        defect_class=dc,
        subject_id=exc.subject_id,
        layer=Layer.L3_LLM,
        confidence=max(0.0, min(1.0, conf)),
        money_impact=abs(verification.delta),
        evidence={
            "reasoning": str(proposal.get("reasoning", ""))[:200],
            "bank_txn_ids": proposal.get("bank_txn_ids", []),
            "verified": verification.reason,
            "credited": str(verification.credited),
        },
    )


def get_client(
    use_real: bool,
    provider: str = "anthropic",
    base_url: str | None = None,
    model: str | None = None,
) -> LLMClient:
    """Pick the layer 3 client.

    Three real providers and one stand-in. The stand-in is the only one whose
    numbers must never be reported, and it is the only one that is not a model.
    """
    if not use_real:
        return MockClient()

    if provider == "ollama":
        return OpenAICompatibleClient(
            base_url=base_url or "http://localhost:11434/v1",
            model=model or "qwen2.5:14b",
        )

    if provider == "openai-compatible":
        if not base_url:
            raise RuntimeError("--llm-base-url is required for an openai-compatible provider")
        key_env = os.environ.get("LLM_API_KEY")
        return OpenAICompatibleClient(
            base_url=base_url, model=model or "", api_key=key_env
        )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Options: --llm ollama to run layer 3 "
            "locally, --llm openai-compatible with --llm-base-url for a hosted "
            "provider, or --mock for the offline heuristic - whose numbers must "
            "not be reported as model results."
        )
    return AnthropicClient()
