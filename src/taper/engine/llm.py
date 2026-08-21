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
from decimal import InvalidOperation
from typing import Any, Protocol

from ..models import AMOUNT_TOLERANCE, DefectClass, Money
from .results import Exception_, Finding, Layer

MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are the exception-resolution layer of a settlement reconciliation \
system for an Indian payment gateway.

You are shown ONE unresolved item that deterministic matching could not settle. \
Deterministic layers have already handled every unambiguous case, so assume the \
easy explanation has been ruled out.

Return STRICT JSON, no prose, with these keys:
  defect_class   one of: unrecorded_adjustment, split_settlement, narration_drift,
                 missing_utr, timing_shift, or "unknown" if you cannot tell
  bank_txn_ids   list of bank credit ids you believe pay this batch (may be empty)
  reasoning      one short sentence
  confidence     0.0-1.0, calibrated: use <0.7 when genuinely unsure
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
    if claimed == "split_settlement":
        # A split must actually reconcile to the batch total. No tolerance for
        # "close enough" - that is how a wrong pairing gets waved through.
        if abs(delta) <= AMOUNT_TOLERANCE:
            return Verification(True, "credits sum to expected net", credited, delta)
        return Verification(False, f"claimed split is off by {delta}", credited, delta)

    if claimed == "unrecorded_adjustment":
        # A shortfall is only an adjustment if money is actually *missing*.
        # An overpayment is a different problem and must not be relabelled.
        if delta > AMOUNT_TOLERANCE:
            return Verification(True, f"shortfall of {delta} confirmed", credited, delta)
        return Verification(False, f"no shortfall to explain (delta {delta})", credited, delta)

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


def get_client(use_real: bool) -> LLMClient:
    if use_real:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Run with --mock to use the offline "
                "heuristic, but do not report those numbers as LLM results."
            )
        return AnthropicClient()
    return MockClient()
