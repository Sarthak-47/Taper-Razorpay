"""Learned rules, and the gate that decides which ones are allowed to exist.

This is the thesis of the project in one file.

When a human resolves an exception, the model proposes a *typed* rule that
would have resolved it automatically. If the rule is admitted, the same
situation next month is handled at layer 2 - deterministically, for free, with
no model call. So LLM calls per 100 records fall month over month while match
rate rises. The agent's job is to make itself unnecessary.

The dangerous part is obvious: a bad rule silently corrupts every future close.
So no rule is admitted on the model's say-so. Every candidate is replayed
against the full history of cases already confirmed correct, and if it would
change even one of them, it is rejected. The model proposes; history decides.

Rules are deliberately *typed*, not arbitrary code. Four shapes, each with a
small parameter set. A model that can only fill in blanks in a known form
cannot invent a rule that does something the system was never designed to do.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

RuleKind = Literal["bank_timing", "narration_alias", "fee_variant", "adjustment_pattern"]

# Longest reference token we will lift out of a narration. Bank references are
# 8-24 digits; anything longer is a parse gone wrong, not a reference.
MAX_REF_DIGITS = 24

# Bumped when the on-disk shape changes. The reader still accepts the original
# bare-list format, so an existing store keeps loading.
STORE_FORMAT_VERSION = 2


def _alias_extract(params: dict[str, Any], narration: str) -> str | None:
    """Lift a settlement reference out of a narration, by marker and prefix.

    Deliberately *not* a learned regex. The model names a marker token and a
    prefix; this function does the extraction with fixed code. A rule store that
    accepted model-authored patterns would be executing text the model wrote
    against every future narration - an injection surface for no benefit, since
    every real case is "find the label, take the number after it".
    """
    marker = str(params.get("marker", "")).upper().strip()
    if not marker:
        return None
    text = narration.upper()
    idx = text.find(marker)
    if idx < 0:
        return None

    rest = text[idx + len(marker):].lstrip(" :-/#")
    digits = ""
    for ch in rest:
        if ch.isdigit():
            digits += ch
            if len(digits) > MAX_REF_DIGITS:
                return None
        else:
            break
    if len(digits) < 6:
        return None
    return f"{params.get('prefix', '')}{digits}"


@dataclass(frozen=True)
class Rule:
    """One learned, typed rule.

    ``origin_exception`` and ``learned_on`` exist so any rule in the store can
    be traced back to the specific human decision that created it. A rule
    nobody can explain the provenance of is a liability.
    """

    rule_id: str
    kind: RuleKind
    params: dict[str, Any]
    origin_exception: str
    learned_on: str
    confidence: float = 0.9

    # ---- behaviour ------------------------------------------------------
    def applies_to(self, context: dict[str, Any]) -> bool:
        """Does this rule have an opinion about this case?"""
        if self.kind == "bank_timing":
            return context.get("bank") == self.params.get("bank")
        if self.kind == "narration_alias":
            return _alias_extract(self.params, str(context.get("narration", ""))) is not None
        if self.kind == "fee_variant":
            return context.get("method") == self.params.get("method")
        if self.kind == "adjustment_pattern":
            return self.params.get("keyword", "").upper() in str(
                context.get("narration", "")
            ).upper()
        return False

    def summary(self) -> str:
        """One line naming what this rule asserts, for a human reading a list.

        Every kind carries different params, so a single template cannot
        describe them. The campaign's rule list used to read the
        ``keyword``/``category`` pair that only ``adjustment_pattern`` has,
        which meant a learned rate card and a learned narration alias both
        printed as an empty arrow - the two kinds hardest to believe were
        learned were the two the output could not show.
        """
        p = self.params
        if self.kind == "bank_timing":
            return f"{p.get('bank', '?')} settles T+{p.get('offset_days', '?')}"
        if self.kind == "narration_alias":
            marker = p.get("marker") or p.get("prefix") or "?"
            return f"reference follows '{marker}'"
        if self.kind == "fee_variant":
            try:
                rate = f"{Decimal(str(p['rate'])):.2%}"
            except (ArithmeticError, KeyError, ValueError):
                rate = str(p.get("rate", "?"))
            return f"{p.get('method', '?')} is billed at {rate}"
        if self.kind == "adjustment_pattern":
            return f"{p.get('keyword', '?')} -> {p.get('category', '?')}"
        return self.kind

    def verdict(self, context: dict[str, Any]) -> dict[str, Any]:
        """What this rule concludes. Pure - no side effects, no I/O."""
        if self.kind == "bank_timing":
            return {"expected_offset_days": self.params["offset_days"]}
        if self.kind == "narration_alias":
            return {"utr": _alias_extract(self.params, str(context.get("narration", "")))}
        if self.kind == "fee_variant":
            return {"rate": self.params["rate"]}
        if self.kind == "adjustment_pattern":
            # The amount is the load-bearing claim, not the category. A rule
            # asserting only a label cannot be contradicted by anything, which
            # made the admission gate unable to reject it.
            verdict: dict[str, Any] = {"category": self.params["category"]}
            if "amount" in self.params:
                verdict["amount"] = str(self.params["amount"])
            return verdict
        return {}


@dataclass(frozen=True)
class ConfirmedCase:
    """A case whose correct answer is already settled.

    Built from two sources: findings the deterministic layers produced at
    confidence 1.0, and exceptions a human has resolved. Together they form the
    regression suite that every candidate rule must survive.
    """

    subject_id: str
    context: dict[str, Any]
    correct: dict[str, Any]


@dataclass
class AdmissionResult:
    admitted: bool
    rule: Rule
    reason: str
    regressions: list[str] = field(default_factory=list)


class RuleStore:
    """The learned-rule corpus, plus the gate guarding entry to it."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.rules: list[Rule] = []
        self.rejected: list[AdmissionResult] = []
        # Rules that were true and stopped being true, with why. Kept rather
        # than deleted so a past close can still be explained.
        self.retired: list[tuple[Rule, str]] = []
        if path and path.exists():
            self.load()

    # ---- the gate -------------------------------------------------------
    def propose(self, rule: Rule, history: list[ConfirmedCase]) -> AdmissionResult:
        """Replay a candidate rule against everything we already know.

        A rule earns admission by being *harmless* on the past, not by being
        persuasive about the present. If it contradicts a single confirmed
        case, it is rejected and the originating item stays an exception - a
        human looks at it again next month.

        This is strictly more conservative than it needs to be, and that is the
        point: the cost of a missed automation is one manual review, while the
        cost of a bad rule is silent corruption across every future close.
        """
        regressions: list[str] = []
        for case in history:
            if not rule.applies_to(case.context):
                continue
            verdict = rule.verdict(case.context)
            for key, value in verdict.items():
                if key in case.correct and case.correct[key] != value:
                    regressions.append(
                        f"{case.subject_id}: {key} would become {value!r}, "
                        f"confirmed value is {case.correct[key]!r}"
                    )

        if regressions:
            result = AdmissionResult(
                admitted=False,
                rule=rule,
                reason=f"rejected: contradicts {len(regressions)} confirmed case(s)",
                regressions=regressions[:5],
            )
            self.rejected.append(result)
            return result

        applicable = sum(1 for c in history if rule.applies_to(c.context))
        self.rules.append(rule)
        return AdmissionResult(
            admitted=True,
            rule=rule,
            reason=f"admitted: consistent with {applicable} confirmed case(s)",
        )

    def retire(self, rule_id: str, reason: str) -> Rule | None:
        """Remove a rule that has stopped being true, keeping the record of it.

        Retirement is not deletion. A rule that was right for six months and
        then wrong is evidence about the world changing, and a store that
        silently drops it loses the ability to explain why last quarter's close
        reconciled differently. It moves to ``retired`` with its reason.
        """
        for i, rule in enumerate(self.rules):
            if rule.rule_id == rule_id:
                self.retired.append((self.rules.pop(i), reason))
                return rule
        return None

    # ---- use ------------------------------------------------------------
    def resolve(self, context: dict[str, Any]) -> tuple[Rule, dict[str, Any]] | None:
        """First matching rule wins. Rules are append-ordered, oldest first."""
        for rule in self.rules:
            if rule.applies_to(context):
                return rule, rule.verdict(context)
        return None

    # ---- persistence ----------------------------------------------------
    def save(self) -> None:
        """Persist the whole store, retirements included.

        Writing only the live rules loses more than history. Rule ids are
        allocated by counting live *and* retired rules of a kind, so a store
        that forgets its retirements reissues an id that a live rule already
        holds - and every provenance trail through the two becomes ambiguous.
        A round-trip test guards exactly that.
        """
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "version": STORE_FORMAT_VERSION,
                    "rules": [asdict(r) for r in self.rules],
                    "retired": [
                        {"rule": asdict(r), "reason": reason} for r, reason in self.retired
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))

        # A bare list is the pre-versioning format, which had no retirements.
        # Reading it still works; it just starts with an empty retired list.
        if isinstance(data, list):
            self.rules = [Rule(**r) for r in data]
            self.retired = []
            return

        self.rules = [Rule(**r) for r in data.get("rules", [])]
        self.retired = [
            (Rule(**entry["rule"]), entry.get("reason", ""))
            for entry in data.get("retired", [])
            if isinstance(entry, dict) and "rule" in entry
        ]

    def __len__(self) -> int:
        return len(self.rules)


def build_history(findings, bundle) -> list[ConfirmedCase]:
    """Turn what we already know for certain into the regression suite.

    Each case has to be stated in the *same terms a rule asserts*, or the gate
    has nothing to compare and silently admits everything. That is not
    hypothetical: an earlier version recorded only ``defect_class`` while every
    verdict returned ``rate``, ``amount`` or ``utr``. The keys never overlapped,
    so across a real close of a hundred confirmed cases not one candidate could
    ever be rejected - the gate was mechanically correct and completely vacuous.

    So each entry below pairs a context that makes some rule applicable with the
    value that rule must not contradict.
    """
    from .matching import UTR_PATTERN

    cases: list[ConfirmedCase] = []
    bank_by_id = {c.bank_txn_id: c for c in bundle.bank}

    # 1. References we can already read. A narration_alias that extracts a
    #    different value from a narration the strict regex already resolves is
    #    wrong, whatever else it might fix.
    for credit in bundle.bank:
        hit = UTR_PATTERN.search(credit.narration.upper())
        if hit:
            cases.append(
                ConfirmedCase(
                    subject_id=credit.bank_txn_id,
                    context={"narration": credit.narration},
                    correct={"utr": hit.group(0)},
                )
            )

    # 2. Contracted rates and charge amounts confirmed by this close.
    for f in findings:
        if f.confidence < 1.0:
            continue
        evidence = f.evidence

        # Deliberately NOT recorded: contracted rates. The rate in a fee
        # finding is the rate the engine *assumed*, not one anybody confirmed -
        # a contracted rate is only knowable from the merchant agreement. An
        # earlier version treated it as fact and the gate then rejected the
        # correct 3% international-card rule for contradicting the 2% default
        # it was there to replace. The gate must replay against confirmed
        # facts, never against its own current assumptions.

        # Observed shortfalls, but only where a rule did not produce them. A
        # rule-derived amount would just be the rule confirming itself.
        amount = evidence.get("amount")
        bank_id = evidence.get("bank_txn_id")
        credit = bank_by_id.get(bank_id) if bank_id else None
        if amount and credit is not None and f.rule_id is None:
            cases.append(
                ConfirmedCase(
                    subject_id=f.subject_id,
                    context={"narration": credit.narration},
                    correct={"amount": str(amount)},
                )
            )

        offset = evidence.get("offset_days")
        if offset is not None and evidence.get("bank"):
            cases.append(
                ConfirmedCase(
                    subject_id=f.subject_id,
                    context={"bank": evidence["bank"]},
                    correct={"expected_offset_days": offset},
                )
            )

    return cases


def next_rule_id(store: RuleStore, kind: RuleKind) -> str:
    """Monotonic per kind, counting retired rules too.

    Counting only live rules would reissue the id of a rule that was just
    retired, so the replacement and the thing it replaced would share an
    identifier - and every provenance trail through them would be ambiguous
    exactly when someone most needs to follow it.
    """
    n = sum(1 for r in store.rules if r.kind == kind)
    n += sum(1 for r, _ in store.retired if r.kind == kind)
    return f"{kind}_{n + 1:03d}"


def rule_from_proposal(
    proposal: dict[str, Any], origin: str, today: date | None = None
) -> Rule | None:
    """Coerce a model proposal into a typed rule, or refuse it.

    Anything the model returns that is not one of the four known shapes, or
    that is missing a required parameter, is dropped here rather than being
    half-understood downstream. Parsing is a trust boundary.
    """
    kind = proposal.get("kind")
    params = proposal.get("params") or {}
    required = {
        "bank_timing": {"bank", "offset_days"},
        "narration_alias": {"marker", "prefix"},
        "fee_variant": {"method", "rate"},
        "adjustment_pattern": {"keyword", "category"},
    }
    if kind not in required:
        return None
    if not required[kind] <= set(params):
        return None
    return Rule(
        rule_id=f"{kind}_pending",
        kind=kind,  # type: ignore[arg-type]
        params=params,
        origin_exception=origin,
        learned_on=str(today or date.today()),
        confidence=float(proposal.get("confidence", 0.9)),
    )
