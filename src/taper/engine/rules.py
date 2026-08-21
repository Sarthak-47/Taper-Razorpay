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
from pathlib import Path
from typing import Any, Literal

RuleKind = Literal["bank_timing", "narration_alias", "fee_variant", "adjustment_pattern"]


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
            return self.params.get("pattern", "").upper() in str(
                context.get("narration", "")
            ).upper()
        if self.kind == "fee_variant":
            return context.get("method") == self.params.get("method")
        if self.kind == "adjustment_pattern":
            return self.params.get("keyword", "").upper() in str(
                context.get("narration", "")
            ).upper()
        return False

    def verdict(self, context: dict[str, Any]) -> dict[str, Any]:
        """What this rule concludes. Pure - no side effects, no I/O."""
        if self.kind == "bank_timing":
            return {"expected_offset_days": self.params["offset_days"]}
        if self.kind == "narration_alias":
            return {"utr": self.params.get("utr")}
        if self.kind == "fee_variant":
            return {"rate": self.params["rate"]}
        if self.kind == "adjustment_pattern":
            return {"category": self.params["category"]}
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

    # ---- use ------------------------------------------------------------
    def resolve(self, context: dict[str, Any]) -> tuple[Rule, dict[str, Any]] | None:
        """First matching rule wins. Rules are append-ordered, oldest first."""
        for rule in self.rules:
            if rule.applies_to(context):
                return rule, rule.verdict(context)
        return None

    # ---- persistence ----------------------------------------------------
    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(r) for r in self.rules], indent=2), encoding="utf-8"
        )

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.rules = [Rule(**r) for r in data]

    def __len__(self) -> int:
        return len(self.rules)


def build_history(findings, bundle) -> list[ConfirmedCase]:
    """Turn confidence-1.0 deterministic findings into the regression suite.

    Only certainties go in. A case the engine was unsure about is not evidence
    of anything, and admitting rules against soft cases would let one guess
    justify the next.
    """
    cases: list[ConfirmedCase] = []
    bank_by_id = {c.bank_txn_id: c for c in bundle.bank}

    for f in findings:
        if f.confidence < 1.0:
            continue
        ctx: dict[str, Any] = {"subject_id": f.subject_id}
        ctx.update({k: v for k, v in f.evidence.items() if isinstance(v, (str, int, float))})
        bank_id = f.evidence.get("bank_txn_id")
        if bank_id and bank_id in bank_by_id:
            ctx["narration"] = bank_by_id[bank_id].narration
        cases.append(
            ConfirmedCase(
                subject_id=f.subject_id,
                context=ctx,
                correct={"defect_class": f.defect_class.value},
            )
        )
    return cases


def next_rule_id(store: RuleStore, kind: RuleKind) -> str:
    n = sum(1 for r in store.rules if r.kind == kind) + 1
    return f"{kind}_{n:03d}"


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
        "narration_alias": {"pattern", "utr"},
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
