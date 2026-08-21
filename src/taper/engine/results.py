"""What the engine emits.

Two ideas carry the whole submission and both live here:

  * **Provenance.** Every finding records which layer produced it, on what
    evidence, at what tolerance. A number a controller cannot trace back to
    source rows is a number they will not sign off on.

  * **Layer attribution.** Every finding knows whether it came from
    deterministic code or from the model. That single field is what makes the
    deterministic-vs-LLM ablation computable, and the ablation is the evidence
    for "the right tool in the right place, and where we chose not to use one".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any

from ..models import DefectClass, Money


class Layer(StrEnum):
    """Which tier resolved this. Ordered cheapest-and-surest first."""

    L0_EXACT = "L0_exact"        # ID joins and arithmetic. No model, no ambiguity.
    L1_FUZZY = "L1_fuzzy"        # Date windows, regex, subset netting. Still no model.
    L2_RULE = "L2_learned_rule"  # A rule the system learned from a past human decision.
    L3_LLM = "L3_llm"            # The model. Proposes only; arithmetic disposes.

    @property
    def is_deterministic(self) -> bool:
        return self is not Layer.L3_LLM


@dataclass
class Finding:
    """One thing the engine believes is wrong, with its receipts."""

    defect_class: DefectClass
    subject_id: str
    layer: Layer
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)
    money_impact: Money = Money("0.00")
    rule_id: str | None = None

    def key(self) -> tuple[DefectClass, str]:
        """Identity for scoring against ground truth."""
        return (self.defect_class, self.subject_id)


@dataclass
class BatchMatch:
    """A settlement batch tied to the bank credit(s) that paid it."""

    batch_id: str
    bank_txn_ids: list[str]
    expected_net: Money
    credited: Money
    layer: Layer
    confidence: float
    method: str = ""

    @property
    def delta(self) -> Money:
        return self.credited - self.expected_net

    @property
    def is_clean(self) -> bool:
        return abs(self.delta) <= Decimal("1.00")


@dataclass
class Exception_:
    """An item no deterministic layer could resolve.

    This is the queue the LLM sees, and - critically - the queue that is
    reported honestly when the LLM cannot resolve it either. The brief is
    explicit that a cherry-picked match proves nothing; the exception list is
    what makes the match rate believable.
    """

    subject_id: str
    kind: str
    context: dict[str, Any] = field(default_factory=dict)
    candidates: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class ReconResult:
    matches: list[BatchMatch] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    exceptions: list[Exception_] = field(default_factory=list)
    llm_calls: int = 0
    records_processed: int = 0
    elapsed_s: float = 0.0

    # ---- headline numbers a controller actually reads --------------------
    @property
    def match_rate(self) -> float:
        if not self.matches:
            return 0.0
        return sum(1 for m in self.matches if m.is_clean) / len(self.matches)

    @property
    def deterministic_share(self) -> float:
        """Fraction of findings resolved without touching a model.

        This is the number that should climb month over month as the rule
        store grows. It is the thesis of the project expressed as a float.
        """
        if not self.findings:
            return 1.0
        det = sum(1 for f in self.findings if f.layer.is_deterministic)
        return det / len(self.findings)

    @property
    def llm_calls_per_100(self) -> float:
        if not self.records_processed:
            return 0.0
        return 100.0 * self.llm_calls / self.records_processed

    def findings_by_layer(self) -> dict[Layer, int]:
        out: dict[Layer, int] = dict.fromkeys(Layer, 0)
        for f in self.findings:
            out[f.layer] += 1
        return out
