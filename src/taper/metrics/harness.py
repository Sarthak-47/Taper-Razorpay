"""Scoring against ground truth: per-class precision/recall, calibration, ablation.

Three artifacts come out of this file, and they are the submission's evidence:

  1. **Per-class precision/recall** on a held-out seed. A single blended number
     hides which defect the engine is actually bad at.
  2. **Calibration** - when the engine says 0.9, is it right about 90% of the
     time? An uncalibrated confidence is not a confidence, it is a vibe, and no
     auto-clear threshold built on it can be trusted.
  3. **Ablation** - deterministic-only versus the full stack. This is what makes
     "we used AI only where it earns its place" a measurement rather than a claim.

Also computed: **false-positive cost**. Precision alone understates the damage,
because every false flag is a human opening a spreadsheet. Costed in review
minutes it becomes a number a controller can weigh against the money recovered.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..engine.results import ReconResult
from ..generator import GeneratedCase
from ..models import DefectClass, Money

# Minutes a human spends dispositioning one flagged item. Used to convert
# false positives into a cost a finance team would actually recognise.
REVIEW_MINUTES_PER_FLAG = 4.0


@dataclass
class ClassScore:
    defect_class: DefectClass
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else float("nan")

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p != p or r != r or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    @property
    def support(self) -> int:
        return self.tp + self.fn


@dataclass
class Scorecard:
    per_class: dict[DefectClass, ClassScore] = field(default_factory=dict)
    match_rate: float = 0.0
    llm_calls: int = 0
    llm_calls_per_100: float = 0.0
    deterministic_share: float = 1.0
    exceptions: int = 0
    records: int = 0
    elapsed_s: float = 0.0
    money_flagged: Money = Money("0.00")
    client_name: str = "unknown"

    @property
    def tp(self) -> int:
        return sum(s.tp for s in self.per_class.values())

    @property
    def fp(self) -> int:
        return sum(s.fp for s in self.per_class.values())

    @property
    def fn(self) -> int:
        return sum(s.fn for s in self.per_class.values())

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else float("nan")

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else float("nan")

    @property
    def false_positive_cost_minutes(self) -> float:
        return self.fp * REVIEW_MINUTES_PER_FLAG

    @property
    def throughput(self) -> float:
        return self.records / self.elapsed_s if self.elapsed_s else float("inf")


def score(case: GeneratedCase, result: ReconResult, client_name: str = "n/a") -> Scorecard:
    """Compare findings to injected ground truth on (class, subject_id) identity."""
    truth = {(d.defect_class, d.subject_id) for d in case.defects}
    pred = {(f.defect_class, f.subject_id) for f in result.findings}

    card = Scorecard(
        match_rate=result.match_rate,
        llm_calls=result.llm_calls,
        llm_calls_per_100=result.llm_calls_per_100,
        deterministic_share=result.deterministic_share,
        exceptions=len(result.exceptions),
        records=result.records_processed,
        elapsed_s=result.elapsed_s,
        money_flagged=sum((f.money_impact for f in result.findings), Money("0.00")),
        client_name=client_name,
    )

    for dc in DefectClass:
        t = {k for k in truth if k[0] is dc}
        p = {k for k in pred if k[0] is dc}
        card.per_class[dc] = ClassScore(
            defect_class=dc, tp=len(t & p), fp=len(p - t), fn=len(t - p)
        )
    return card


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass
class CalibrationBin:
    lo: float
    hi: float
    n: int = 0
    correct: int = 0

    @property
    def observed(self) -> float:
        return self.correct / self.n if self.n else float("nan")

    @property
    def midpoint(self) -> float:
        return (self.lo + self.hi) / 2

    @property
    def gap(self) -> float:
        """Signed calibration error. Positive means over-confident."""
        obs = self.observed
        return float("nan") if obs != obs else self.midpoint - obs


def calibration(case: GeneratedCase, result: ReconResult, n_bins: int = 5) -> list[CalibrationBin]:
    """Are stated confidences honest? Bin by confidence, compare to hit rate."""
    truth = {(d.defect_class, d.subject_id) for d in case.defects}
    edges = [i / n_bins for i in range(n_bins + 1)]
    bins = [CalibrationBin(edges[i], edges[i + 1]) for i in range(n_bins)]

    for f in result.findings:
        idx = min(int(f.confidence * n_bins), n_bins - 1)
        bins[idx].n += 1
        if f.key() in truth:
            bins[idx].correct += 1
    return [b for b in bins if b.n]


def auto_clear_operating_point(
    case: GeneratedCase, result: ReconResult, target_precision: float = 0.99
) -> dict[str, Any]:
    """Highest-coverage confidence threshold that still holds target precision.

    This is the number a controller actually buys: how much of the close can I
    stop looking at, and what does that cost me in missed defects?
    """
    truth = {(d.defect_class, d.subject_id) for d in case.defects}
    scored = sorted(result.findings, key=lambda f: -f.confidence)
    total = len(scored)
    best: dict[str, Any] = {
        "threshold": 1.01, "coverage": 0.0, "precision": float("nan"),
        "auto_cleared": 0, "routed_to_human": total,
    }
    if not total:
        return best

    tp = fp = 0
    for i, f in enumerate(scored, start=1):
        if f.key() in truth:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        if precision >= target_precision:
            best = {
                "threshold": round(f.confidence, 3),
                "coverage": i / total,
                "precision": precision,
                "auto_cleared": i,
                "routed_to_human": total - i,
                "review_minutes_saved": round(i * REVIEW_MINUTES_PER_FLAG, 1),
            }
    return best


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

@dataclass
class Ablation:
    deterministic_only: Scorecard
    full_stack: Scorecard

    @property
    def recall_delta(self) -> float:
        return self.full_stack.recall - self.deterministic_only.recall

    @property
    def precision_delta(self) -> float:
        return self.full_stack.precision - self.deterministic_only.precision

    @property
    def exceptions_closed(self) -> int:
        return self.deterministic_only.exceptions - self.full_stack.exceptions

    def verdict(self) -> str:
        """State plainly whether the model earned its place in the pipeline."""
        if self.recall_delta <= 0.001 and self.exceptions_closed <= 0:
            return ("The model added nothing measurable on this run. On this data the "
                    "deterministic layers are sufficient and layer 3 should be disabled.")
        return (
            f"The model closed {self.exceptions_closed} exception(s) the deterministic "
            f"layers could not, lifting recall by {self.recall_delta:+.3f} at "
            f"{self.full_stack.llm_calls_per_100:.2f} calls per 100 records "
            f"(precision {self.precision_delta:+.3f})."
        )


def layer_breakdown(result: ReconResult) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for f in result.findings:
        out[f.layer.value] += 1
    return dict(out)
