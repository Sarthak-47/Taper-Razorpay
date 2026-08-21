"""Train and evaluate the exception-risk model on disjoint seed sets.

The one rule this module exists to enforce: **the seeds used to train are never
the seeds used to report.** A model evaluated on data it was fitted to will show
a flawless reliability curve and predict nothing in production, and it is easy
to do by accident when both sets come from the same generator. So the split is
asserted here rather than left to whoever calls it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..engine.pipeline import RunConfig, reconcile
from ..generator import generate
from .confidence import (
    ExceptionRiskModel,
    auc,
    brier,
    reliability,
    review_budget_curve,
)
from .features import build_dataset

TRAIN_SEEDS = [11, 12, 13, 14, 15, 16, 17, 18]
HOLDOUT_SEEDS = [901, 902, 903, 904]


@dataclass
class RiskReport:
    backend: str
    n_train: int
    n_holdout: int
    positives_train: int
    positives_holdout: int
    brier_model: float
    brier_baseline: float
    auc: float
    reliability: list[tuple[float, float, float, int]] = field(default_factory=list)
    budget: list[tuple[float, float, int]] = field(default_factory=list)
    importances: list[tuple[str, float]] = field(default_factory=list)

    @property
    def skill(self) -> float:
        """Brier skill score against always predicting the base rate.

        Positive means the model beats simply quoting how often batches escalate
        on average. A model that cannot clear this bar is not worth shipping,
        and reporting it stops a good-looking Brier from hiding a useless model.
        """
        if self.brier_baseline <= 0:
            return float("nan")
        return 1 - (self.brier_model / self.brier_baseline)


def collect(seeds: list[int], n_batches: int = 40) -> tuple[list[list[float]], list[int]]:
    """Run the deterministic pipeline over each seed and label every batch."""
    X: list[list[float]] = []
    y: list[int] = []
    for seed in seeds:
        case = generate(n_batches=n_batches, seed=seed)
        result = reconcile(case.bundle, config=RunConfig(use_llm=False))
        Xi, yi, _ = build_dataset(case, result)
        X.extend(Xi)
        y.extend(yi)
    return X, y


def train_and_evaluate(
    train_seeds: list[int] | None = None,
    holdout_seeds: list[int] | None = None,
    n_batches: int = 40,
    seed: int = 0,
    prefer: str = "logistic",
) -> tuple[ExceptionRiskModel, RiskReport]:
    train_seeds = train_seeds or TRAIN_SEEDS
    holdout_seeds = holdout_seeds or HOLDOUT_SEEDS

    overlap = set(train_seeds) & set(holdout_seeds)
    if overlap:
        raise ValueError(
            f"train and holdout seeds overlap on {sorted(overlap)} - every reported "
            "number would be measured on data the model was fitted to"
        )

    X_tr, y_tr = collect(train_seeds, n_batches)
    X_ho, y_ho = collect(holdout_seeds, n_batches)

    model = ExceptionRiskModel().fit(X_tr, y_tr, seed=seed, prefer=prefer)
    probs = model.predict(X_ho)

    base_rate = sum(y_tr) / len(y_tr) if y_tr else 0.0
    report = RiskReport(
        backend=model.backend,
        n_train=len(X_tr),
        n_holdout=len(X_ho),
        positives_train=sum(y_tr),
        positives_holdout=sum(y_ho),
        brier_model=brier(probs, y_ho),
        brier_baseline=brier([base_rate] * len(y_ho), y_ho),
        auc=auc(probs, y_ho),
        reliability=reliability(probs, y_ho),
        budget=review_budget_curve(probs, y_ho),
        importances=model.importances(),
    )
    return model, report
