"""Exception-risk model: which batches will need a human, predicted up front.

Two deliberate choices.

**Calibration is the product, not accuracy.** A controller does not act on a
ranking, they act on a threshold - "anything above 0.8, I'll look at myself".
That decision is only sound if a stated 0.8 means the batch really does escalate
about 80% of the time. So the model is scored with Brier and a reliability
curve, not with accuracy, and every raw score is passed through isotonic
regression fitted on a held-out split.

**Nothing here needs a dependency.** Isotonic regression via pool-adjacent-
violators is about thirty lines, and the shipped model is a logistic regression
trained by gradient descent. scikit-learn is an optional extra used only to
reproduce the comparison that led to this choice - gradient boosting was tried
first and measured worse on both ranking and calibration.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .features import FEATURE_NAMES

# ---------------------------------------------------------------------------
# Isotonic calibration - pool adjacent violators
# ---------------------------------------------------------------------------

@dataclass
class IsotonicCalibrator:
    """Maps raw scores to probabilities via a monotone step function.

    Chosen over Platt scaling because it assumes only monotonicity - that a
    higher score never means a lower true probability - rather than a particular
    sigmoid shape. With a tree model and a small validation split that is the
    safer assumption.
    """

    thresholds: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def fit(self, scores: list[float], labels: list[int]) -> IsotonicCalibrator:
        if not scores:
            return self
        pairs = sorted(zip(scores, labels, strict=True))
        xs = [p[0] for p in pairs]
        ys = [float(p[1]) for p in pairs]

        # Pool adjacent violators: repeatedly merge any block whose mean is
        # lower than the block before it, until the sequence is non-decreasing.
        blocks: list[tuple[float, int]] = [(y, 1) for y in ys]
        i = 0
        while i < len(blocks) - 1:
            total_a, n_a = blocks[i]
            total_b, n_b = blocks[i + 1]
            if total_a / n_a > total_b / n_b:
                blocks[i:i + 2] = [(total_a + total_b, n_a + n_b)]
                i = max(i - 1, 0)
            else:
                i += 1

        fitted: list[float] = []
        for total, n in blocks:
            fitted.extend([total / n] * n)

        self.thresholds = xs
        self.values = fitted
        return self

    def predict(self, score: float) -> float:
        """Step-function lookup with linear ends. Clamped to [0, 1]."""
        if not self.thresholds:
            return min(max(score, 0.0), 1.0)
        if score <= self.thresholds[0]:
            return self.values[0]
        if score >= self.thresholds[-1]:
            return self.values[-1]
        lo, hi = 0, len(self.thresholds) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.thresholds[mid] < score:
                lo = mid + 1
            else:
                hi = mid
        return self.values[lo]


# ---------------------------------------------------------------------------
# The shipped model - logistic regression, no dependencies
# ---------------------------------------------------------------------------

@dataclass
class LogisticModel:
    """Plain logistic regression by gradient descent, with L2 and standardisation.

    This is the *shipped* model, not a consolation prize. Gradient boosting was
    tried first and lost on this problem - worse ranking and worse calibration -
    so the default is the one with no dependencies. See ``taper risk --compare``.
    """

    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)
    epochs: int = 400
    lr: float = 0.15
    l2: float = 1e-4

    name = "logistic (no dependencies)"

    def _standardise(self, row: list[float]) -> list[float]:
        return [
            (v - m) / s if s > 1e-9 else 0.0
            for v, m, s in zip(row, self.mean, self.std, strict=True)
        ]

    def fit(self, X: list[list[float]], y: list[int]) -> LogisticModel:
        n, d = len(X), len(X[0])
        self.mean = [sum(r[j] for r in X) / n for j in range(d)]
        self.std = [
            math.sqrt(sum((r[j] - self.mean[j]) ** 2 for r in X) / n) or 1.0
            for j in range(d)
        ]
        Z = [self._standardise(r) for r in X]
        self.weights = [0.0] * d
        self.bias = 0.0

        for _ in range(self.epochs):
            gw = [0.0] * d
            gb = 0.0
            for row, label in zip(Z, y, strict=True):
                p = self._sigmoid(sum(w * v for w, v in zip(self.weights, row, strict=True))
                                  + self.bias)
                err = p - label
                for j in range(d):
                    gw[j] += err * row[j]
                gb += err
            for j in range(d):
                self.weights[j] -= self.lr * (gw[j] / n + self.l2 * self.weights[j])
            self.bias -= self.lr * gb / n
        return self

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-min(z, 60)))
        e = math.exp(max(z, -60))
        return e / (1.0 + e)

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        out = []
        for row in X:
            z = self._standardise(row)
            out.append(self._sigmoid(
                sum(w * v for w, v in zip(self.weights, z, strict=True)) + self.bias
            ))
        return out

    def importances(self) -> list[tuple[str, float]]:
        pairs = list(zip(FEATURE_NAMES, (abs(w) for w in self.weights), strict=True))
        return sorted(pairs, key=lambda p: -p[1])


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

@dataclass
class ExceptionRiskModel:
    """Predicts P(this batch needs a human), calibrated."""

    model: object | None = None
    calibrator: IsotonicCalibrator = field(default_factory=IsotonicCalibrator)
    backend: str = "unfitted"
    _sklearn: bool = False

    def fit(
        self, X: list[list[float]], y: list[int], seed: int = 0, prefer: str = "logistic"
    ) -> ExceptionRiskModel:
        """Fit the model, then fit the calibrator on a *disjoint* split.

        Calibrating on the training rows would produce a beautiful reliability
        curve that means nothing, because the model has already seen those
        labels. The split is enforced here rather than left to the caller.
        """
        rng = random.Random(seed)
        idx = list(range(len(X)))
        rng.shuffle(idx)
        cut = int(len(idx) * 0.75) or 1
        fit_idx, cal_idx = idx[:cut], idx[cut:] or idx[:1]

        Xf = [X[i] for i in fit_idx]
        yf = [y[i] for i in fit_idx]

        if len(set(yf)) < 2:
            # One class in the fit split - nothing learnable. Say so instead of
            # returning a model that always predicts the majority.
            self.backend = "degenerate (single class in training split)"
            self.model = None
            return self

        self.model, self._sklearn = _make_model(seed, prefer)
        self.model.fit(Xf, yf)
        self.backend = (
            "sklearn GradientBoostingClassifier" if self._sklearn else LogisticModel.name
        )

        cal_scores = self._raw([X[i] for i in cal_idx])
        self.calibrator.fit(cal_scores, [y[i] for i in cal_idx])
        return self

    def _raw(self, X: list[list[float]]) -> list[float]:
        if self.model is None:
            return [0.5] * len(X)
        if self._sklearn:
            return [float(p[1]) for p in self.model.predict_proba(X)]
        return self.model.predict_proba(X)

    def predict(self, X: list[list[float]]) -> list[float]:
        return [self.calibrator.predict(s) for s in self._raw(X)]

    def importances(self) -> list[tuple[str, float]]:
        if self.model is None:
            return []
        if self._sklearn:
            pairs = list(zip(FEATURE_NAMES, self.model.feature_importances_, strict=True))
            return sorted(((n, float(v)) for n, v in pairs), key=lambda p: -p[1])
        return self.model.importances()


def _make_model(seed: int, prefer: str = "logistic") -> tuple[object, bool]:
    """Build the underlying model. Logistic is the default because it measured better.

    Gradient boosting was the obvious first choice and lost on this problem -
    worse ranking *and* worse calibration than a standardised logistic
    regression (see ``taper risk --compare``). The signal here is close to
    linear in a couple of strong features, so the extra capacity buys variance
    rather than accuracy on a few hundred rows.

    So the shipped model is the one with no dependencies, and that is a
    measurement rather than a preference. Pass ``prefer="gbm"`` to reproduce the
    comparison.
    """
    if prefer == "gbm":
        try:
            from sklearn.ensemble import GradientBoostingClassifier
        except ImportError:
            return LogisticModel(), False
        return (
            GradientBoostingClassifier(
                n_estimators=120, max_depth=3, learning_rate=0.08, random_state=seed
            ),
            True,
        )
    return LogisticModel(), False


# ---------------------------------------------------------------------------
# Evaluation - calibration first, ranking second
# ---------------------------------------------------------------------------

def brier(probs: list[float], labels: list[int]) -> float:
    """Mean squared error of the probabilities. Lower is better; 0.25 is a coin."""
    if not probs:
        return float("nan")
    return sum((p - y) ** 2 for p, y in zip(probs, labels, strict=True)) / len(probs)


def auc(probs: list[float], labels: list[int]) -> float:
    """Probability a random escalated batch outranks a random clean one."""
    pos = [p for p, y in zip(probs, labels, strict=True) if y == 1]
    neg = [p for p, y in zip(probs, labels, strict=True) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum(
        1.0 if a > b else 0.5 if a == b else 0.0
        for a in pos for b in neg
    )
    return wins / (len(pos) * len(neg))


def reliability(
    probs: list[float], labels: list[int], bins: int = 5
) -> list[tuple[float, float, float, int]]:
    """(bin_mid, mean_predicted, observed_rate, n) for a reliability curve."""
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [
            (p, y) for p, y in zip(probs, labels, strict=True)
            if (lo <= p < hi) or (b == bins - 1 and p == 1.0)
        ]
        if not sel:
            continue
        mean_p = sum(p for p, _ in sel) / len(sel)
        obs = sum(y for _, y in sel) / len(sel)
        out.append(((lo + hi) / 2, mean_p, obs, len(sel)))
    return out


def review_budget_curve(
    probs: list[float], labels: list[int]
) -> list[tuple[float, float, int]]:
    """(fraction reviewed, share of escalations caught, count) by risk rank.

    The operational question is not "is the model accurate" but "if I only have
    time for the riskiest 20% of batches, how much of the mess do I actually
    catch?"
    """
    order = sorted(zip(probs, labels, strict=True), key=lambda t: -t[0])
    total_pos = sum(labels) or 1
    out, caught = [], 0
    for i, (_, y) in enumerate(order, start=1):
        caught += y
        if i % max(1, len(order) // 10) == 0 or i == len(order):
            out.append((i / len(order), caught / total_pos, i))
    return out
