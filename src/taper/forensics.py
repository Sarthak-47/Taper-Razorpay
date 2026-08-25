"""First-digit analysis: does this population of amounts look like real money?

Every other check here asks whether a *transaction* is wrong. This one asks
whether a *population* was lived or authored - a different question, answered by
a technique forensic accountants have used for decades and which needs no model
at all.

Genuine financial amounts follow **Benford's law**: leading digit 1 about 30% of
the time, decaying to under 5% for 9. It falls out of amounts spanning several
orders of magnitude, which real payment volume does - many small transactions,
a long tail of large ones. Invented figures do not. A person typing plausible
totals reaches for 5,000 far more often than chance and almost never for
1,043.75, so fabricated sets skew round and favour middle digits.

**Chi-square is reported and never used to decide.** It scales with sample size,
so on a few thousand rows it flags deviations far too small to matter - the
classic way this technique is misapplied.

**Nigrini's MAD bands are reported and also do not decide.** They are the
recognised vocabulary, so they appear in every profile, but they were
calibrated on datasets of many thousands. At the few hundred rows a monthly
payment channel actually produces, sampling noise alone lands near 0.010 and
regularly tips past the 0.015 "nonconformity" line. Using them as the test
produced 21 false alarms across 30 clean periods - precision 0.30.

**What decides is the null distribution at the observed sample size.** Draw
from Benford at exactly this n, many times, and take the 99th percentile of the
resulting MADs. A segment is flagged only when it exceeds that by a margin, so
deviation is judged against what chance produces *here* rather than against a
constant borrowed from a much larger study. Same data, same technique:
precision 1.00 with zero false alarms.

**This is a screening signal and never proof.** A nonconforming segment earns a
human's attention, not an accusation - which is exactly how the technique is
used in practice, and why it produces exceptions here rather than findings.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field

from .models import Money

# P(first digit = d) = log10(1 + 1/d)
BENFORD = {d: math.log10(1 + 1 / d) for d in range(1, 10)}

# Nigrini's conformance bands for the mean absolute deviation of first digits.
# These are the published forensic-accounting thresholds, not tuned here.
MAD_CLOSE = 0.006
MAD_ACCEPTABLE = 0.012
MAD_MARGINAL = 0.015

# Below this the test says nothing. Benford needs a few hundred observations;
# running it on a handful of rows produces confident noise, which is worse than
# no answer at all.
MIN_SAMPLE = 150

# How far past the 99th percentile of chance a population must sit before it is
# worth a human's time. 1.0 would flag one clean segment in a hundred; the
# margin buys quiet at the cost of only catching larger fabrications.
EXCESS_TO_FLAG = 1.5

# Chi-square critical value, 8 degrees of freedom, p = 0.05. Reported for
# completeness and explicitly not used to decide anything.
CHI2_CRITICAL_P05 = 15.507

# Null-distribution thresholds are deterministic per sample size, so they are
# computed once and reused. Keyed by every input that changes the answer.
_NULL_CACHE: dict[tuple, float] = {}


@dataclass
class DigitProfile:
    """First-digit distribution of one population of amounts."""

    segment: str
    n: int
    observed: dict[int, float] = field(default_factory=dict)
    mad: float = 0.0
    chi_square: float = 0.0
    # What chance alone produces at this sample size. Set by `profile`.
    null_mad: float = 0.0

    @property
    def band(self) -> str:
        """Nigrini's published band. Reported, but not what decides anything."""
        if self.n < MIN_SAMPLE:
            return "too few"
        if self.mad < MAD_CLOSE:
            return "close"
        if self.mad < MAD_ACCEPTABLE:
            return "acceptable"
        if self.mad < MAD_MARGINAL:
            return "marginal"
        return "nonconformity"

    @property
    def excess(self) -> float:
        """How far past chance this population sits, as a multiple."""
        return self.mad / self.null_mad if self.null_mad else 0.0

    @property
    def verdict(self) -> str:
        if self.n < MIN_SAMPLE:
            return "too few"
        if self.mad <= self.null_mad:
            return "within chance"
        return "nonconformity" if self.excess >= EXCESS_TO_FLAG else "elevated"

    @property
    def flagged(self) -> bool:
        return self.verdict == "nonconformity"

    def worst_digits(self, top: int = 3) -> list[tuple[int, float, float]]:
        """(digit, observed share, expected share), largest excess first."""
        gaps = [
            (d, self.observed.get(d, 0.0), BENFORD[d])
            for d in range(1, 10)
        ]
        return sorted(gaps, key=lambda g: -(g[1] - g[2]))[:top]


def null_threshold(n: int, percentile: float = 0.99, trials: int = 400,
                   seed: int = 0) -> float:
    """The MAD a *conforming* population of this size reaches by chance alone.

    Nigrini's fixed bands are the recognised vocabulary and are reported as
    such, but they were calibrated on datasets of many thousands. At the few
    hundred rows a monthly payment channel actually produces, sampling noise
    alone lands around 0.010 and regularly tips past the 0.015 "nonconformity"
    line - which is how this screen produced 21 false alarms across 30 clean
    periods on its first run, a precision of 0.30. A screen that cries wolf
    seven times in ten is worse than no screen.

    So the decision threshold is derived rather than fixed: draw from Benford
    at this exact sample size many times, and take a high percentile of the
    resulting MADs. Deviation is then measured against what chance produces
    *here*, not against a constant borrowed from a much larger study.
    """
    key = (n, percentile, trials, seed)
    cached = _NULL_CACHE.get(key)
    if cached is not None:
        return cached

    rng = random.Random(seed)
    digits = list(range(1, 10))
    weights = [BENFORD[d] for d in digits]

    mads: list[float] = []
    for _ in range(trials):
        sample = rng.choices(digits, weights=weights, k=n)
        counts = Counter(sample)
        mads.append(
            sum(abs(counts.get(d, 0) / n - BENFORD[d]) for d in digits) / 9
        )
    mads.sort()
    threshold = mads[min(int(percentile * trials), trials - 1)]
    _NULL_CACHE[key] = threshold
    return threshold


def _first_digit(amount: Money) -> int | None:
    for ch in str(abs(amount)):
        if ch.isdigit() and ch != "0":
            return int(ch)
        if ch not in "0.":
            break
    return None


def profile(amounts: list[Money], segment: str = "all") -> DigitProfile:
    """Measure one population against Benford."""
    digits = [d for d in (_first_digit(a) for a in amounts if a > 0) if d]
    n = len(digits)
    prof = DigitProfile(segment=segment, n=n)
    if n == 0:
        return prof

    counts = Counter(digits)
    prof.observed = {d: counts.get(d, 0) / n for d in range(1, 10)}
    prof.mad = sum(abs(prof.observed[d] - BENFORD[d]) for d in range(1, 10)) / 9
    prof.chi_square = sum(
        n * (prof.observed[d] - BENFORD[d]) ** 2 / BENFORD[d] for d in range(1, 10)
    )
    if n >= MIN_SAMPLE:
        prof.null_mad = null_threshold(n)
    return prof


def profile_by_segment(rows, key, amount_of) -> list[DigitProfile]:
    """Profile each segment separately.

    Segmenting is the whole technique in practice. Fabrication is normally
    concentrated - one channel, one operator, one period - and averaging it
    into a large honest population hides it. Testing each segment is what turns
    a blunt aggregate into something that can actually point somewhere.
    """
    buckets: dict[str, list[Money]] = {}
    for row in rows:
        buckets.setdefault(str(key(row)), []).append(amount_of(row))
    return sorted(
        (profile(v, segment=k) for k, v in buckets.items()),
        key=lambda p: -p.mad,
    )


def describe(prof: DigitProfile) -> str:
    """A reason a controller can act on, without overclaiming."""
    worst = ", ".join(
        f"{d} at {obs:.0%} against {exp:.0%} expected"
        for d, obs, exp in prof.worst_digits(2)
    )
    return (
        f"First-digit distribution for {prof.segment} does not look like naturally "
        f"occurring amounts: MAD {prof.mad:.4f} over {prof.n} rows, {prof.excess:.1f}x "
        f"what chance produces at this sample size ({prof.null_mad:.4f}), and in "
        f"Nigrini's terms {prof.band}. Most skewed: {worst}. This is a "
        f"screening signal, not proof - genuine causes include price lists, "
        f"fixed-fee products and rounding policy. It says look here, nothing more."
    )
