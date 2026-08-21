"""Push the matcher until it breaks, and report exactly where.

Stating your own limits precisely is stronger than claiming you have none. This
walks the engine through escalating conditions and finds the point at which it
stops working - then says which way it failed, which matters more than when.

There are two ways a reconciliation engine can fail under pressure:

  **fail-safe**  - it stops asserting and starts escalating. Match rate falls,
                   exceptions rise, precision holds. The close takes longer and
                   a human does more, but no wrong number is ever signed off.

  **fail-wrong** - it keeps asserting into ambiguity. Match rate looks fine and
                   the findings are quietly incorrect.

The first is survivable, the second is what a payments company cannot tolerate.
The design rule in ``matching.py`` - assert only when unambiguous - is a bet
that the engine fails the first way. This harness is the test of that bet.

The stress knobs are the ones that remove the matcher's easy path:

  * ``ambiguity`` scales the rates that strip identifying references, so more
    batches fall through the UTR join onto amount-and-date matching.
  * ``spacing`` packs batches closer together, so those fallback matches face
    more competing candidates in the same settlement window.

Scaling every defect rate uniformly would be the obvious move and would mostly
add noise the engine already handles. These two are chosen because they attack
the fallback path specifically.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .engine.pipeline import RunConfig, reconcile
from .generator import DefectRates, generate
from .metrics.harness import Scorecard, score


@dataclass(frozen=True)
class StressLevel:
    name: str
    ambiguity: float       # multiplier on reference-stripping defect rates
    spacing: int           # days between batches; smaller = more window overlap

    def rates(self) -> DefectRates:
        base = DefectRates()
        return replace(
            base,
            missing_utr=min(base.missing_utr * self.ambiguity, 1.0),
            narration_drift=min(base.narration_drift * self.ambiguity, 1.0),
            split_settlement=min(base.split_settlement * self.ambiguity, 1.0),
            unrecorded_adjustment=min(base.unrecorded_adjustment * self.ambiguity, 1.0),
        )


LADDER = [
    StressLevel("baseline", 1.0, 2),
    StressLevel("mild", 1.5, 2),
    StressLevel("moderate", 2.5, 1),
    StressLevel("heavy", 4.0, 1),
    StressLevel("severe", 6.0, 0),
    StressLevel("absurd", 6.6, 0),
]


@dataclass
class StressResult:
    """Metrics for one rung, averaged over seeds.

    A dedicated type rather than a reused ``Scorecard``: that class derives
    precision and recall from its per-class table, and a mean of per-class
    tables across seeds is not a per-class table. Averaging the derived numbers
    directly is the honest operation.
    """

    level: StressLevel
    seeds: int
    precision: float
    recall: float
    match_rate: float
    exceptions: float
    false_positives: int
    records: float


@dataclass
class StressReport:
    results: list[StressResult] = field(default_factory=list)

    @property
    def precision_floor(self) -> float:
        return min(r.precision for r in self.results)

    @property
    def broke_at(self) -> StressResult | None:
        """First level where the engine asserted something untrue."""
        for r in self.results:
            if r.false_positives:
                return r
        return None

    def verdict(self) -> str:
        first, last = self.results[0], self.results[-1]
        broke = self.broke_at
        if broke is not None:
            return (
                f"FAIL-WRONG at '{broke.level.name}': precision fell to "
                f"{broke.precision:.3f} and the engine asserted "
                f"{broke.false_positives} finding(s) that are not true. That is "
                f"the failure mode a payments close cannot tolerate."
            )
        return (
            f"FAIL-SAFE across the whole ladder. Precision never left 1.000 - "
            f"zero false findings at any level. Under {last.level.ambiguity:.1f}x "
            f"ambiguity the engine gave up rather than guessed: recall "
            f"{first.recall:.3f} -> {last.recall:.3f}, match rate "
            f"{first.match_rate:.1%} -> {last.match_rate:.1%}, exceptions "
            f"{first.exceptions:.0f} -> {last.exceptions:.0f}. The close gets "
            f"slower and a human does more; no wrong number is ever signed off."
        )


def run_stress(
    seeds: list[int] | None = None,
    n_batches: int = 40,
    ladder: list[StressLevel] | None = None,
) -> StressReport:
    """Walk the ladder, averaging each level over several seeds."""
    seeds = seeds or [301, 302, 303]
    ladder = ladder or LADDER
    report = StressReport()

    for level in ladder:
        cards: list[Scorecard] = []
        for seed in seeds:
            case = generate(
                n_batches=n_batches,
                seed=seed,
                rates=level.rates(),
                batch_spacing_days=level.spacing,
            )
            result = reconcile(case.bundle, config=RunConfig(use_llm=False))
            cards.append(score(case, result))
        report.results.append(_summarise(level, cards))
    return report


def _summarise(level: StressLevel, cards: list[Scorecard]) -> StressResult:
    n = len(cards)
    return StressResult(
        level=level,
        seeds=n,
        precision=sum(c.precision for c in cards) / n,
        recall=sum(c.recall for c in cards) / n,
        match_rate=sum(c.match_rate for c in cards) / n,
        exceptions=sum(c.exceptions for c in cards) / n,
        # Summed, not averaged: one false finding anywhere is the thing we are
        # looking for, and a mean would dilute it below the rounding threshold.
        false_positives=sum(c.fp for c in cards),
        records=sum(c.records for c in cards) / n,
    )
