"""Forward cash: when the money that is owed will actually land.

The cash position in ``cashflow.py`` answers *what do I have*. This answers the
next question, which is the one that decides whether payroll clears: **what will
I have, and on which day**.

The whole model is one empirical fact the reconciler already produces for free.
Every matched batch carries two dates - when the gateway said it settled, and
when the bank actually credited it - and the difference between them is a
measurement, not an assumption. Enough of those and you have a distribution.
That is the forecast: settled-but-uncredited batches, pushed forward by the lag
this merchant's banks have actually shown.

**Three commitments, because a forecast is the easiest thing in this repository
to fake.**

*It is a range, not a number.* Lag varies, so an arrival date is reported as
p10/p50/p90 from the observed distribution. A single date implies a confidence
nothing here has earned.

*It is backtested, and the backtest can fail.* ``backtest()`` fits the lag model
on early batches, predicts the later ones, and reports both the median absolute
error in days and the **coverage** - how often reality landed inside the band the
model claimed. A p10-p90 band should contain about 80% of outcomes. If it
contains 45%, the forecast is overconfident, and saying so is the point.

*Money that is not scheduled is not forecast.* Withheld funds resolve when a
dispute resolves, and unsettled revenue has no settlement date at all. Neither
has a knowable arrival day, so neither is smeared across the horizon to make the
total look better. They are reported separately, outside the curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from .engine.results import ReconResult
from .models import DefectClass, Money, SourceBundle

# Below this many observations the distribution is noise wearing a percentile's
# clothes, and the forecast says so instead of quoting one.
MIN_OBSERVATIONS = 8

# The band the forecast claims. Reported alongside measured coverage so the two
# can be compared rather than assumed equal.
BAND = (10, 90)
EXPECTED_COVERAGE = 0.80


def _percentile(values: list[int], pct: int) -> int:
    """Nearest-rank percentile on a small integer sample.

    Deliberately not interpolated: lags are whole days, and inventing a 2.4-day
    lag to make a curve smoother would report a precision the data does not
    have.
    """
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(pct / 100 * len(ordered) + 0.5)))
    return ordered[rank - 1]


@dataclass
class LagModel:
    """How long this merchant's money has actually taken to arrive."""

    observations: list[int] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.observations)

    @property
    def usable(self) -> bool:
        return self.n >= MIN_OBSERVATIONS

    @property
    def median(self) -> int:
        return _percentile(self.observations, 50)

    @property
    def low(self) -> int:
        return _percentile(self.observations, BAND[0])

    @property
    def high(self) -> int:
        return _percentile(self.observations, BAND[1])

    def describe(self) -> str:
        if not self.usable:
            return (
                f"only {self.n} matched batch(es) to learn from - fewer than the "
                f"{MIN_OBSERVATIONS} needed before quoting a percentile"
            )
        return (
            f"T+{self.median} typical, T+{self.low} to T+{self.high} across "
            f"{self.n} matched batch(es)"
        )


def observe_lags(result: ReconResult, bundle: SourceBundle) -> LagModel:
    """Measure settle-to-credit lag on every batch that actually matched.

    Only matched batches are used, and that is the honest choice rather than a
    convenient one: an unmatched batch has no confirmed arrival date, so
    including a guess about it would fit the model to its own assumptions.
    """
    settled_on = {
        row.settlement_batch_id: row.settled_on for row in bundle.settlement
    }
    credited_on = {credit.bank_txn_id: credit.value_date for credit in bundle.bank}

    lags: list[int] = []
    for match in result.matches:
        start = settled_on.get(match.batch_id)
        arrivals = [
            credited_on[txn] for txn in match.bank_txn_ids if txn in credited_on
        ]
        if start and arrivals:
            # The last credit is when the batch was actually whole.
            lags.append((max(arrivals) - start).days)
    return LagModel(observations=[lag for lag in lags if lag >= 0])


@dataclass
class Arrival:
    """One expected movement of money, with the uncertainty on its date."""

    batch_id: str
    amount: Money
    settled_on: date
    earliest: date
    expected: date
    latest: date
    direction: str = "in"


@dataclass
class Unscheduled:
    """Real money with no knowable arrival date."""

    label: str
    amount: Money
    count: int
    why: str


@dataclass
class Forecast:
    as_of: date = date.today()
    horizon_days: int = 14
    lag: LagModel = field(default_factory=LagModel)
    arrivals: list[Arrival] = field(default_factory=list)
    overdue: list[Arrival] = field(default_factory=list)
    unscheduled: list[Unscheduled] = field(default_factory=list)

    @property
    def total_expected(self) -> Money:
        return sum((a.amount for a in self.arrivals), Money("0.00"))

    @property
    def overdue_total(self) -> Money:
        """Money that should already have landed and has not.

        Kept off the forward curve on purpose. A batch settled in June whose
        slowest observed arrival was July is not cash arriving in October - it
        is a batch nobody chased, and forecasting it as income would put money
        in a plan that no longer has a reason to show up.
        """
        return sum((a.amount for a in self.overdue), Money("0.00"))

    def days_overdue(self, arrival: Arrival) -> int:
        return (self.as_of - arrival.latest).days

    @property
    def unscheduled_total(self) -> Money:
        return sum((u.amount for u in self.unscheduled), Money("0.00"))

    def by_day(self) -> list[tuple[date, Money, int]]:
        """The curve: expected inflow per day across the horizon."""
        buckets: dict[date, list[Money]] = {}
        for arrival in self.arrivals:
            buckets.setdefault(arrival.expected, []).append(arrival.amount)
        days = [
            self.as_of + timedelta(days=offset)
            for offset in range(self.horizon_days + 1)
        ]
        return [
            (day, sum(buckets.get(day, []), Money("0.00")), len(buckets.get(day, [])))
            for day in days
        ]

    @property
    def worst_case(self) -> Money:
        """What has landed if every batch takes its slowest observed path."""
        cutoff = self.as_of + timedelta(days=self.horizon_days)
        return sum(
            (a.amount for a in self.arrivals if a.latest <= cutoff), Money("0.00")
        )

    def verdict(self) -> str:
        if not self.arrivals and self.overdue:
            return (
                f"Nothing is due to arrive, but Rs.{self.overdue_total:,.2f} "
                f"across {len(self.overdue)} batch(es) is already past its "
                f"slowest observed arrival and has still not landed. That is a "
                f"collections problem, not a cash-flow one."
            )
        if not self.arrivals:
            return (
                "Nothing is scheduled to arrive. Every settled batch in this "
                "close has already been credited."
            )
        if not self.lag.usable:
            return (
                f"Rs.{self.total_expected:,.2f} is owed across "
                f"{len(self.arrivals)} batch(es), but there were only "
                f"{self.lag.n} matched batches to learn a lag from - too few to "
                f"quote a date. The amount is real; the timing is not forecast."
            )
        return (
            f"Rs.{self.total_expected:,.2f} across {len(self.arrivals)} batch(es), "
            f"typically landing T+{self.lag.median} and between T+{self.lag.low} "
            f"and T+{self.lag.high} - measured across {self.lag.n} matched "
            f"batches, not assumed. If every one takes its slowest observed "
            f"path, Rs.{self.worst_case:,.2f} has arrived by day "
            f"{self.horizon_days}, and that is the number to plan against."
        )


def build(
    result: ReconResult,
    bundle: SourceBundle,
    as_of: date | None = None,
    horizon_days: int = 14,
) -> Forecast:
    """Project settled-but-uncredited money forward onto dates."""
    lag = observe_lags(result, bundle)
    matched = {match.batch_id for match in result.matches}

    settled_on: dict[str, date] = {}
    owed: dict[str, Money] = {}
    for row in bundle.settlement:
        batch = row.settlement_batch_id
        if batch in matched or batch == "UNSETTLED":
            continue
        settled_on.setdefault(batch, row.settled_on)
        settled_on[batch] = min(settled_on[batch], row.settled_on)
        owed[batch] = owed.get(batch, Money("0.00")) + row.net_amount

    anchor = as_of or (
        max(settled_on.values()) if settled_on
        else max((r.settled_on for r in bundle.settlement), default=date.today())
    )

    forecast = Forecast(as_of=anchor, horizon_days=horizon_days, lag=lag)
    for batch, amount in sorted(owed.items()):
        if amount <= Money("0.00"):
            continue
        start = settled_on[batch]
        arrival = Arrival(
            batch_id=batch,
            amount=amount,
            settled_on=start,
            earliest=start + timedelta(days=lag.low),
            expected=start + timedelta(days=lag.median),
            latest=start + timedelta(days=lag.high),
        )
        # Past its slowest observed arrival and still not credited. That is not
        # a forecast, it is an escalation, and putting it on the curve would be
        # the single most misleading thing this module could do.
        if arrival.latest < anchor:
            forecast.overdue.append(arrival)
        else:
            forecast.arrivals.append(arrival)

    forecast.arrivals.sort(key=lambda a: (a.expected, -a.amount))
    forecast.overdue.sort(key=lambda a: a.latest)

    # --- money that is real and has no date ------------------------------
    for defect, label, why in (
        (DefectClass.CHARGEBACK_HOLD, "Withheld pending disputes",
         "Released when the dispute resolves, which is not a date anybody can "
         "predict from a settlement file."),
        (DefectClass.UNSETTLED_REVENUE, "Sold but never settled",
         "No settlement date exists to project from. This is the amount most "
         "worth chasing and the least worth forecasting."),
    ):
        hits = [f for f in result.findings if f.defect_class is defect]
        if hits:
            forecast.unscheduled.append(Unscheduled(
                label=label,
                amount=sum((f.money_impact for f in hits), Money("0.00")),
                count=len(hits),
                why=why,
            ))
    return forecast


# ---------------------------------------------------------------------------
# The part that makes it a forecast rather than a chart
# ---------------------------------------------------------------------------

@dataclass
class Backtest:
    """Fit on early batches, predict later ones, compare against what happened."""

    n_fit: int = 0
    n_tested: int = 0
    errors_days: list[int] = field(default_factory=list)
    inside_band: int = 0

    @property
    def median_error_days(self) -> float:
        if not self.errors_days:
            return float("nan")
        ordered = sorted(abs(e) for e in self.errors_days)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[mid])
        return (ordered[mid - 1] + ordered[mid]) / 2

    @property
    def coverage(self) -> float:
        """Share of outcomes that fell inside the band the model claimed."""
        if not self.n_tested:
            return float("nan")
        return self.inside_band / self.n_tested

    @property
    def honest(self) -> bool:
        """Does the band deliver roughly what it promises?

        A p10-p90 band should hold about 80% of outcomes. Materially under that
        and the forecast is overconfident - which is worse than a wide band,
        because somebody plans against it.
        """
        return self.coverage >= EXPECTED_COVERAGE - 0.15

    def verdict(self) -> str:
        if not self.n_tested:
            return "Not enough matched batches to hold any back for testing."
        line = (
            f"Fitted on {self.n_fit} batch(es), tested on {self.n_tested}. "
            f"Median error {self.median_error_days:.1f} day(s). The p10-p90 band "
            f"held {self.coverage:.0%} of outcomes against {EXPECTED_COVERAGE:.0%} "
            f"claimed."
        )
        if self.honest:
            return line + " The band is honest about what it does not know."
        return (
            line + " That is overconfident: the band is narrower than reality, "
            "and a plan built on it would be late."
        )


def backtest(result: ReconResult, bundle: SourceBundle, fit_share: float = 0.6) -> Backtest:
    """Split matched batches by settlement date, fit on the earlier ones.

    Split by *date* rather than at random, because that is the only split that
    matches how the forecast is used: you always predict forward from what you
    have already seen, never from a sample that includes the future.
    """
    settled_on = {row.settlement_batch_id: row.settled_on for row in bundle.settlement}
    credited_on = {credit.bank_txn_id: credit.value_date for credit in bundle.bank}

    observed: list[tuple[date, int]] = []
    for match in result.matches:
        start = settled_on.get(match.batch_id)
        arrivals = [c for txn in match.bank_txn_ids if (c := credited_on.get(txn))]
        if start and arrivals:
            lag = (max(arrivals) - start).days
            if lag >= 0:
                observed.append((start, lag))

    observed.sort(key=lambda pair: pair[0])
    split = int(len(observed) * fit_share)
    fit, test = observed[:split], observed[split:]

    report = Backtest(n_fit=len(fit), n_tested=len(test))
    if len(fit) < MIN_OBSERVATIONS or not test:
        report.n_tested = 0
        return report

    model = LagModel(observations=[lag for _, lag in fit])
    for _, actual in test:
        report.errors_days.append(actual - model.median)
        if model.low <= actual <= model.high:
            report.inside_band += 1
    return report
