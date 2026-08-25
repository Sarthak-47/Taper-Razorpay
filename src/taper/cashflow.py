"""The cash position: how much money is actually there.

Reconciliation answers "do the books agree". A controller's next question is
different and more urgent - *what do I actually have* - and the two are not the
same number. Money can be settled and still not arrive, arrive and still be
owed back, or be deducted for a reason nobody has explained.

Nothing here is new detection. Every input is a finding the engine already
produced; this assembles them into the question a CFO asks first.

**Three groups, and they are not slices of one pot.** Conflating them is how a
cash position ends up wrong in the direction that hurts:

  * **In the bank** - credits actually matched to a settlement batch.
  * **Not arriving yet** - settled but uncredited, or withheld pending a
    dispute. Real money, absent from the balance.
  * **Claims** - amounts owed *to* the merchant (unpaid revenue, recoverable
    overcharges) and owed *by* them (duplicate captures awaiting refund). These
    sit against the balance rather than inside it.

A duplicate capture is both received *and* owed back. Reporting it only as
received overstates the position; reporting it only as a liability loses the
cash. It appears in both, labelled, because that is what is true.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .engine.results import ReconResult
from .models import DefectClass, Money


@dataclass
class Line:
    """One line of the position, with what produced it."""

    label: str
    amount: Money
    count: int
    note: str
    direction: str = "neutral"  # inflow | outflow | neutral


@dataclass
class CashPosition:
    period: str = ""
    in_bank: Money = Money("0.00")
    matched_batches: int = 0

    not_arriving: list[Line] = field(default_factory=list)
    claims_in: list[Line] = field(default_factory=list)
    claims_out: list[Line] = field(default_factory=list)
    unreconciled: Money = Money("0.00")
    unreconciled_count: int = 0

    @property
    def withheld_total(self) -> Money:
        return sum((line.amount for line in self.not_arriving), Money("0.00"))

    @property
    def owed_to_us(self) -> Money:
        return sum((line.amount for line in self.claims_in), Money("0.00"))

    @property
    def owed_by_us(self) -> Money:
        return sum((line.amount for line in self.claims_out), Money("0.00"))

    @property
    def net_position(self) -> Money:
        """Cash in the bank, adjusted for what is owed each way.

        Deliberately excludes ``withheld_total``. Money held against a dispute
        is not the merchant's to count until the dispute resolves, and money
        still in transit has not arrived. Both are shown, neither is added -
        counting either would overstate the position in exactly the direction
        that causes an overdraft.
        """
        return self.in_bank + self.owed_to_us - self.owed_by_us

    @property
    def confidence_note(self) -> str:
        if self.unreconciled_count == 0:
            return "Every credit is attributed. This position is complete."
        return (
            f"Rs.{self.unreconciled:,.0f} across {self.unreconciled_count} item(s) "
            f"could not be attributed to a batch. The position is stated without "
            f"them, so treat it as a floor rather than a total."
        )


def _sum_findings(result: ReconResult, defect: DefectClass) -> tuple[Money, int]:
    hits = [f for f in result.findings if f.defect_class is defect]
    return sum((f.money_impact for f in hits), Money("0.00")), len(hits)


def assemble(result: ReconResult, period: str = "") -> CashPosition:
    """Build the position from findings the engine already produced."""
    pos = CashPosition(period=period)

    # --- what actually landed ------------------------------------------
    pos.in_bank = sum((m.credited for m in result.matches), Money("0.00"))
    pos.matched_batches = len(result.matches)

    # --- real money that is not in the balance --------------------------
    held, n_held = _sum_findings(result, DefectClass.CHARGEBACK_HOLD)
    if n_held:
        pos.not_arriving.append(Line(
            "Withheld pending disputes", held, n_held,
            "Deducted from payouts by the gateway. Not lost, not available - "
            "and invisible in the payout total unless somebody looks.",
            "inflow",
        ))

    late, n_late = _sum_findings(result, DefectClass.TIMING_SHIFT)
    if n_late:
        pos.not_arriving.append(Line(
            "Settled late", Money("0.00"), n_late,
            "Batches whose money landed after the settlement date. A timing "
            "fact, not a loss - it moves when cash is available, not how much.",
            "neutral",
        ))

    # --- owed to the merchant -------------------------------------------
    unpaid, n_unpaid = _sum_findings(result, DefectClass.UNSETTLED_REVENUE)
    if n_unpaid:
        pos.claims_in.append(Line(
            "Sold but never paid for", unpaid, n_unpaid,
            "Orders in the ledger with no settlement behind them. Commercially "
            "the most important number here: money that never arrived at all.",
            "inflow",
        ))

    overcharged, n_over = _sum_findings(result, DefectClass.FEE_OVERCHARGE)
    if n_over:
        pos.claims_in.append(Line(
            "Overcharged fees", overcharged, n_over,
            "Billed above the contracted rate. Recoverable from the gateway.",
            "inflow",
        ))

    deducted, n_deducted = _sum_findings(result, DefectClass.UNRECORDED_ADJUSTMENT)
    if n_deducted:
        pos.claims_in.append(Line(
            "Unexplained deductions", deducted, n_deducted,
            "Taken off payouts with nothing in the report to explain them. "
            "Query with the bank, or learn them as a standing charge.",
            "inflow",
        ))

    # --- owed by the merchant -------------------------------------------
    dupes, n_dupes = _sum_findings(result, DefectClass.DUPLICATE_CAPTURE)
    if n_dupes:
        pos.claims_out.append(Line(
            "Duplicate captures", dupes, n_dupes,
            "Customers charged twice. This money is in the bank and is owed "
            "back - counting it as revenue is how a refund run becomes a "
            "surprise.",
            "outflow",
        ))

    # --- what could not be placed ---------------------------------------
    orphans = [e for e in result.exceptions if e.kind == "unclaimed_credit"]
    pos.unreconciled_count = len(orphans)
    pos.unreconciled = sum(
        (Money(str(e.context.get("amount", "0"))) for e in orphans), Money("0.00")
    )
    return pos
