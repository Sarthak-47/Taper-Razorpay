"""Core domain model.

Three sources of truth that never agree:

  1. ``SettlementRow``  - the PG settlement report (per-transaction, with fees)
  2. ``BankCredit``     - the bank statement (one lump credit per batch, messy narration)
  3. ``LedgerEntry``    - the merchant's own order/ledger system

Reconciliation closes two loops:

  A. ledger  <-> settlement   did every order we think we sold actually settle?
  B. settlement <-> bank      does the netted batch equal the money that landed?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Any

# Money is Decimal everywhere. Never float. A cent of drift across 500 records
# is the difference between a clean close and a week of manual hunting.
Money = Decimal

CONTRACTED_FEE_RATE = Decimal("0.02")   # 2% of gross
GST_RATE = Decimal("0.18")              # 18% GST on the fee itself
AMOUNT_TOLERANCE = Decimal("1.00")      # +/- Rs.1 rounding tolerance


class TxnType(StrEnum):
    PAYMENT = "payment"
    REFUND = "refund"
    ADJUSTMENT = "adjustment"
    CHARGEBACK_HOLD = "chargeback_hold"


class DefectClass(StrEnum):
    """Defects the generator injects, and the engine is scored on catching.

    Every injected defect carries ground truth, so precision/recall is
    computable per class rather than as a single blended number.
    """

    DUPLICATE_CAPTURE = "duplicate_capture"
    CROSS_CYCLE_REFUND = "cross_cycle_refund"
    FEE_OVERCHARGE = "fee_overcharge"
    MISSING_UTR = "missing_utr"
    NARRATION_DRIFT = "narration_drift"
    TIMING_SHIFT = "timing_shift"
    SPLIT_SETTLEMENT = "split_settlement"
    UNRECORDED_ADJUSTMENT = "unrecorded_adjustment"
    MISSING_LEDGER_ENTRY = "missing_ledger_entry"


@dataclass(frozen=True)
class LedgerEntry:
    """What the merchant's own system believes happened."""

    order_id: str
    txn_id: str
    amount: Money
    created_at: date
    status: str = "captured"


@dataclass(frozen=True)
class SettlementRow:
    """One line of the PG settlement report."""

    txn_id: str
    settlement_batch_id: str
    utr: str | None
    txn_type: TxnType
    gross_amount: Money
    fee: Money
    gst_on_fee: Money
    settled_on: date
    order_id: str | None = None
    # Real merchants are on a rate card, not one rate: UPI, domestic cards and
    # international cards are all priced differently. The settlement report
    # names the method but never the contracted rate for it, which is why a
    # legitimately higher fee is indistinguishable from an overcharge until
    # somebody tells the system what the rate card says.
    method: str = "card"

    @property
    def net_amount(self) -> Money:
        """Signed contribution of this row to the bank credit for its batch."""
        if self.txn_type is TxnType.PAYMENT:
            return self.gross_amount - self.fee - self.gst_on_fee
        # Refunds, holds and adjustments all pull money back out of the payout.
        return -(self.gross_amount)

    @property
    def expected_fee(self) -> Money:
        """Fee the contract says should have been charged, recomputed independently."""
        if self.txn_type is not TxnType.PAYMENT:
            return Money("0.00")
        return (self.gross_amount * CONTRACTED_FEE_RATE).quantize(Money("0.01"))

    @property
    def expected_gst(self) -> Money:
        return (self.expected_fee * GST_RATE).quantize(Money("0.01"))


@dataclass(frozen=True)
class BankCredit:
    """One credit line on the bank statement.

    ``narration`` is deliberately messy - the UTR may be embedded in free text,
    prefixed, truncated or absent. Parsing it is the one place an LLM genuinely
    beats a regex, which is why it lives at layer 2 and not layer 0.
    """

    bank_txn_id: str
    credit_amount: Money
    value_date: date
    narration: str
    utr: str | None = None  # populated only when the bank gave us a clean field


@dataclass
class SourceBundle:
    """Everything the agent is handed for one reconciliation run."""

    ledger: list[LedgerEntry] = field(default_factory=list)
    settlement: list[SettlementRow] = field(default_factory=list)
    bank: list[BankCredit] = field(default_factory=list)
    period: str = ""

    def __len__(self) -> int:
        return len(self.ledger) + len(self.settlement) + len(self.bank)


@dataclass(frozen=True)
class InjectedDefect:
    """Ground truth. The generator knows what it broke; the engine must find it."""

    defect_class: DefectClass
    subject_id: str          # txn_id / batch_id / bank_txn_id the defect attaches to
    detail: dict[str, Any] = field(default_factory=dict)
