"""Synthetic three-source data with *injected, labelled* defects.

The whole metrics story rests on this file. Because the generator knows exactly
what it broke and where, precision and recall are computable per defect class
against real ground truth - not eyeballed off a demo run.

Two things matter for the submission:

  * ``seed`` controls everything. The tuning set and the held-out set differ
    only by seed, so "we never tuned on the held-out set" is a checkable claim.
  * Defect *rates* are declared up front in ``DefectRates``, so the expected
    count of each defect class is known before the engine ever runs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal

from .models import (
    CONTRACTED_FEE_RATE,
    GST_RATE,
    BankCredit,
    DefectClass,
    InjectedDefect,
    LedgerEntry,
    Money,
    SettlementRow,
    SourceBundle,
    TxnType,
)


@dataclass(frozen=True)
class BankProfile:
    """A bank's *persistent* behaviour, stable across months.

    This is what makes the learning loop real rather than staged. If timing and
    deductions were re-randomised every month there would be no structure to
    learn, and a rule store would be theatre. Real banks are boringly
    consistent: the same settlement lag, the same narration format, the same
    recurring charge, month after month. That consistency is the thing worth
    learning once and never paying a model to re-derive.
    """

    name: str
    settle_offset_days: int
    narration_template: str
    # A recurring deduction the bank takes off every payout. Absent from the
    # PG settlement report, so it always shows up as an unexplained shortfall
    # until somebody works out what it is - the archetypal learnable exception.
    adjustment_amount: Money | None = None
    adjustment_keyword: str = ""


BANK_PROFILES = [
    BankProfile("HDFC", 1, "NEFT-{utr}-RAZORPAY SOFTWARE PVT LTD"),
    BankProfile("ICICI", 2, "UPI/{utr}/SETTLEMENT/RAZORPAY"),
    BankProfile(
        "AXIS", 0, "IMPS AXIS REF {utr} CR PROC CHG",
        adjustment_amount=Money("250.00"), adjustment_keyword="PROC CHG",
    ),
    BankProfile(
        "SBI", 3, "RTGS RCVD FRM RAZORPAY SOFTWARE UTR {utr} SVC CHG",
        adjustment_amount=Money("500.00"), adjustment_keyword="SVC CHG",
    ),
]

NARRATION_NO_UTR = [
    "NEFT CR-RAZORPAY SOFTWARE PVT LTD-SETTLEMENT",
    "MERCHANT SETTLEMENT CREDIT",
]


@dataclass
class DefectRates:
    """Injection rates. Declared, not discovered.

    Note the two different denominators, which is easy to get wrong:

      * per-transaction  - duplicate_capture, fee_overcharge, missing_ledger_entry
      * per-batch        - missing_utr, timing_shift, split_settlement,
                           unrecorded_adjustment, narration_drift

    Batch-level rates are set an order of magnitude higher because there are
    ~40 batches to a run against ~600 transactions. Rates are chosen so every
    defect class lands enough samples for its precision/recall to mean
    something - a class with three instances has no measurable recall.

    ``cross_cycle_refund`` is conditional on a refund existing at all (~12% of
    transactions), so its effective rate is roughly a quarter of the value here.
    """

    duplicate_capture: float = 0.03
    cross_cycle_refund: float = 0.25
    fee_overcharge: float = 0.02
    missing_utr: float = 0.15
    narration_drift: float = 0.15
    timing_shift: float = 0.25
    split_settlement: float = 0.15
    unrecorded_adjustment: float = 0.10
    missing_ledger_entry: float = 0.02


@dataclass
class GeneratedCase:
    bundle: SourceBundle
    defects: list[InjectedDefect]

    def defects_of(self, cls: DefectClass) -> list[InjectedDefect]:
        return [d for d in self.defects if d.defect_class is cls]


def _money(rng: random.Random, low: int, high: int) -> Money:
    """Amounts with realistic paise, not round numbers - rounding bugs hide there."""
    rupees = rng.randint(low, high)
    paise = rng.choice([0, 0, 25, 50, 49, 99, 75])
    return (Money(rupees) + Money(paise) / Money(100)).quantize(Money("0.01"))


def _fee_for(gross: Money, rate: Decimal = CONTRACTED_FEE_RATE) -> tuple[Money, Money]:
    fee = (gross * rate).quantize(Money("0.01"))
    gst = (fee * GST_RATE).quantize(Money("0.01"))
    return fee, gst


def generate(
    n_batches: int = 12,
    txns_per_batch: tuple[int, int] = (8, 22),
    seed: int = 7,
    rates: DefectRates | None = None,
    start: date = date(2026, 6, 1),
) -> GeneratedCase:
    """Build one reconciliation period across all three sources."""
    rng = random.Random(seed)
    rates = rates or DefectRates()

    ledger: list[LedgerEntry] = []
    settlement: list[SettlementRow] = []
    bank: list[BankCredit] = []
    defects: list[InjectedDefect] = []

    # Refunds deliberately pushed into the *next* cycle are parked here.
    carried_refunds: list[SettlementRow] = []
    counter = 0

    for b in range(n_batches):
        batch_id = f"setl_{seed}_{b:03d}"
        utr = f"UTR{seed}{b:04d}{rng.randint(1000, 9999)}"
        settled_on = start + timedelta(days=b * 2)
        profile = rng.choice(BANK_PROFILES)
        bank_name = profile.name
        rows: list[SettlementRow] = []

        # Refunds carried from the previous cycle land here. This is the
        # cross-cycle defect materialising: the refund's order belongs to an
        # earlier batch but the money comes out of this one.
        for carried in carried_refunds:
            rows.append(
                replace(carried, settlement_batch_id=batch_id, utr=utr, settled_on=settled_on)
            )
        carried_refunds = []

        for _ in range(rng.randint(*txns_per_batch)):
            counter += 1
            txn_id = f"pay_{seed}_{counter:05d}"
            order_id = f"ord_{seed}_{counter:05d}"
            gross = _money(rng, 250, 12000)
            fee, gst = _fee_for(gross)

            # --- fee_overcharge: PG quietly bills above the contracted rate ---
            if rng.random() < rates.fee_overcharge:
                bad_rate = CONTRACTED_FEE_RATE + Decimal(str(rng.choice([0.003, 0.005, 0.01])))
                fee, gst = _fee_for(gross, bad_rate)
                fair_fee, fair_gst = _fee_for(gross)
                defects.append(
                    InjectedDefect(
                        DefectClass.FEE_OVERCHARGE,
                        txn_id,
                        {
                            "charged_rate": str(bad_rate),
                            "contracted_rate": str(CONTRACTED_FEE_RATE),
                            "overcharge": str((fee + gst) - (fair_fee + fair_gst)),
                        },
                    )
                )

            rows.append(
                SettlementRow(
                    txn_id=txn_id,
                    settlement_batch_id=batch_id,
                    utr=utr,
                    txn_type=TxnType.PAYMENT,
                    gross_amount=gross,
                    fee=fee,
                    gst_on_fee=gst,
                    settled_on=settled_on,
                    order_id=order_id,
                )
            )

            # --- missing_ledger_entry: settled money with no order behind it --
            if rng.random() < rates.missing_ledger_entry:
                defects.append(
                    InjectedDefect(DefectClass.MISSING_LEDGER_ENTRY, txn_id, {"amount": str(gross)})
                )
            else:
                ledger.append(
                    LedgerEntry(
                        order_id=order_id,
                        txn_id=txn_id,
                        amount=gross,
                        created_at=settled_on - timedelta(days=2),
                    )
                )

            # --- duplicate_capture: same order captured twice -----------------
            if rng.random() < rates.duplicate_capture:
                counter += 1
                dup_id = f"pay_{seed}_{counter:05d}"
                d_fee, d_gst = _fee_for(gross)
                rows.append(
                    SettlementRow(
                        txn_id=dup_id,
                        settlement_batch_id=batch_id,
                        utr=utr,
                        txn_type=TxnType.PAYMENT,
                        gross_amount=gross,
                        fee=d_fee,
                        gst_on_fee=d_gst,
                        settled_on=settled_on,
                        order_id=order_id,
                    )
                )
                defects.append(
                    InjectedDefect(
                        DefectClass.DUPLICATE_CAPTURE,
                        dup_id,
                        {"original_txn_id": txn_id, "order_id": order_id, "amount": str(gross)},
                    )
                )

            # --- refunds, some of which slip into the next cycle --------------
            if rng.random() < 0.12:
                counter += 1
                refund_id = f"rfnd_{seed}_{counter:05d}"
                refund_amt = (gross / Money(rng.choice([1, 2, 4]))).quantize(Money("0.01"))
                refund = SettlementRow(
                    txn_id=refund_id,
                    settlement_batch_id=batch_id,
                    utr=utr,
                    txn_type=TxnType.REFUND,
                    gross_amount=refund_amt,
                    fee=Money("0.00"),
                    gst_on_fee=Money("0.00"),
                    settled_on=settled_on,
                    order_id=order_id,
                )
                if rng.random() < rates.cross_cycle_refund and b + 1 < n_batches:
                    carried_refunds.append(refund)
                    defects.append(
                        InjectedDefect(
                            DefectClass.CROSS_CYCLE_REFUND,
                            refund_id,
                            {"origin_batch": batch_id, "order_id": order_id},
                        )
                    )
                else:
                    rows.append(refund)

        settlement.extend(rows)
        expected_credit = sum((r.net_amount for r in rows), Money("0.00"))

        # --- unrecorded_adjustment ------------------------------------------
        # Two flavours, and the difference is the whole point of the rule store.
        #
        #   recurring  - this bank deducts the same charge every single payout.
        #                Learnable once, then free forever.
        #   one-off    - a genuine anomaly. Never learnable, always a human's
        #                problem. A system that "learns" these is overfitting.
        if profile.adjustment_amount is not None:
            expected_credit -= profile.adjustment_amount
            defects.append(
                InjectedDefect(
                    DefectClass.UNRECORDED_ADJUSTMENT, batch_id,
                    {
                        "amount": str(profile.adjustment_amount),
                        "bank": bank_name,
                        "recurring": True,
                        "keyword": profile.adjustment_keyword,
                    },
                )
            )
        elif rng.random() < rates.unrecorded_adjustment:
            adj = _money(rng, 50, 900)
            expected_credit -= adj
            defects.append(
                InjectedDefect(
                    DefectClass.UNRECORDED_ADJUSTMENT, batch_id,
                    {"amount": str(adj), "bank": bank_name, "recurring": False},
                )
            )

        # --- missing_utr: settlement report shipped without the UTR field -----
        if rng.random() < rates.missing_utr:
            settlement[-len(rows):] = [replace(r, utr=None) for r in rows]
            defects.append(InjectedDefect(DefectClass.MISSING_UTR, batch_id, {"true_utr": utr}))

        # --- timing_shift: driven by the bank's fixed lag, plus rare jitter ----
        # The lag itself is a property of the bank, not a random event, so it is
        # learnable. The occasional extra day on top is genuine noise and stays
        # unlearnable by construction - the rule store must not absorb it.
        offset = profile.settle_offset_days
        if rng.random() < rates.timing_shift * 0.2:
            offset += rng.randint(1, 2)
        value_date = settled_on + timedelta(days=offset)
        if offset:
            defects.append(
                InjectedDefect(
                    DefectClass.TIMING_SHIFT, batch_id,
                    {"offset_days": offset, "bank": bank_name,
                     "expected_offset": profile.settle_offset_days},
                )
            )

        # --- narration_drift: UTR absent from the narration entirely ----------
        if rng.random() < rates.narration_drift:
            narration = rng.choice(NARRATION_NO_UTR)
            clean_utr = None
            defects.append(InjectedDefect(DefectClass.NARRATION_DRIFT, batch_id, {"true_utr": utr}))
        else:
            narration = profile.narration_template.format(utr=utr, bank=bank_name)
            # Even when the narration carries the UTR, the bank only populates
            # the structured field about half the time.
            clean_utr = utr if rng.random() < 0.55 else None

        # --- split_settlement: one batch arrives as two bank credits ----------
        if rng.random() < rates.split_settlement:
            first = (expected_credit / Money("2")).quantize(Money("0.01"))
            second = expected_credit - first
            for i, part in enumerate((first, second)):
                bank.append(
                    BankCredit(
                        bank_txn_id=f"bank_{seed}_{b:03d}_{i}",
                        credit_amount=part,
                        value_date=value_date + timedelta(days=i),
                        narration=narration,
                        utr=clean_utr,
                    )
                )
            defects.append(
                InjectedDefect(
                    DefectClass.SPLIT_SETTLEMENT, batch_id,
                    {"parts": 2, "total": str(expected_credit)},
                )
            )
        else:
            bank.append(
                BankCredit(
                    bank_txn_id=f"bank_{seed}_{b:03d}_0",
                    credit_amount=expected_credit,
                    value_date=value_date,
                    narration=narration,
                    utr=clean_utr,
                )
            )

    bundle = SourceBundle(
        ledger=ledger, settlement=settlement, bank=bank, period=f"{start:%Y-%m}"
    )
    return GeneratedCase(bundle=bundle, defects=defects)
