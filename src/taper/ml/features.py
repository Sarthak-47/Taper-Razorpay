"""Features for predicting which batches will need a human.

Everything here is computed from the **raw three sources only** - settlement
rows, bank credits, ledger entries - before the matcher runs. Nothing derived
from a matching decision is allowed in, because a feature that encodes the
outcome makes the model look brilliant and predict nothing.

The prediction target is deliberately not "is this finding correct". The
deterministic layers run at precision 1.000, so that label has one class and
nothing to learn. What genuinely varies, and what a controller genuinely wants
to know before the close starts, is: **which batches are going to land on my
desk?**
"""

from __future__ import annotations

import math

from ..engine.matching import (
    AMOUNT_TOLERANCE,
    MAX_SETTLEMENT_LAG,
    _has_any_reference,
    extract_utr,
)
from ..models import BankCredit, Money, SourceBundle, TxnType

FEATURE_NAMES = [
    "n_txns",
    "n_payments",
    "n_refunds",
    "refund_ratio",
    "n_holds",
    "hold_ratio",
    "log_deductions",
    "log_net",
    "log_max_txn",
    "settlement_has_utr",
    "n_credits_in_window",
    "any_window_utr_parseable",
    "any_window_reference_present",
    "exact_amount_available",
    "closest_amount_gap_log",
    "window_date_spread",
]


def _log1p_money(m: Money) -> float:
    return math.log1p(float(abs(m)))


def batch_features(bundle: SourceBundle, batch_id: str) -> dict[str, float]:
    """Describe one settlement batch using only what the sources say."""
    rows = [r for r in bundle.settlement if r.settlement_batch_id == batch_id]
    if not rows:
        return dict.fromkeys(FEATURE_NAMES, 0.0)

    payments = [r for r in rows if r.txn_type is TxnType.PAYMENT]
    refunds = [r for r in rows if r.txn_type is TxnType.REFUND]
    # Holds and refunds both pull money back out of the payout, so a batch full
    # of them looks short before anything has gone wrong. Without these the
    # model reads a legitimately reduced payout as a matching problem - which
    # is what happened when chargeback holds entered the data and AUC fell from
    # 0.885 to 0.803.
    holds = [r for r in rows if r.txn_type is TxnType.CHARGEBACK_HOLD]
    deductions = sum((r.gross_amount for r in refunds + holds), Money("0.00"))
    net = sum((r.net_amount for r in rows), Money("0.00"))
    settled_on = rows[0].settled_on
    utr = next((r.utr for r in rows if r.utr), None)

    window: list[BankCredit] = [
        c for c in bundle.bank
        if settled_on <= c.value_date <= settled_on + MAX_SETTLEMENT_LAG
    ]

    gaps = [abs(c.credit_amount - net) for c in window]
    closest = min(gaps) if gaps else Money("999999")
    spread = 0
    if window:
        spread = (max(c.value_date for c in window) - min(c.value_date for c in window)).days

    return {
        "n_txns": float(len(rows)),
        "n_payments": float(len(payments)),
        "n_refunds": float(len(refunds)),
        "refund_ratio": len(refunds) / len(rows) if rows else 0.0,
        "n_holds": float(len(holds)),
        "hold_ratio": len(holds) / len(rows) if rows else 0.0,
        "log_deductions": _log1p_money(deductions),
        "log_net": _log1p_money(net),
        "log_max_txn": _log1p_money(max((r.gross_amount for r in rows), default=Money("0"))),
        "settlement_has_utr": 1.0 if utr else 0.0,
        "n_credits_in_window": float(len(window)),
        "any_window_utr_parseable": 1.0 if any(extract_utr(c) for c in window) else 0.0,
        "any_window_reference_present": (
            1.0 if any(_has_any_reference(c.narration) for c in window) else 0.0
        ),
        # The single strongest signal, and reported as such rather than hidden:
        # if some credit in the window already equals the expected payout, the
        # batch almost certainly matches without help.
        "exact_amount_available": 1.0 if closest <= AMOUNT_TOLERANCE else 0.0,
        "closest_amount_gap_log": _log1p_money(closest),
        "window_date_spread": float(spread),
    }


def vectorise(features: dict[str, float]) -> list[float]:
    return [features.get(name, 0.0) for name in FEATURE_NAMES]


def build_dataset(case, result) -> tuple[list[list[float]], list[int], list[str]]:
    """One row per settlement batch; label 1 if it required human review.

    The label comes from the pipeline's own exception list, so it is exactly the
    thing a controller cares about - not a proxy for it.
    """
    # Label only batches the pipeline actually escalated as unmatched.
    #
    # An earlier version also attributed every *unclaimed bank credit* back to
    # any batch sharing its settlement window. That looked like better coverage
    # and was in fact label noise: an orphan credit sits in the window of
    # several perfectly clean batches, so it marked them all as needing review.
    # Roughly a fifth of the positive labels were batches that never escalated.
    escalated: set[str] = {
        exc.subject_id for exc in result.exceptions if exc.kind == "unmatched_batch"
    }

    batch_dates = {
        r.settlement_batch_id: r.settled_on for r in case.bundle.settlement
    }

    X: list[list[float]] = []
    y: list[int] = []
    ids: list[str] = []
    for batch_id in sorted(batch_dates):
        X.append(vectorise(batch_features(case.bundle, batch_id)))
        y.append(1 if batch_id in escalated else 0)
        ids.append(batch_id)
    return X, y, ids
