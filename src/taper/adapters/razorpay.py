"""Read Razorpay's actual settlement recon schema.

Everything else in this repository reconciles data it invented. This module
reads the shape Razorpay really publishes, so the engine can be pointed at a
file a merchant genuinely downloads rather than only at the generator's idea of
one.

Field names and semantics are taken from the public API documentation for the
settlement recon report:

    https://razorpay.com/docs/api/settlements/fetch-recon/

**No real merchant data is used anywhere in this project**, and none is needed
to make this worth having: the schema is public, and reading it correctly is a
separate problem from having somebody's transactions.

Four things about this format would silently corrupt a close if the generic CSV
loader were pointed at it. They are the reason this file exists.

**Amounts are in currency subunits.** ``amount: 150000`` is one thousand five
hundred rupees, not a hundred and fifty thousand. The generic loader reads the
integer at face value and produces a close that is wrong by a factor of a
hundred, with no error and no warning - every total plausible, every total
false. Conversion is exact: an integer number of paise divided by exactly 100,
never a float.

**Timestamps are Unix epoch seconds**, not dates. ``settled_at: 1748563200`` is
not a date string in any format the generic parser tries, so those rows would be
rejected - which at least fails loudly, unlike the paise.

**``on_hold`` and ``settled`` are load-bearing.** A row can appear in the recon
report and not be money you have. ``on_hold`` is the gateway withholding against
a dispute; ``settled: false`` is revenue recognised but not paid. Both map onto
defect classes this engine already reports, and dropping them would turn money
that has not arrived into money that has.

**``type`` includes ``transfer``**, which has no counterpart in a plain
settlement model. Route transfers move money to a linked account: real, and not
a payment. It is mapped to an adjustment and flagged, rather than guessed at.
"""

from __future__ import annotations

import csv
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ..io import LoadReport
from ..models import Money, SettlementRow, TxnType

# The subunit divisor for INR. Named rather than inlined because the whole
# correctness of this adapter rests on it being applied exactly once.
SUBUNITS_PER_RUPEE = Decimal("100")

# Razorpay's transaction types, mapped onto this engine's. `transfer` is a Route
# movement to a linked account - it is not a customer payment and it is not a
# refund, so it is carried as an adjustment rather than forced into either.
TYPE_MAP = {
    "payment": TxnType.PAYMENT,
    "refund": TxnType.REFUND,
    "adjustment": TxnType.ADJUSTMENT,
    "transfer": TxnType.ADJUSTMENT,
}

TRUTHY = {"true", "1", "yes", "y", "t"}


def _paise(raw: str | None, field: str, line: int) -> Money:
    """Currency subunits to rupees, exactly.

    Integer paise over an exact Decimal 100. No float ever touches a rupee
    amount, so 1 is 0.01 and not 0.010000000000000000208166817117216851.
    """
    text = (raw or "").strip()
    if not text:
        return Money("0.00")
    try:
        subunits = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"line {line}: {field} is not a number: {text!r}") from exc
    if subunits != subunits.to_integral_value():
        # A fractional paisa means somebody has already divided by 100 - the
        # file is not in subunits and reading it as though it were would be
        # wrong by a factor of a hundred in the other direction.
        raise ValueError(
            f"line {line}: {field} is {text}, which is not a whole number of "
            "paise - this file may already be in rupees"
        )
    return (subunits / SUBUNITS_PER_RUPEE).quantize(Money("0.01"))


def _epoch(raw: str | None, field: str, line: int) -> date:
    """Unix seconds to a date, in UTC.

    UTC on purpose and worth stating: reading these in local time moves a
    settlement across a date boundary for anything posted near midnight, which
    silently changes which close it belongs to.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError(f"line {line}: {field} is empty")
    try:
        seconds = int(Decimal(text))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"line {line}: {field} is not a timestamp: {text!r}") from exc
    return datetime.fromtimestamp(seconds, tz=UTC).date()


def _flag(raw: str | None) -> bool:
    return (raw or "").strip().lower() in TRUTHY


def load_recon(path: Path) -> tuple[list[SettlementRow], LoadReport]:
    """Load a Razorpay settlement recon export into settlement rows.

    Rows fail individually with their line number, as everywhere else here: one
    malformed amount on line 400 must not cost the other 399.
    """
    report = LoadReport()
    rows: list[SettlementRow] = []

    with path.open(newline="", encoding="utf-8-sig") as handle:
        for line, raw in enumerate(csv.DictReader(handle), start=2):
            record = {(k or "").strip().lower(): (v or "") for k, v in raw.items()}
            try:
                rows.append(_row(record, line))
            except ValueError as exc:
                report.errors.append(str(exc))

    report.loaded = len(rows)
    return rows, report


def _row(record: dict[str, str], line: int) -> SettlementRow:
    kind = record.get("type", "").strip().lower()
    if kind not in TYPE_MAP:
        raise ValueError(f"line {line}: unknown transaction type {kind!r}")

    txn_type = TYPE_MAP[kind]
    # A held row is money the gateway has kept back against a dispute. It is in
    # the report and it is not in the bank, which is exactly the distinction
    # CHARGEBACK_HOLD exists to carry.
    if _flag(record.get("on_hold")):
        txn_type = TxnType.CHARGEBACK_HOLD

    settlement_id = record.get("settlement_id", "").strip()
    settled = _flag(record.get("settled"))
    if not settlement_id:
        # Recognised but not paid. Kept, with a synthetic batch id that reads as
        # what it is, because dropping the row would turn revenue that never
        # arrived into revenue that was never sold.
        settlement_id = "UNSETTLED" if not settled else ""
        if not settlement_id:
            raise ValueError(f"line {line}: settled row carries no settlement_id")

    entity_id = record.get("entity_id", "").strip()
    if not entity_id:
        raise ValueError(f"line {line}: entity_id is empty")

    # `amount` is the total; `credit` and `debit` are its signed halves. Prefer
    # the explicit total and fall back to whichever half is populated.
    gross_raw = record.get("amount") or record.get("credit") or record.get("debit")

    when = record.get("settled_at") or record.get("created_at")
    return SettlementRow(
        txn_id=entity_id,
        settlement_batch_id=settlement_id,
        utr=(record.get("settlement_utr", "").strip() or None),
        txn_type=txn_type,
        gross_amount=_paise(gross_raw, "amount", line),
        fee=_paise(record.get("fee"), "fee", line),
        gst_on_fee=_paise(record.get("tax"), "tax", line),
        settled_on=_epoch(when, "settled_at", line),
        order_id=(record.get("order_id", "").strip() or None),
        method=(record.get("method", "").strip().lower() or "card"),
    )


# ---------------------------------------------------------------------------
# The other direction, for fixtures and for the round-trip test
# ---------------------------------------------------------------------------

RECON_COLUMNS = [
    "entity_id", "type", "debit", "credit", "amount", "currency", "fee", "tax",
    "on_hold", "settled", "created_at", "settled_at", "settlement_id",
    "posted_at", "description", "notes", "payment_id", "settlement_utr",
    "order_id", "order_receipt", "method", "card_network", "card_issuer",
    "card_type", "dispute_id",
]


def write_recon(rows: list[SettlementRow], path: Path) -> Path:
    """Express settlement rows in Razorpay's recon schema.

    Exists so the adapter can be tested against a file in the real shape, and
    so the repository can ship one. Amounts go back out in paise, which is what
    makes the round trip a real check: if either direction applied the divisor
    twice, or not at all, the reconciled close would not match.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECON_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            held = row.txn_type is TxnType.CHARGEBACK_HOLD
            unsettled = row.settlement_batch_id == "UNSETTLED"
            kind = "payment" if held else row.txn_type.value
            epoch = int(
                datetime(
                    row.settled_on.year, row.settled_on.month, row.settled_on.day,
                    tzinfo=UTC,
                ).timestamp()
            )
            amount = int((row.gross_amount * SUBUNITS_PER_RUPEE).to_integral_value())
            writer.writerow({
                "entity_id": row.txn_id,
                "type": kind,
                "debit": 0 if row.txn_type is TxnType.PAYMENT else amount,
                "credit": amount if row.txn_type is TxnType.PAYMENT else 0,
                "amount": amount,
                "currency": "INR",
                "fee": int((row.fee * SUBUNITS_PER_RUPEE).to_integral_value()),
                "tax": int((row.gst_on_fee * SUBUNITS_PER_RUPEE).to_integral_value()),
                "on_hold": "true" if held else "false",
                "settled": "false" if unsettled else "true",
                "created_at": epoch,
                "settled_at": epoch,
                "settlement_id": "" if unsettled else row.settlement_batch_id,
                "posted_at": epoch,
                "description": f"{row.txn_type.value} settled",
                "notes": "",
                "payment_id": row.txn_id,
                "settlement_utr": row.utr or "",
                "order_id": row.order_id or "",
                "order_receipt": "",
                "method": row.method,
                "card_network": "",
                "card_issuer": "",
                "card_type": "",
                "dispute_id": "",
            })
    return path
