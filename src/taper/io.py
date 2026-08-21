"""Read and write the three sources as CSV.

Until this existed the engine could only reconcile data it had generated itself,
which makes it a simulation rather than a tool. A reviewer's first fair question
is "can it read my settlement file", and the answer had to be yes.

Two things this file takes seriously, both learned from the shape of real
finance exports rather than invented:

  * **Headers drift.** The same column is ``utr``, ``UTR``, ``bank_reference``
    or ``Bank Ref No`` depending on who exported it and when. A loader that
    demands one spelling fails on the second file it ever sees, so columns are
    resolved through an alias table.

  * **Rows fail individually.** One malformed amount in row 400 must not lose
    the other 399. Bad rows are collected with their line number and reported,
    because silently dropping a payment is how a reconciliation tool produces a
    confidently wrong total.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import InvalidOperation
from pathlib import Path
from typing import Any

from .models import (
    BankCredit,
    LedgerEntry,
    Money,
    SettlementRow,
    SourceBundle,
    TxnType,
)

# Column aliases seen across gateway and bank exports. First match wins.
ALIASES: dict[str, tuple[str, ...]] = {
    "txn_id": ("txn_id", "transaction_id", "payment_id", "id", "entity_id"),
    "settlement_batch_id": ("settlement_batch_id", "settlement_id", "batch_id", "payout_id"),
    "utr": ("utr", "bank_reference", "bank_ref_no", "reference", "rrn"),
    "txn_type": ("txn_type", "type", "entity_type", "transaction_type"),
    "gross_amount": ("gross_amount", "amount", "gross", "transaction_amount"),
    "fee": ("fee", "fees", "commission", "mdr"),
    "gst_on_fee": ("gst_on_fee", "gst", "tax", "gst_amount"),
    "settled_on": ("settled_on", "settlement_date", "settled_at", "date"),
    "order_id": ("order_id", "order", "receipt", "merchant_order_id"),
    "method": ("method", "payment_method", "instrument"),
    "bank_txn_id": ("bank_txn_id", "statement_id", "txn_ref", "id"),
    "credit_amount": ("credit_amount", "credit", "amount", "deposit"),
    "value_date": ("value_date", "date", "posting_date", "txn_date"),
    "narration": ("narration", "description", "particulars", "remarks"),
    "created_at": ("created_at", "date", "order_date", "created"),
    "status": ("status", "state"),
}

DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d-%b-%Y")


@dataclass
class LoadReport:
    """What loaded, what did not, and why."""

    loaded: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        if self.ok:
            return f"{self.loaded} row(s), no errors"
        return f"{self.loaded} row(s), {len(self.errors)} rejected"


def _normalise(header: str) -> str:
    """Reduce a header to a comparable key.

    Exports write the same column as ``value_date``, ``Value Date``,
    ``Posting Date`` and ``bank-ref-no``. Lowercasing alone is not enough -
    separators differ too - so spaces, hyphens and dots all fold to
    underscores before the alias table is consulted.
    """
    key = (header or "").strip().lower()
    for ch in (" ", "-", ".", "/"):
        key = key.replace(ch, "_")
    while "__" in key:
        key = key.replace("__", "_")
    return key.strip("_")


def _pick(row: dict[str, Any], field_name: str) -> str | None:
    """Resolve a logical field to whatever this file happens to call it."""
    normalised = {_normalise(k): v for k, v in row.items()}
    for alias in ALIASES.get(field_name, (field_name,)):
        if alias in normalised:
            value = normalised[alias]
            if value is not None and str(value).strip() != "":
                return str(value).strip()
    return None


def _money(raw: str | None, field_name: str, line: int) -> Money:
    """Parse an amount the way exports actually write them."""
    if raw is None:
        return Money("0.00")
    text = raw.replace(",", "").replace("₹", "").replace("INR", "").strip()
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    if not text:
        return Money("0.00")
    try:
        value = Money(text).quantize(Money("0.01"))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"line {line}: {field_name} {raw!r} is not an amount") from exc
    return -value if negative else value


def _date(raw: str | None, field_name: str, line: int) -> date:
    if not raw:
        raise ValueError(f"line {line}: {field_name} is empty")
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"line {line}: {field_name} {raw!r} is not a recognised date")


def _rows(path: Path) -> list[tuple[int, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(enumerate(csv.DictReader(fh), start=2))  # line 1 is the header


def load_settlement(path: Path) -> tuple[list[SettlementRow], LoadReport]:
    out: list[SettlementRow] = []
    report = LoadReport()
    for line, row in _rows(path):
        try:
            raw_type = (_pick(row, "txn_type") or "payment").lower()
            try:
                txn_type = TxnType(raw_type)
            except ValueError as exc:
                raise ValueError(f"line {line}: unknown txn_type {raw_type!r}") from exc
            out.append(
                SettlementRow(
                    txn_id=_pick(row, "txn_id") or f"row_{line}",
                    settlement_batch_id=_pick(row, "settlement_batch_id") or "",
                    utr=_pick(row, "utr"),
                    txn_type=txn_type,
                    gross_amount=_money(_pick(row, "gross_amount"), "gross_amount", line),
                    fee=_money(_pick(row, "fee"), "fee", line),
                    gst_on_fee=_money(_pick(row, "gst_on_fee"), "gst_on_fee", line),
                    settled_on=_date(_pick(row, "settled_on"), "settled_on", line),
                    order_id=_pick(row, "order_id"),
                    method=(_pick(row, "method") or "card").lower(),
                )
            )
            report.loaded += 1
        except ValueError as exc:
            report.errors.append(str(exc))
    return out, report


def load_bank(path: Path) -> tuple[list[BankCredit], LoadReport]:
    out: list[BankCredit] = []
    report = LoadReport()
    for line, row in _rows(path):
        try:
            out.append(
                BankCredit(
                    bank_txn_id=_pick(row, "bank_txn_id") or f"bank_{line}",
                    credit_amount=_money(_pick(row, "credit_amount"), "credit_amount", line),
                    value_date=_date(_pick(row, "value_date"), "value_date", line),
                    narration=_pick(row, "narration") or "",
                    utr=_pick(row, "utr"),
                )
            )
            report.loaded += 1
        except ValueError as exc:
            report.errors.append(str(exc))
    return out, report


def load_ledger(path: Path) -> tuple[list[LedgerEntry], LoadReport]:
    out: list[LedgerEntry] = []
    report = LoadReport()
    for line, row in _rows(path):
        try:
            out.append(
                LedgerEntry(
                    order_id=_pick(row, "order_id") or f"order_{line}",
                    txn_id=_pick(row, "txn_id") or "",
                    amount=_money(_pick(row, "gross_amount"), "amount", line),
                    created_at=_date(_pick(row, "created_at"), "created_at", line),
                    status=_pick(row, "status") or "captured",
                )
            )
            report.loaded += 1
        except ValueError as exc:
            report.errors.append(str(exc))
    return out, report


def load_bundle(
    settlement: Path, bank: Path, ledger: Path | None = None
) -> tuple[SourceBundle, dict[str, LoadReport]]:
    """Load a close from files. The ledger is optional; the other two are not."""
    settlement_rows, s_rep = load_settlement(settlement)
    bank_rows, b_rep = load_bank(bank)
    ledger_rows: list[LedgerEntry] = []
    reports = {"settlement": s_rep, "bank": b_rep}
    if ledger is not None:
        ledger_rows, l_rep = load_ledger(ledger)
        reports["ledger"] = l_rep

    period = ""
    if settlement_rows:
        period = f"{min(r.settled_on for r in settlement_rows):%Y-%m}"
    bundle = SourceBundle(
        ledger=ledger_rows, settlement=settlement_rows, bank=bank_rows, period=period
    )
    return bundle, reports


# ---------------------------------------------------------------------------
# Writing - so the expected shape is documented by example, not prose
# ---------------------------------------------------------------------------

def write_bundle(bundle: SourceBundle, out_dir: Path) -> dict[str, Path]:
    """Write the three sources as CSV, in the canonical column names."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "settlement": out_dir / "settlement.csv",
        "bank": out_dir / "bank.csv",
        "ledger": out_dir / "ledger.csv",
    }

    with paths["settlement"].open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["txn_id", "settlement_batch_id", "utr", "txn_type", "gross_amount",
                    "fee", "gst_on_fee", "settled_on", "order_id", "method"])
        for r in bundle.settlement:
            w.writerow([r.txn_id, r.settlement_batch_id, r.utr or "", r.txn_type.value,
                        r.gross_amount, r.fee, r.gst_on_fee, r.settled_on,
                        r.order_id or "", r.method])

    with paths["bank"].open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["bank_txn_id", "credit_amount", "value_date", "narration", "utr"])
        for c in bundle.bank:
            w.writerow([c.bank_txn_id, c.credit_amount, c.value_date, c.narration,
                        c.utr or ""])

    with paths["ledger"].open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["order_id", "txn_id", "gross_amount", "created_at", "status"])
        for e in bundle.ledger:
            w.writerow([e.order_id, e.txn_id, e.amount, e.created_at, e.status])

    return paths
