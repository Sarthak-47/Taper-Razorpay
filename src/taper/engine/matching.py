"""Layers 0 and 1: everything resolvable without a model.

Design rule for this file: **assert only when unambiguous.** Anything with two
plausible explanations is pushed to the exception queue rather than guessed at.
That conservatism is deliberate - it keeps deterministic precision near 1.0 and
leaves the genuinely ambiguous residue for layer 3, which is exactly the split
the ablation is meant to demonstrate.

Layer 0 - ID joins and arithmetic. Cannot be wrong if the inputs are sane.
Layer 1 - date windows, narration regex, subset netting. Bounded heuristics.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import timedelta
from decimal import Decimal
from itertools import combinations
from typing import Any

from ..models import (
    AMOUNT_TOLERANCE,
    CONTRACTED_FEE_RATE,
    GST_RATE,
    BankCredit,
    DefectClass,
    Money,
    SettlementRow,
    SourceBundle,
    TxnType,
)
from .results import BatchMatch, Exception_, Finding, Layer

# Matches the UTR shapes the banks emit in clean narrations. Deliberately
# strict: a loose pattern that half-matches messy narrations would produce
# confident wrong answers, which is worse than admitting we do not know.
UTR_PATTERN = re.compile(r"\bUTR\d{6,}\b")

# How far past the settlement date we will still consider a credit the same
# batch. T+3 covers observed bank behaviour; beyond that we refuse to guess.
MAX_SETTLEMENT_LAG = timedelta(days=4)

# Hard cap on the subset-netting search. Without this the solver is 2^n and a
# single fat batch hangs the run. Anything wider is an exception, not a stall.
SUBSET_MAX_CREDITS = 3

# How many payments must share one above-base rate before it reads as pricing
# rather than error. Three is deliberately low: the cost of asking is one
# question, while the cost of monthly false overcharge flags is a controller who
# stops reading the report.
SYSTEMATIC_RATE_MIN_ROWS = 3

# ...and it must also cover a majority of that method's payments. Pricing is
# what nearly everything on a method is billed at; an overcharge is a minority.
SYSTEMATIC_RATE_MIN_SHARE = 0.5

# How many payouts must disagree with a stored charge before we call the rule
# stale rather than the payout odd. One mismatch proves nothing; a repeated one
# is the world having moved.
STALE_RULE_MIN_ROWS = 2

# How long an order may sit unsettled before it counts as unpaid rather than
# merely recent. Wider than MAX_SETTLEMENT_LAG on purpose: the cost of asking
# early about a sale that was always going to settle is a wasted investigation.
SETTLEMENT_HORIZON = timedelta(days=7)


# ---------------------------------------------------------------------------
# Layer 0 - transaction-level arithmetic
# ---------------------------------------------------------------------------

def check_fees(
    bundle: SourceBundle, store=None
) -> tuple[list[Finding], list[Exception_]]:
    """Recompute every fee from the contract and separate two very different things.

    Independent recomputation, not verification of the PG's own arithmetic - the
    point is to catch a wrong *rate*, which a self-consistent report will never
    reveal.

    But "charged more than the base rate" has two causes, and calling both an
    overcharge is how a reconciliation tool loses a controller's trust:

      * **A rate card we do not have.** International cards genuinely cost more.
        If every one of them is billed at the same higher rate, that is pricing,
        not theft - and flagging it monthly is noise. It becomes one exception
        asking what the rate card says, and one learned rule ends it forever.

      * **An actual overcharge.** An isolated row priced above what everything
        else on that method was charged.

    Systematic is a question; isolated is a finding. The engine cannot tell which
    from one row, so it looks at the method's whole population before asserting.
    """
    findings: list[Finding] = []
    exceptions: list[Exception_] = []

    by_method: dict[str, list[SettlementRow]] = defaultdict(list)
    for row in bundle.settlement:
        if row.txn_type is TxnType.PAYMENT and row.gross_amount > 0:
            by_method[row.method].append(row)

    for method, rows in sorted(by_method.items()):
        expected_rate = _contracted_rate(method, store)
        over: list[tuple[SettlementRow, Decimal]] = []
        for row in rows:
            implied = (row.fee / row.gross_amount).quantize(Decimal("0.0001"))
            if implied > expected_rate + Decimal("0.0005"):  # above rounding noise
                over.append((row, implied))

        if not over:
            continue

        # Is one rate shared by enough rows to look like pricing rather than error?
        #
        # Count alone is not enough. Three overcharges that happen to land on the
        # same rate look identical to a rate card if you only count them - and
        # that is not hypothetical, it happened on two seeds. What separates them
        # is *share*: pricing applies to nearly every payment on a method, while
        # an overcharge is a minority of them. Requiring a majority keeps the
        # 45-of-50 international-card case systematic and correctly leaves
        # 3-of-161 domestic-card rows as the overcharges they are.
        counts = Counter(implied for _, implied in over)
        modal_rate, modal_n = counts.most_common(1)[0]
        systematic = (
            modal_n >= SYSTEMATIC_RATE_MIN_ROWS
            and modal_n / len(rows) >= SYSTEMATIC_RATE_MIN_SHARE
        )

        if systematic:
            exceptions.append(
                Exception_(
                    subject_id=f"ratecard::{method}",
                    kind="unknown_rate_card",
                    context={
                        "method": method,
                        "implied_rate": str(modal_rate),
                        "assumed_rate": str(expected_rate),
                        "rows_at_this_rate": modal_n,
                    },
                    reason=(
                        f"{modal_n} {method} payments all billed at {modal_rate}, "
                        f"above the assumed {expected_rate}. Consistent enough to be a "
                        f"rate card we do not have rather than an overcharge - needs a "
                        f"human to confirm the contracted rate for this method."
                    ),
                )
            )

        for row, implied in over:
            if systematic and implied == modal_rate:
                continue  # covered by the rate-card question above
            charged = row.fee + row.gst_on_fee
            fair_fee = (row.gross_amount * expected_rate).quantize(Money("0.01"))
            fair = fair_fee + (fair_fee * GST_RATE).quantize(Money("0.01"))
            findings.append(
                Finding(
                    defect_class=DefectClass.FEE_OVERCHARGE,
                    subject_id=row.txn_id,
                    layer=Layer.L2_RULE if _has_rate_rule(method, store) else Layer.L0_EXACT,
                    confidence=1.0,
                    money_impact=charged - fair,
                    evidence={
                        "method": method,
                        "gross": str(row.gross_amount),
                        "fee_charged": str(row.fee),
                        "implied_rate": str(implied),
                        "contracted_rate": str(expected_rate),
                    },
                )
            )

    return findings, exceptions


def _contracted_rate(method: str, store) -> Decimal:
    """The rate this method should be billed at: learned if known, else the base."""
    if store is not None and len(store):
        for rule in store.rules:
            if rule.kind == "fee_variant" and rule.params.get("method") == method:
                try:
                    return Decimal(str(rule.params["rate"]))
                except (ArithmeticError, KeyError, ValueError):
                    break
    return CONTRACTED_FEE_RATE


def _has_rate_rule(method: str, store) -> bool:
    if store is None or not len(store):
        return False
    return any(
        r.kind == "fee_variant" and r.params.get("method") == method for r in store.rules
    )


def check_duplicates(bundle: SourceBundle) -> list[Finding]:
    """Same order, same amount, captured more than once in a batch."""
    findings: list[Finding] = []
    groups: dict[tuple[str, Money], list[SettlementRow]] = defaultdict(list)
    for row in bundle.settlement:
        if row.txn_type is TxnType.PAYMENT and row.order_id:
            groups[(row.order_id, row.gross_amount)].append(row)

    for (order_id, amount), rows in groups.items():
        if len(rows) < 2:
            continue
        # Earliest txn_id is the original capture; everything after is a dup.
        rows.sort(key=lambda r: r.txn_id)
        original, dups = rows[0], rows[1:]
        for dup in dups:
            findings.append(
                Finding(
                    defect_class=DefectClass.DUPLICATE_CAPTURE,
                    subject_id=dup.txn_id,
                    layer=Layer.L0_EXACT,
                    confidence=1.0,
                    money_impact=amount,
                    evidence={
                        "order_id": order_id,
                        "original_txn_id": original.txn_id,
                        "duplicate_txn_id": dup.txn_id,
                        "amount": str(amount),
                    },
                )
            )
    return findings


def check_ledger_coverage(
    bundle: SourceBundle, known_duplicates: set[str] | None = None
) -> list[Finding]:
    """Settled payments with no corresponding entry in the merchant's ledger.

    ``known_duplicates`` must be supplied, or this check double-counts: a
    duplicate capture has no ledger entry *by definition*, so every duplicate
    would also surface here as a missing entry. Two findings, one underlying
    problem, and a controller chasing the same rupee twice.
    """
    known = {e.txn_id for e in bundle.ledger}
    dupes = known_duplicates or set()
    findings: list[Finding] = []
    for row in bundle.settlement:
        if row.txn_id in dupes:
            continue
        if row.txn_type is TxnType.PAYMENT and row.txn_id not in known:
            findings.append(
                Finding(
                    defect_class=DefectClass.MISSING_LEDGER_ENTRY,
                    subject_id=row.txn_id,
                    layer=Layer.L0_EXACT,
                    confidence=1.0,
                    money_impact=row.gross_amount,
                    evidence={
                        "order_id": row.order_id or "",
                        "amount": str(row.gross_amount),
                        "batch": row.settlement_batch_id,
                    },
                )
            )
    return findings


def check_unsettled_revenue(bundle: SourceBundle) -> list[Finding]:
    """Orders the merchant recorded and was never paid for.

    Commercially the most important thing here. Every other class is money in
    the wrong place; this is money that never arrived.

    The nuance is aging. An order captured yesterday has not settled *yet* and
    flagging it would bury the real cases in noise, so only entries older than
    the settlement window count. A tool that cries wolf about every recent sale
    gets muted in a week.
    """
    if not bundle.ledger:
        return []

    settled = {r.txn_id for r in bundle.settlement}
    # The period's own horizon, not today's date: a close is re-run months
    # later and must reach the same answer it did on the day.
    latest = max((r.settled_on for r in bundle.settlement), default=None)
    if latest is None:
        return []
    cutoff = latest - SETTLEMENT_HORIZON

    findings: list[Finding] = []
    for entry in bundle.ledger:
        if entry.txn_id in settled or entry.created_at > cutoff:
            continue
        findings.append(
            Finding(
                defect_class=DefectClass.UNSETTLED_REVENUE,
                subject_id=entry.txn_id,
                layer=Layer.L0_EXACT,
                confidence=1.0,
                money_impact=entry.amount,
                evidence={
                    "order_id": entry.order_id,
                    "amount": str(entry.amount),
                    "created_at": str(entry.created_at),
                    "aged_days": (latest - entry.created_at).days,
                    "status": entry.status,
                },
            )
        )
    return findings


def check_chargeback_holds(bundle: SourceBundle) -> list[Finding]:
    """Money the gateway is withholding pending a dispute.

    Not lost and not available, and invisible in the payout total unless
    somebody looks for it. Reported because a cash position that counts held
    money as received is wrong in the direction that hurts.
    """
    return [
        Finding(
            defect_class=DefectClass.CHARGEBACK_HOLD,
            subject_id=row.txn_id,
            layer=Layer.L0_EXACT,
            confidence=1.0,
            money_impact=row.gross_amount,
            evidence={
                "order_id": row.order_id or "",
                "amount": str(row.gross_amount),
                "batch": row.settlement_batch_id,
            },
        )
        for row in bundle.settlement
        if row.txn_type is TxnType.CHARGEBACK_HOLD
    ]


def check_cross_cycle_refunds(bundle: SourceBundle) -> list[Finding]:
    """A refund settling in a different batch than the payment it reverses."""
    payment_batch: dict[str, str] = {
        r.order_id: r.settlement_batch_id
        for r in bundle.settlement
        if r.txn_type is TxnType.PAYMENT and r.order_id
    }
    findings: list[Finding] = []
    for row in bundle.settlement:
        if row.txn_type is not TxnType.REFUND or not row.order_id:
            continue
        origin = payment_batch.get(row.order_id)
        if origin and origin != row.settlement_batch_id:
            findings.append(
                Finding(
                    defect_class=DefectClass.CROSS_CYCLE_REFUND,
                    subject_id=row.txn_id,
                    layer=Layer.L0_EXACT,
                    confidence=1.0,
                    money_impact=row.gross_amount,
                    evidence={
                        "order_id": row.order_id,
                        "payment_batch": origin,
                        "refund_batch": row.settlement_batch_id,
                        "amount": str(row.gross_amount),
                    },
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Layer 1 - batch to bank matching
# ---------------------------------------------------------------------------

def batch_nets(bundle: SourceBundle) -> dict[str, Money]:
    """Expected payout per settlement batch: payments less refunds, fees, GST."""
    nets: dict[str, Money] = defaultdict(lambda: Money("0.00"))
    for row in bundle.settlement:
        nets[row.settlement_batch_id] += row.net_amount
    return dict(nets)


def extract_utr(credit: BankCredit, store=None) -> str | None:
    """Structured field, then a strict regex, then a learned alias. No guessing.

    The strict regex stays strict. Rather than loosening it to cover banks that
    label the reference differently - which would produce confident wrong joins
    everywhere else - an alias learned from one human decision handles that bank
    specifically. Precision is preserved by narrowing the rule, not the pattern.
    """
    if credit.utr:
        return credit.utr
    hit = UTR_PATTERN.search(credit.narration.upper())
    if hit:
        return hit.group(0)

    if store is not None and len(store):
        resolved = store.resolve({"narration": credit.narration})
        if resolved:
            rule, verdict = resolved
            if rule.kind == "narration_alias" and verdict.get("utr"):
                return verdict["utr"]
    return None


def learned_adjustment(credit: BankCredit, store) -> tuple[Money, str] | None:
    """Ask the rule store whether this credit carries a known recurring charge.

    This is where a rule learned in a previous close pays for itself. Month one,
    an AXIS payout short by Rs.250 is an unexplained shortfall that costs an
    exception, a model call and a human's attention. Once the charge is learned,
    the same shortfall is *expected* - matched deterministically, reported as a
    known charge, and never escalated again.
    """
    if store is None or not len(store):
        return None
    hit = store.resolve({"narration": credit.narration})
    if not hit:
        return None
    rule, verdict = hit
    if rule.kind != "adjustment_pattern":
        return None
    raw = rule.params.get("amount")
    if raw is None:
        return None
    try:
        return Money(str(raw)), rule.rule_id
    except (ArithmeticError, ValueError):
        return None


def _effective_expected(
    credits: list[BankCredit], expected: Money, credited: Money, store
) -> Money:
    """Reduce the expected payout by a learned charge, but only if it explains
    the gap exactly. A rule that merely matches the narration is not licence to
    write off an arbitrary difference."""
    shortfall = expected - credited
    if shortfall <= AMOUNT_TOLERANCE:
        return expected
    for c in credits:
        hit = learned_adjustment(c, store)
        if hit and abs(hit[0] - shortfall) <= AMOUNT_TOLERANCE:
            return expected - hit[0]
    return expected


def match_batches(
    bundle: SourceBundle,
    store=None,
) -> tuple[list[BatchMatch], list[Finding], list[Exception_]]:
    """Tie settlement batches to the bank credits that paid them.

    Escalation order, cheapest and surest first:
      1. UTR join                      (L0)
      2. exact amount + date window    (L1)
      3. bounded subset netting        (L1, for split settlements)
      4. exception queue               (-> layer 3)
    """
    nets = batch_nets(bundle)
    batch_dates = {
        r.settlement_batch_id: r.settled_on for r in bundle.settlement
    }
    batch_utrs: dict[str, str | None] = {}
    for row in bundle.settlement:
        batch_utrs.setdefault(row.settlement_batch_id, row.utr)
        if row.utr:
            batch_utrs[row.settlement_batch_id] = row.utr

    matches: list[BatchMatch] = []
    findings: list[Finding] = []
    exceptions: list[Exception_] = []
    # Two indexes and a claimed-set, built once.
    #
    # The obvious implementation - a list of unclaimed credits, rescanned per
    # batch - is quadratic, and `taper bench` found it: per-record cost went
    # from 2.5us at 1.4k records to 13.3us at 68k, a 5x degradation that a real
    # month would have hit long before a reviewer did. Worse, the UTR scan ran
    # a regex against every unclaimed credit for every batch.
    #
    # Indexing by reference and by value date makes both lookups proportional
    # to the candidates that could actually match, and membership in a set
    # replaces O(n) list removal.
    claimed: set[str] = set()
    by_utr: dict[str, list[BankCredit]] = defaultdict(list)
    by_date: dict[Any, list[BankCredit]] = defaultdict(list)
    for credit in bundle.bank:
        by_date[credit.value_date].append(credit)
        found = extract_utr(credit, store)
        if found:
            by_utr[found].append(credit)

    for batch_id, expected in nets.items():
        settled_on = batch_dates[batch_id]
        utr = batch_utrs.get(batch_id)

        # --- 1. UTR join ---------------------------------------------------
        if utr:
            hits = [c for c in by_utr.get(utr, []) if c.bank_txn_id not in claimed]
            if hits:
                credited = sum((c.credit_amount for c in hits), Money("0.00"))
                for c in hits:
                    claimed.add(c.bank_txn_id)
                # A batch whose entire shortfall is a charge we already
                # understand is reconciled, not broken. Scoring it as unclean
                # would permanently cap the match rate at the fraction of banks
                # that take no recurring charge - and would hide the very
                # improvement the rule store exists to produce.
                effective = _effective_expected(hits, expected, credited, store)
                matches.append(
                    BatchMatch(
                        batch_id=batch_id,
                        bank_txn_ids=[c.bank_txn_id for c in hits],
                        expected_net=effective,
                        credited=credited,
                        layer=Layer.L2_RULE if effective != expected else Layer.L0_EXACT,
                        confidence=1.0,
                        method="utr_join" if effective == expected
                        else "utr_join+learned_charge",
                    )
                )
                # One batch paid as several credits is a split, whether we found
                # it by UTR or by subset search. Reporting it only on the subset
                # path would understate splits by however many carry a clean UTR.
                if len(hits) > 1:
                    findings.append(
                        Finding(
                            defect_class=DefectClass.SPLIT_SETTLEMENT,
                            subject_id=batch_id,
                            layer=Layer.L0_EXACT,
                            confidence=1.0,
                            evidence={
                                "parts": len(hits),
                                "bank_txn_ids": [c.bank_txn_id for c in hits],
                                "total": str(credited),
                                "found_by": "utr_join",
                            },
                        )
                    )
                findings.extend(
                    _post_match_checks(batch_id, hits, expected, credited, settled_on, store)
                )
                continue
        else:
            # The settlement report shipped without a UTR at all - a defect in
            # its own right, independent of whether we can still match it.
            findings.append(
                Finding(
                    defect_class=DefectClass.MISSING_UTR,
                    subject_id=batch_id,
                    layer=Layer.L0_EXACT,
                    confidence=1.0,
                    evidence={"batch": batch_id, "fallback": "amount+date matching"},
                )
            )

        # --- 2. exact amount inside the settlement window -------------------
        # A learned timing rule extends the window for banks known to settle
        # slowly. Note the honest limitation: which bank paid a batch is only
        # known *after* it is matched, so the extension cannot be applied
        # per-bank here - it takes the widest learned offset, hard-capped. That
        # is why the cap matters: an unbounded window turns amount matching into
        # guesswork and the subset search into a combinatorial problem.
        lag = MAX_SETTLEMENT_LAG + timedelta(days=_learned_extra_lag(store))
        window = [
            c
            for offset in range(lag.days + 1)
            for c in by_date.get(settled_on + timedelta(days=offset), [])
            if c.bank_txn_id not in claimed
        ]
        exact = [c for c in window if abs(c.credit_amount - expected) <= AMOUNT_TOLERANCE]
        if len(exact) == 1:
            c = exact[0]
            claimed.add(c.bank_txn_id)
            matches.append(
                BatchMatch(
                    batch_id=batch_id, bank_txn_ids=[c.bank_txn_id],
                    expected_net=expected, credited=c.credit_amount,
                    layer=Layer.L1_FUZZY, confidence=0.95, method="amount_date_window",
                )
            )
            findings.extend(
                _post_match_checks(batch_id, [c], expected, c.credit_amount, settled_on, store)
            )
            continue

        # --- 2b. exact amount *net of a learned recurring charge* (L2) --------
        # Only reachable once a previous close taught us this bank's deduction.
        matched_by_rule = False
        for c in window:
            hit = learned_adjustment(c, store)
            if not hit:
                continue
            adj, rule_id = hit
            if abs(c.credit_amount - (expected - adj)) <= AMOUNT_TOLERANCE:
                claimed.add(c.bank_txn_id)
                matches.append(
                    BatchMatch(
                        batch_id=batch_id, bank_txn_ids=[c.bank_txn_id],
                        expected_net=expected - adj, credited=c.credit_amount,
                        layer=Layer.L2_RULE, confidence=0.97,
                        method=f"amount_net_of_learned_charge:{rule_id}",
                    )
                )
                findings.append(
                    Finding(
                        defect_class=DefectClass.UNRECORDED_ADJUSTMENT,
                        subject_id=batch_id,
                        layer=Layer.L2_RULE,
                        confidence=0.97,
                        money_impact=adj,
                        rule_id=rule_id,
                        evidence={
                            "amount": str(adj),
                            "explained_by": rule_id,
                            "narration": c.narration,
                            "status": "known recurring charge",
                        },
                    )
                )
                findings.extend(
                    _post_match_checks(
                        batch_id, [c], expected - adj, c.credit_amount, settled_on, store
                    )
                )
                matched_by_rule = True
                break

        if matched_by_rule:
            continue

        # --- 3. bounded subset netting, optionally net of a learned charge ---
        # A batch can be split across credits *and* short by a recurring charge
        # at the same time. Searching only for the full expected total misses
        # every such batch, which was the single largest source of exceptions.
        # Each candidate target is an exact sum we can defend, not a tolerance
        # widened until something fits.
        subset, applied_adj = None, None
        for target, adj in _candidate_targets(window, expected, store):
            subset = _find_subset(window, target)
            if subset:
                applied_adj = adj
                break

        if subset:
            credited = sum((c.credit_amount for c in subset), Money("0.00"))
            for c in subset:
                claimed.add(c.bank_txn_id)
            effective = expected - (applied_adj[0] if applied_adj else Money("0.00"))
            matches.append(
                BatchMatch(
                    batch_id=batch_id, bank_txn_ids=[c.bank_txn_id for c in subset],
                    expected_net=effective, credited=credited,
                    layer=Layer.L2_RULE if applied_adj else Layer.L1_FUZZY,
                    confidence=0.90,
                    method="subset_netting" if not applied_adj
                    else f"subset_netting+learned_charge:{applied_adj[1]}",
                )
            )
            findings.append(
                Finding(
                    defect_class=DefectClass.SPLIT_SETTLEMENT,
                    subject_id=batch_id,
                    layer=Layer.L1_FUZZY,
                    confidence=0.90,
                    evidence={
                        "parts": len(subset),
                        "bank_txn_ids": [c.bank_txn_id for c in subset],
                        "total": str(credited),
                    },
                )
            )
            if applied_adj:
                findings.append(
                    Finding(
                        defect_class=DefectClass.UNRECORDED_ADJUSTMENT,
                        subject_id=batch_id,
                        layer=Layer.L2_RULE,
                        confidence=0.95,
                        money_impact=applied_adj[0],
                        rule_id=applied_adj[1],
                        evidence={
                            "amount": str(applied_adj[0]),
                            "explained_by": applied_adj[1],
                            "status": "known recurring charge, batch also split",
                        },
                    )
                )
            findings.extend(
                _post_match_checks(batch_id, subset, effective, credited, settled_on, store)
            )
            continue

        # --- 4. give up honestly -------------------------------------------
        # Under-payment could be an unrecorded adjustment, or the first half of
        # a split whose sibling has not arrived. Two explanations, so we refuse
        # to pick one here and hand it to layer 3 with the candidate set.
        exceptions.append(
            Exception_(
                subject_id=batch_id,
                kind="unmatched_batch",
                context={
                    "expected_net": str(expected),
                    "settled_on": str(settled_on),
                    "utr": utr,
                },
                candidates=[c.bank_txn_id for c in window],
                reason="no UTR join, no exact amount match, no bounded subset",
            )
        )

    for leftover in (c for c in bundle.bank if c.bank_txn_id not in claimed):
        exceptions.append(
            Exception_(
                subject_id=leftover.bank_txn_id,
                kind="unclaimed_credit",
                context={
                    "amount": str(leftover.credit_amount),
                    "value_date": str(leftover.value_date),
                    "narration": leftover.narration,
                },
                reason="bank credit not claimed by any settlement batch",
            )
        )

    return matches, findings, exceptions


def _learned_extra_lag(store) -> int:
    """Extra days of settlement window granted by a learned timing rule.

    Capped hard. A timing rule is evidence about a bank's habits, not a licence
    to search an unbounded date range - a wide window turns amount matching into
    guesswork and the subset search into a combinatorial problem.
    """
    if store is None or not len(store):
        return 0
    best = 0
    for rule in store.rules:
        if rule.kind != "bank_timing":
            continue
        try:
            offset = int(rule.params.get("offset_days", 0))
        except (TypeError, ValueError):
            continue
        best = max(best, min(offset, 5))
    return best


def _has_any_reference(narration: str) -> bool:
    """Does this narration contain something that could be a settlement reference?

    A run of six or more digits. Used only to tell "unreadable label" apart from
    "genuinely no reference", which are different problems with different fixes.
    """
    return re.search(r"\d{6,}", narration) is not None


def _candidate_targets(
    window: list[BankCredit], expected: Money, store
) -> list[tuple[Money, tuple[Money, str] | None]]:
    """Sums worth searching for: the full payout, then payout less each learned charge.

    Kept to a handful of *exact* targets on purpose. The alternative - widening
    the tolerance until something fits - would manufacture matches that reconcile
    to nothing in particular, which is worse than an honest exception.
    """
    targets: list[tuple[Money, tuple[Money, str] | None]] = [(expected, None)]
    seen: set[Money] = {expected}
    for c in window:
        hit = learned_adjustment(c, store)
        if not hit:
            continue
        reduced = expected - hit[0]
        if reduced not in seen:
            seen.add(reduced)
            targets.append((reduced, hit))
    return targets


def _find_subset(credits: list[BankCredit], target: Money) -> list[BankCredit] | None:
    """Smallest set of credits summing to target, searched under a hard cap.

    Bounded at ``SUBSET_MAX_CREDITS`` on purpose. Unbounded subset-sum over a
    fat settlement window is exponential and will hang the run; a timeout that
    routes to the exception list is strictly better than a solver that stalls.
    """
    if len(credits) > 12:  # window too wide to search safely
        return None
    for size in range(2, SUBSET_MAX_CREDITS + 1):
        for combo in combinations(credits, size):
            total = sum((c.credit_amount for c in combo), Money("0.00"))
            if abs(total - target) <= AMOUNT_TOLERANCE:
                return list(combo)
    return None


def _post_match_checks(
    batch_id: str,
    credits: list[BankCredit],
    expected: Money,
    credited: Money,
    settled_on,
    store=None,
) -> list[Finding]:
    """Checks that only become possible once a batch is tied to its credits."""
    findings: list[Finding] = []

    # Timing: did the money start landing later than the settlement date?
    # Measured against the *earliest* credit on purpose. Using the latest would
    # score every split settlement as a timing shift too, since the second leg
    # arrives a day behind by construction - one event, two findings.
    first = min(c.value_date for c in credits)
    if first > settled_on:
        findings.append(
            Finding(
                defect_class=DefectClass.TIMING_SHIFT,
                subject_id=batch_id,
                layer=Layer.L1_FUZZY,
                confidence=0.95,
                evidence={
                    "settled_on": str(settled_on),
                    "value_date": str(first),
                    "offset_days": (first - settled_on).days,
                },
            )
        )

    # Narration: matched, but the narration carried no reference at all.
    #
    # "No reference we could parse" and "no reference present" are different
    # conditions and only the second is a defect. A bank that labels the same
    # reference as REF instead of UTR has not lost anything - we simply cannot
    # read it yet, and one learned alias fixes that permanently. Reporting those
    # as drift would flag a bank convention as a data problem every month.
    for c in credits:
        if extract_utr(c, store) is None and not _has_any_reference(c.narration):
            findings.append(
                Finding(
                    defect_class=DefectClass.NARRATION_DRIFT,
                    subject_id=batch_id,
                    layer=Layer.L1_FUZZY,
                    confidence=0.85,
                    evidence={"bank_txn_id": c.bank_txn_id, "narration": c.narration},
                )
            )
            break

    # Shortfall: bank paid less than the settlement report explains.
    shortfall = expected - credited
    if shortfall > AMOUNT_TOLERANCE:
        # Before calling it unexplained, ask whether a previous close already
        # explained it. Same finding either way - the money really was deducted -
        # but a *known* charge is reported at L2 and never escalated, while an
        # unknown one costs a human. The distinction is the entire thesis.
        explained = None
        for c in credits:
            explained = learned_adjustment(c, store)
            if explained and abs(explained[0] - shortfall) <= AMOUNT_TOLERANCE:
                break
            explained = None

        findings.append(
            Finding(
                defect_class=DefectClass.UNRECORDED_ADJUSTMENT,
                subject_id=batch_id,
                layer=Layer.L2_RULE if explained else Layer.L0_EXACT,
                confidence=0.97 if explained else 1.0,
                money_impact=shortfall,
                rule_id=explained[1] if explained else None,
                evidence={
                    "expected_net": str(expected),
                    "credited": str(credited),
                    "shortfall": str(shortfall),
                    "status": "known recurring charge" if explained else "unexplained",
                    **({"explained_by": explained[1]} if explained else {}),
                },
            )
        )
    return findings


def check_rule_health(
    matches: list[BatchMatch], bundle: SourceBundle, store
) -> list[Exception_]:
    """Notice when something we learned has stopped being true.

    Learning is only half a lifecycle. Until now the store could add rules and
    never question them, so when a bank changed its processing charge the rule
    kept asserting the old amount, quietly failed to explain the gap, and the
    batch became an ordinary exception. The engine degraded safely but said
    nothing about *why* - and a human re-investigating a solved problem every
    month is exactly the waste the rule store exists to remove.

    The signal is specific: a charge rule whose narration still matches, on a
    batch that still has a shortfall, where that shortfall is a consistent
    amount that is *not* the one the rule stores. Consistency is what separates
    "the bank repriced" from "one odd payout" - a single mismatch proves
    nothing and is left alone.
    """
    if store is None or not len(store):
        return []

    credit_by_id = {c.bank_txn_id: c for c in bundle.bank}
    observed: dict[str, list[Money]] = defaultdict(list)

    for m in matches:
        shortfall = m.expected_net - m.credited
        if shortfall <= AMOUNT_TOLERANCE:
            continue
        for bank_id in m.bank_txn_ids:
            credit = credit_by_id.get(bank_id)
            if not credit:
                continue
            hit = learned_adjustment(credit, store)
            if hit and abs(hit[0] - shortfall) > AMOUNT_TOLERANCE:
                observed[hit[1]].append(shortfall)
            break

    exceptions: list[Exception_] = []
    for rule_id, amounts in sorted(observed.items()):
        if len(amounts) < STALE_RULE_MIN_ROWS:
            continue
        counts = Counter(amounts)
        modal_amount, modal_n = counts.most_common(1)[0]
        if modal_n < STALE_RULE_MIN_ROWS:
            continue
        rule = next((r for r in store.rules if r.rule_id == rule_id), None)
        stored = rule.params.get("amount") if rule else "?"
        exceptions.append(
            Exception_(
                subject_id=f"stale::{rule_id}",
                kind="stale_rule",
                context={
                    "rule_id": rule_id,
                    "stored_amount": str(stored),
                    "observed_amount": str(modal_amount),
                    "occurrences": modal_n,
                    # The retiring rule's own keyword travels with the exception.
                    # Without it the replacement has to be reconstructed by
                    # guesswork downstream, and with two recurring charges in
                    # play that guess picks the wrong bank's label.
                    "keyword": str(rule.params.get("keyword", "")) if rule else "",
                },
                reason=(
                    f"Rule {rule_id} still matches these narrations but no longer "
                    f"explains them: it stores {stored} while {modal_n} payouts are "
                    f"short by {modal_amount}. The underlying charge appears to have "
                    f"changed - confirm and update the rule rather than resolving "
                    f"each payout again."
                ),
            )
        )
    return exceptions


def run_deterministic(
    bundle: SourceBundle,
    store=None,
) -> tuple[list[BatchMatch], list[Finding], list[Exception_]]:
    """Everything layers 0-2 can establish, with no model in the loop.

    ``store`` carries rules learned in previous closes. Passing it is what makes
    this month cheaper than last month.
    """
    findings: list[Finding] = []
    fee_findings, fee_exceptions = check_fees(bundle, store)
    findings += fee_findings

    dup_findings = check_duplicates(bundle)
    findings += dup_findings
    # Order matters: duplicates must be known before ledger coverage runs, or
    # every duplicate is reported twice under two different labels.
    findings += check_ledger_coverage(bundle, {f.subject_id for f in dup_findings})

    findings += check_cross_cycle_refunds(bundle)
    findings += check_unsettled_revenue(bundle)
    findings += check_chargeback_holds(bundle)

    matches, match_findings, exceptions = match_batches(bundle, store)
    findings += match_findings

    # Ask whether anything we learned has stopped being true. Runs last because
    # it reads the matches the earlier layers produced.
    stale = check_rule_health(matches, bundle, store)
    return matches, findings, fee_exceptions + stale + exceptions
