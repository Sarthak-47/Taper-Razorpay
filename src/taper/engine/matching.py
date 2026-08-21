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
from collections import defaultdict
from datetime import timedelta
from decimal import Decimal
from itertools import combinations

from ..models import (
    AMOUNT_TOLERANCE,
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


# ---------------------------------------------------------------------------
# Layer 0 - transaction-level arithmetic
# ---------------------------------------------------------------------------

def check_fees(bundle: SourceBundle) -> list[Finding]:
    """Recompute every fee from the contract and flag overcharges.

    Independent recomputation, not verification of the PG's own arithmetic -
    the point is to catch a wrong *rate*, which a self-consistent report will
    never reveal. This check alone finds real money.
    """
    findings: list[Finding] = []
    for row in bundle.settlement:
        if row.txn_type is not TxnType.PAYMENT:
            continue
        charged = row.fee + row.gst_on_fee
        expected = row.expected_fee + row.expected_gst
        overcharge = charged - expected
        if overcharge > Decimal("0.05"):  # above rounding noise
            findings.append(
                Finding(
                    defect_class=DefectClass.FEE_OVERCHARGE,
                    subject_id=row.txn_id,
                    layer=Layer.L0_EXACT,
                    confidence=1.0,
                    money_impact=overcharge,
                    evidence={
                        "gross": str(row.gross_amount),
                        "fee_charged": str(row.fee),
                        "fee_expected": str(row.expected_fee),
                        "gst_charged": str(row.gst_on_fee),
                        "gst_expected": str(row.expected_gst),
                        "implied_rate": str((row.fee / row.gross_amount).quantize(Money("0.0001"))),
                    },
                )
            )
    return findings


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
    unclaimed: list[BankCredit] = list(bundle.bank)

    for batch_id, expected in nets.items():
        settled_on = batch_dates[batch_id]
        utr = batch_utrs.get(batch_id)

        # --- 1. UTR join ---------------------------------------------------
        if utr:
            hits = [c for c in unclaimed if extract_utr(c, store) == utr]
            if hits:
                credited = sum((c.credit_amount for c in hits), Money("0.00"))
                for c in hits:
                    unclaimed.remove(c)
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
        lag = MAX_SETTLEMENT_LAG + timedelta(days=_learned_extra_lag(batch_id, store))
        window = [
            c for c in unclaimed
            if settled_on <= c.value_date <= settled_on + lag
        ]
        exact = [c for c in window if abs(c.credit_amount - expected) <= AMOUNT_TOLERANCE]
        if len(exact) == 1:
            c = exact[0]
            unclaimed.remove(c)
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
                unclaimed.remove(c)
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
                unclaimed.remove(c)
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

    for leftover in unclaimed:
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


def _learned_extra_lag(batch_id: str, store) -> int:
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


def run_deterministic(
    bundle: SourceBundle,
    store=None,
) -> tuple[list[BatchMatch], list[Finding], list[Exception_]]:
    """Everything layers 0-2 can establish, with no model in the loop.

    ``store`` carries rules learned in previous closes. Passing it is what makes
    this month cheaper than last month.
    """
    findings: list[Finding] = []
    findings += check_fees(bundle)

    dup_findings = check_duplicates(bundle)
    findings += dup_findings
    # Order matters: duplicates must be known before ledger coverage runs, or
    # every duplicate is reported twice under two different labels.
    findings += check_ledger_coverage(bundle, {f.subject_id for f in dup_findings})

    findings += check_cross_cycle_refunds(bundle)

    matches, match_findings, exceptions = match_batches(bundle, store)
    findings += match_findings
    return matches, findings, exceptions
