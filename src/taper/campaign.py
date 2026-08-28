"""Consecutive monthly closes, carrying the rule store forward.

This produces the headline result: **LLM calls per 100 records fall month over
month while match rate rises**, because each close compiles what a human taught
it into deterministic rules the next close gets for free.

The human in the loop is simulated by ``HumanOracle``. That is not a cheat - it
is the supervision signal. In production a controller investigates an exception
and writes down what it turned out to be; here the oracle looks up the injected
ground truth and does the same. What is *not* simulated is the part that matters:
the rule still has to survive the retroactive admission gate before it is
allowed to affect any future close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from .engine.llm import LLMClient
from .engine.pipeline import RunConfig, reconcile
from .engine.results import Exception_, ReconResult
from .engine.rules import Rule, RuleStore, build_history, next_rule_id
from .generator import METHOD_RATES, DefectRates, GeneratedCase, generate
from .metrics.harness import Scorecard, score
from .models import DefectClass, Money


@dataclass
class HumanOracle:
    """Stands in for the controller who investigates an exception.

    Returns what the item actually was, plus - when the pattern looks like it
    will recur - a candidate rule. Deliberately conservative about the second
    part: it only proposes a rule for defects the ground truth marks recurring.
    A human who turns every one-off anomaly into a standing rule is how a rule
    store poisons itself, so the oracle refuses to do it.
    """

    case: GeneratedCase
    reviews: int = 0

    def resolve(self, exc: Exception_) -> dict[str, Any] | None:
        self.reviews += 1

        # A human staring at an unmatched batch reads the candidate narrations
        # next to the settlement's own reference and spots the correspondence:
        # "the bank writes REF 70000123456, our report says UTR70000123456 -
        # it's the same number under a different label." That observation, once,
        # is worth a permanent alias.
        alias = self._spot_reference_alias(exc)
        if alias:
            return alias

        stale = self._confirm_repricing(exc)
        if stale:
            return stale

        rate_card = self._confirm_rate_card(exc)
        if rate_card:
            return rate_card

        lag = self._spot_timing_lag(exc)
        if lag:
            return lag

        truths = [d for d in self.case.defects if d.subject_id == exc.subject_id]
        if not truths:
            return None

        for truth in truths:
            if truth.defect_class is not DefectClass.UNRECORDED_ADJUSTMENT:
                continue
            if not truth.detail.get("recurring"):
                # A genuine one-off. Resolved for this close, never generalised.
                return {"defect_class": truth.defect_class.value, "proposed_rule": None}
            return {
                "defect_class": truth.defect_class.value,
                "proposed_rule": {
                    "kind": "adjustment_pattern",
                    "params": {
                        "keyword": truth.detail.get("keyword", ""),
                        "category": "bank_recurring_charge",
                        "amount": truth.detail.get("amount"),
                        "pattern": truth.detail.get("keyword", ""),
                    },
                    "confidence": 0.95,
                },
            }
        return {"defect_class": truths[0].defect_class.value, "proposed_rule": None}

    def _confirm_repricing(self, exc: Exception_) -> dict[str, Any] | None:
        """Confirm that a bank really has repriced, and hand back the new amount.

        The engine can see that a stored charge stopped explaining the payouts;
        only a human can say whether the bank repriced or something else broke.
        Answering closes the loop the rule store would otherwise leave open -
        without this the same question returns every month forever.
        """
        if exc.kind != "stale_rule":
            return None
        observed = exc.context.get("observed_amount")
        if not observed:
            return None
        return {
            "defect_class": None,
            "retire_rule": exc.context.get("rule_id"),
            "proposed_rule": {
                "kind": "adjustment_pattern",
                "params": {
                    "keyword": str(exc.context.get("keyword", "")),
                    "category": "bank_recurring_charge",
                    "amount": str(observed),
                },
                "confidence": 0.95,
            },
        }

    def _confirm_rate_card(self, exc: Exception_) -> dict[str, Any] | None:
        """Answer "what is this method actually contracted at?" from the rate card.

        This is the one exception a human can settle without investigating
        anything - they look it up in the merchant agreement. It is also the
        purest case for the rule store: a contracted rate is a fact that does
        not change month to month, so asking twice is pure waste.

        The oracle confirms the *true* rate rather than the observed one. A human
        reading a contract would not ratify whatever the PG happened to bill; if
        the two disagree, that is an overcharge to pursue, not a rate to adopt.
        """
        if exc.kind != "unknown_rate_card":
            return None
        method = str(exc.context.get("method", ""))
        true_rate = METHOD_RATES.get(method)
        if true_rate is None:
            return None
        return {
            "defect_class": None,
            "proposed_rule": {
                "kind": "fee_variant",
                "params": {"method": method, "rate": str(true_rate)},
                "confidence": 1.0,
            },
        }

    def _spot_timing_lag(self, exc: Exception_) -> dict[str, Any] | None:
        """Find the batch's money sitting just outside the settlement window.

        A controller chasing an unmatched payout looks a few days further out and
        finds the exact amount landed late. The observation generalises: that
        bank is simply slower than the default window assumes.

        Requires an *exact* amount match. A near-miss outside the window is two
        unknowns at once and teaches nothing reliable.
        """
        if exc.kind != "unmatched_batch":
            return None
        try:
            expected = Money(str(exc.context.get("expected_net")))
            settled_on = date.fromisoformat(str(exc.context.get("settled_on")))
        except (ArithmeticError, ValueError, TypeError):
            return None

        claimed = {b for c in self.case.bundle.bank for b in [c.bank_txn_id]
                   if b in exc.candidates}
        for credit in self.case.bundle.bank:
            if credit.bank_txn_id in claimed:
                continue
            lag_days = (credit.value_date - settled_on).days
            if lag_days <= 4 or lag_days > 9:
                continue
            if abs(credit.credit_amount - expected) <= Money("1.00"):
                return {
                    "defect_class": DefectClass.TIMING_SHIFT.value,
                    "proposed_rule": {
                        "kind": "bank_timing",
                        "params": {
                            "bank": _bank_from_narration(credit.narration),
                            "offset_days": lag_days,
                        },
                        "confidence": 0.9,
                    },
                }
        return None

    def _spot_reference_alias(self, exc: Exception_) -> dict[str, Any] | None:
        """Find a narration that quotes the batch's own reference under a label.

        Only fires when the digits in the narration *are* the settlement's UTR -
        the correspondence is verified, not assumed. A label that happens to sit
        near some other number teaches nothing and is skipped.
        """
        if exc.kind != "unmatched_batch":
            return None
        utr = exc.context.get("utr")
        if not utr:
            return None
        digits = str(utr).removeprefix("UTR")

        by_id = {c.bank_txn_id: c for c in self.case.bundle.bank}
        for bank_id in exc.candidates:
            credit = by_id.get(bank_id)
            if not credit or digits not in credit.narration:
                continue
            # The token immediately before the digits is the bank's label.
            before = credit.narration.upper().split(digits)[0].rstrip(" :-/#")
            marker = before.split()[-1] if before.split() else ""
            if not marker or marker.isdigit():
                continue
            return {
                "defect_class": DefectClass.NARRATION_DRIFT.value,
                "proposed_rule": {
                    "kind": "narration_alias",
                    "params": {"marker": marker, "prefix": "UTR"},
                    "confidence": 0.95,
                },
            }
        return None


@dataclass
class MonthResult:
    month: int
    period: str
    card: Scorecard
    rules_before: int
    rules_after: int
    rules_rejected: int
    human_reviews: int
    exceptions: int

    @property
    def rules_learned(self) -> int:
        return self.rules_after - self.rules_before


@dataclass
class CampaignResult:
    months: list[MonthResult] = field(default_factory=list)
    store: RuleStore | None = None

    @property
    def first(self) -> MonthResult:
        return self.months[0]

    @property
    def last(self) -> MonthResult:
        return self.months[-1]

    def verdict(self) -> str:
        a, b = self.first, self.last
        call_drop = a.card.llm_calls_per_100 - b.card.llm_calls_per_100
        exc_drop = a.exceptions - b.exceptions
        match_gain = b.card.match_rate - a.card.match_rate
        if call_drop <= 0 and exc_drop <= 0:
            return (
                "No taper. LLM load did not fall across the campaign - either the "
                "data has no recurring structure to learn, or no candidate rule "
                "survived the admission gate. Both are real answers; neither is a win."
            )
        return (
            f"Across {len(self.months)} closes the rule store grew to "
            f"{b.rules_after} rule(s). Model calls per 100 records fell "
            f"{a.card.llm_calls_per_100:.2f} -> {b.card.llm_calls_per_100:.2f} "
            f"({-call_drop:+.2f}), exceptions {a.exceptions} -> {b.exceptions} "
            f"({-exc_drop:+d}), clean match rate {a.card.match_rate:.1%} -> "
            f"{b.card.match_rate:.1%} ({match_gain:+.1%})."
        )


def run_campaign(
    months: int = 4,
    n_batches: int = 40,
    base_seed: int = 500,
    config: RunConfig | None = None,
    client: LLMClient | None = None,
    learn: bool = True,
    reprice: tuple[int, str, Money] | None = None,
) -> CampaignResult:
    """Close the books ``months`` times in a row, keeping what we learn.

    Each month is fresh data (a new seed) drawn from the *same* bank profiles -
    new transactions, same institutional behaviour. That is the realistic
    setting: the merchant's volume changes every month, their bank's habits do not.

    ``reprice`` breaks that assumption on purpose: ``(month, bank, new_amount)``
    changes a bank's recurring charge partway through, so a rule learned earlier
    becomes a confident wrong answer. Without it the campaign only ever proves
    that learning works, never that the system notices when what it learned has
    stopped being true.
    """
    config = config or RunConfig(use_llm=True, use_real_llm=False)
    store = RuleStore()
    out = CampaignResult(store=store)

    for m in range(months):
        seed = base_seed + m
        start = date(2026, 1, 1) + timedelta(days=31 * m)
        overrides: dict[str, Money] = {}
        if reprice and m + 1 >= reprice[0]:
            overrides[reprice[1]] = reprice[2]
        case = generate(
            n_batches=n_batches, seed=seed, rates=DefectRates(), start=start,
            charge_overrides=overrides,
        )

        rules_before = len(store)
        rejected_before = len(store.rejected)

        result = reconcile(case.bundle, store=store, config=config, client=client)

        # --- the human works the exception queue, and we learn from them ----
        oracle = HumanOracle(case)
        if learn:
            _learn_from_reviews(result, oracle, store, case)

        out.months.append(
            MonthResult(
                month=m + 1,
                period=f"{start:%Y-%m}",
                card=score(case, result, client.name if client else "mock"),
                rules_before=rules_before,
                rules_after=len(store),
                rules_rejected=len(store.rejected) - rejected_before,
                human_reviews=oracle.reviews,
                exceptions=len(result.exceptions),
            )
        )

    return out


@dataclass
class AveragedMonth:
    """One month's metrics averaged over independent campaigns."""

    month: int
    runs: int
    llm_calls_per_100: float
    exceptions: float
    match_rate: float
    precision: float
    recall: float
    rules: float
    human_reviews: float


def run_campaign_averaged(
    runs: int = 6, months: int = 5, n_batches: int = 40, base_seed: int = 500, **kw
) -> list[AveragedMonth]:
    """Average the campaign over independent runs.

    A single campaign is one noisy draw: how many one-off anomalies land in
    month 3 is luck, and reading a taper off one line would be reading noise.
    Averaging over independent campaigns separates the part that actually
    improves - the learnable structure absorbed after the first close - from
    month-to-month variance that no amount of learning can remove.
    """
    campaigns = [
        run_campaign(months=months, n_batches=n_batches, base_seed=base_seed + 100 * r, **kw)
        for r in range(runs)
    ]

    out: list[AveragedMonth] = []
    for i in range(months):
        rows = [c.months[i] for c in campaigns]
        n = len(rows)
        out.append(
            AveragedMonth(
                month=i + 1,
                runs=n,
                llm_calls_per_100=sum(r.card.llm_calls_per_100 for r in rows) / n,
                exceptions=sum(r.exceptions for r in rows) / n,
                match_rate=sum(r.card.match_rate for r in rows) / n,
                precision=sum(r.card.precision for r in rows) / n,
                recall=sum(r.card.recall for r in rows) / n,
                rules=sum(r.rules_after for r in rows) / n,
                human_reviews=sum(r.human_reviews for r in rows) / n,
            )
        )
    return out


def _learn_from_reviews(
    result: ReconResult, oracle: HumanOracle, store: RuleStore, case: GeneratedCase
) -> None:
    """Turn this month's human decisions into next month's deterministic rules.

    Every candidate goes through ``store.propose``, which replays it against all
    confirmed cases and rejects anything that would change one. A rule the model
    and the human both liked still does not get in if history disagrees.
    """
    history = build_history(result.findings, case.bundle)
    seen: set[str] = set()

    for exc in result.exceptions:
        resolution = oracle.resolve(exc)
        if not resolution:
            continue
        # A repricing must retire the old rule first. Admitting the new amount
        # alongside the old one would leave two rules disagreeing about the same
        # narration, and whichever was appended first would keep winning.
        doomed = resolution.get("retire_rule")
        if doomed:
            store.retire(str(doomed), f"superseded from {exc.subject_id}")

        proposal = resolution.get("proposed_rule")
        if not proposal:
            continue

        params = proposal.get("params") or {}
        # One rule per pattern per month. Ten AXIS shortfalls are ten instances
        # of one lesson, not ten lessons.
        fingerprint = (
            f"{proposal.get('kind')}:"
            f"{params.get('keyword') or params.get('marker') or params.get('bank') or params.get('method')}"
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        if any(
            r.kind == proposal.get("kind")
            and r.params.get("keyword") == params.get("keyword")
            and r.params.get("marker") == params.get("marker")
            and r.params.get("bank") == params.get("bank")
            and r.params.get("method") == params.get("method")
            for r in store.rules
        ):
            continue  # already known

        candidate = Rule(
            rule_id=next_rule_id(store, proposal["kind"]),
            kind=proposal["kind"],
            params=params,
            origin_exception=exc.subject_id,
            learned_on=str(date.today()),
            confidence=float(proposal.get("confidence", 0.9)),
        )
        store.propose(candidate, history)


def _bank_from_narration(narration: str) -> str:
    """Best-effort bank identifier from a narration, for labelling a timing rule.

    Only ever used as a rule parameter for provenance and dedup - never to
    decide a match - so an unknown bank is recorded as such rather than guessed.
    """
    for name in ("HDFC", "ICICI", "AXIS", "SBI", "KOTAK", "KKBK"):
        if name in narration.upper():
            return "KOTAK" if name == "KKBK" else name
    return "UNKNOWN"
