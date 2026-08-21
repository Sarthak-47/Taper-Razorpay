"""The invariants that must never break.

These are not coverage tests. Each one guards a claim the writeup makes, so if
one fails, a sentence in the submission has become false.

Run: python -m pytest tests/ -v
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from taper.engine.llm import verify_proposal
from taper.engine.matching import run_deterministic
from taper.engine.pipeline import RunConfig, reconcile
from taper.engine.rules import ConfirmedCase, Rule, RuleStore
from taper.generator import generate
from taper.metrics.harness import score
from taper.models import Money

SEEDS = [7, 99, 1234]


# ---------------------------------------------------------------------------
# Claim: "deterministic layers never produce a false positive"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_deterministic_precision_is_perfect(seed: int) -> None:
    """Layers 0 and 1 assert only when unambiguous, so precision must be 1.0.

    If this fails, the conservatism argument is dead and the auto-clear
    operating point cannot be trusted.
    """
    case = generate(n_batches=40, seed=seed)
    _, findings, _ = run_deterministic(case.bundle)
    truth = {(d.defect_class, d.subject_id) for d in case.defects}
    false_positives = [f for f in findings if f.key() not in truth]
    assert not false_positives, (
        f"{len(false_positives)} false positive(s), e.g. "
        f"{[(f.defect_class.value, f.subject_id) for f in false_positives[:3]]}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_no_finding_is_reported_twice(seed: int) -> None:
    """One underlying problem, one finding. Double-counting sends a controller
    chasing the same rupee twice - the duplicate/missing-ledger overlap bug."""
    case = generate(n_batches=40, seed=seed)
    _, findings, _ = run_deterministic(case.bundle)
    keys = [f.key() for f in findings]
    assert len(keys) == len(set(keys)), "duplicate findings emitted"


# ---------------------------------------------------------------------------
# Claim: "the model never decides a match; arithmetic does"
# ---------------------------------------------------------------------------

def test_verify_rejects_wrong_sum() -> None:
    """A confident, well-formed, numerically wrong proposal must be refused."""
    proposal = {
        "defect_class": "split_settlement",
        "bank_txn_ids": ["b1", "b2"],
        "confidence": 0.99,  # maximum confidence, still wrong
    }
    credits = {"b1": Money("100.00"), "b2": Money("200.00")}
    v = verify_proposal(proposal, expected_net=Money("5000.00"), credit_amounts=credits)
    assert not v.ok
    assert "off by" in v.reason


def test_verify_rejects_unknown_credit_ids() -> None:
    """A hallucinated bank_txn_id must not be silently ignored."""
    proposal = {"defect_class": "split_settlement", "bank_txn_ids": ["ghost"], "confidence": 0.9}
    v = verify_proposal(proposal, Money("100.00"), {"b1": Money("100.00")})
    assert not v.ok and "unknown credits" in v.reason


def test_verify_accepts_correct_sum() -> None:
    proposal = {"defect_class": "split_settlement", "bank_txn_ids": ["b1", "b2"], "confidence": 0.8}
    credits = {"b1": Money("2000.00"), "b2": Money("3000.00")}
    v = verify_proposal(proposal, Money("5000.00"), credits)
    assert v.ok


def test_verify_will_not_relabel_an_overpayment_as_a_shortfall() -> None:
    """An adjustment means money is missing. More money than expected is a
    different problem and must not be absorbed into the wrong bucket."""
    proposal = {"defect_class": "unrecorded_adjustment", "bank_txn_ids": ["b1"], "confidence": 0.9}
    v = verify_proposal(proposal, Money("100.00"), {"b1": Money("500.00")})
    assert not v.ok


# ---------------------------------------------------------------------------
# Claim: "no rule enters the store if it contradicts a confirmed case"
# ---------------------------------------------------------------------------

def _rule(offset: int) -> Rule:
    return Rule(
        rule_id="bank_timing_001",
        kind="bank_timing",
        params={"bank": "HDFC", "offset_days": offset},
        origin_exception="setl_test_001",
        learned_on="2026-06-30",
    )


def test_admission_gate_rejects_regressing_rule() -> None:
    history = [
        ConfirmedCase("setl_a", {"bank": "HDFC"}, {"expected_offset_days": 2}),
        ConfirmedCase("setl_b", {"bank": "HDFC"}, {"expected_offset_days": 2}),
    ]
    store = RuleStore()
    result = store.propose(_rule(offset=3), history)  # contradicts both
    assert not result.admitted
    assert len(store) == 0, "a rejected rule must not enter the store"
    assert result.regressions


def test_admission_gate_admits_consistent_rule() -> None:
    history = [ConfirmedCase("setl_a", {"bank": "HDFC"}, {"expected_offset_days": 2})]
    store = RuleStore()
    result = store.propose(_rule(offset=2), history)
    assert result.admitted and len(store) == 1


def test_rule_that_touches_nothing_is_still_admitted() -> None:
    """A rule about a bank we have no history for is unproven, not wrong."""
    history = [ConfirmedCase("setl_a", {"bank": "ICICI"}, {"expected_offset_days": 1})]
    store = RuleStore()
    assert store.propose(_rule(offset=3), history).admitted


# ---------------------------------------------------------------------------
# Claim: "reruns are identical" - a close a controller can re-derive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_run_is_deterministic(seed: int) -> None:
    case = generate(n_batches=20, seed=seed)
    cfg = RunConfig(use_llm=False)
    a = reconcile(case.bundle, config=cfg)
    b = reconcile(case.bundle, config=cfg)
    assert sorted(f.key() for f in a.findings) == sorted(f.key() for f in b.findings)
    assert a.match_rate == b.match_rate


@pytest.mark.parametrize("seed", SEEDS)
def test_generator_is_reproducible(seed: int) -> None:
    a, b = generate(n_batches=15, seed=seed), generate(n_batches=15, seed=seed)
    assert len(a.bundle) == len(b.bundle)
    assert [d.subject_id for d in a.defects] == [d.subject_id for d in b.defects]


# ---------------------------------------------------------------------------
# Claim: money arithmetic is exact
# ---------------------------------------------------------------------------

def test_money_is_decimal_not_float() -> None:
    case = generate(n_batches=5, seed=7)
    for row in case.bundle.settlement:
        assert isinstance(row.gross_amount, Decimal)
        assert isinstance(row.net_amount, Decimal)


@pytest.mark.parametrize("seed", SEEDS)
def test_every_exception_carries_a_reason(seed: int) -> None:
    """An exception list with no reasons is not an honest exception list."""
    case = generate(n_batches=40, seed=seed)
    result = reconcile(case.bundle, config=RunConfig(use_llm=False))
    for exc in result.exceptions:
        assert exc.reason.strip(), f"{exc.subject_id} has no reason"


@pytest.mark.parametrize("seed", SEEDS)
def test_scorecard_totals_are_consistent(seed: int) -> None:
    case = generate(n_batches=40, seed=seed)
    result = reconcile(case.bundle, config=RunConfig(use_llm=False))
    card = score(case, result)
    assert card.tp + card.fn == len(case.defects)
    assert card.tp + card.fp == len(result.findings)


# ---------------------------------------------------------------------------
# Claim: "learning does not cost precision" - the campaign's core promise
# ---------------------------------------------------------------------------

def test_campaign_never_trades_precision_for_automation() -> None:
    """The rule store may only make the system cheaper, never looser.

    This guards the bug that dropped precision to 0.966: a rule whose verdict
    did not determine a defect class was stamping a label onto any exception
    whose narration it happened to match.
    """
    from taper.campaign import run_campaign

    run = run_campaign(months=5)
    for m in run.months:
        assert m.card.precision == 1.0, (
            f"month {m.month} precision fell to {m.card.precision:.3f} "
            f"with {m.rules_after} rule(s) active"
        )


def test_campaign_learns_the_recurring_charges() -> None:
    """Both persistent bank charges must be learned, with correct amounts."""
    from taper.campaign import run_campaign

    run = run_campaign(months=4)
    learned = {r.params.get("keyword"): r.params.get("amount") for r in run.store.rules}
    assert learned.get("PROC CHG") == "250.00"
    assert learned.get("SVC CHG") == "500.00"


def test_campaign_does_not_learn_one_off_anomalies() -> None:
    """One-off adjustments must never become standing rules - that is overfitting.

    There are exactly three learnable patterns in the generated world: the AXIS
    and SBI recurring charges, and the KOTAK reference label. Anything beyond
    those means the store generalised a random anomaly, so the ceiling is part
    of the assertion rather than a loose sanity bound.
    """
    from taper.campaign import run_campaign

    run = run_campaign(months=5)
    recurring_amounts = {"250.00", "500.00"}

    for rule in run.store.rules:
        if rule.kind == "adjustment_pattern":
            assert rule.params.get("category") == "bank_recurring_charge", (
                f"{rule.rule_id} generalised a non-recurring event"
            )
            assert rule.params.get("amount") in recurring_amounts, (
                f"{rule.rule_id} learned a one-off amount {rule.params.get('amount')}"
            )
        elif rule.kind == "narration_alias":
            assert rule.params.get("prefix") == "UTR"
            assert rule.params.get("marker"), "alias rule with no marker"
        else:
            raise AssertionError(f"unexpected rule kind learned: {rule.kind}")

    assert len(run.store) <= 3, "learned more rules than there are recurring patterns"


def test_learning_reduces_load() -> None:
    """Averaged over campaigns, month 1 must cost more than the steady state."""
    from taper.campaign import run_campaign_averaged

    rows = run_campaign_averaged(runs=6, months=4)
    first, later = rows[0], rows[-1]
    assert later.match_rate > first.match_rate
    assert later.exceptions < first.exceptions


# ---------------------------------------------------------------------------
# Claim: "the close report is self-contained and honest about its resolver"
# ---------------------------------------------------------------------------

def test_report_is_self_contained() -> None:
    """No external asset may be referenced - the report must open from disk,
    offline, years from now. A chart that needs a CDN is not a work product."""
    from taper.engine.llm import MockClient
    from taper.metrics.harness import score
    from taper.report import render

    case = generate(n_batches=20, seed=99)
    result = reconcile(case.bundle, config=RunConfig(use_llm=True), client=MockClient())
    html = render(result, score(case, result, "mock:offline-heuristic"), case, "2026-06")

    # The SVG xmlns is a namespace identifier, not a fetch, so it is stripped
    # before checking. Everything else that looks like a URL is a real dependency
    # on the network and must not exist.
    stripped = html.replace('xmlns="http://www.w3.org/2000/svg"', "")

    for forbidden in ("http://", "https://", "<script", "cdn.", "<link", "@import"):
        assert forbidden not in stripped, f"report references external resource: {forbidden}"
    assert "<svg" in html and "</html>" in html


def test_report_warns_when_run_on_the_mock() -> None:
    """A reader must never mistake heuristic output for model output."""
    from taper.engine.llm import MockClient
    from taper.metrics.harness import score
    from taper.report import render

    case = generate(n_batches=15, seed=7)
    result = reconcile(case.bundle, config=RunConfig(use_llm=True), client=MockClient())
    html = render(result, score(case, result, "mock:offline-heuristic"), case, "2026-06")
    assert "Offline heuristic, not a model" in html


def test_report_escapes_untrusted_text() -> None:
    """Bank narrations are external input and land in the report verbatim."""
    from taper.report import _esc

    assert _esc("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"


# ---------------------------------------------------------------------------
# Claim: "alias extraction is typed, not a model-authored pattern"
# ---------------------------------------------------------------------------

def test_alias_extraction_is_bounded_and_typed() -> None:
    """The model names a marker and a prefix; fixed code does the extraction.

    Guards against the store ever executing a model-authored regex against
    every future narration - an injection surface for no benefit.
    """
    from taper.engine.rules import _alias_extract

    p = {"marker": "REF", "prefix": "UTR"}
    assert _alias_extract(p, "KKBK NEFT REF 70000123456 RAZORPAY") == "UTR70000123456"
    assert _alias_extract(p, "KKBK NEFT REF: 70000123456 X") == "UTR70000123456"
    # marker absent
    assert _alias_extract(p, "NEFT 70000123456 RAZORPAY") is None
    # too few digits to be a reference
    assert _alias_extract(p, "NEFT REF 123 RAZORPAY") is None
    # absurdly long run is a parse gone wrong, not a reference
    assert _alias_extract(p, "REF " + "9" * 40) is None
    # no marker configured
    assert _alias_extract({"marker": "", "prefix": "UTR"}, "REF 70000123456") is None


def test_alias_never_widens_the_strict_regex() -> None:
    """Learning an alias for one bank must not change any other bank's parsing."""
    from datetime import date

    from taper.engine.matching import extract_utr
    from taper.engine.rules import Rule, RuleStore
    from taper.models import BankCredit, Money

    store = RuleStore()
    store.rules.append(Rule("narration_alias_001", "narration_alias",
                            {"marker": "REF", "prefix": "UTR"}, "setl_x", "2026-01-01"))

    clean = BankCredit("b1", Money("100.00"), date(2026, 1, 1),
                       "NEFT-UTR70000123456-RAZORPAY")
    assert extract_utr(clean, store) == "UTR70000123456"

    none_at_all = BankCredit("b2", Money("100.00"), date(2026, 1, 1),
                             "MERCHANT SETTLEMENT CREDIT")
    assert extract_utr(none_at_all, store) is None
