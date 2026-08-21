"""Command line entry point.

    python -m taper.cli reconcile --seed 99
    python -m taper.cli ablate    --seed 99
    python -m taper.cli evaluate  --tune-seed 7 --holdout-seed 99
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine.pipeline import RunConfig, reconcile
from .engine.results import Layer
from .generator import generate
from .metrics.harness import (
    Ablation,
    auto_clear_operating_point,
    calibration,
    layer_breakdown,
    score,
)
from .models import DefectClass

RULE_STORE_PATH = Path("data/rules.json")
BAR = "=" * 72


def _client_and_config(args) -> tuple[RunConfig, str]:
    use_real = not args.mock
    cfg = RunConfig(use_llm=not args.no_llm, use_real_llm=use_real)
    name = "anthropic:claude-sonnet-5" if use_real else "mock:offline-heuristic"
    if args.no_llm:
        name = "none (deterministic only)"
    return cfg, name


def _warn_mock(args) -> None:
    if args.mock and not args.no_llm:
        print(
            "\n  !! Running on the OFFLINE HEURISTIC, not a model.\n"
            "     These numbers are for CI and smoke-testing only. Do not report\n"
            "     them as LLM results - rerun without --mock and with\n"
            "     ANTHROPIC_API_KEY set before putting numbers in the writeup.\n",
            file=sys.stderr,
        )


def cmd_reconcile(args) -> None:
    case = generate(n_batches=args.batches, seed=args.seed)
    cfg, client_name = _client_and_config(args)
    _warn_mock(args)

    from .engine.rules import RuleStore

    store = RuleStore(RULE_STORE_PATH if args.persist_rules else None)
    result = reconcile(case.bundle, store=store, config=cfg)
    card = score(case, result, client_name)

    print(BAR)
    print(f"  RECONCILIATION - period {case.bundle.period}  (seed {args.seed})")
    print(BAR)
    print(f"  records processed     {card.records}")
    print(f"  batches matched       {len(result.matches)}")
    print(f"  clean match rate      {card.match_rate:6.1%}")
    print(f"  findings              {len(result.findings)}")
    print(f"  exceptions (to human) {card.exceptions}")
    print(f"  money flagged         Rs.{card.money_flagged:,}")
    print(f"  throughput            {card.throughput:,.0f} records/sec")
    print(f"  llm calls             {card.llm_calls}  ({card.llm_calls_per_100:.2f} per 100 records)")
    print(f"  resolved w/o a model  {card.deterministic_share:6.1%}")
    print(f"  client                {card.client_name}")

    print(f"\n  {'-' * 68}")
    print("  ACCURACY vs INJECTED GROUND TRUTH")
    print(f"  {'-' * 68}")
    print(f"  {'defect class':<26}{'sup':>5}{'TP':>5}{'FP':>5}{'FN':>5}{'prec':>8}{'rec':>8}")
    for dc in DefectClass:
        s = card.per_class[dc]
        if not s.support and not s.fp:
            continue
        print(f"  {dc.value:<26}{s.support:>5}{s.tp:>5}{s.fp:>5}{s.fn:>5}"
              f"{s.precision:>8.3f}{s.recall:>8.3f}")
    print(f"  {'OVERALL':<26}{card.tp + card.fn:>5}{card.tp:>5}{card.fp:>5}{card.fn:>5}"
          f"{card.precision:>8.3f}{card.recall:>8.3f}")
    print(f"\n  false-positive cost   {card.false_positive_cost_minutes:.0f} review-minutes "
          f"({card.fp} false flags x 4 min)")

    print(f"\n  {'-' * 68}")
    print("  WHERE THE WORK HAPPENED")
    print(f"  {'-' * 68}")
    for layer in Layer:
        n = layer_breakdown(result).get(layer.value, 0)
        share = n / len(result.findings) if result.findings else 0
        print(f"  {layer.value:<22}{n:>5}  {share:6.1%}")

    op = auto_clear_operating_point(case, result)
    print(f"\n  {'-' * 68}")
    print("  AUTO-CLEAR OPERATING POINT (target precision 0.99)")
    print(f"  {'-' * 68}")
    if op["coverage"]:
        print(f"  at confidence >= {op['threshold']}: auto-clear {op['coverage']:.1%} of findings "
              f"at {op['precision']:.3f} precision")
        print(f"  {op['auto_cleared']} auto-cleared, {op['routed_to_human']} to a human, "
              f"{op['review_minutes_saved']:.0f} review-minutes saved")
    else:
        print("  no threshold reaches target precision - route everything to a human")

    bins = calibration(case, result)
    if bins:
        print(f"\n  {'-' * 68}")
        print("  CALIBRATION  (stated confidence vs observed hit rate)")
        print(f"  {'-' * 68}")
        print(f"  {'bucket':<14}{'n':>6}{'stated':>9}{'observed':>10}{'gap':>8}")
        for b in bins:
            print(f"  {f'{b.lo:.1f}-{b.hi:.1f}':<14}{b.n:>6}{b.midpoint:>9.2f}"
                  f"{b.observed:>10.2f}{b.gap:>+8.2f}")

    if result.exceptions:
        print(f"\n  {'-' * 68}")
        print(f"  EXCEPTION LIST - {len(result.exceptions)} item(s) a human must look at")
        print(f"  {'-' * 68}")
        for exc in result.exceptions[: args.max_exceptions]:
            print(f"  [{exc.kind}] {exc.subject_id}")
            print(f"      {exc.reason}")
        if len(result.exceptions) > args.max_exceptions:
            print(f"      ... and {len(result.exceptions) - args.max_exceptions} more")

    if args.persist_rules:
        store.save()
        print(f"\n  rule store: {len(store)} rule(s) -> {RULE_STORE_PATH}")
    print(BAR)


def cmd_ablate(args) -> None:
    case = generate(n_batches=args.batches, seed=args.seed)
    _warn_mock(args)
    use_real = not args.mock

    det = reconcile(case.bundle, config=RunConfig(use_llm=False))
    full = reconcile(case.bundle, config=RunConfig(use_llm=True, use_real_llm=use_real))

    ab = Ablation(
        deterministic_only=score(case, det, "none"),
        full_stack=score(case, full, "anthropic" if use_real else "mock"),
    )
    a, b = ab.deterministic_only, ab.full_stack

    print(BAR)
    print(f"  ABLATION - does the model earn its place?  (seed {args.seed})")
    print(BAR)
    print(f"  {'':<26}{'deterministic':>15}{'+ model':>12}{'delta':>10}")
    rows = [
        ("precision", a.precision, b.precision),
        ("recall", a.recall, b.recall),
        ("match rate", a.match_rate, b.match_rate),
    ]
    for label, x, y in rows:
        print(f"  {label:<26}{x:>15.3f}{y:>12.3f}{y - x:>+10.3f}")
    print(f"  {'exceptions left':<26}{a.exceptions:>15}{b.exceptions:>12}"
          f"{b.exceptions - a.exceptions:>+10}")
    print(f"  {'llm calls':<26}{a.llm_calls:>15}{b.llm_calls:>12}")
    print(f"  {'llm calls / 100 records':<26}{a.llm_calls_per_100:>15.2f}"
          f"{b.llm_calls_per_100:>12.2f}")
    print(f"\n  VERDICT\n  {ab.verdict()}")
    print(BAR)


def cmd_evaluate(args) -> None:
    """Tune-vs-holdout. The holdout seed is never used to change anything."""
    _warn_mock(args)
    use_real = not args.mock
    cfg = RunConfig(use_llm=not args.no_llm, use_real_llm=use_real)

    print(BAR)
    print("  EVALUATION - tuning set vs held-out set")
    print(BAR)
    print(f"  {'set':<12}{'seed':>6}{'records':>9}{'prec':>8}{'rec':>8}"
          f"{'match':>8}{'exc':>6}{'llm/100':>9}")
    for label, seed in (("tune", args.tune_seed), ("HELD-OUT", args.holdout_seed)):
        case = generate(n_batches=args.batches, seed=seed)
        res = reconcile(case.bundle, config=cfg)
        c = score(case, res)
        print(f"  {label:<12}{seed:>6}{c.records:>9}{c.precision:>8.3f}{c.recall:>8.3f}"
              f"{c.match_rate:>8.1%}{c.exceptions:>6}{c.llm_calls_per_100:>9.2f}")
    print("\n  The held-out seed was never inspected while tuning thresholds.")
    print(BAR)


def cmd_campaign(args) -> None:
    """The headline result: consecutive closes, carrying what we learn."""
    from .campaign import run_campaign, run_campaign_averaged

    _warn_mock(args)
    cfg, _ = _client_and_config(args)

    run = run_campaign(months=args.months, n_batches=args.batches,
                       base_seed=args.base_seed, config=cfg)

    print(BAR)
    print(f"  CAMPAIGN - {args.months} consecutive closes, rule store carried forward")
    print(BAR)
    print(f"  {'month':<9}{'period':<10}{'rules':>7}{'learned':>9}{'rejected':>10}"
          f"{'llm/100':>9}{'exc':>6}{'match':>9}{'prec':>8}")
    for m in run.months:
        print(f"  {m.month:<9}{m.period:<10}{m.rules_after:>7}{m.rules_learned:>9}"
              f"{m.rules_rejected:>10}{m.card.llm_calls_per_100:>9.2f}{m.exceptions:>6}"
              f"{m.card.match_rate:>9.1%}{m.card.precision:>8.3f}")

    print("\n  RULES LEARNED")
    if run.store and len(run.store):
        for r in run.store.rules:
            amt = r.params.get("amount")
            detail = f"{r.params.get('keyword', '')} -> {r.params.get('category', '')}"
            print(f"    {r.rule_id:<26}{detail:<44}{'Rs.' + str(amt) if amt else ''}")
    else:
        print("    none admitted")
    if run.store and run.store.rejected:
        print(f"\n  RULES REJECTED BY THE ADMISSION GATE: {len(run.store.rejected)}")
        for rej in run.store.rejected[:3]:
            print(f"    {rej.rule.rule_id}: {rej.reason}")
            for reg in rej.regressions[:2]:
                print(f"        {reg}")

    print(f"\n  VERDICT\n  {run.verdict()}")

    if args.average_runs > 1:
        rows = run_campaign_averaged(
            runs=args.average_runs, months=args.months,
            n_batches=args.batches, base_seed=args.base_seed, config=cfg,
        )
        print(f"\n  {'-' * 68}")
        print(f"  AVERAGED OVER {args.average_runs} INDEPENDENT CAMPAIGNS")
        print("  (a single campaign is one noisy draw; this separates learning from variance)")
        print(f"  {'-' * 68}")
        print(f"  {'month':<9}{'rules':>7}{'llm/100':>10}{'exc':>8}{'match':>9}"
              f"{'prec':>8}{'rec':>8}{'reviews':>9}")
        for m in rows:
            print(f"  {m.month:<9}{m.rules:>7.1f}{m.llm_calls_per_100:>10.2f}"
                  f"{m.exceptions:>8.1f}{m.match_rate:>9.1%}{m.precision:>8.3f}"
                  f"{m.recall:>8.3f}{m.human_reviews:>9.1f}")
        a, b = rows[0], rows[-1]
        drop = 1 - (b.llm_calls_per_100 / a.llm_calls_per_100) if a.llm_calls_per_100 else 0
        print(f"\n  model calls / 100 records  {a.llm_calls_per_100:.2f} -> "
              f"{b.llm_calls_per_100:.2f}   ({drop:.0%} reduction)")
        print(f"  clean match rate           {a.match_rate:.1%} -> {b.match_rate:.1%}")
        print(f"  human reviews per close    {a.human_reviews:.1f} -> {b.human_reviews:.1f}")
        print(f"  precision                  {a.precision:.3f} -> {b.precision:.3f}   "
              f"(held while automating)")
    print(BAR)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="taper", description=__doc__)
    p.add_argument("--mock", action="store_true",
                   help="use the offline heuristic instead of a real model")
    p.add_argument("--no-llm", action="store_true", help="deterministic layers only")
    p.add_argument("--batches", type=int, default=40)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("reconcile", help="run one close and print the report")
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--persist-rules", action="store_true")
    r.add_argument("--max-exceptions", type=int, default=10)
    r.set_defaults(func=cmd_reconcile)

    a = sub.add_parser("ablate", help="deterministic vs full stack")
    a.add_argument("--seed", type=int, default=99)
    a.set_defaults(func=cmd_ablate)

    c = sub.add_parser("campaign", help="consecutive closes; the rule-growth curve")
    c.add_argument("--months", type=int, default=5)
    c.add_argument("--base-seed", type=int, default=500)
    c.add_argument("--average-runs", type=int, default=8,
                   help="independent campaigns to average over (1 = single run only)")
    c.set_defaults(func=cmd_campaign)

    e = sub.add_parser("evaluate", help="tuning set vs held-out set")
    e.add_argument("--tune-seed", type=int, default=7)
    e.add_argument("--holdout-seed", type=int, default=99)
    e.set_defaults(func=cmd_evaluate)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
