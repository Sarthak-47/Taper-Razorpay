# Taper

**A settlement reconciliation agent that shrinks its own AI usage over time.**

Razorpay AI Buildathon — Track 04, AI Finance Controller.

---

## The problem

A merchant on a payment gateway has three sources of truth that never agree:

| Source | What it says |
|---|---|
| **PG settlement report** | Individual payments, refunds, fees, GST on fees, chargeback holds |
| **Bank statement** | One lump credit per batch, messy free-text narration, T+1..T+3 lag |
| **Internal ledger** | Order IDs and the amounts the merchant *thinks* it sold |

The hard part is not row matching. One bank credit equals *N* payments minus *M*
refunds minus fees minus GST minus holds, arriving one to three days late, under
a narration that may not carry the UTR at all. It is many-to-one netting across
a timing lag, and finance teams do it by hand in Excel every month.

## The thesis

> **The agent's job is to make itself unnecessary.**

Every exception a human resolves is compiled back into a *typed, regression-tested
rule*. So LLM calls per 100 records fall month over month while match rate holds.
The system uses AI to manufacture deterministic logic, rather than putting AI on
the critical path forever.

This is a direct answer to the track brief's framing — that verification capacity,
not generation speed, is the bottleneck. The response to that bottleneck is not to
generate more. It is to shrink the surface that needs verifying.

## Architecture

Four layers, cheapest and surest first. Each item stops at the first layer that
can answer it.

```
  L0  exact       ID joins, fee/GST recomputation, duplicate detection
                  Arithmetic. Cannot be wrong if inputs are sane.          no model
  L1  fuzzy       date windows, narration regex, bounded subset netting
                  Bounded heuristics. Refuses to guess when ambiguous.     no model
  L2  rule        rules learned from past human decisions, replay-gated    no model
  L3  llm         narration parsing + ambiguous exception classification      model
                  Proposes only. Never decides.
```

**The design rule for L0/L1 is: assert only when unambiguous.** Anything with two
plausible explanations goes to the exception queue rather than being guessed at.
That conservatism is why deterministic precision is 1.000, and it is what makes
the auto-clear threshold trustworthy.

### The model never does arithmetic

Layer 3 proposes; `verify_proposal()` disposes. The model states *which credits it
believes belong together*, and the system sums them itself. If the numbers do not
reconcile, the proposal is refused and the item stays on the exception list with
the rejection reason recorded.

This gate exists because of a specific failure during development: the model
proposed a batch match whose credits were off by roughly ₹4,000, and the pipeline
accepted it because the response *looked* well-formed. Structural validity is not
numerical truth. Nothing is accepted on plausibility alone now.

You can watch the gate fire on a live run — several proposals are rejected with
`claimed split is off by ...` and correctly routed to a human.

### The admission gate (the part that makes learning safe)

A learned rule that is wrong silently corrupts every future close, so no rule
enters the store on the model's say-so:

1. The model proposes a **typed** rule — one of four shapes (`bank_timing`,
   `narration_alias`, `fee_variant`, `adjustment_pattern`), each with a fixed
   parameter set. A model that can only fill blanks in a known form cannot invent
   a rule that does something the system was never designed to do.
2. The candidate is **replayed against every case already confirmed correct**.
3. If it would change even one of them, it is **rejected**, and the originating
   item stays an exception.

Deliberately more conservative than necessary. A missed automation costs one
manual review; a bad rule costs silent corruption across every close after it.

## Results

### The taper

Five consecutive monthly closes, rule store carried forward, **averaged over 8
independent campaigns** (a single campaign is one noisy draw — averaging
separates learning from month-to-month variance):

| Month | Rules | Model calls / 100 records | Exceptions | Clean match rate | Precision | Recall |
|---|---|---|---|---|---|---|
| 1 | 1.6 | 0.96 | 12.0 | 48.6% | 1.000 | 0.902 |
| 2 | 2.0 | 0.76 | 9.4 | 88.7% | 1.000 | 0.908 |
| 3 | 2.0 | 0.69 | 8.4 | 93.3% | 1.000 | 0.917 |
| 4 | 2.0 | 0.64 | 7.8 | 93.7% | 1.000 | 0.931 |
| 5 | 2.0 | 0.71 | 8.4 | **97.6%** | 1.000 | 0.921 |

**Model calls per 100 records fall 27%. Clean match rate goes 48.6% → 97.6%.
Human reviews per close drop 12.0 → 8.4. Precision stays at 1.000 the whole way.**

The system learns exactly two rules — the AXIS ₹250 processing charge and the
SBI ₹500 service charge — and those two rules are worth ~49 points of match rate,
because before they exist every payout from those banks is an unexplained
shortfall that costs an exception, a model call, and a human's attention.

Note what does **not** happen: the one-off adjustments are never learned. They are
genuine anomalies, they stay anomalies, and a test enforces that the store never
generalises one. A rule store that absorbed those would be overfitting to noise.

```bash
python -m taper.cli campaign --months 5 --average-runs 8
```

### Single-close accuracy

Deterministic layers only, 40 batches (~1,300 records) per seed:

| Seed | Records | Precision | Recall | Clean match rate |
|---|---|---|---|---|
| 7 | 1,234 | 1.000 | 1.000 | 90.0% |
| **99** *(held out)* | 1,334 | **1.000** | **0.931** | 91.7% |
| 1234 | 1,301 | 1.000 | 0.920 | 89.2% |
| 2025 | 1,261 | 1.000 | 0.927 | 97.2% |

Precision is 1.000 across every seed — **zero false positives**, enforced as a
test rather than merely observed. Recall sits at 0.92–0.93 on unseen seeds, and
the ~7% it misses is not silently dropped: it lands on the exception list with a
stated reason.

Per-class precision and recall, calibration, false-positive cost in review-minutes,
and the auto-clear operating point all print from `reconcile`.

### Two bugs the harness caught, and what they cost

Both were found by ground-truth scoring, not by reading the code:

1. **Duplicate captures were double-reported** as missing ledger entries — 20 false
   positives on one seed. A duplicate has no ledger entry *by definition*, so two
   findings described one problem and a controller would chase the same rupee twice.
2. **A learned rule fabricated a defect class.** When an `adjustment_pattern` rule
   matched a narration, the pipeline defaulted its verdict to
   `UNRECORDED_ADJUSTMENT` — stamping that label onto any exception it happened to
   match, including orphan bank credits it had no opinion about. Precision decayed
   from 1.000 to 0.966 *as the rule store grew*, which is the worst possible
   failure shape: the system got less trustworthy the more it learned.

The fix for the second one became a design rule — **a rule with no verdict on an
item must leave the item alone** — and a test that fails if any month of a campaign
drops below 1.000 precision.

### Honest limitations

- **Combined defects defeat the deterministic solver.** A batch that is *both*
  split across two credits *and* short by an unrecorded adjustment satisfies
  neither the exact-sum nor the subset path. These are the bulk of the remaining
  exceptions. `verify_proposal` also does not yet verify a combined claim, so the
  model cannot currently resolve them either.
- **Subset netting is capped** at 3 credits and a 12-credit window. Unbounded
  subset-sum is exponential and would hang the run; a bounded search that routes
  overflow to the exception list is strictly better than a solver that stalls.
- **Injection rates are higher than production base rates**, deliberately, so each
  defect class has enough samples for its recall to mean anything. A class with
  three instances has no measurable recall.
- **The offline mock is not a model.** See below.

## Running it

Everything installs into a project-local virtualenv — nothing touches the global
Python.

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

On macOS or Linux use `.venv/bin/python` instead. The deterministic layers and the
offline mock have **no third-party dependencies** — `pip install -e .` alone is
enough to run everything except layer 3 against a real model.

Deterministic only — no API key needed, and the honest baseline:

```bash
python -m taper.cli --no-llm reconcile --seed 99
```

Full stack against a real model:

```bash
cp .env.example .env   # then add your key
python -m taper.cli reconcile --seed 99
```

The headline result — consecutive closes, rule store carried forward:

```bash
python -m taper.cli campaign --months 5 --average-runs 8
```

Does the model earn its place?

```bash
python -m taper.cli ablate --seed 99
```

Tuning set vs held-out set:

```bash
python -m taper.cli evaluate --tune-seed 7 --holdout-seed 99
```

Tests:

```bash
python -m pytest tests/ -v
```

### On `--mock`

`--mock` runs an offline heuristic so the repo works end-to-end with no API key,
for CI and for reviewers. **It is not a model, and ablation numbers produced
against it are heuristic-vs-heuristic and prove nothing about LLM value.** Every
metrics artifact records which client produced it, and the CLI prints a warning
banner whenever the mock is in use. Reported figures must come from the real path.

## Why the ground truth is trustworthy

`generator.py` injects defects at declared rates and records exactly what it broke
and where. Precision and recall are therefore computed against real ground truth
rather than eyeballed off a demo. The tuning seed and the held-out seed differ
only by RNG seed, so "we never tuned on the held-out set" is a checkable claim,
not a promise.

Nine injected defect classes: `duplicate_capture`, `cross_cycle_refund`,
`fee_overcharge`, `missing_utr`, `narration_drift`, `timing_shift`,
`split_settlement`, `unrecorded_adjustment`, `missing_ledger_entry`.

## Layout

```
src/taper/
  models.py            domain types; Decimal money, never float
  generator.py         synthetic 3-source data + labelled defect injection
                       incl. persistent BankProfiles - the structure worth learning
  campaign.py          consecutive closes, HumanOracle, the rule-growth curve
  engine/
    results.py         Finding / BatchMatch / Exception_, with layer attribution
    matching.py        L0 + L1, the deterministic core
    rules.py           learned rules + the retroactive admission gate
    llm.py             L3 client, prompt, and the propose-then-verify gate
    pipeline.py        layer orchestration
  metrics/harness.py   per-class scoring, calibration, ablation, FP cost
  cli.py               reconcile / ablate / evaluate
tests/                 invariants that guard the claims above
.github/workflows/     lint, test on 3.11-3.13, CLI smoke runs, metrics re-assertion
```

## The human in the loop

`HumanOracle` in `campaign.py` simulates the controller who investigates an
exception. That is not a cheat — it is the supervision signal. In production a
human works the exception queue and writes down what each item turned out to be;
the oracle looks up the injected ground truth and does the same.

What is *not* simulated is the part that matters: a rule the human and the model
both endorsed still has to survive the retroactive admission gate before it can
affect a single future close.

## Status

Running end-to-end: 30 tests green, five CLI commands, no API key required.

Remaining work: real-model evaluation runs (everything to date is either
deterministic or against the offline mock), combined-defect verification, and
`bank_timing` / `narration_alias` rule learning — the rule types exist and are
gated, but only `adjustment_pattern` is currently proposed by the oracle.
