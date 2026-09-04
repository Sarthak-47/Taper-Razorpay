# Taper

**A settlement reconciliation agent that shrinks its own AI usage over time.**

[![CI](https://github.com/Sarthak-47/Taper-Razorpay/actions/workflows/ci.yml/badge.svg)](https://github.com/Sarthak-47/Taper-Razorpay/actions/workflows/ci.yml)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%20%E2%80%93%203.13-blue)](pyproject.toml)
[![Dependencies: none](https://img.shields.io/badge/runtime%20dependencies-none-success)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

Razorpay AI Buildathon — Track 04, AI Finance Controller.

---

## Review this in five minutes

A merchant's settlement report, their bank statement and their own ledger never
agree. Taper reconciles all three, reports what is wrong, and is honest about
what it could not resolve. **Every exception a human answers is compiled into a
typed, regression-tested rule — so it calls the model less every month while
getting more accurate.**

### Start with the number that argues against this project

This is an AI buildathon entry whose own ablation says the AI is not earning its
place. That is the first thing you should see, not the last:

```bash
python -m taper.cli --mock ablate --seed 99
```

```
                    deterministic     + model     delta
precision                   1.000       1.000    +0.000
recall                      0.928       0.928    +0.000
match rate                  0.778       0.778    +0.000
exceptions left                10          10        +0
llm calls                       0          10

VERDICT: the model added nothing measurable on this run. On this data the
deterministic layers are sufficient and layer 3 should be disabled.
```

I could have deleted that command. Three reasons it leads instead.

**It is the thesis, not a bug.** The deterministic layers were built to make the
model unnecessary. An ablation showing +0.000 is that design working — and the
only way to say so honestly is to measure the model's contribution and report it
even when the answer is zero. A system that cannot tell you its AI is useless
also cannot tell you when it is essential.

**What the model actually did is the finding.** The block above is the offline
resolver; [the same ablation against a real local model](docs/RESULTS.md#layer-3-on-a-local-model--the-ablation-in-full)
(qwen2.5:14b, 13 calls) is where it gets interesting. Five calls were honest
declines. The other **eight were confabulations** — adjustments of ₹1,123,
₹511.60, ₹98.12 proposed against gaps of forty to seventy thousand.
Plausible-looking numbers that explain nothing.

**All eight were refused by `verify_proposal()`**, which re-derives every claim
from the source amounts in code the model never touches. So the honest summary
is not "the AI did nothing". It is *the model tried to put wrong numbers in the
books eight times and the architecture stopped it eight times, and precision
never left 1.000.* That is a load-bearing result about where a language model
belongs near money — and you only get it by measuring the thing you hoped would
work.

**It is a claim about this data, not about models.** The tested model is a local
qwen2.5:14b, because there is no API key here and the writeup will not pretend
otherwise. A stronger model may well clear the bar; the ablation is the
instrument that would say so, and it is wired to Anthropic, Ollama, or any
OpenAI-compatible endpoint without a code change.

**The taper is what happens once you accept this.** If the expensive layer earns
little, the correct engineering response is to need it less every month — which
is the rest of this README.

```bash
python -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"
```

```bash
python -m taper.cli doctor
```

`doctor` reports what your machine can run and prints the exact command to type
next — a local model if Ollama is up, an API key if one is set, otherwise the
deterministic path. **Nothing is required.** The deterministic layers have zero
third-party dependencies and reproduce most of what follows.

Then the headline result:

```bash
python -m taper.cli campaign --months 6 --average-runs 8
```

| | Month 1 | Month 6 |
|---|---|---|
| Model calls / 100 records | 1.12 | **0.50** *(−56%)* |
| Clean match rate | 65.4% | **95.9%** |
| Human reviews per close | 14.1 | **5.8** |
| **Precision** | **1.000** | **1.000** |

Four more things worth thirty seconds each:

| Command | What it shows |
|---|---|
| `taper reconcile --clean` | The negative control — a period where nothing is wrong. It finds nothing, escalates nothing, calls no model |
| `taper aging` | Is the queue shrinking, or the same items every month? The control that makes the taper falsifiable |
| `taper materiality` | Which findings deserve a person — and the aggregation rule that stops a floor from hiding a pattern |
| `taper signoff` | The stopping rules — and each one refusing a close it should refuse |
| `taper forecast` | Forward cash — when the money owed will land, as a backtested range, with overdue money kept off the curve |
| `taper cash` | The position a CFO reads first — in the bank, withheld, owed each way, and what could not be placed |
| `taper stress` | Under 6.6× ambiguity it matches *nothing* and escalates everything — **zero false findings at any level.** It fails safe, not wrong |
| `taper risk` | Reviewing the riskiest 10% of batches catches **56%** of all escalations (5.6×) |
| `taper forensics` | Benford first-digit analysis — does this population look **lived, or authored**? |
| `taper risk --compare` | Why the shipped model has no dependencies — three measurements, not a preference |
| `taper ingest --format razorpay` | Reads the settlement recon schema **Razorpay actually publishes** — paise, epochs, `on_hold` — to the same close digest |
| `taper ingest` | Reconciles real CSV files — drifting headers, per-row error reporting, and a re-derivable close digest |
| `taper resolve` | Teach the store what a human worked out — through the admission gate |
| `taper doctor` | What this machine can run, and the exact command to type next |
| `taper --llm ollama reconcile` | Layer 3 against a **local model, no API key** — and an ablation that honestly reports it added nothing |
| `taper redteam` | A prompt-injection payload in a bank narration — and proof a **fully compromised model** still moves nothing |
| `taper drift` | A bank reprices mid-campaign — the engine names the rule that went stale, retires it, relearns |
| `taper report` | The close package a controller actually receives — one self-contained HTML file, sortable and filterable, that still reads with JavaScript off |

---

## When it refuses to sign

Everything else here decides what is *true* about a close. This decides whether
the close is fit to be signed at all — which is the thing a controller is
actually accountable for.

**An engine that always produces a report produces one for a truncated bank
statement too**, and it looks identical to a good close: a match rate, a findings
table, an exception list. It is simply wrong, and nothing in it says so. A number
that is always emitted carries no information about whether it should have been.

```bash
python -m taper.cli signoff --seed 99
```

```
settlement file never arrived     DO NOT SIGN
   !! source_present: settlement is empty
   !! unattributed_money: Rs.16,344,577.29 (100.0%) tied to no batch

bank statement truncated          DO NOT SIGN
   !! match_rate_floor: 12.5% of batches matched

wrong period exported             DO NOT SIGN
   !! unattributed_money: Rs.14,573,970.55 (89.2%) tied to no batch
```

A close ends in one of three states — **SIGN**, **SIGN WITH CAVEATS**, or **DO
NOT SIGN** — and a refused close carries that banner at the top of the HTML
report, *above* the numbers, because from the KPI row down it looks exactly like
a good one.

### Two stops that change what the close does

The model budget and the refusal streak are enforced inside the pipeline, not
reported after the fact. A stopping rule that only ever appears in a summary is a
comment with a threshold in it.

| | model calls | exceptions |
|---|---|---|
| unbounded | 10 | 10 |
| budget 0.2 / 100 records | **2** | 10 |
| abandon after 2 refusals | **4** | 10 |

Precision is unchanged in all three. **A stop costs recall and human time, never
correctness** — items it holds back land on the exception list rather than being
decided without evidence.

### Why these thresholds

Each was set against measured behaviour on healthy closes, not chosen because it
sounded prudent — *a bound that never binds is decoration, and one that binds
silently is worse.*

The refusal streak is the clearest case, and it is in the repo because it went
wrong first. At 5 consecutive refusals it fired on **perfectly healthy closes**
and pulled month-one model calls from 1.12 to 0.78. That would have been worse
than useless: the taper is a claim about *learning* reducing model calls, and a
stop quietly truncating them would have taken the credit. Measured across eight
seeds, the longest refusal run on a healthy close was 10 — so the default is 12,
and a test fails the build if any default stop fires on a healthy close.

---

## Forward cash

The [cash position](#the-cash-position) says what the merchant *has*. This says
what they will have and on which day — the question that decides whether payroll
clears. The track brief names a forward cash forecaster as one of its example
directions, and a forecast is the easiest thing in this repository to fake: pick
a lag, draw a curve, and nobody can tell it is wrong until the money fails to
arrive.

```bash
python -m taper.cli --no-llm forecast --seed 99 --batches 60
```

**The model is one fact the reconciler already produces for free.** Every matched
batch carries the date the gateway settled it and the date the bank credited it.
The gap between them is a *measurement*, not an assumption — 51 of them on this
seed, giving T+2 typical and T+1 to T+3 across the band.

Three commitments hold it up.

**It is a range, not a number.** A single date implies confidence nothing here
has earned, so arrivals are p10/p50/p90 from the observed distribution. Below
eight observations it refuses to quote a percentile at all and says why.

**It is backtested, and the backtest can fail.** Fit on the earliest batches,
predict the later ones — split by *date*, because that is the only split matching
how a forecast is used. You predict forward from what you have seen, never from a
sample containing the future.

```
Fitted on 30 batch(es), tested on 21. Median error 1.0 day(s).
The p10-p90 band held 90% of outcomes against 80% claimed.
```

A p10–p90 band should hold ~80%. Reporting measured coverage beside the claimed
figure is what stops a narrow band passing for an accurate one, and a test fails
the build if coverage drops far below what the band promises.

**Overdue money is not forecast.** This is the one that matters most:

```
OVERDUE - past the slowest arrival ever observed
  setl_99_005    Rs.282,193   settled 2026-06-11   due by 2026-06-14   105d late
  ...
  Rs.3,400,019.96 owed and late.
```

A batch settled in June whose slowest *observed* arrival was July is not cash
arriving in October — it is a batch nobody chased. Putting it on the curve would
be the single most misleading thing this module could do, so ₹3.4M sits in a
collections bucket while only ₹398k reaches the forward view. Withheld funds and
unsettled revenue are excluded for the same reason: neither has a knowable
arrival date, so neither gets smeared across the horizon to flatter the total.

---

## It reads Razorpay's real settlement schema

Everything else here reconciles data this repository invented, which is the
fairest criticism of the project. So it also reads the shape Razorpay genuinely
publishes — the [settlement recon report](https://razorpay.com/docs/api/settlements/fetch-recon/):
`entity_id`, `settlement_id`, `settlement_utr`, `on_hold`, `settled`, `fee`,
`tax`, `type`.

```bash
python -m taper.cli ingest --settlement data/sample/razorpay-recon.csv     --bank data/sample/bank.csv --ledger data/sample/ledger.csv
```

**Same period, same close digest** as the generic CSV — `taper-close-v1:520993d8d5a0`.
That equality is the test: if the adapter lost a field or misplaced a decimal,
the digest would move.

Four things in that format would silently corrupt a close, and they are why this
is an adapter rather than a column-name alias:

**Amounts are in currency subunits.** `amount: 150000` is ₹1,500.00, not
₹150,000. Point the generic loader at a recon export and every total comes out a
hundred times too large — with no error, no warning, and every figure still
plausible. That is the worst class of bug in a finance tool, so the conversion is
exact integer paise over `Decimal("100")`, and a *fractional* paisa raises rather
than silently dividing a rupee file again.

**Timestamps are Unix epoch seconds**, read in UTC. Local time moves a
near-midnight settlement into the wrong close.

**`on_hold` and `settled` decide whether money exists.** A row can be in the
recon report and not be in your bank. Both map onto defect classes this engine
already reports — dropping them would turn money that has not arrived into money
that has.

**`type` includes `transfer`.** A Route movement to a linked account is real and
is not a payment; it becomes an adjustment rather than being forced into a
category that flatters the total.

The format is detected from the *header*, not the filename, and `--format` always
overrides. **No real merchant data is used anywhere in this project** — the
schema is public, and reading it correctly is a different problem from having
somebody's transactions.

---

## Is the queue actually shrinking?

The headline says human reviews fall 14.1 → 5.8. That is true, and **it is not
sufficient.** Two completely different worlds produce that same chart:

- the engine learns each recurring situation, those items stop appearing, and
  what's left in month six is genuinely new work; or
- the volume of novel problems happens to fall while a handful of items nobody
  can resolve get re-raised every single close.

A falling *count* cannot tell those apart. Only identity can — so `taper aging`
gives an exception a name that survives the period it was raised in, and asks how
many closes it has been asked in.

```bash
python -m taper.cli --mock aging --months 6
```

```
                            standing  recurring   stale  episodic
with the learning loop             3          0       0        54
with learning disabled             3          1       1        65
```

Same six closes, same seeds, same data. The only difference is whether the store
is allowed to keep what a human worked out:

```
WITH THE LEARNING LOOP
  unknown_rate_card::method=intl_card - asked in month 1

WITH LEARNING DISABLED
  unknown_rate_card::method=intl_card - asked in months 1, 2, 3, 4, 5, 6  STALE
```

**That is the control.** If the falling exception count were an artefact of the
data rather than the loop, both columns would look the same. A test asserts the
two runs must *disagree* — if they ever converge, the headline chart means
nothing and CI says so.

The hard part is identity. Subject ids are period-scoped by construction
(`setl_500_004` exists in exactly one close), so keying on them would prove that
nothing ever recurs — a measurement that always passes and therefore says
nothing. Exceptions split honestly instead: **standing** questions outlive their
period ("what rate card covers `intl_card`" is the same question in June and
November), while **episodic** ones are about one specific batch and cannot recur.
They're counted separately, because averaging them together would dilute the
recurrence rate toward zero and make the metric look good for a reason unrelated
to learning.

An item on its third consecutive appearance stops being a queue entry and becomes
a process that has stalled. It gets reported as one.

---

## What is worth chasing

Every finding this engine produces is correct — precision is 1.000 and nothing
here changes that. But **correct is not the same as worth chasing.** A ₹1.77 fee
overcharge is real, and no controller alive is opening a ticket with the gateway
about it. Hand over seventeen of those alongside a ₹470,000 duplicate capture and
you haven't helped; you've buried the one that matters.

```bash
python -m taper.cli --no-llm materiality --seed 99 --batches 60
```

The obvious move is a floor: waive anything under ₹1,000. The obvious move is
also **a blind spot at exactly the size an error would choose.** Systematic
problems present as many small items — that is what *systematic* means.

So waived items are grouped by what a human would claim them as, and any group
clearing the aggregate floor comes back — not as fourteen tickets, but as one:

```
14 x unrecorded_adjustment, none individually above the floor, Rs.5,481.00 together
    largest single item Rs.500.00, below the Rs.1,000.00 floor
```

That is the bank's standing charge, which is trivial per payout and a
conversation worth having per quarter. **The floor removes noise; the aggregate
rule stops the floor from removing a pattern.**

The command sweeps a ladder of floors, because a controller doesn't want a
default, they want the shape of the trade:

| Floor | Items left | Saved | Claims | Waived | % of money |
|---|---|---|---|---|---|
| ₹250 | 174 | 33 | 0 | ₹2,347 | 0.06% |
| ₹1,000 | 151 | 56 | 1 | ₹8,270 | 0.21% |
| ₹2,500 | 137 | 70 | 4 | ₹12,742 | 0.32% |
| ₹5,000 | 122 | 85 | **7** | **₹0** | **0.00%** |

Read the last row twice. Past a point the floor stops removing *money* entirely
while it is still removing *items* — every class clears the aggregate threshold
on its own, so 85 fewer things to work through costs nothing unexamined. Two
curves with different shapes, and the gap between them is the whole feature.

**Two things are never waived, at any floor.** Findings with no rupee amount — a
missing UTR costs nothing and breaks matchability, and materiality is a statement
about money. And exceptions: "too small to investigate" is a judgement you can
only make about something you understand, and by definition those aren't
understood yet.

Nothing is deleted. The waived total is printed beside the floor that produced
it, and a test asserts the close digest and precision are byte-identical before
and after — a review threshold must never be able to change the reported accuracy
of a close.

---

## What it does when nothing is wrong

Every accuracy number in this README is measured on periods that contain
defects. Precision says how often a finding is *wrong*. It cannot say whether a
**clean** period produces findings at all — and that is the failure mode nobody
charges for. An engine that manufactures a dozen exceptions on a quiet month
spends human attention no metric here would ever bill it for, and it teaches the
controller to skim the queue, which is how the one real finding eventually gets
waved through.

```bash
python -m taper.cli --no-llm reconcile --clean --seed 7 --batches 40
```

```
batches matched       40
clean match rate      100.0%
findings              0
exceptions (to human) 0
llm calls             0  (0.00 per 100 records)
```

This is not an easier period. The volumes, the amount distribution, the batch
sizes and the matching problem are unchanged — only the defects are gone, and
the banks are stripped of their habits (no settlement lag, no standing charge,
the UTR written where the strict pattern reads it). The accuracy table is
suppressed rather than printed full of `nan`: precision over zero findings is
undefined, and an empty table with a number in it invites you to read it as a
measurement.

**One rule is pre-loaded**, and it is worth knowing why. International cards
genuinely cost 3%, and the settlement report never says so, so without that rule
a clean period raises exactly one item — a *question about a rate card* rather
than a pile of overcharge accusations. That is the design rule visible at its
limit: [assert only when unambiguous](src/taper/engine/matching.py). Three tests
pin all of it, including that the question disappears once the rate card is
learned and never comes back.

---

## The cash position

The track is titled *run the books **and the cash position***. Those are two
different questions, and the second one is the one a CFO asks first.
Reconciliation says whether the books agree. It does not say how much money is
actually there — money can be settled and still not arrive, arrive and still be
owed back, or be deducted for a reason nobody has explained.

`taper cash` assembles that number out of findings the engine already produces.
No new detection; new arithmetic, with two decisions in it.

```
In the bank                                Rs.14,394,848.01   (36 batches)
Withheld pending disputes                  Rs.    49,599.73   shown, not added
Owed to the merchant                       Rs.   627,955.92
Owed by the merchant                       Rs. 1,333,451.73
--------------------------------------------------------------------------
Net position                               Rs.13,689,352.20
```

**Withheld money is shown and never added.** It is not the merchant's to count
until the dispute resolves. Including it would overstate the position in exactly
the direction that causes an overdraft.

**Duplicate captures are a liability even though the cash is in the bank**,
because it is owed back. A finding can legitimately sit on both sides of a
position, labelled — that is what is true of it. Counting it as revenue is how a
refund run becomes a surprise.

Anything that could not be attributed to a batch is named alongside the total,
and the position is stated as a *floor* rather than a total. The report leads
with this section, ahead of reconciliation, because that is the reading order.

---

**The other documents:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) is the
design on one page, [`docs/RESULTS.md`](docs/RESULTS.md) is every measurement
with the command that produced it, and honest limitations are at the end of it.

**Where to look in the code:** [`matching.py`](src/taper/engine/matching.py) is
the deterministic core and its design rule; [`rules.py`](src/taper/engine/rules.py)
is the learning loop and the gate that makes it safe;
[`llm.py`](src/taper/engine/llm.py) is the model layer and the arithmetic that
overrules it. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) is the whole design
in one page: the four layers, the three independent reasons the model cannot
decide anything, and how the taper is measured.

**184 tests**, CI on Python 3.11–3.13. The tests are not coverage — each one
guards a claim made below, so a failure means a sentence here has become false.

---

## The problem

A merchant on a payment gateway has three sources of truth that never agree:

| Source | What it says |
|---|---|
| **PG settlement report** | Individual payments, refunds, fees, GST on fees, chargeback holds |
| **Rate card** | Different contracted rates per payment method — named nowhere in the report |
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

Every headline claim has a command behind it, and the full evidence — the
ablation, forensics, the red team, the failure boundary, and what the harness
caught — is in **[docs/RESULTS.md](docs/RESULTS.md)**.

The short version:

| | |
|---|---|
| **The taper** | Model calls / 100 records 1.12 → 0.50 over six closes, while clean match rate rises 65.4% → 95.9% |
| **Precision** | 1.000, and it never leaves 1.000 — including under 6.6× ambiguity, where the engine matches nothing rather than guessing |
| **The ablation** | The model adds +0.000. Against a real local model, 8 of 13 calls were confabulations and arithmetic refused all 8 |
| **Red team** | A fully attacker-controlled model asserting everything reconciled moves nothing |
| **Forensics** | Benford screening at precision 1.00 against a derived null threshold — Nigrini's published bands gave 0.30 on this data |
| **Risk model** | Reviewing the riskiest 10% of batches catches 56% of escalations, on held-out seeds |
| **Negative control** | A clean period: nothing found, nothing escalated, no model calls |
| **Aging control** | With learning, a standing question is asked once; without it, every close |

Honest limitations are listed at the [end of that page](docs/RESULTS.md#honest-limitations)
rather than left for you to find.

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
python -m taper.cli --mock ablate --seed 99
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
  adversarial.py       stress ladder; proves it fails safe rather than wrong
  engine/
    results.py         Finding / BatchMatch / Exception_, with layer attribution
    matching.py        L0 + L1, the deterministic core
    rules.py           learned rules + the retroactive admission gate
    llm.py             L3 client, prompt, and the propose-then-verify gate
    pipeline.py        layer orchestration
  metrics/harness.py   per-class scoring, calibration, ablation, FP cost
  ml/
    features.py        batch features from raw sources only - no matching outcome
    confidence.py      logistic model + hand-rolled isotonic calibration
    train.py           disjoint-seed training; raises on any train/holdout overlap
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

Running end-to-end: **184 tests green, 20 CLI commands**, no API key required.

All four layers are wired, and **all four rule types now learn end to end** —
`adjustment_pattern`, `narration_alias`, `bank_timing` and `fee_variant`, the
last being the rate card the settlement report never states. Combined defects —
a batch both split across credits *and* short by a recurring charge — resolve
deterministically via candidate-target subset netting.

The negative control is green: on a period where nothing is wrong, it finds
nothing, escalates nothing and calls no model.

Layer 3 has been validated against a **real model** — a local qwen2.5 via
Ollama — for a single close, the ablation, and a full six-month campaign. The
offline mock is now only a CI stand-in, and every artifact records which
resolver produced it.

Remaining work: the same runs against a **frontier** model. The local result
says a 14B model adds nothing measurable on this workload; that is not evidence
about what a stronger one would do, and the README does not claim otherwise.
