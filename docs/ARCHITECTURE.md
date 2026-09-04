# Architecture

Taper reconciles three sources that never agree — a payment gateway's settlement
report, a bank statement, and the merchant's own ledger — and reports what is
wrong, what it could not resolve, and how much money is actually there.

The design has one governing idea: **the model is the last resort, and it is
never allowed to decide anything.** Everything below follows from that.

---

## 1. The four layers

Work falls through them in order. Each layer is strictly more expensive and
strictly less trusted than the one above it.

```
                 settlement report      bank statement       ledger
                        |                     |                 |
                        +----------+----------+-----------------+
                                   |
                        +----------v-----------+
                        |  L0  exact arithmetic |   UTR join, exact netting
                        |      Decimal, no float|   confidence 1.000
                        +----------+------------+
                                   | unresolved
                        +----------v------------+
                        |  L1  bounded heuristic|   date-window + subset-sum
                        |      declared bounds  |   over <= 3 credits
                        +----------+------------+
                                   | unresolved
                        +----------v------------+
                        |  L2  learned rules    |   typed rules from the store
                        |      free, replayable |   grows every close
                        +----------+------------+
                                   | unresolved
                        +----------v------------+
                        |  L3  language model   |   PROPOSES ONLY
                        |      propose-then-    |   verify_proposal() re-derives
                        |      verify           |   every claim in code
                        +----------+------------+
                                   | unresolved
                             exception queue  ->  a human
```

A finding records which layer produced it, so `taper reconcile` can show where
the work actually happened, and the share resolved without a model is the number
that should climb month over month.

### The design rule: assert only when unambiguous

Every layer may decline. A layer that cannot reach one unambiguous answer passes
the item down rather than choosing between candidates. This is why precision is
1.000 and stays there: ambiguity becomes an exception, never a guess.

The cost is visible and deliberate — under 6.6× ambiguity the engine matches
*nothing* and escalates all 61 batches. It fails safe, not wrong.

---

## 2. Why the model cannot decide anything

Three independent mechanisms, each sufficient on its own.

**Propose-then-verify.** The model never returns a conclusion. It returns a
*proposal* — "batch X is these three credits, short by a recurring charge" — and
`verify_proposal()` re-derives the arithmetic from the source amounts in code the
model never touches. A proposal that does not reconcile to the paisa is refused.
The model names candidates; `Decimal` decides.

**Typed rules, never model-authored code.** The rule store holds exactly four
shapes — `bank_timing`, `narration_alias`, `fee_variant`, `adjustment_pattern` —
each a small parameter set. A narration alias is a *marker token and a prefix*,
and fixed code does the extraction. A learned regex would mean executing text the
model wrote against every future narration: an injection surface for no benefit,
since every real case is "find the label, take the number after it".

**The retroactive admission gate.** A candidate rule is replayed against every
case already confirmed correct. If it would change even one of them, it is
rejected and the originating item stays an exception. A rule earns admission by
being *harmless on the past*, not by being persuasive about the present. This is
deliberately more conservative than necessary: a missed automation costs one
manual review, a bad rule silently corrupts every future close.

**Result:** `taper redteam` runs a prompt-injection payload in a bank narration
against a *fully attacker-controlled* model that asserts everything reconciled at
maximum confidence. It moves nothing. Narration sanitisation is the second line
of defence; the first is that layer 3 was never allowed to decide.

---

## 3. The taper — why cost falls while accuracy rises

When a human resolves an exception, the model proposes a typed rule that would
have resolved it automatically. If the rule survives the admission gate, the same
situation next month is handled at layer 2: deterministically, for free, with no
model call.

Averaged over 8 independent 6-month campaigns:

| | Month 1 | Month 6 |
|---|---|---|
| Model calls / 100 records | 1.12 | **0.50** *(−56%)* |
| Clean match rate | 65.4% | **95.9%** |
| Human reviews per close | 14.1 | **5.8** |
| **Precision** | **1.000** | **1.000** |

All four rule kinds now learn end to end. The agent's job is to make itself
unnecessary.

**Rules also die.** When a bank reprices, a rule that was true becomes a
confident wrong answer. `check_rule_health` detects the drift, retires the rule
with a reason, and relearns the new behaviour — and the retired rule is kept
rather than deleted, so a past close can still be explained.

---

## 4. Measurement

Nothing is eyeballed off a demo run.

**Ground truth is injected.** The generator knows exactly what it broke and
where, at rates declared up front, so precision and recall are computed per
defect class against real labels — not against the engine's own opinion.

**Held-out seeds.** The tuning set and the evaluation set differ only by seed,
and the risk model asserts the split rather than trusting the caller: a model
evaluated on data it was fitted to shows a flawless curve and predicts nothing.

**The negative control.** Every accuracy number is measured on periods that
*contain* defects, so none of them can answer the opposite question — what does
the engine do when nothing is wrong? On a clean period (`reconcile --clean`) it
finds nothing, escalates nothing, and calls no model. An engine that manufactures
work on a quiet month spends human attention no metric here would charge it for.

**The queue is aged, not just counted.** A falling exception count is
compatible with a stuck queue, so exceptions get an identity that survives the
period and are checked for recurrence. Running the same six closes with learning
disabled is the control: with the loop the rate-card question is asked once, and
without it every close. A test asserts the two runs must disagree.

**It refuses closes it cannot stand behind.** Stopping rules decide whether a
close is fit to sign: a missing source, under half the batches matched, a third
of credited money tied to no batch. A refused close says so above its own
numbers, because from the KPI row down it looks exactly like a good one. Two of
the rules — the model budget and the refusal streak — are enforced inside the
pipeline rather than reported after it, and a stop costs recall and human time
but never correctness.

**Calibration, not just accuracy.** Stated confidence is compared against
observed hit rate, and the auto-clear threshold is derived from that curve rather
than picked. Brier skill is reported next to AUC because skill is sensitive to
prior shift in a way AUC is not.

---

## 5. The cash position

Reconciliation says whether the books agree. A controller's first question is
different: *how much money is actually there.* `taper cash` assembles that from
findings the engine already produces — no new detection, new arithmetic.

Three groups kept deliberately apart, because conflating them is how a position
ends up wrong in the direction that hurts: **in the bank**, **real but not in the
balance** (withheld against disputes, or settled and uncredited), and **claims**
each way.

Two decisions carry it. Withheld money is shown and never added — it is not the
merchant's to count until the dispute resolves. And duplicate captures are a
liability *even though the cash is in the bank*, because it is owed back. A
finding can legitimately sit on both sides of a position, labelled.

---

## 6. Materiality — what deserves a person

Everything above decides what is *true*. This decides what is *worth someone's
afternoon*, and it is deliberately downstream: a test asserts the close digest
and precision are identical before and after, because a review threshold must
never be able to change the reported accuracy of a close.

A floor alone would be a blind spot at exactly the size an error would choose —
systematic problems present as many small items, since that is what systematic
means. So waived findings are grouped by what a human would claim them as, and
any group clearing the aggregate floor returns as a single claim rather than as
nothing. Fourteen standing bank charges are trivial per payout and ₹5,481 per
quarter.

Two carve-outs hold at any floor. Findings with no rupee amount are never waived
— a missing UTR costs nothing and breaks matchability, and materiality is a
statement about money. Neither are exceptions: "too small to investigate" is a
judgement you can only make about something you already understand.

---

## 7. Forensics

`taper forensics` screens first-digit distributions against Benford's law per
channel, because fabrication presents concentrated — one compromised channel, one
operator — not sprinkled evenly where it would hide in the honest majority.

The threshold is derived, not borrowed. Nigrini's published bands gave precision
0.30 on this data; simulating the null distribution and scaling it by 1/√n gives
precision 1.00. The output says explicitly that this is a screening signal, not
proof, and names the innocent causes — price lists, fixed-fee products, rounding
policy.

---

## 8. Engineering choices

**The report is the product surface, and it is one file.** Sortable columns, a
filter, a theme override and a section tracker are an inline script at the
bottom of the same document - progressive enhancement, so every number is in the
HTML before it runs and a test asserts the report still reads with the script
stripped. A served dashboard would have traded "opens from disk in eight years"
for conveniences available without giving that up.

**Zero runtime dependencies.** A reviewer clones the repo and reproduces every
number without installing anything. The model client is optional; the shipped
risk model is a hand-written logistic regression, chosen over scikit-learn on
three measurements rather than preference. Charts in the close report are
hand-rolled inline SVG for the same reason.

**`Decimal` everywhere, never float.** Money arithmetic is exact, and tests
enforce it.

**Provider-agnostic layer 3.** Anthropic, Ollama, or any OpenAI-compatible
endpoint, over stdlib `urllib`. The full six-month campaign runs against a local
model at zero cost.

**Re-derivable closes.** Every close emits a SHA-256 digest over its canonical
conclusions — matches, findings, exceptions, not timestamps or wording — so a
regenerated report hashes identically and only real change shows.

**Indexed matching.** Batch matching is indexed by UTR and value date rather than
scanned; ~1.5 µs per record, flat across 50× data growth.

---

## 9. Map of the code

| Path | What lives there |
|---|---|
| `engine/matching.py` | The deterministic core (L0/L1) and every check |
| `engine/rules.py` | Typed rules, the store, and the admission gate |
| `engine/llm.py` | Layer 3, the clients, and `verify_proposal` |
| `engine/sanitize.py` | Narration neutralisation before any prompt |
| `engine/pipeline.py` | Orchestration across the four layers |
| `generator.py` | Synthetic three-source data with labelled defects |
| `metrics/harness.py` | Scoring against injected ground truth |
| `ml/` | Exception-risk model, calibration, held-out evaluation |
| `forensics.py` | Benford screening with a derived threshold |
| `cashflow.py` | The cash position |
| `forecast.py` | Forward cash, the lag model and its backtest |
| `signoff.py` | Stopping rules, and the refusal to sign |
| `materiality.py` | What deserves a person, and the aggregation rule |
| `aging.py` | Exception identity across closes, and the stale check |
| `report.py` | The self-contained HTML close package |
| `adapters/razorpay.py` | Razorpay's real settlement recon schema |
| `tests/` | 184 invariants, organised by the claim each one guards |

The test suite is organised by **claim**, not by module: each section guards a
sentence the writeup makes, so a failure means the writeup has become false.
