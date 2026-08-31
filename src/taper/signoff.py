"""Stopping rules: when the agent refuses to sign the close.

Everything else here decides what is *true* about a close. This decides whether
the close is fit to be signed at all, which is a different question and the one
a controller is actually accountable for.

The failure this exists to prevent is specific. An engine that always produces a
report will always produce a report - including for a period where the bank
statement was truncated, or the settlement file arrived in the wrong currency, or
half the batches never matched. The output looks the same as a good close: a
match rate, a findings table, an exception list. It is just wrong, and nothing in
it says so. **A number that is always emitted carries no information about
whether it should have been.**

So a close ends in one of three states, and the agent picks:

  * **SIGN** - nothing tripped. The close stands on its own.
  * **SIGN WITH CAVEATS** - something degraded. The conclusions hold, but a
    layer was disabled or a queue is beyond what a human will actually work,
    and the caveat travels with the close rather than being discovered later.
  * **DO NOT SIGN** - a halt condition. The close is not evidence of anything
    and should not be filed. Refusing is the useful output.

**Two design rules, both learned the hard way.**

*A bound that never binds is decoration.* Every threshold here was set against
measured behaviour on healthy closes, not chosen because it sounded prudent, and
each one carries the number it was calibrated against. A rule that cannot fire
is worse than no rule, because it looks like a safeguard.

*A stop must change what happens, not just what is printed.* The model budget
and the refusal streak are enforced inside the pipeline - they hold calls back
and abandon a layer - and are reported here. A "stopping rule" that only ever
appears in a summary is a comment with a threshold in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .engine.results import ReconResult
from .models import Money, SourceBundle


class Severity(StrEnum):
    HALT = "halt"
    DEGRADE = "degrade"


# --- thresholds, each calibrated against observed healthy closes ------------

# A close matching under half its batches is not a close with a low match rate,
# it is a close that did not happen. Healthy runs sit at 65-97%; the adversarial
# stress ladder reaches 0% only under 6x ambiguity, and that run should be
# refused rather than reported.
MIN_MATCH_RATE = 0.50

# Bank credits that could not be attributed to any batch, as a share of all
# money credited. Some unattributed money is normal - timing, cross-period
# refunds. A third of it means the two sources are not describing the same
# period, which is a filing error rather than a reconciliation finding.
MAX_UNATTRIBUTED_SHARE = 0.33

# Exceptions a human will realistically work in one close. Past this the queue
# stops being a queue and becomes a backlog nobody reads, and the review-time
# figures elsewhere in this repo quietly stop being true.
MAX_HUMAN_QUEUE = 40


@dataclass
class Trip:
    """One stopping rule that fired, and what it means for the close."""

    rule: str
    severity: Severity
    observed: str
    threshold: str
    why: str
    action: str

    def line(self) -> str:
        return f"[{self.severity.value.upper()}] {self.rule}: {self.observed}"


@dataclass
class SignOff:
    trips: list[Trip] = field(default_factory=list)
    checked: int = 0

    @property
    def halts(self) -> list[Trip]:
        return [t for t in self.trips if t.severity is Severity.HALT]

    @property
    def degrades(self) -> list[Trip]:
        return [t for t in self.trips if t.severity is Severity.DEGRADE]

    @property
    def signed(self) -> bool:
        return not self.halts

    @property
    def decision(self) -> str:
        if self.halts:
            return "DO NOT SIGN"
        if self.degrades:
            return "SIGN WITH CAVEATS"
        return "SIGN"

    def verdict(self) -> str:
        if self.halts:
            return (
                f"{len(self.halts)} halt condition(s) tripped. This close is not "
                f"evidence of anything and should not be filed. Refusing is the "
                f"useful output - a report produced anyway would look exactly "
                f"like a good one."
            )
        if self.degrades:
            return (
                f"No halt conditions. {len(self.degrades)} caveat(s) travel with "
                f"this close rather than waiting to be discovered by whoever "
                f"relies on it."
            )
        return (
            f"All {self.checked} stopping rules passed. The close stands on its "
            f"own, and each threshold was set against measured behaviour rather "
            f"than chosen because it sounded prudent."
        )


def evaluate(result: ReconResult, bundle: SourceBundle) -> SignOff:
    """Run every stopping rule against a finished close."""
    report = SignOff(checked=5)

    # --- halt: a source never arrived ------------------------------------
    missing = [
        name for name, rows in (
            ("settlement", bundle.settlement),
            ("bank", bundle.bank),
        ) if not rows
    ]
    if missing:
        report.trips.append(Trip(
            rule="source_present",
            severity=Severity.HALT,
            observed=f"{', '.join(missing)} is empty",
            threshold="both settlement and bank are required",
            why=(
                "With a source missing every check returns empty and the close "
                "comes back spotless. That is the most dangerous output this "
                "system can produce: a controller reads 'no exceptions' and "
                "signs off on a period nobody actually checked."
            ),
            action="Obtain the missing file. Nothing here is reconcilable without it.",
        ))

    # --- halt: too little matched to call it a close ---------------------
    batches = {row.settlement_batch_id for row in bundle.settlement}
    if batches:
        matched_share = len(result.matches) / len(batches)
        if matched_share < MIN_MATCH_RATE:
            report.trips.append(Trip(
                rule="match_rate_floor",
                severity=Severity.HALT,
                observed=f"{matched_share:.1%} of batches matched",
                threshold=f"{MIN_MATCH_RATE:.0%}",
                why=(
                    "Below half, this is not a close with a low match rate - it "
                    "is a close that did not happen. Healthy runs sit between "
                    "65% and 97%, and the only thing that reaches 0% here is the "
                    "adversarial stress ladder, which should be refused rather "
                    "than reported."
                ),
                action=(
                    "Check the two files describe the same period and the same "
                    "account before reading anything below."
                ),
            ))

    # --- halt: the sources are not describing the same period ------------
    credited = sum((c.credit_amount for c in bundle.bank), Money("0.00"))
    claimed = {txn for match in result.matches for txn in match.bank_txn_ids}
    unattributed = sum(
        (c.credit_amount for c in bundle.bank if c.bank_txn_id not in claimed),
        Money("0.00"),
    )
    if credited > Money("0.00"):
        share = float(unattributed / credited)
        if share > MAX_UNATTRIBUTED_SHARE:
            report.trips.append(Trip(
                rule="unattributed_money",
                severity=Severity.HALT,
                observed=f"Rs.{unattributed:,.2f} ({share:.1%}) tied to no batch",
                threshold=f"{MAX_UNATTRIBUTED_SHARE:.0%}",
                why=(
                    "Some unattributed money is ordinary - timing, cross-period "
                    "refunds. A third of it means the settlement report and the "
                    "bank statement are not describing the same period, which is "
                    "a filing mistake rather than a reconciliation finding."
                ),
                action="Confirm the date ranges on both exports match.",
            ))

    # --- degrade: a queue nobody will work --------------------------------
    if len(result.exceptions) > MAX_HUMAN_QUEUE:
        report.trips.append(Trip(
            rule="human_queue_depth",
            severity=Severity.DEGRADE,
            observed=f"{len(result.exceptions)} exception(s)",
            threshold=f"{MAX_HUMAN_QUEUE}",
            why=(
                "Past this a queue stops being a queue and becomes a backlog "
                "nobody reads - and every review-time figure this project "
                "reports quietly stops being true, because they all assume the "
                "queue gets worked."
            ),
            action=(
                "Triage by materiality before assigning: `taper materiality` "
                "groups the small items into one claim each."
            ),
        ))

    # --- degrade: stops the pipeline actually enforced ---------------------
    for stop in result.stops:
        rule, _, detail = stop.partition(": ")
        report.trips.append(Trip(
            rule=rule,
            severity=Severity.DEGRADE,
            observed=detail or stop,
            threshold="enforced during the run",
            why=(
                "This one changed what the close did rather than only what it "
                "says: calls were held back or a layer was abandoned mid-run."
            ),
            action="Conclusions stand; the model contributed less than usual.",
        ))
    return report
