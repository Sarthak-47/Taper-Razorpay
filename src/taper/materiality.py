"""Materiality: which findings are worth a human's time, and which are not.

Every finding this engine produces is correct - precision is 1.000 and nothing
here changes that. But correct is not the same as *worth chasing*. A Rs.1.77 fee
overcharge is real, and no controller alive is going to open a ticket with the
gateway about it. An engine that hands over seventeen of those alongside a
Rs.470,000 duplicate capture has not helped; it has buried the one that matters.

So this module answers a different question from the rest of the codebase. Not
"is this wrong" - that is settled by the time we get here - but "does this
deserve a person".

**The whole feature is one idea, and it is the second half that makes it safe.**

Set a floor and waive everything under it, and you have built a blind spot at
exactly the size an error would choose. Seventeen fee overcharges at a median of
Rs.84 are individually beneath contempt and collectively Rs.13,435, which is a
claim worth making. Systematic errors are *precisely* the ones that present as
many small items, because that is what "systematic" means.

So waived items are grouped by what a human would actually claim them as, and
any group whose total clears the aggregate floor comes back - not as seventeen
tickets, but as one: "fee overcharges, 17 items, Rs.13,435, chase as a single
claim against the gateway". The floor removes noise. The aggregate rule stops
the floor from removing a pattern.

Two things are never waived, whatever the floor:

  * **Findings with no rupee amount.** A missing UTR costs nothing and breaks
    matchability; a timing shift moves when cash arrives, not how much.
    Materiality is a statement about money, and applying it to something that
    was never about money is a category error.
  * **Exceptions.** Those are items the engine could not resolve at all. "Too
    small to investigate" is a judgement you can only make about something you
    understand, and by definition we do not understand these yet.

Nothing is deleted or hidden. The waived total is reported next to the floor
that produced it, because the honest form of this feature is "here is what I
did not look at, and here is how much it was" - not a shorter list with no
explanation of what left it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .engine.results import Finding, ReconResult
from .models import DefectClass, Money

# Defaults sized to this data rather than to a standard. Real materiality is set
# by the audit committee as a share of revenue or profit, not by a library - so
# these are a starting point for the sweep, not a recommendation.
DEFAULT_FLOOR = Money("500.00")

# Deliberately not a multiple of the floor. The aggregate question is "is this
# pattern worth one conversation", which has a different answer from "is this
# item worth one ticket", and tying them together would hide that.
DEFAULT_AGGREGATE_FLOOR = Money("5000.00")


@dataclass(frozen=True)
class MaterialityPolicy:
    """What a controller decided is worth pursuing."""

    floor: Money = DEFAULT_FLOOR
    aggregate_floor: Money = DEFAULT_AGGREGATE_FLOOR

    def describe(self) -> str:
        return (
            f"chase anything at or above Rs.{self.floor:,.2f}; below that, chase "
            f"a class only if it totals Rs.{self.aggregate_floor:,.2f} or more"
        )


@dataclass
class AggregateClaim:
    """Small items that add up to something worth one conversation."""

    defect_class: DefectClass
    findings: list[Finding] = field(default_factory=list)

    @property
    def total(self) -> Money:
        return sum((f.money_impact for f in self.findings), Money("0.00"))

    @property
    def count(self) -> int:
        return len(self.findings)

    @property
    def largest(self) -> Money:
        return max((f.money_impact for f in self.findings), default=Money("0.00"))

    def line(self) -> str:
        return (
            f"{self.count} x {self.defect_class.value}, none individually above "
            f"the floor, Rs.{self.total:,.2f} together"
        )


@dataclass
class MaterialityReport:
    """What a policy did to a close, stated so it can be argued with."""

    policy: MaterialityPolicy
    chased: list[Finding] = field(default_factory=list)
    not_about_money: list[Finding] = field(default_factory=list)
    aggregated: list[AggregateClaim] = field(default_factory=list)
    waived: list[Finding] = field(default_factory=list)

    # ---- totals ---------------------------------------------------------
    @property
    def chased_total(self) -> Money:
        return sum((f.money_impact for f in self.chased), Money("0.00"))

    @property
    def aggregated_total(self) -> Money:
        return sum((c.total for c in self.aggregated), Money("0.00"))

    @property
    def waived_total(self) -> Money:
        """Money the policy decided not to look at.

        The number that makes this feature honest. A materiality policy without
        it is just a shorter list.
        """
        return sum((f.money_impact for f in self.waived), Money("0.00"))

    @property
    def examined_total(self) -> Money:
        return self.chased_total + self.aggregated_total

    @property
    def items_before(self) -> int:
        return (
            len(self.chased) + len(self.waived) + len(self.not_about_money)
            + sum(c.count for c in self.aggregated)
        )

    @property
    def items_after(self) -> int:
        """One line per chased finding, plus one per aggregate claim."""
        return len(self.chased) + len(self.aggregated) + len(self.not_about_money)

    @property
    def items_saved(self) -> int:
        return self.items_before - self.items_after

    @property
    def waived_share(self) -> float:
        """Waived money as a share of everything with a rupee amount.

        The number to argue about. If this climbs past a percent or two, the
        floor is too high whatever it saves in review time.
        """
        total = self.examined_total + self.waived_total
        if not total:
            return 0.0
        return float(self.waived_total / total)

    def verdict(self) -> str:
        if not self.items_saved:
            return (
                "No effect. Nothing fell below the floor, so this policy is "
                "costing a line of config and buying nothing."
            )
        parts = [
            f"{self.items_saved} fewer item(s) to work through "
            f"({self.items_before} -> {self.items_after})."
        ]
        if self.aggregated:
            parts.append(
                f"{len(self.aggregated)} pattern(s) came back as a single claim "
                f"worth Rs.{self.aggregated_total:,.2f} - individually all of it "
                f"was below the floor."
            )
        if self.waived_total:
            parts.append(
                f"Rs.{self.waived_total:,.2f} was waived and not examined "
                f"({self.waived_share:.2%} of identified money)."
            )
        else:
            parts.append("Nothing was waived: everything small enough to skip "
                         "aggregated into something worth chasing.")
        return " ".join(parts)


def assess(result: ReconResult, policy: MaterialityPolicy | None = None) -> MaterialityReport:
    """Sort a close's findings into chase, claim together, and leave.

    Pure: reads the result, changes nothing about it. Precision, recall and the
    close digest are all computed upstream of this and are unaffected - which is
    the point. This decides presentation of work, not truth.
    """
    policy = policy or MaterialityPolicy()
    report = MaterialityReport(policy=policy)

    below: dict[DefectClass, list[Finding]] = defaultdict(list)
    for finding in result.findings:
        if finding.money_impact <= Money("0.00"):
            # Never waived. A missing UTR is a matchability problem, not a
            # rupee problem, and a floor has nothing to say about it.
            report.not_about_money.append(finding)
        elif finding.money_impact >= policy.floor:
            report.chased.append(finding)
        else:
            below[finding.defect_class].append(finding)

    for defect_class, findings in below.items():
        claim = AggregateClaim(defect_class=defect_class, findings=findings)
        if claim.total >= policy.aggregate_floor:
            report.aggregated.append(claim)
        else:
            report.waived.extend(findings)

    report.chased.sort(key=lambda f: f.money_impact, reverse=True)
    report.aggregated.sort(key=lambda c: c.total, reverse=True)
    report.waived.sort(key=lambda f: f.money_impact, reverse=True)
    return report


def sweep(
    result: ReconResult,
    floors: list[Money] | None = None,
    aggregate_floor: Money | None = None,
) -> list[MaterialityReport]:
    """Run the policy at a ladder of floors so the choice can be seen.

    A controller does not want a default, they want the shape of the trade:
    how many items disappear, and how much money stops being looked at. Those
    two curves have very different shapes, and the useful floor is where the
    first is still falling and the second has not started to climb.
    """
    floors = floors or [
        Money("50.00"), Money("100.00"), Money("250.00"), Money("500.00"),
        Money("1000.00"), Money("2500.00"), Money("5000.00"), Money("10000.00"),
    ]
    agg = aggregate_floor if aggregate_floor is not None else DEFAULT_AGGREGATE_FLOOR
    return [
        assess(result, MaterialityPolicy(floor=floor, aggregate_floor=agg))
        for floor in floors
    ]
