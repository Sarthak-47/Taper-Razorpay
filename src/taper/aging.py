"""Exception aging: is the queue shrinking, or is it the same items every month?

The campaign reports human reviews falling from 14.1 to 5.8. That number is
true and it is not sufficient, because there are two completely different
worlds behind it.

In the good one, the engine learns each recurring situation, those items stop
appearing, and what remains in month six is genuinely new work. In the bad one,
the *volume* of novel problems happens to fall while a handful of items nobody
can resolve get re-raised every single close - and the headline chart looks
identical. A falling count cannot tell those apart. Only identity can.

So this module gives an exception a name that survives the period it was raised
in, and then asks how many closes it has been asked in. An item on its sixth
consecutive appearance is not an exception any more; it is a question the system
keeps asking and nobody has answered, and it should be reported as one.

**The hard part is identity.** Subject ids are period-scoped by construction -
``setl_500_004`` exists in exactly one close - so keying on them would prove
that nothing ever recurs, which is a measurement that always passes and
therefore says nothing. Exceptions split into two honest categories instead:

  * **Standing** - the thing being asked about outlives the period. "What rate
    card covers intl_card" is the same question in June and in November, and
    the answer settles it permanently. These can recur, and it means something
    when they do.
  * **Episodic** - the thing being asked about *is* a specific batch or credit
    in one period. A batch that could not be matched in June has no counterpart
    in July. These cannot recur by construction and are counted separately
    rather than folded in, because averaging them together would dilute the
    recurrence rate toward zero and make the metric look good for a reason that
    has nothing to do with learning.

The point of separating them is to keep the measurement capable of failing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .engine.results import Exception_

# Kinds whose subject outlives the close that raised it. Everything not named
# here is treated as episodic, which is the conservative default: wrongly
# calling something standing would invent recurrences that are not real.
STANDING_KINDS = {
    "unknown_rate_card",     # a rate card is a contract, not an event
    "fabricated_amounts",    # a channel's amount distribution, not one payment
    "stale_rule",            # a rule that stopped being true
    "missing_source",        # a file that did not arrive, by name
}

# Consecutive appearances after which an item stops being a queue entry and
# starts being a process failure. Three closes is a quarter: long enough that
# a busy month is not the explanation, short enough to still be actionable.
STALE_AFTER = 3


def identity(exc: Exception_) -> str | None:
    """A name for this exception that survives the period, or None if episodic.

    Built from the kind plus the dimension the question is actually about -
    the method for a rate card, the rule id for a stale rule. Never the subject
    id, which is period-scoped and would make every item look brand new.
    """
    if exc.kind not in STANDING_KINDS:
        return None
    ctx = exc.context
    for dimension in ("method", "rule_id", "bank", "source", "segment"):
        value = ctx.get(dimension)
        if value:
            return f"{exc.kind}::{dimension}={value}"
    # A standing kind carrying no dimension still has a stable subject, because
    # the subject is the thing rather than a row in this period's data.
    return f"{exc.kind}::{exc.subject_id}"


@dataclass
class AgedItem:
    """One standing question, and every close it has been asked in."""

    identity: str
    months: list[int] = field(default_factory=list)
    last_reason: str = ""

    @property
    def appearances(self) -> int:
        return len(self.months)

    @property
    def consecutive(self) -> int:
        """Longest unbroken run ending at the most recent appearance."""
        if not self.months:
            return 0
        run = 1
        for earlier, later in zip(self.months, self.months[1:], strict=False):
            run = run + 1 if later == earlier + 1 else 1
        return run

    @property
    def is_stale(self) -> bool:
        return self.consecutive >= STALE_AFTER

    def line(self) -> str:
        span = f"month{'s' if self.appearances > 1 else ''} " + ", ".join(
            str(m) for m in self.months
        )
        return f"{self.identity} - asked in {span}"


@dataclass
class AgingReport:
    """What recurred across a campaign, and what that says about the taper."""

    months_observed: int = 0
    items: list[AgedItem] = field(default_factory=list)
    episodic_per_month: list[int] = field(default_factory=list)

    @property
    def standing_total(self) -> int:
        return len(self.items)

    @property
    def recurring(self) -> list[AgedItem]:
        return [i for i in self.items if i.appearances > 1]

    @property
    def stale(self) -> list[AgedItem]:
        return [i for i in self.items if i.is_stale]

    @property
    def resolved(self) -> list[AgedItem]:
        """Standing questions that stopped being asked.

        The ones that appeared and then went away are the learning loop working
        on exactly the items it is supposed to work on.
        """
        return [
            i for i in self.items
            if i.months and max(i.months) < self.months_observed
        ]

    @property
    def episodic_total(self) -> int:
        return sum(self.episodic_per_month)

    def verdict(self) -> str:
        if not self.items:
            return (
                "No standing questions arose at all, so there is nothing here "
                "that could have gone stale. The queue was episodic throughout."
            )
        parts = [
            f"{self.standing_total} standing question(s) across "
            f"{self.months_observed} closes, against {self.episodic_total} "
            f"episodic item(s) that cannot recur by construction."
        ]
        if self.stale:
            parts.append(
                f"{len(self.stale)} has been asked {STALE_AFTER} or more closes "
                f"running and is still unanswered - that is not an exception "
                f"queue, it is a process that has stalled."
            )
        else:
            parts.append(
                f"Nothing was asked {STALE_AFTER} closes running: every standing "
                f"question was either answered or newly raised."
            )
        if self.resolved:
            parts.append(
                f"{len(self.resolved)} stopped being asked after being resolved, "
                f"which is the learning loop doing the thing it claims to do."
            )
        return " ".join(parts)


def build(monthly_exceptions: list[list[Exception_]]) -> AgingReport:
    """Age a campaign's exception queues. Index 0 is month 1."""
    report = AgingReport(months_observed=len(monthly_exceptions))
    seen: dict[str, AgedItem] = {}

    for index, exceptions in enumerate(monthly_exceptions, start=1):
        episodic = 0
        for exc in exceptions:
            key = identity(exc)
            if key is None:
                episodic += 1
                continue
            item = seen.setdefault(key, AgedItem(identity=key))
            # Guard against one close raising the same standing question twice;
            # it is one question, however many rows provoked it.
            if index not in item.months:
                item.months.append(index)
            item.last_reason = exc.reason
        report.episodic_per_month.append(episodic)

    report.items = sorted(
        seen.values(), key=lambda i: (-i.consecutive, -i.appearances, i.identity)
    )
    return report
