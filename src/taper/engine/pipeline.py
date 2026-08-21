"""Orchestration: run the layers cheapest-first and stop as soon as one answers.

The ordering is the argument. Layer 0 is arithmetic and cannot be wrong. Layer 1
is bounded heuristics. Layer 2 is rules the system has already learned and
regression-tested. Only what survives all three reaches the model, and whatever
the model says is verified numerically before it counts.

The number to watch across runs is ``llm_calls_per_100``. It should fall as the
rule store grows while match rate holds or improves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..models import Money, SourceBundle
from . import sanitize
from .llm import LLMClient, get_client, to_finding, verify_proposal
from .matching import batch_nets, run_deterministic
from .results import Exception_, Layer, ReconResult
from .rules import RuleStore, build_history, next_rule_id, rule_from_proposal


@dataclass
class RunConfig:
    use_llm: bool = True
    use_real_llm: bool = False
    learn_rules: bool = True
    confidence_floor: float = 0.60  # below this we keep the exception instead


def reconcile(
    bundle: SourceBundle,
    store: RuleStore | None = None,
    config: RunConfig | None = None,
    client: LLMClient | None = None,
) -> ReconResult:
    config = config or RunConfig()
    store = store if store is not None else RuleStore()
    started = time.perf_counter()

    # --- layers 0-2 (the store carries last month's lessons) --------------
    matches, findings, exceptions = run_deterministic(bundle, store)

    nets = batch_nets(bundle)
    credit_amounts: dict[str, Money] = {c.bank_txn_id: c.credit_amount for c in bundle.bank}
    bank_by_id = {c.bank_txn_id: c for c in bundle.bank}

    result = ReconResult(
        matches=matches,
        findings=list(findings),
        exceptions=[],
        records_processed=len(bundle),
    )

    # --- report attacker-shaped narrations --------------------------------
    # An exception, not a finding: it needs a human, and the generator does not
    # inject this class, so emitting a finding would be a false positive by
    # construction and would break the precision guarantee everything else rests on.
    for credit in bundle.bank:
        hits = sanitize.scan(credit.narration)
        if hits:
            result.exceptions.append(
                Exception_(
                    subject_id=credit.bank_txn_id,
                    kind="suspicious_narration",
                    context={"patterns": ",".join(hits), "narration": credit.narration[:200]},
                    reason=sanitize.describe(hits),
                )
            )

    # --- layer 2: rules learned from previous closes ----------------------
    still_open: list[Exception_] = []
    for exc in exceptions:
        ctx = _context_for(exc, bank_by_id)
        hit = store.resolve(ctx) if len(store) else None
        if hit:
            rule, verdict = hit
            finding = _rule_finding(exc, rule.rule_id, verdict, rule.confidence)
            if finding is not None:
                result.findings.append(finding)
                continue
            # The rule matched the narration but has no verdict on this item.
            # Not an answer - fall through and let a later layer try.
        still_open.append(exc)

    # --- layer 3: the model, on whatever is left --------------------------
    if not config.use_llm or not still_open:
        # extend, never assign: anything already queued - notably the
        # suspicious-narration warnings raised above - would otherwise be
        # silently discarded on every deterministic-only run, which is exactly
        # the path a security warning most needs to survive.
        result.exceptions.extend(still_open)
        result.elapsed_s = time.perf_counter() - started
        return result

    client = client or get_client(config.use_real_llm)
    history = build_history(result.findings, bundle)

    for exc in still_open:
        # Narrations are attacker-influenced text on their way into a prompt.
        # Neutralised here rather than at the edge, because the deterministic
        # layers match on the raw string and must keep seeing it unchanged -
        # only the model gets the defanged version.
        candidates = [
            {
                "bank_txn_id": bid,
                "amount": str(credit_amounts[bid]),
                "value_date": str(bank_by_id[bid].value_date),
                "narration": sanitize.neutralise(bank_by_id[bid].narration),
            }
            for bid in exc.candidates
            if bid in credit_amounts
        ]
        proposal = client.classify(exc, {"candidates": candidates})
        result.llm_calls += 1

        if float(proposal.get("confidence", 0)) < config.confidence_floor:
            exc.reason += f" | llm declined (confidence {proposal.get('confidence')})"
            result.exceptions.append(exc)
            continue

        expected = nets.get(exc.subject_id, Money("0.00"))
        verdict = verify_proposal(proposal, expected, credit_amounts)
        if not verdict.ok:
            # The model was confident and wrong, or confident and unverifiable.
            # Either way the item stays a human's problem, and we record why.
            exc.reason += f" | llm proposal rejected: {verdict.reason}"
            result.exceptions.append(exc)
            continue

        finding = to_finding(proposal, exc, verdict)
        if finding is None:
            exc.reason += " | llm proposal unparseable"
            result.exceptions.append(exc)
            continue
        result.findings.append(finding)

        # --- learn from it, if it survives the admission gate --------------
        if config.learn_rules and proposal.get("proposed_rule"):
            candidate = rule_from_proposal(proposal["proposed_rule"], exc.subject_id)
            if candidate:
                candidate = _with_id(candidate, next_rule_id(store, candidate.kind))
                store.propose(candidate, history)

    result.elapsed_s = time.perf_counter() - started
    return result


def _context_for(exc: Exception_, bank_by_id) -> dict[str, Any]:
    ctx: dict[str, Any] = {"subject_id": exc.subject_id, "kind": exc.kind}
    ctx.update({k: v for k, v in exc.context.items() if isinstance(v, (str, int, float))})
    for bid in exc.candidates:
        credit = bank_by_id.get(bid)
        if credit:
            ctx.setdefault("narration", credit.narration)
            break
    return ctx


def _rule_finding(exc: Exception_, rule_id: str, verdict: dict[str, Any], confidence: float):
    """Build a finding from a rule verdict, or refuse to.

    Returns ``None`` when the verdict does not actually determine a defect
    class. An earlier version defaulted to ``UNRECORDED_ADJUSTMENT`` here, which
    meant an ``adjustment_pattern`` rule - whose verdict only names a *category* -
    silently stamped that label onto any exception whose narration it happened to
    match, including orphan bank credits it has no opinion about. That produced
    confident findings on subjects with no defect at all, and dropped precision
    from 1.000 to 0.966 as the rule store grew.

    A rule that has no verdict on an item must leave the item alone.
    """
    from ..models import DefectClass
    from .results import Finding

    dc = verdict.get("defect_class")
    if not dc:
        return None
    try:
        defect = DefectClass(dc)
    except ValueError:
        return None
    return Finding(
        defect_class=defect,
        subject_id=exc.subject_id,
        layer=Layer.L2_RULE,
        confidence=confidence,
        rule_id=rule_id,
        evidence={"resolved_by_rule": rule_id, "verdict": verdict},
    )


def _with_id(rule, rule_id: str):
    from dataclasses import replace

    return replace(rule, rule_id=rule_id)
