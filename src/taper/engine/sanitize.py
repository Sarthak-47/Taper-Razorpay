"""Untrusted text hardening for anything that reaches the model.

Bank narrations are attacker-influenced. A merchant's own customer can put text
into a payment reference, and that string travels through the bank statement
into this system and, at layer 3, into a prompt. That is a prompt-injection
surface in a component that reasons about money.

The defence here is deliberately the *second* line, not the first. The first is
architectural: layer 3 only ever proposes, and ``verify_proposal`` re-derives
every claim arithmetically before it counts. A fully compromised model - one
that returns exactly what an attacker asked for - still cannot make a batch
reconcile, because the numbers are checked by code the model never touches.

So this module does not try to make injection impossible. It reduces the
attack surface, and more importantly it *reports* the attempt: a narration
carrying imperative instructions is itself worth a human's attention, whether
or not it worked.
"""

from __future__ import annotations

import re

# Phrases that have no business in a bank narration and every business in an
# injected instruction. Matched case-insensitively on word boundaries.
#
# Deliberately narrow. A loose pattern here would flag ordinary narrations
# ("REVERSAL AS INSTRUCTED BY BRANCH") and train a controller to ignore the
# warning, which is worse than not warning at all.
INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("instruction_override", re.compile(
        r"\b(ignore|disregard|forget)\b.{0,20}\b(previous|prior|above|all)\b", re.I)),
    ("role_switch", re.compile(
        r"\b(you are now|act as|system prompt|as an ai|new instructions?)\b", re.I)),
    ("directive", re.compile(
        r"\b(mark|treat|classify|report)\b.{0,30}\b(as )?(reconciled|matched|clean|resolved)\b",
        re.I)),
    ("confidence_forcing", re.compile(
        r"\b(confidence|certainty)\b.{0,15}\b(1\.0|100%|maximum|highest)\b", re.I)),
    ("delimiter_break", re.compile(r"(```|\"\"\"|</?\s*(system|instructions?|prompt)\s*>)", re.I)),
    ("json_injection", re.compile(r'"\s*(defect_class|bank_txn_ids|confidence)\s*"\s*:', re.I)),
]

# Bank narrations are short. Anything past this is not a reference, it is a
# payload, and truncating costs nothing real.
MAX_NARRATION = 160


def scan(text: str) -> list[str]:
    """Names of the injection patterns this text matches. Empty means clean."""
    if not text:
        return []
    return [name for name, pattern in INJECTION_PATTERNS if pattern.search(text)]


def neutralise(text: str) -> str:
    """Render a narration inert for inclusion in a prompt.

    Three things, none of which change how the string is *matched* by the
    deterministic layers - only how it is presented to a model:

      * collapse newlines, which is how a payload fakes the end of a field
      * strip the delimiters used to escape a quoted context
      * truncate to a length a real narration never exceeds
    """
    if not text:
        return ""
    flattened = re.sub(r"[\r\n\t]+", " ", text)
    flattened = flattened.replace("```", "").replace('"""', "")
    flattened = re.sub(r"</?\s*(system|instructions?|prompt)\s*>", "", flattened, flags=re.I)
    flattened = re.sub(r"\s{2,}", " ", flattened).strip()
    if len(flattened) > MAX_NARRATION:
        flattened = flattened[:MAX_NARRATION] + "...[truncated]"
    return flattened


def describe(hits: list[str]) -> str:
    """A reason string a controller can act on."""
    return (
        "Bank narration contains text shaped like instructions to an AI system "
        f"({', '.join(hits)}). The narration was neutralised before any model saw "
        "it, and no proposal can move money without passing arithmetic "
        "verification - but a payment reference carrying this is worth "
        "investigating on its own."
    )
