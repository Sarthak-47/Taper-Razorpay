"""A digest that identifies one close exactly.

Finance runs on re-derivability. "This is the same close I signed off in June"
has to be checkable in a second, not by reading two reports side by side, and
"nothing changed except the code" has to be distinguishable from "the numbers
moved".

The hash covers **conclusions, not commentary**: matches, findings and
exceptions, each reduced to the fields that would change what somebody signs.
Timestamps, wording, ordering and runtime are excluded on purpose - a report
regenerated an hour later must hash identically, or the value is gone and
everyone learns to ignore it.

Sorting is what makes it stable. Dictionaries and set iteration are not ordered
by anything meaningful across runs, so every collection is sorted into a
canonical sequence before it is hashed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .engine.results import ReconResult

# Bumped only when the *meaning* of the digest changes - a new field entering
# the canonical form, or one leaving it. Without this a hash from an older
# version would silently compare unequal for reasons nobody could reconstruct.
DIGEST_VERSION = "taper-close-v1"


@dataclass(frozen=True)
class Attestation:
    digest: str
    version: str
    matches: int
    findings: int
    exceptions: int

    @property
    def short(self) -> str:
        return self.digest[:12]

    def line(self) -> str:
        return (
            f"{self.version}:{self.short} "
            f"({self.matches}m/{self.findings}f/{self.exceptions}e)"
        )


def canonical_form(result: ReconResult) -> dict[str, Any]:
    """The parts of a close that would change what someone signs."""
    matches = sorted(
        [
            [m.batch_id, sorted(m.bank_txn_ids), str(m.expected_net), str(m.credited)]
            for m in result.matches
        ]
    )
    findings = sorted(
        [
            [f.defect_class.value, f.subject_id, str(f.money_impact)]
            for f in result.findings
        ]
    )
    # Exceptions carry only kind and subject. The reason text is prose that gets
    # reworded, and hashing prose would make every copy-edit look like a
    # restated close.
    exceptions = sorted([[e.kind, e.subject_id] for e in result.exceptions])
    return {
        "version": DIGEST_VERSION,
        "matches": matches,
        "findings": findings,
        "exceptions": exceptions,
    }


def attest(result: ReconResult) -> Attestation:
    """Digest one close."""
    blob = json.dumps(canonical_form(result), separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return Attestation(
        digest=digest,
        version=DIGEST_VERSION,
        matches=len(result.matches),
        findings=len(result.findings),
        exceptions=len(result.exceptions),
    )
