"""What this machine can actually run, and what to type next.

A reviewer clones the repo, runs a command from the README, and it fails
because Ollama is not up or a key is not set. That is a bad first minute, and
it is avoidable: the system knows which of its layers are available and can
simply say so.

Everything here is read-only and offline except one short probe of a local
Ollama socket. No check may take long enough that someone stops waiting.
"""

from __future__ import annotations

import importlib.util
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

OLLAMA_TAGS = "http://localhost:11434/api/tags"
PROBE_TIMEOUT = 2.0


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = False


@dataclass
class Diagnosis:
    checks: list[Check] = field(default_factory=list)
    ollama_models: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        """True only if something *required* is missing."""
        return any(c.required and not c.ok for c in self.checks)

    def recommended_command(self) -> str:
        """The best command this machine can actually run right now."""
        if self.ollama_models:
            preferred = _best_ollama_model(self.ollama_models)
            return (
                f"python -m taper.cli --llm ollama --llm-model {preferred} "
                f"reconcile --seed 99"
            )
        if any(c.name == "ANTHROPIC_API_KEY" and c.ok for c in self.checks):
            return "python -m taper.cli reconcile --seed 99"
        return "python -m taper.cli --no-llm reconcile --seed 99"

    def why(self) -> str:
        if self.ollama_models:
            return (
                "Ollama is running locally, so layer 3 costs nothing and needs no key."
            )
        if any(c.name == "ANTHROPIC_API_KEY" and c.ok for c in self.checks):
            return "An Anthropic key is set, so layer 3 will use it."
        return (
            "No model provider is available, so this runs the deterministic layers "
            "only - which is the honest baseline and reproduces most of the README."
        )


def _best_ollama_model(models: list[str]) -> str:
    """Prefer a mid-size instruct model: big enough to follow the schema, small
    enough to answer before anyone gives up waiting."""
    for preference in ("qwen2.5:14b", "qwen2.5:7b", "qwen3:8b", "llama3.1:8b"):
        if preference in models:
            return preference
    return models[0]


def _probe_ollama() -> tuple[bool, list[str], str]:
    try:
        with urllib.request.urlopen(OLLAMA_TAGS, timeout=PROBE_TIMEOUT) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return False, [], "not reachable on localhost:11434"

    models = sorted(m.get("name", "") for m in data.get("models", []) if m.get("name"))
    if not models:
        return True, [], "running, but no models pulled (try: ollama pull qwen2.5:7b)"
    return True, models, f"{len(models)} model(s): {', '.join(models[:4])}"


def run() -> Diagnosis:
    result = Diagnosis()

    import sys

    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    result.checks.append(
        Check(
            "Python",
            sys.version_info >= (3, 11),
            f"{version} (3.11+ required)",
            required=True,
        )
    )

    # The deterministic core is the whole point: it must never need anything.
    result.checks.append(
        Check("Deterministic layers", True, "no dependencies, always available",
              required=True)
    )

    up, models, detail = _probe_ollama()
    result.ollama_models = models
    result.checks.append(Check("Ollama (local layer 3)", up and bool(models), detail))

    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    result.checks.append(
        Check("ANTHROPIC_API_KEY", has_key,
              "set" if has_key else "not set (optional - see .env.example)")
    )

    result.checks.append(
        Check("LLM_API_KEY", bool(os.environ.get("LLM_API_KEY")),
              "set" if os.environ.get("LLM_API_KEY")
              else "not set (only needed for a hosted OpenAI-compatible provider)")
    )

    sklearn = importlib.util.find_spec("sklearn") is not None
    result.checks.append(
        Check("scikit-learn", sklearn,
              "available - risk model uses gradient boosting" if sklearn
              else "absent - risk model falls back to logistic regression")
    )

    anthropic_sdk = importlib.util.find_spec("anthropic") is not None
    result.checks.append(
        Check("anthropic SDK", anthropic_sdk,
              "available" if anthropic_sdk else "absent (only needed for --llm anthropic)")
    )

    return result
