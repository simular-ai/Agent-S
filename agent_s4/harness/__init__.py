"""Harness registry. Lookup by string key; instantiate on demand."""

from __future__ import annotations

from agent_s4.harness.base import (
    BuilderHarness,
    EvaluatorHarness,
    BuilderResult,
    EvalResults,
)
from agent_s4.harness.opencode.builder import OpenCodeBuilder
from agent_s4.harness.prime.builder import PrimeAgentBuilder
from agent_s4.harness.ubuntu_vm import UbuntuVMEvaluator

BUILDER_HARNESSES: dict[str, type[BuilderHarness]] = {
    "prime_agent": PrimeAgentBuilder,
    "opencode": OpenCodeBuilder,
}

# The evaluator class is benchmark-bound (VM provisioning + scoring), not
# model-bound. OSWorld V2 and ScienceBoard both run through the shared worker
# (agent_s4.eval.osworld_v2 / agent_s4.eval.scienceboard_prime), which
# UbuntuVMEvaluator selects by benchmark.
EVALUATOR_HARNESSES: dict[str, type[EvaluatorHarness]] = {
    "prime_agent": UbuntuVMEvaluator,
    "opencode": UbuntuVMEvaluator,
}


def resolve_builder_harness(key: str) -> BuilderHarness:
    try:
        cls = BUILDER_HARNESSES[key]
    except KeyError:
        raise KeyError(
            f"unknown builder harness {key!r}; known: {sorted(BUILDER_HARNESSES)}"
        )
    return cls()


def resolve_evaluator_harness(key: str, benchmark: str = "osworld") -> EvaluatorHarness:
    try:
        cls = EVALUATOR_HARNESSES[key]
    except KeyError:
        raise KeyError(
            f"unknown evaluator harness {key!r}; known: {sorted(EVALUATOR_HARNESSES)}"
        )
    return cls()


__all__ = [
    "BUILDER_HARNESSES",
    "EVALUATOR_HARNESSES",
    "resolve_builder_harness",
    "resolve_evaluator_harness",
    "BuilderHarness",
    "EvaluatorHarness",
    "BuilderResult",
    "EvalResults",
]
