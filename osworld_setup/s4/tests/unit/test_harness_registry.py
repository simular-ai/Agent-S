"""Harness registry lookup tests."""
import pytest

from agent_s4.harness import (
    BUILDER_HARNESSES, EVALUATOR_HARNESSES,
    resolve_builder_harness, resolve_evaluator_harness,
)
from agent_s4.harness.base import BuilderHarness, EvaluatorHarness


def test_registries_contain_prime_agent_and_opencode():
    assert set(BUILDER_HARNESSES) == {"prime_agent", "opencode"}
    assert set(EVALUATOR_HARNESSES) == {"prime_agent", "opencode"}


@pytest.mark.parametrize("key", ["prime_agent", "opencode"])
def test_resolve_returns_instance(key):
    b = resolve_builder_harness(key)
    assert isinstance(b, BuilderHarness)
    e = resolve_evaluator_harness(key)
    assert isinstance(e, EvaluatorHarness)


def test_unknown_key_raises_keyerror_with_helpful_message():
    with pytest.raises(KeyError) as exc:
        resolve_builder_harness("nonexistent")
    assert "nonexistent" in str(exc.value)
    assert "prime_agent" in str(exc.value)


@pytest.mark.parametrize("benchmark", ["osworld_v2", "scienceboard"])
def test_evaluator_resolves_to_shared_worker_for_every_benchmark(benchmark):
    """OSWorld V2 and ScienceBoard both run through UbuntuVMEvaluator, which
    picks the eval module by benchmark."""
    from agent_s4.harness.ubuntu_vm import UbuntuVMEvaluator

    harness = resolve_evaluator_harness("prime_agent", benchmark=benchmark)
    assert isinstance(harness, UbuntuVMEvaluator)
