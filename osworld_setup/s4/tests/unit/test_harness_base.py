"""Sanity tests for harness base interfaces + result dataclasses."""
from pathlib import Path

import pytest

from agent_s4.harness.base import (
    BuilderHarness, EvaluatorHarness,
    BuilderResult, EvalResults,
)


def test_builder_result_fields():
    r = BuilderResult(
        exit_code=0,
        cost_usd=1.23,
        log_path=Path("/tmp/log.jsonl"),
        crashed=False,
    )
    assert r.exit_code == 0
    assert r.cost_usd == 1.23
    assert r.log_path == Path("/tmp/log.jsonl")
    assert r.crashed is False


def test_eval_results_fields_and_default():
    r = EvalResults(
        pass_rate=0.4, total=25, passed=10, failed=15, results=[],
    )
    assert r.timed_out is False
    r2 = EvalResults(
        pass_rate=0, total=25, passed=0, failed=25, results=[], timed_out=True,
    )
    assert r2.timed_out is True


def test_builder_harness_is_abstract():
    with pytest.raises(TypeError):
        BuilderHarness()


def test_evaluator_harness_is_abstract():
    with pytest.raises(TypeError):
        EvaluatorHarness()
