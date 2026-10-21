"""Unit tests for IslandConfig.

v2: IslandConfig only carries per-island evaluator config. Builder
model/harness and seed/prompt are global on ExperimentConfig.
"""
from agent_s4.config.island import IslandConfig


def test_minimal_island_config():
    cfg = IslandConfig(
        name="gui",
        evaluator_harness="claude_code",
        eval_model="haiku",
    )
    assert cfg.name == "gui"
    assert cfg.evaluator_harness == "claude_code"
    assert cfg.eval_model == "haiku"


def test_evaluator_harness_can_differ_per_island():
    a = IslandConfig(name="haiku", evaluator_harness="claude_code", eval_model="haiku")
    b = IslandConfig(name="qwen", evaluator_harness="opencode", eval_model="qwen3")
    assert a.evaluator_harness != b.evaluator_harness
