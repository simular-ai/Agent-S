"""Unit tests for ExperimentConfig: construction, defaulting, island lookup.

v2: IslandConfig only carries name + evaluator harness/model. Builder
harness/model + seed_dir/prompt_template live on ExperimentConfig.
"""
from pathlib import Path

import pytest

from agent_s4.config.experiment import ExperimentConfig
from agent_s4.config.island import IslandConfig


def _mk(name, **overrides):
    defaults = dict(evaluator_harness="prime_agent", eval_model="haiku")
    defaults.update(overrides)
    return IslandConfig(name=name, **defaults)


def test_defaults():
    cfg = ExperimentConfig(islands=[_mk("gui")])
    assert cfg.builder_model == "opus"
    assert cfg.builder_harness == "prime_agent"
    assert cfg.builder_timeout == 600
    assert cfg.builder_no_time_limit is False
    assert cfg.eval_timeout == 600
    assert cfg.region == "us-east-1"
    assert cfg.max_steps is None
    assert cfg.max_cost is None
    assert cfg.seed is None
    assert cfg.held_out_tasks_path == Path("eval_sets/osworld_v2_test.json")


def test_explicit_overrides():
    cfg = ExperimentConfig(
        islands=[_mk("gui")],
        builder_model="sonnet",
        max_steps=30,
        seed=42,
        region="us-west-2",
    )
    assert cfg.builder_model == "sonnet"
    assert cfg.max_steps == 30
    assert cfg.seed == 42
    assert cfg.region == "us-west-2"


def test_island_by_name():
    cfg = ExperimentConfig(islands=[_mk("gui"), _mk("cli")])
    assert cfg.island_by_name("gui").name == "gui"
    assert cfg.island_by_name("cli").name == "cli"
    with pytest.raises(KeyError):
        cfg.island_by_name("nonexistent")


def test_resolve_builder_model_uniform_in_v2():
    cfg = ExperimentConfig(islands=[_mk("gui")], builder_model="opus")
    assert cfg.resolve_builder_model(cfg.islands[0]) == "opus"


def test_resolve_eval_model_from_island():
    cfg = ExperimentConfig(islands=[_mk("gui", eval_model="sonnet")])
    assert cfg.resolve_eval_model(cfg.islands[0]) == "sonnet"


def test_resolve_evaluator_harness_from_island():
    cfg = ExperimentConfig(
        islands=[_mk("qwen", evaluator_harness="prime_agent")],
    )
    assert cfg.resolve_evaluator_harness_key(cfg.islands[0]) == "prime_agent"


def test_selection_signal_validates():
    with pytest.raises(ValueError, match="selection_signal"):
        ExperimentConfig(islands=[_mk("gui")], selection_signal="other")  # type: ignore[arg-type]


def test_selection_signal_val_requires_val_set():
    with pytest.raises(ValueError, match="val_set_file=none"):
        ExperimentConfig(
            islands=[_mk("gui")],
            val_set_file="none",
            selection_signal="val",
        )


def test_benchmark_defaults_to_osworld_v2():
    from agent_s4.config.experiment import ExperimentConfig
    cfg = ExperimentConfig(islands=[])
    assert cfg.benchmark == "osworld_v2"


@pytest.mark.parametrize("benchmark", ["osworld", "scienceboard"])
def test_benchmarks_without_shipped_splits_require_them(benchmark):
    with pytest.raises(ValueError, match="no default train split"):
        ExperimentConfig(islands=[], benchmark=benchmark)
    with pytest.raises(ValueError, match="no default val split"):
        ExperimentConfig(islands=[], benchmark=benchmark,
                         train_set_file="train.json")
    cfg = ExperimentConfig(islands=[], benchmark=benchmark,
                           train_set_file="train.json", val_set_file="val.json")
    assert cfg.train_set_file == "train.json"
    assert cfg.val_set_file == "val.json"


def test_v1_osworld_has_no_held_out_default():
    cfg = ExperimentConfig(islands=[], benchmark="osworld",
                           train_set_file="train.json", val_set_file="val.json")
    assert cfg.held_out_tasks_path is None


def test_osworld_v2_is_a_supported_benchmark():
    cfg = ExperimentConfig(islands=[], benchmark="osworld_v2")
    assert cfg.benchmark == "osworld_v2"
    assert cfg.train_set_file == "osworld_v2_train_50.json"
    assert cfg.val_set_file == "osworld_v2_val_25.json"
    assert cfg.held_out_tasks_path == Path("eval_sets/osworld_v2_test.json")


def test_osworld_v2_preserves_explicit_legacy_split_names():
    cfg = ExperimentConfig(
        islands=[], benchmark="osworld_v2",
        train_set_file="eval_island_100.json",
        val_set_file="eval_heldout_50.json",
        allow_v2_noncanonical_splits=True,
    )
    assert cfg.train_set_file == "eval_island_100.json"
    assert cfg.val_set_file == "eval_heldout_50.json"


def test_invalid_benchmark_raises():
    from agent_s4.config.experiment import ExperimentConfig
    with pytest.raises(ValueError):
        ExperimentConfig(islands=[], benchmark="bogus")


def test_builder_no_time_limit_accepts_opencode_builder():
    cfg = ExperimentConfig(
        islands=[_mk("gui")], builder_harness="opencode", builder_no_time_limit=True
    )
    assert cfg.builder_no_time_limit


def test_builder_no_time_limit_rejects_other_builders():
    with pytest.raises(ValueError, match="builder_no_time_limit"):
        ExperimentConfig(
            islands=[_mk("gui")], builder_harness="other", builder_no_time_limit=True
        )
