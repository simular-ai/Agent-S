import pytest

from agent_s4.cli import _parse_scheduler
from agent_s4.config.scheduler import (
    DeterministicScheduler, ProbabilisticScheduler,
)


def test_parse_scheduler_deterministic_default():
    s = _parse_scheduler("deterministic")
    assert isinstance(s, DeterministicScheduler)
    assert s.migration_cadence == 5
    assert s.crossover_cadence == 6
    assert s.elimination_cadence == 10


def test_parse_scheduler_deterministic_with_cadences():
    s = _parse_scheduler("deterministic:7,11,13")
    assert s.migration_cadence == 7
    assert s.crossover_cadence == 11
    assert s.elimination_cadence == 13


def test_parse_scheduler_probabilistic():
    s = _parse_scheduler("probabilistic:mutation=0.7,migration=0.15,crossover=0.1,elimination=0.05")
    assert isinstance(s, ProbabilisticScheduler)
    assert s.kinds == ["mutation", "migration", "crossover", "elimination"]
    assert abs(s.probs[0] - 0.7) < 1e-9


def test_cli_benchmark_flag_parsed():
    from agent_s4.cli import parse_args, build_config_from_args
    args = parse_args([
        "run", "--island", "name=sci1,evaluator_harness=prime_agent,eval_model=sonnet",
        "--benchmark", "scienceboard",
        "--train-set", "train.json", "--val-set", "val.json",
    ])
    cfg = build_config_from_args(args)
    assert cfg.benchmark == "scienceboard"


def test_cli_benchmark_defaults_osworld_v2():
    from agent_s4.cli import parse_args, build_config_from_args
    args = parse_args([
        "run", "--island", "name=o1,evaluator_harness=prime_agent,eval_model=sonnet",
    ])
    cfg = build_config_from_args(args)
    assert cfg.benchmark == "osworld_v2"
    assert cfg.train_set_file == "osworld_v2_train_50.json"
    assert cfg.val_set_file == "osworld_v2_val_25.json"


def test_cli_v1_benchmark_requires_explicit_splits():
    from agent_s4.cli import parse_args, build_config_from_args
    args = parse_args([
        "run", "--island", "name=o1,evaluator_harness=prime_agent,eval_model=sonnet",
        "--benchmark", "osworld",
    ])
    with pytest.raises(ValueError, match="--train-set"):
        build_config_from_args(args)


def test_cli_builder_claude_bin_defaults_none():
    from agent_s4.cli import parse_args, build_config_from_args
    args = parse_args([
        "run", "--island", "name=o1,evaluator_harness=prime_agent,eval_model=sonnet",
    ])
    cfg = build_config_from_args(args)
    assert cfg.builder_claude_bin is None
