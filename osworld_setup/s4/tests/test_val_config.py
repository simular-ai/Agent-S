"""Tests for ExperimentConfig val_set_file + selection_signal fields."""
import pytest

from agent_s4.config.experiment import ExperimentConfig
from agent_s4.config.island import IslandConfig
from agent_s4.cli import parse_args, build_config_from_args


def _minimal_island():
    return IslandConfig(
        name="t",
        evaluator_harness="claude_code",
        eval_model="haiku",
    )


def _island_argv() -> str:
    return "name=t,evaluator_harness=claude_code,eval_model=haiku"


def test_defaults_have_val_enabled_with_val_selection():
    cfg = ExperimentConfig(islands=[_minimal_island()])
    assert cfg.val_set_file == "osworld_v2_val_25.json"
    assert cfg.selection_signal == "val"


def test_disable_val_with_val_selection_is_rejected():
    with pytest.raises(ValueError, match="val_set_file=none"):
        ExperimentConfig(
            islands=[_minimal_island()],
            val_set_file="none", selection_signal="val",
        )


def test_disable_val_with_train_selection_is_allowed():
    cfg = ExperimentConfig(
        islands=[_minimal_island()],
        val_set_file="none", selection_signal="train",
    )
    assert cfg.val_set_file == "none"
    assert cfg.selection_signal == "train"


def test_invalid_selection_signal_rejected():
    with pytest.raises(ValueError, match="selection_signal"):
        ExperimentConfig(
            islands=[_minimal_island()],
            selection_signal="something",
        )


def test_cli_parses_val_flags():
    argv = [
        "run",
        "--island", _island_argv(),
        "--val-set", "my_val.json",
        "--selection-signal", "train",
    ]
    args = parse_args(argv)
    cfg = build_config_from_args(args)
    assert cfg.val_set_file == "my_val.json"
    assert cfg.selection_signal == "train"


def test_cli_val_set_none():
    argv = [
        "run",
        "--island", _island_argv(),
        "--val-set", "none",
        "--selection-signal", "train",
    ]
    cfg = build_config_from_args(parse_args(argv))
    assert cfg.val_set_file == "none"


def test_cli_default_when_flag_omitted():
    argv = [
        "run",
        "--island", _island_argv(),
    ]
    cfg = build_config_from_args(parse_args(argv))
    assert cfg.val_set_file == "osworld_v2_val_25.json"
    assert cfg.selection_signal == "val"


# --- migration_mode (copy vs move migration) -------------------------------


def test_default_migration_mode_is_move():
    cfg = ExperimentConfig(islands=[_minimal_island()])
    assert cfg.migration_mode == "move"


def test_invalid_migration_mode_rejected():
    with pytest.raises(ValueError, match="migration_mode"):
        ExperimentConfig(
            islands=[_minimal_island()],
            migration_mode="duplicate",
        )


def test_cli_parses_migration_mode():
    argv = [
        "run",
        "--island", _island_argv(),
        "--migration-mode", "copy",
    ]
    cfg = build_config_from_args(parse_args(argv))
    assert cfg.migration_mode == "copy"


def test_cli_default_migration_mode_is_move():
    argv = ["run", "--island", _island_argv()]
    cfg = build_config_from_args(parse_args(argv))
    assert cfg.migration_mode == "move"


def test_migration_mode_survives_rerun_roundtrip():
    """`shadow-evol rerun` reconstructs config from config.json via
    _config_to_dict -> _config_from_dict; migration_mode must round-trip."""
    from agent_s4.orchestrator import _config_to_dict
    from agent_s4.cli import _config_from_dict

    cfg = ExperimentConfig(islands=[_minimal_island()], migration_mode="copy")
    restored = _config_from_dict(_config_to_dict(cfg))
    assert restored.migration_mode == "copy"


# --- migration_topology (ring vs complete/all-to-all migration) ------------


def test_default_migration_topology_is_ring():
    cfg = ExperimentConfig(islands=[_minimal_island()])
    assert cfg.migration_topology == "ring"


def test_invalid_migration_topology_rejected():
    with pytest.raises(ValueError, match="migration_topology"):
        ExperimentConfig(
            islands=[_minimal_island()],
            migration_topology="star",
        )


def test_cli_parses_migration_topology():
    argv = [
        "run",
        "--island", _island_argv(),
        "--migration-topology", "complete",
    ]
    cfg = build_config_from_args(parse_args(argv))
    assert cfg.migration_topology == "complete"


def test_cli_default_migration_topology_is_ring():
    argv = ["run", "--island", _island_argv()]
    cfg = build_config_from_args(parse_args(argv))
    assert cfg.migration_topology == "ring"


def test_migration_topology_survives_rerun_roundtrip():
    from agent_s4.orchestrator import _config_to_dict
    from agent_s4.cli import _config_from_dict

    cfg = ExperimentConfig(islands=[_minimal_island()],
                           migration_topology="complete")
    restored = _config_from_dict(_config_to_dict(cfg))
    assert restored.migration_topology == "complete"
