from unittest.mock import patch

import pytest

from agent_s4.cli import _config_from_dict, build_config_from_args, parse_args
from agent_s4.config.experiment import ExperimentConfig
from agent_s4.config.island import IslandConfig
from agent_s4.config.scheduler import DeterministicScheduler
from agent_s4.db import get_current_step, get_variant, insert_variant, set_current_step
from agent_s4.driver import Driver
from agent_s4.orchestrator import _config_to_dict


def config(**kwargs):
    return ExperimentConfig(
        islands=[IslandConfig(name="a", evaluator_harness="claude_code", eval_model="m")],
        scheduler=DeterministicScheduler(0, 2, 10),
        reevaluation_interval=5, **kwargs,
    )


def test_maintenance_preserves_budget_and_scheduler_cadences(tmp_path):
    driver = Driver(config(max_steps=12), tmp_path)
    calls = []
    with patch.object(driver, "run_op_on_all_islands", side_effect=lambda op, step: calls.append((op, step))):
        driver.run()
    regular = [entry for entry in calls if entry[0] != "reevaluation"]
    assert regular == [(driver.config.scheduler.choose_op(step, None), step) for step in range(1, 13)]
    assert [(op, step) for op, step in calls if op == "reevaluation"] == [
        ("reevaluation", 5), ("reevaluation", 10),
    ]
    assert calls.index(("reevaluation", 5)) == calls.index(("mutation", 5)) + 1
    assert calls.index(("reevaluation", 10)) == calls.index(("elimination", 10)) + 1
    assert get_current_step(driver.conn) == 12
    rows = driver.conn.execute("SELECT step, op_kind, status FROM dispatch_log").fetchall()
    assert len(rows) == 12
    assert all(row["op_kind"] != "reevaluation" and row["status"] == "complete" for row in rows)
    driver.conn.close()


def test_resume_runs_due_maintenance_before_next_evolution_step(tmp_path):
    driver = Driver(config(max_steps=6), tmp_path)
    set_current_step(driver.conn, 5)
    with patch.object(driver, "run_op_on_all_islands") as run:
        driver.run()
    assert [call.args for call in run.call_args_list] == [("reevaluation", 5), ("crossover", 6)]
    driver.conn.close()


def test_final_budgeted_step_still_gets_maintenance(tmp_path):
    driver = Driver(config(max_steps=5), tmp_path)
    set_current_step(driver.conn, 5)
    with patch.object(driver, "run_op_on_all_islands") as run:
        driver.run()
    run.assert_called_once_with("reevaluation", 5)
    assert get_current_step(driver.conn) == 5
    driver.conn.close()


def test_maintenance_failure_never_rolls_back_a_completed_interface(tmp_path):
    driver = Driver(config(max_steps=5), tmp_path)
    set_current_step(driver.conn, 4)
    def run(op, step):
        if op == "reevaluation":
            raise RuntimeError("failed repeat")
        insert_variant(driver.conn, {
            "id": "a-if-005", "island": "a", "tag": "evol/a/if-005",
            "origin": "mutation", "step": step, "score": 1, "val_score": 2,
        })
    with patch.object(driver, "run_op_on_all_islands", side_effect=run):
        with pytest.raises(RuntimeError, match="failed repeat"):
            driver.run()
    assert get_current_step(driver.conn) == 5
    driver.conn.close()

    resumed = Driver(config(max_steps=6), tmp_path)
    with patch.object(resumed, "run_op_on_all_islands") as run:
        resumed.run()
    assert get_variant(resumed.conn, "a-if-005") is not None
    assert [call.args for call in run.call_args_list] == [("reevaluation", 5), ("crossover", 6)]
    resumed.conn.close()


def test_reevaluation_cli_and_saved_config_round_trip():
    args = parse_args([
        "run", "--island", "name=a,evaluator_harness=claude_code,eval_model=m",
        "--reevaluation-interval", "5", "--reevaluation-top-k", "3",
    ])
    cfg = build_config_from_args(args)
    restored = _config_from_dict(_config_to_dict(cfg))
    assert restored.reevaluation_interval == 5
    assert restored.reevaluation_top_k == 3
    assert restored.max_steps == cfg.max_steps
    assert restored.scheduler.__dict__ == cfg.scheduler.__dict__
    old = _config_to_dict(cfg)
    del old["reevaluation_interval"], old["reevaluation_top_k"]
    assert _config_from_dict(old).reevaluation_interval == 0


@pytest.mark.parametrize("kwargs", [
    {"reevaluation_interval": -1}, {"reevaluation_interval": 1.5},
    {"reevaluation_top_k": 0}, {"reevaluation_top_k": True},
    {"reevaluation_interval": 5, "val_set_file": "none", "selection_signal": "train"},
])
def test_invalid_reevaluation_config(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(islands=[], **kwargs)
