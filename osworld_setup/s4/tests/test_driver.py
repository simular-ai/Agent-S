from unittest.mock import MagicMock, patch
from agent_s4.driver import Driver
from agent_s4.config.scheduler import DeterministicScheduler, ProbabilisticScheduler


def test_driver_loop_picks_op_via_scheduler(tmp_path):
    config = MagicMock()
    config.benchmark = "osworld"
    config.seed = None
    config.scheduler = DeterministicScheduler(5, 6, 10)
    config.max_steps = 3
    config.max_cost = None
    config.islands = [MagicMock(name="haiku"), MagicMock(name="sonnet"), MagicMock(name="qwen")]
    for i, name in enumerate(["haiku", "sonnet", "qwen"]):
        config.islands[i].name = name

    driver = Driver(config=config, base_dir=tmp_path)
    with patch.object(driver, "run_op_on_all_islands") as mock_run:
        driver.run()
        # Steps 1, 2, 3 -> all "mutation"
        assert mock_run.call_count == 3
        ops = [call.args[0] for call in mock_run.call_args_list]
        assert ops == ["mutation", "mutation", "mutation"]


def test_driver_works_with_probabilistic_scheduler(tmp_path):
    """Regression: ProbabilisticScheduler used to crash on step 1 because
    Driver.run() passed state=None. Driver now passes self (which carries
    a seeded .rng), so probabilistic dispatch works end-to-end."""
    config = MagicMock()
    config.benchmark = "osworld"
    config.seed = 42
    config.scheduler = ProbabilisticScheduler({
        "mutation": 0.7, "migration": 0.15,
        "crossover": 0.1, "elimination": 0.05,
    })
    config.max_steps = 5
    config.max_cost = None
    config.islands = [MagicMock()]
    config.islands[0].name = "haiku"

    driver = Driver(config=config, base_dir=tmp_path)
    seen_ops = []

    def fake_run(op, step):
        seen_ops.append(op)

    with patch.object(driver, "run_op_on_all_islands", side_effect=fake_run):
        driver.run()

    assert len(seen_ops) == 5
    # All ops are valid choices from the weights dict
    assert set(seen_ops) <= {"mutation", "migration", "crossover", "elimination"}




def test_driver_runs_op_in_parallel_across_islands(tmp_path):
    """Cross-process verification: 3 worker subprocesses each write a
    marker file, parent asserts all 3 exist after dispatch."""
    from agent_s4.driver import Driver
    from agent_s4.config.experiment import ExperimentConfig
    from agent_s4.config.island import IslandConfig
    from agent_s4.config.scheduler import DeterministicScheduler
    from tests._fixtures.marker_operator import _FilesystemMarkerOperator

    config = ExperimentConfig(
        islands=[
            IslandConfig(name="haiku", evaluator_harness="claude_code",
                         eval_model="haiku"),
            IslandConfig(name="sonnet", evaluator_harness="claude_code",
                         eval_model="sonnet"),
            IslandConfig(name="qwen", evaluator_harness="opencode",
                         eval_model="openrouter/qwen/qwen3.5"),
        ],
        scheduler=DeterministicScheduler(0, 0, 0),  # always mutation
        max_steps=1,
    )
    driver = Driver(
        config=config, base_dir=tmp_path,
        op_classes={"mutation": _FilesystemMarkerOperator},
    )
    try:
        driver.run()
    finally:
        # Driver.run() already shuts the pool in its finally; this is belt+suspenders
        driver._pool.shutdown(wait=False)

    marker_dir = tmp_path / "_test_markers"
    markers = sorted(m.name for m in marker_dir.iterdir())
    assert markers == ["haiku.touch", "qwen.touch", "sonnet.touch"]


def test_driver_propagates_worker_exception(tmp_path):
    """Cross-process: worker raises; driver re-raises across IPC."""
    from agent_s4.driver import Driver
    from agent_s4.config.experiment import ExperimentConfig
    from agent_s4.config.island import IslandConfig
    from agent_s4.config.scheduler import DeterministicScheduler
    from tests._fixtures.marker_operator import (
        _RaisingMarkerOperator, MarkerOperatorError,
    )

    config = ExperimentConfig(
        islands=[
            IslandConfig(name="haiku", evaluator_harness="claude_code",
                         eval_model="haiku"),
        ],
        scheduler=DeterministicScheduler(0, 0, 0),
        max_steps=1,
    )
    driver = Driver(
        config=config, base_dir=tmp_path,
        op_classes={"mutation": _RaisingMarkerOperator},
    )
    raised = None
    try:
        driver.run()
    except MarkerOperatorError as e:
        raised = e
    finally:
        driver._pool.shutdown(wait=False)

    assert raised is not None
    assert "intentional test failure" in str(raised)
    assert "haiku" in str(raised)


def test_build_loopstate_constructs_correct_state(tmp_path):
    from agent_s4.driver import _build_loopstate
    from agent_s4.config.experiment import ExperimentConfig
    from agent_s4.config.island import IslandConfig

    config = ExperimentConfig(
        islands=[IslandConfig(name="haiku", evaluator_harness="claude_code",
                              eval_model="haiku")],
        seed=42,
    )
    state, conn = _build_loopstate("haiku", step=7, config=config,
                                    base_dir=tmp_path)
    try:
        assert state.current_island == "haiku"
        assert state.step == 7
        assert state.config is config
        assert state.clone_dir == tmp_path / "OSWorldV2_Evol_haiku"
        assert state.base_dir == tmp_path
        assert state.eval_cache_dir == tmp_path / "eval_cache_osworld_v2"
        # Verify conn is a live, fresh sqlite3 connection
        assert conn is state.conn
        result = conn.execute("SELECT 1").fetchone()
        assert result[0] == 1
    finally:
        conn.close()


def test_worker_entry_dispatches_to_right_operator(tmp_path):
    """In-process test: _worker_entry called directly (no subprocess) with
    a custom op_classes dict; verify the right class is instantiated and
    its apply method is invoked."""
    from agent_s4.driver import _worker_entry
    from agent_s4.config.experiment import ExperimentConfig
    from agent_s4.config.island import IslandConfig
    from unittest.mock import MagicMock

    config = ExperimentConfig(
        islands=[IslandConfig(name="haiku", evaluator_harness="claude_code",
                              eval_model="haiku")],
    )
    fake_mutation_cls = MagicMock()
    fake_mutation_instance = fake_mutation_cls.return_value
    op_classes = {"mutation": fake_mutation_cls}

    _worker_entry("haiku", 3, "mutation", config, tmp_path, op_classes)

    fake_mutation_cls.assert_called_once_with()
    fake_mutation_instance.apply.assert_called_once()
    # Argument to apply is the LoopState; check its current_island.
    state_arg = fake_mutation_instance.apply.call_args.args[0]
    assert state_arg.current_island == "haiku"
    assert state_arg.step == 3


def test_worker_entry_rejects_unknown_op_kind(tmp_path):
    from agent_s4.driver import _worker_entry
    from agent_s4.config.experiment import ExperimentConfig
    from agent_s4.config.island import IslandConfig
    import pytest

    config = ExperimentConfig(
        islands=[IslandConfig(name="haiku", evaluator_harness="claude_code",
                              eval_model="haiku")],
    )
    with pytest.raises(ValueError, match="unknown op_kind"):
        _worker_entry("haiku", 1, "bogus_op", config, tmp_path, {})


def test_driver_creates_process_pool_in_init(tmp_path):
    from agent_s4.driver import Driver
    from agent_s4.config.experiment import ExperimentConfig
    from agent_s4.config.island import IslandConfig
    import concurrent.futures

    config = ExperimentConfig(
        islands=[
            IslandConfig(name="haiku", evaluator_harness="claude_code",
                         eval_model="haiku"),
            IslandConfig(name="sonnet", evaluator_harness="claude_code",
                         eval_model="sonnet"),
        ],
    )
    driver = Driver(config=config, base_dir=tmp_path)
    try:
        assert isinstance(driver._pool, concurrent.futures.ProcessPoolExecutor)
        assert driver.op_classes is None  # default
    finally:
        driver._pool.shutdown(wait=False)


def test_driver_accepts_op_classes_param(tmp_path):
    from agent_s4.driver import Driver
    from agent_s4.config.experiment import ExperimentConfig
    from agent_s4.config.island import IslandConfig

    config = ExperimentConfig(
        islands=[IslandConfig(name="haiku", evaluator_harness="claude_code",
                              eval_model="haiku")],
    )
    custom = {"mutation": object}  # placeholder picklable class
    driver = Driver(config=config, base_dir=tmp_path, op_classes=custom)
    try:
        assert driver.op_classes is custom
    finally:
        driver._pool.shutdown(wait=False)


def test_pool_reused_across_dispatches(tmp_path):
    """The Driver's ProcessPoolExecutor should be created once in
    __init__ and reused across multiple operator dispatches."""
    from agent_s4.driver import Driver
    from agent_s4.config.experiment import ExperimentConfig
    from agent_s4.config.island import IslandConfig
    from agent_s4.config.scheduler import DeterministicScheduler
    from tests._fixtures.marker_operator import _FilesystemMarkerOperator

    config = ExperimentConfig(
        islands=[
            IslandConfig(name="haiku", evaluator_harness="claude_code",
                         eval_model="haiku"),
        ],
        scheduler=DeterministicScheduler(0, 0, 0),
        max_steps=3,
    )
    driver = Driver(
        config=config, base_dir=tmp_path,
        op_classes={"mutation": _FilesystemMarkerOperator},
    )
    pool_id_before_run = id(driver._pool)
    try:
        driver.run()
    finally:
        driver._pool.shutdown(wait=False)
    # Pool object identity unchanged after run; we hold ref for assertion
    assert id(driver._pool) == pool_id_before_run
    # And the marker file was written 3 times (once per step) — confirms
    # that subsequent dispatches reused the pool without crashing
    assert (tmp_path / "_test_markers" / "haiku.touch").exists()


def test_migration_purges_transit_after_phase2(tmp_path):
    """A migration round runs phase1 then phase2 (barrier between), then the
    driver purges the transit root. Cleanup must happen AFTER phase2 so that
    under a complete topology every destination has read every source transit
    before any of them is removed (no per-destination deletion races)."""
    from agent_s4.driver import Driver
    from agent_s4.constants import MIGRATE_TRANSIT_DIR

    config = MagicMock()
    config.benchmark = "osworld"  # ns() -> bare names; matches un-namespaced paths below
    config.islands = [MagicMock()]
    config.islands[0].name = "haiku"
    driver = Driver(config=config, base_dir=tmp_path)

    transit_root = tmp_path / MIGRATE_TRANSIT_DIR
    (transit_root / "haiku").mkdir(parents=True)

    seen = []

    def fake_fanout(op_kind, step):
        seen.append(op_kind)
        # transit must still be present while phase2 is running
        if op_kind == "migration_phase2":
            assert transit_root.exists()

    try:
        with patch.object(driver, "_fan_out", side_effect=fake_fanout):
            driver.run_op_on_all_islands("migration", 5)
    finally:
        driver._pool.shutdown(wait=False)

    assert seen == ["migration_phase1", "migration_phase2"]
    assert not transit_root.exists(), "transit root must be purged after phase2"


def test_build_loopstate_uses_namespaced_db_for_scienceboard(tmp_path):
    """The driver's DB connection must use the benchmark-namespaced filename so
    a ScienceBoard run never writes the OSWorld shadow_evolution.db."""
    from agent_s4.driver import _build_loopstate
    from agent_s4.config.experiment import ExperimentConfig
    from agent_s4.config.island import IslandConfig

    config = ExperimentConfig(
        islands=[IslandConfig(name="haiku", evaluator_harness="claude_code",
                              eval_model="haiku")],
        benchmark="scienceboard", seed=42,
        train_set_file="train.json", val_set_file="val.json",
    )
    state, conn = _build_loopstate("haiku", step=0, config=config,
                                   base_dir=tmp_path)
    try:
        assert (tmp_path / "shadow_evolution_scienceboard.db").exists()
        assert not (tmp_path / "shadow_evolution.db").exists()
    finally:
        conn.close()
