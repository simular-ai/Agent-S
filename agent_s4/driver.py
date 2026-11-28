"""Global lockstep driver.

One process replaces N tmux panes. The driver maintains a single global
step counter, dispatches one operator per step via a pluggable Scheduler,
and fans the operator out to all islands in parallel with a barrier.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import random
import shutil
import subprocess
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from agent_s4.config.experiment import ExperimentConfig
from agent_s4.benchmark_spec import ns
from agent_s4.constants import (
    DB_FILENAME,
    EVAL_CACHE_DIR,
    MIGRATE_TRANSIT_DIR,
)
from agent_s4.db import (
    connect,
    delete_orphan_variants_at_step,
    find_pending_dispatch,
    get_current_step,
    init_schema,
    log_dispatch_complete,
    log_dispatch_start,
)
from agent_s4.operators.crossover import Crossover
from agent_s4.operators.elimination import Elimination
from agent_s4.operators.migration import Migration
from agent_s4.operators.mutation import Mutation
from agent_s4.operators.reevaluation import Reevaluation
from agent_s4.state import LoopState

logger = logging.getLogger("agent_s4.driver")


def derive_seed(global_seed: int | None, island_name: str) -> int:
    """Per-island rng seed derived deterministically from the global seed +
    island name. `None` global seed → fresh randomness."""
    if global_seed is None:
        return int.from_bytes(os.urandom(8), "big")
    h = sha256(f"{global_seed}:{island_name}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def _build_loopstate(
    island_name: str, step: int, config: ExperimentConfig, base_dir: Path
):
    """Construct a LoopState inside a worker subprocess.

    Returns (state, conn) so the caller can close conn in a finally block.
    Picklable inputs only — conn is opened locally, never passed in."""
    from agent_s4.benchmark_spec import BENCHMARKS

    template = BENCHMARKS[config.benchmark]["template"]
    worker_conn = connect(str(base_dir / ns(DB_FILENAME, config.benchmark)))
    rng = random.Random(derive_seed(config.seed, island_name))
    state = LoopState(
        conn=worker_conn,
        current_island=island_name,
        step=step,
        rng=rng,
        config=config,
        clone_dir=base_dir / f"{template}_{island_name}",
        base_dir=base_dir,
        eval_cache_dir=base_dir / ns(EVAL_CACHE_DIR, config.benchmark),
    )
    return state, worker_conn


# Default operator registry used by _worker_entry when no op_classes
# injection is provided. Tests pass a custom op_classes dict to substitute
# marker operators that the production registry doesn't know about.
_DEFAULT_OP_CLASSES = {
    "mutation": Mutation,
    "reevaluation": Reevaluation,
    "crossover": Crossover,
    "elimination": Elimination,
    "migration_phase1": Migration,
    "migration_phase2": Migration,
}


def _worker_entry(
    island_name: str,
    step: int,
    op_kind: str,
    config: ExperimentConfig,
    base_dir: Path,
    op_classes: dict | None = None,
) -> None:
    """ProcessPoolExecutor worker entry. Picklable; runs in subprocess.

    Builds own LoopState (with own DB conn), dispatches the operator
    named by op_kind, closes conn in finally. op_classes lets tests
    substitute marker operators across the process boundary; production
    callers pass None to use the default registry."""
    registry = op_classes if op_classes is not None else _DEFAULT_OP_CLASSES
    state, conn = _build_loopstate(island_name, step, config, base_dir)
    try:
        if op_kind == "elimination":
            cls = registry.get("elimination", Elimination)
            cls(count=config.elimination_count).apply(state)
        elif op_kind == "migration_phase1":
            cls = registry.get("migration_phase1", Migration)
            cls().apply_phase1(state)
        elif op_kind == "migration_phase2":
            cls = registry.get("migration_phase2", Migration)
            cls().apply_phase2(state)
        elif op_kind in ("mutation", "crossover", "reevaluation"):
            cls = registry.get(op_kind)
            if cls is None:
                cls = _DEFAULT_OP_CLASSES[op_kind]
            cls().apply(state)
        else:
            raise ValueError(f"unknown op_kind: {op_kind!r}")
    finally:
        conn.close()


def _total_cost_for_run(conn) -> float:
    row = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM cost_events").fetchone()
    return float(row[0] if row else 0)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Driver:
    def __init__(
        self, config: ExperimentConfig, base_dir: Path, op_classes: dict | None = None
    ):
        self.config = config
        self.base_dir = Path(base_dir)
        self.conn = connect(str(self.base_dir / ns(DB_FILENAME, self.config.benchmark)))
        init_schema(self.conn)
        # Scheduler RNG: shared across choose_op calls so probabilistic
        # schedulers produce reproducible sequences for a given seed.
        # Coerce non-seed values (e.g. MagicMock in tests that don't care)
        # to None so random.Random falls through to fresh entropy.
        seed = getattr(config, "seed", None)
        if not isinstance(seed, (int, float, str, bytes, bytearray, type(None))):
            seed = None
        self.rng = random.Random(seed)
        # Optional operator class registry; passed to _worker_entry across
        # the process boundary. None → workers use _DEFAULT_OP_CLASSES.
        self.op_classes = op_classes
        # One warm ProcessPoolExecutor reused across all dispatches.
        # ProcessPoolExecutor (not ThreadPoolExecutor) because OSWorld's
        # DesktopEnv.__init__ calls signal.signal() which is main-thread-only;
        # each worker needs its own main thread.
        self._pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=len(config.islands)
        )

    def stop_condition(self, step: int) -> bool:
        if self.config.max_steps is not None and step > self.config.max_steps:
            return True
        if self.config.max_cost is not None:
            if _total_cost_for_run(self.conn) >= self.config.max_cost:
                return True
        return False

    def _fan_out(self, op_kind: str, step: int) -> None:
        """Submit one _worker_entry per island to the warm pool, barrier on
        all futures. Exceptions from worker subprocesses are re-raised in the
        driver via f.result() (pickled across IPC with _RemoteTraceback)."""
        futures = [
            self._pool.submit(
                _worker_entry,
                i.name,
                step,
                op_kind,
                self.config,
                self.base_dir,
                self.op_classes,
            )
            for i in self.config.islands
        ]
        for f in futures:
            f.result()

    def run_op_on_all_islands(self, op: str, step: int) -> None:
        """Fan an operator dispatch out across islands. For migration the
        driver runs phase 1 to completion (barrier) before phase 2 starts;
        other ops are a single fan-out."""
        if op == "migration":
            self._fan_out("migration_phase1", step)
            self._fan_out("migration_phase2", step)
            # Purge transit dirs only AFTER the phase-2 barrier. Under a
            # complete topology a source transit is read by several
            # destinations in parallel, so cleanup can't live in phase 2; by
            # here every destination has consumed every source. A mid-phase-2
            # crash skips this line, leaving transits for idempotent re-dispatch.
            transit_root = self.base_dir / ns(
                MIGRATE_TRANSIT_DIR, self.config.benchmark
            )
            if transit_root.exists():
                shutil.rmtree(transit_root, ignore_errors=True)
            return
        self._fan_out(op, step)

    def _resume_check(self) -> bool:
        """If a previous run crashed mid-dispatch, clean up partial state.

        Returns True if cleanup happened, False if no pending dispatch.
        """
        pending = find_pending_dispatch(self.conn)
        if pending is None:
            return False

        pending_id = pending["id"]
        step = pending["step"]
        op_kind = pending["op_kind"]
        started_at = pending["started_at"]
        logger.warning(
            "Resume: found pending dispatch step=%d op=%s started_at=%s — cleaning up",
            step,
            op_kind,
            started_at,
        )

        # Delete orphan variants from the partial step
        from agent_s4.benchmark_spec import BENCHMARKS

        template = BENCHMARKS[self.config.benchmark]["template"]
        orphans = delete_orphan_variants_at_step(self.conn, step)
        for orphan in orphans:
            # Best-effort git tag deletion on that island's clone
            clone_dir = self.base_dir / f"{template}_{orphan['island']}"
            if clone_dir.exists():
                subprocess.run(
                    ["git", "tag", "-d", orphan["tag"]],
                    cwd=str(clone_dir),
                    capture_output=True,
                    check=False,
                )
            logger.info(
                "Resume: deleted orphan variant %s (tag %s)",
                orphan["id"],
                orphan["tag"],
            )

        # For migration: PRESERVE transit dirs. Phase 2 will read them on
        # re-dispatch and complete the receive. Only purge transit dirs
        # if we're rolling back a non-migration op (defensive — shouldn't happen).
        if op_kind != "migration":
            transit_root = self.base_dir / ns(
                MIGRATE_TRANSIT_DIR, self.config.benchmark
            )
            if transit_root.exists():
                logger.warning("Resume: non-migration crash had transit dirs; purging")
                shutil.rmtree(transit_root, ignore_errors=True)

        # Delete the pending dispatch log row
        self.conn.execute("DELETE FROM dispatch_log WHERE id = ?", (pending_id,))
        self.conn.commit()
        return True

    def _reevaluate_if_due(self, completed_step: int) -> None:
        interval = getattr(self.config, "reevaluation_interval", 0)
        if (
            isinstance(interval, int)
            and interval > 0
            and completed_step > 0
            and completed_step % interval == 0
        ):
            # Separate from dispatch_log/current_step: a failed maintenance
            # round must not roll back the completed evolutionary operator.
            self.run_op_on_all_islands("reevaluation", completed_step)

    def run(self) -> None:
        self._resume_check()
        step = get_current_step(self.conn) + 1
        try:
            # Includes resuming an experiment paused after step 5, and a
            # retry after a maintenance failure at the final budgeted step.
            self._reevaluate_if_due(step - 1)
            while not self.stop_condition(step):
                # Pass self as state — Scheduler.choose_op only needs `.rng`,
                # which Driver provides. DeterministicScheduler ignores state;
                # ProbabilisticScheduler uses state.rng.choices().
                op = self.config.scheduler.choose_op(step, state=self)
                logger.info("[step=%d] dispatching op=%s", step, op)
                dispatch_id = log_dispatch_start(self.conn, step, op, _now())
                self.run_op_on_all_islands(op, step)
                log_dispatch_complete(self.conn, dispatch_id, _now())
                self._reevaluate_if_due(step)
                step += 1
        finally:
            self._pool.shutdown(wait=True)
        logger.info("Driver stopping at step=%d", step)
