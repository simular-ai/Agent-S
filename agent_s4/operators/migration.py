"""Migration operator (two-phase, v2).

Phase 1 (source): each island tournament-picks one variant and writes its
shadow_interface to a transit dir. In move mode it then deletes the pick from
its own pop; in copy mode the pick stays (see config.migration_mode).

Phase 2 (destination): after a barrier, each island receives one migrant per
source island, re-evaluates it under its own evaluator, and inserts it. Which
sources a destination reads depends on config.migration_topology — ring (the
single predecessor) or complete (every other island; see sources_for).

Transit cleanup is owned by the driver, not phase 2: under a complete topology
a source transit is read by several destinations in parallel, so no single
destination may delete it. The driver purges the transit root once after the
phase-2 barrier (Driver.run_op_on_all_islands). On a mid-migration crash the
transit dirs survive, and re-dispatch is idempotent (phase 1 skips if its
transit exists; phase 2 skips a source whose migrant already exists at this step).

Phase order matters for crash recovery: write transit before the source-side
delete (move mode). See spec section "Phase 1 order of operations".
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import ClassVar

from agent_s4.costs import record_eval_costs
from agent_s4.db import (
    append_history,
    delete_variant,
    get_population,
    insert_eval_result,
    insert_variant,
    recompute_score_from_train_set,
    recompute_val_score,
    update_variant,
)
from agent_s4.harness import resolve_evaluator_harness
from agent_s4.operators.base import Operator
from agent_s4.state import LoopState
from agent_s4.tournament import _collect_per_task_records, tournament

from agent_s4.benchmark_spec import ns
from agent_s4.constants import (
    EVAL_CACHE_DIR,
    EVAL_SETS_DIR,
    MIGRATE_TRANSIT_DIR,
)
from agent_s4.git_utils import git_archive_variant, git_tag_variant
from agent_s4.utils import _load_val_tasks, get_next_variant_num, now_iso

logger = logging.getLogger("agent_s4.operators.migration")


def successor_of(island_name: str, island_names: list[str]) -> str:
    """Ring successor: islands[i] -> islands[(i+1) % N]."""
    i = island_names.index(island_name)
    return island_names[(i + 1) % len(island_names)]


def predecessor_of(island_name: str, island_names: list[str]) -> str:
    """Ring predecessor: islands[i] -> islands[(i-1) % N]."""
    i = island_names.index(island_name)
    return island_names[(i - 1) % len(island_names)]


def sources_for(island_name: str, island_names: list[str], topology: str) -> list[str]:
    """Islands whose transit this destination consumes in phase 2.

    ring (default): [predecessor] — the single upstream neighbor.
    complete:       every other island — all-to-all gene flow (= the
                    bidirectional ring at N=3), so the destination's
                    intra-island crossover pool can hold migrants from every
                    island at once. That is the structural precondition for a
                    single variant to acquire ancestry from all islands.
    """
    if topology == "complete":
        return [n for n in island_names if n != island_name]
    return [predecessor_of(island_name, island_names)]


def _atomic_write_transit(
    src_clone_dir: str, src_tag: str, src_id: str, transit_dir: Path
) -> None:
    """git_archive src_tag from src_clone into transit_dir/.

    Writes via tmp+rename so partial writes are never visible. Also
    writes a metadata.json with the source variant id and tag for
    phase 2 to read.
    """
    tmp_dir = str(transit_dir) + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    if os.path.exists(str(transit_dir)):
        shutil.rmtree(str(transit_dir))
    os.makedirs(os.path.dirname(str(transit_dir)), exist_ok=True)
    git_archive_variant(src_clone_dir, src_tag, tmp_dir)
    # write metadata into the tmp dir before the atomic rename
    meta = {"source_id": src_id, "source_tag": src_tag}
    with open(os.path.join(tmp_dir, "metadata.json"), "w") as f:
        json.dump(meta, f)
    os.rename(tmp_dir, str(transit_dir))


class Migration(Operator):
    kind: ClassVar[str] = "migration"

    def __init__(self) -> None:
        pass

    def apply_phase1(self, state: LoopState) -> None:
        """Source side: pick variant, write transit (atomic), then delete."""
        island_name = state.current_island
        island_cfg = state.current_config()
        clone_dir = str(state.clone_dir)
        conn = state.conn

        signal = state.config.selection_signal
        score_key = "val_score" if signal == "val" else "score"
        pop = [v for v in get_population(conn, island_name) if v[score_key] is not None]
        if not pop:
            logger.warning("Migration phase1: %s has no scored variants", island_name)
            return

        # Resume idempotency: if transit dir already exists from a prior
        # crashed attempt, phase 1 is already done for this source.
        transit_dir = (
            Path(state.base_dir)
            / ns(MIGRATE_TRANSIT_DIR, state.config.benchmark)
            / island_name
        )
        if transit_dir.exists():
            logger.info(
                "Migration phase1: transit dir for %s already exists, skipping pick",
                island_name,
            )
            return

        eval_model = state.config.resolve_eval_model(island_cfg)
        picked = tournament(
            conn,
            pop,
            k=2,
            t=1,
            clone_dir=clone_dir,
            eval_model=eval_model,
            eval_timeout=state.config.eval_timeout,
            region=state.config.region,
            selection_signal=signal,
            benchmark=state.config.benchmark,
        )[0]

        # Order: write transit BEFORE the source-side mutation. If we crash
        # between these, the transit dir exists (= phase 1 done for this
        # source) and the next dispatch will skip the pick.
        _atomic_write_transit(clone_dir, picked["tag"], picked["id"], transit_dir)

        # Move mode (default): the shipped variant is removed from its source
        # island. Copy mode: it stays, so a known-good variant remains home
        # while the neighbor still gets seeded. The source-side history event
        # differs by type. 'migration_out' marks "this variant left the pool"
        # — orchestrator._seed_was_eliminated reads it to suppress seed
        # re-insertion on resume (also guards against a phase-1-only crash
        # re-creating a shipped seed). 'migration_copy_out' must NOT be read
        # that way: a copied variant is still present.
        if state.config.migration_mode == "copy":
            event_type, verb = "migration_copy_out", "copied"
        else:
            delete_variant(conn, picked["id"])
            event_type, verb = "migration_out", "shipped"
        append_history(
            conn,
            {
                "type": event_type,
                "island": island_name,
                "parent": picked["id"],
                "child": picked["id"],
                "score": picked.get("score"),
                "cost": 0,
                "details": None,
                "timestamp": now_iso(),
            },
        )
        logger.info(
            "Migration phase1: %s %s %s via transit", island_name, verb, picked["id"]
        )

    def apply_phase2(self, state: LoopState) -> None:
        """Destination side: receive one migrant per source island.

        ring: the single predecessor. complete: every other island. Each
        present source transit is handled by _receive_from_source. Transit
        dirs are NOT deleted here — under a complete topology a source is read
        by multiple destinations running phase 2 in parallel, so the driver
        purges the transit root once after the phase-2 barrier (see
        Driver.run_op_on_all_islands).
        """
        island_name = state.current_island
        base_dir = str(state.base_dir)
        island_names = [i.name for i in state.config.islands]
        topology = getattr(state.config, "migration_topology", "ring")

        for source_island in sources_for(island_name, island_names, topology):
            transit_dir = (
                Path(base_dir)
                / ns(MIGRATE_TRANSIT_DIR, state.config.benchmark)
                / source_island
            )
            if not transit_dir.exists():
                logger.warning(
                    "Migration phase2: no transit dir from %s for %s — skipping",
                    source_island,
                    island_name,
                )
                continue
            self._receive_from_source(state, source_island)

    def _receive_from_source(self, state: LoopState, source_island: str) -> None:
        """Read source_island's transit, re-eval under this island's evaluator,
        and insert the migrant. Idempotent per (source, step)."""
        island_name = state.current_island
        island_cfg = state.current_config()
        clone_dir = str(state.clone_dir)
        base_dir = str(state.base_dir)
        conn = state.conn

        transit_dir = (
            Path(base_dir)
            / ns(MIGRATE_TRANSIT_DIR, state.config.benchmark)
            / source_island
        )

        with open(transit_dir / "metadata.json") as f:
            meta = json.load(f)
        source_id = meta["source_id"]

        # Resume idempotency: if a migrant variant from this source already
        # exists on this island at this step, this receive is already done.
        existing = conn.execute(
            "SELECT id FROM variants WHERE island = ? AND parent_id = ? AND step = ?",
            (island_name, source_id, state.step),
        ).fetchone()
        if existing:
            logger.info(
                "Migration phase2: migrant %s on %s already exists, skipping",
                existing["id"],
                island_name,
            )
            return

        # Overwrite this island's working tree with the migrant's code
        dst = os.path.join(clone_dir, "shadow_interface")
        for item in os.listdir(dst):
            if item == ".git":
                continue
            p = os.path.join(dst, item)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        src = os.path.join(str(transit_dir), "shadow_interface")
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)

        pop = get_population(conn, island_name)
        variant_num = get_next_variant_num({"population": pop}, clone_dir)
        new_id = f"{island_name}-if-{variant_num:03d}"
        new_tag = f"evol/{island_name}/if-{variant_num:03d}"
        git_tag_variant(
            clone_dir,
            new_tag,
            f"{new_tag}: migrated from {source_island}/{source_id}",
        )

        # Re-eval under destination's evaluator
        train_set_file = state.config.train_set_file
        train_path = os.path.join(base_dir, EVAL_SETS_DIR, train_set_file)
        with open(train_path) as f:
            train_tasks = json.load(f)
        val_tasks = _load_val_tasks(base_dir, state.config.val_set_file)
        all_tasks = train_tasks + val_tasks

        eval_model = state.config.resolve_eval_model(island_cfg)
        evaluator_key = state.config.resolve_evaluator_harness_key(island_cfg)
        evaluator = resolve_evaluator_harness(evaluator_key, state.config.benchmark)
        result_dir = os.path.join(
            base_dir,
            ns(EVAL_CACHE_DIR, state.config.benchmark),
            new_id,
        )
        eval_out = evaluator.run_tasks(
            clone_dir=state.clone_dir,
            shadow_dir=state.clone_dir / "shadow_interface",
            result_dir=result_dir,
            task_ids=all_tasks,
            model=eval_model,
            region=state.config.region,
            timeout=state.config.eval_timeout,
            max_steps=state.config.eval_max_steps,
            no_time_limit=state.config.eval_no_time_limit,
            harness=evaluator_key,
            benchmark=state.config.benchmark,
        )
        if eval_out.timed_out:
            raise RuntimeError(
                f"Migration eval timed out for {new_id}; refusing partial zero fitness"
            )

        insert_variant(
            conn,
            {
                "id": new_id,
                "island": island_name,
                "tag": new_tag,
                "parent_id": source_id,
                "parent_b_id": None,
                "origin": "migration",
                "generation": 0,
                "score": None,
                "eval_level": 0,
                "step": state.step,
                "builder_cost": 0.0,
                "created_at": now_iso(),
            },
        )
        per_task = _collect_per_task_records(
            clone_dir,
            new_id,
            all_tasks,
            eval_model,
            benchmark=state.config.benchmark,
        )
        now = now_iso()
        for tid, record in per_task.items():
            insert_eval_result(
                conn,
                new_id,
                tid,
                record["passed"],
                now,
                raw_score=record["raw_score"],
                fitness_score=record["fitness_score"],
                details_json=record["details_json"],
                failure_type=record["failure_type"],
            )
        record_eval_costs(
            conn,
            variant_id=new_id,
            island=island_name,
            clone_dir=clone_dir,
            task_ids=all_tasks,
            event_type="migration_eval_task",
            model_fallback=eval_model,
            timestamp=now,
            benchmark=state.config.benchmark,
        )

        score = recompute_score_from_train_set(conn, new_id, train_tasks)
        val_score = recompute_val_score(conn, new_id)
        update_variant(conn, new_id, score=score, val_score=val_score)

        append_history(
            conn,
            {
                "type": "migration",
                "island": island_name,
                "parent": source_id,
                "child": new_id,
                "score": score,
                "cost": 0,
                "details": json.dumps({"source_island": source_island}),
                "timestamp": now_iso(),
            },
        )
        logger.info(
            "Migration phase2: %s received %s from %s (score=%.3f, val=%.3f)",
            island_name,
            new_id,
            source_island,
            score,
            val_score,
        )

    def apply(self, state: LoopState) -> None:
        """Single-island apply is unused under the v2 driver — the driver
        calls apply_phase1 and apply_phase2 directly with a barrier between.
        Kept here so the Operator interface remains uniform.
        """
        raise NotImplementedError(
            "Migration in v2 is driven by apply_phase1 + apply_phase2; "
            "the global driver calls these directly."
        )
