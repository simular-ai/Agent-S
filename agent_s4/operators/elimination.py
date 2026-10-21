"""Elimination operator.

Ports run_island.check_elimination. Culls the bottom-N variants from this
island's population via tournament(k=all, t=keep). The surviving variants
are the keep highest-scoring per tournament's selection; eliminated
variants are deleted and a history row is appended for each.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import ClassVar

from agent_s4.db import (
    append_history,
    delete_variant,
    get_population,
    update_island,
)
from agent_s4.operators.base import Operator
from agent_s4.state import LoopState
from agent_s4.tournament import tournament

from agent_s4.utils import now_iso

logger = logging.getLogger("agent_s4.operators.elimination")

BACKUP_DIR = "runs/db_backups"


def backup_before_cull(conn, base_dir, step: int) -> Path:
    """Snapshot the database before an elimination deletes from it.

    ``delete_variant`` cascades to ``eval_results``, so a cull discards the
    per-task record of every variant it drops -- the rows every held-out
    comparison and paper table is computed from. Taking the snapshot here
    rather than only when a run is launched is what covers variants created
    after that launch: at step 10 of a resumed run, four of the ten
    candidates were built during the run itself.

    A failure to write the backup raises, because deleting without one is the
    outcome this exists to prevent.
    """
    row = conn.execute("PRAGMA database_list").fetchone()
    source = row[2] if row and row[2] else None
    if not source:
        raise RuntimeError(
            "Refusing to eliminate: could not resolve the database path to "
            "back up before deleting variants"
        )
    source = Path(source)
    target_dir = Path(base_dir) / BACKUP_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{source.stem}.pre_elimination_step{step}{source.suffix}"
    # The connection is in autocommit mode, so the snapshot is a consistent
    # view of everything recorded up to this point.
    dest = sqlite3.connect(str(target))
    try:
        conn.backup(dest)
    finally:
        dest.close()
    logger.info("Backed up %s -> %s before elimination", source.name, target)
    return target


class Elimination(Operator):
    kind: ClassVar[str] = "elimination"

    def __init__(self, count: int):
        if count < 0:
            raise ValueError(f"elimination count must be >= 0, got {count}")
        self.count = count

    def apply(self, state: LoopState) -> None:
        island_name = state.current_island
        island_cfg = state.current_config()
        conn = state.conn

        pop = get_population(conn, island_name)
        # Eligibility filter mirrors the configured selection signal (same
        # semantic as Mutation.apply): a variant must have a ranked-side
        # score populated to enter the elimination pool. Both score and
        # val_score are set atomically by the same eval call so in steady
        # state these filters select the same set.
        if state.config.selection_signal == "val":
            scored = [v for v in pop if v["val_score"] is not None]
        else:
            scored = [v for v in pop if v["score"] is not None]
        if self.count >= len(scored):
            logger.warning(
                "Elimination skipped: count=%d >= scored_pop=%d",
                self.count,
                len(scored),
            )
            update_island(
                conn,
                island_name,
                last_elimination_at=state.step,
                last_updated=now_iso(),
            )
            return

        keep = len(scored) - self.count
        logger.info("Elimination triggered: keep top %d of %d", keep, len(scored))
        eval_model = state.config.resolve_eval_model(island_cfg)
        survivors = tournament(
            conn,
            scored,
            k=len(scored),
            t=keep,
            clone_dir=str(state.clone_dir),
            eval_model=eval_model,
            eval_timeout=state.config.eval_timeout,
            region=state.config.region,
            selection_signal=state.config.selection_signal,
            benchmark=state.config.benchmark,
        )
        survivor_ids = {v["id"] for v in survivors}
        backup_before_cull(conn, state.base_dir, state.step)
        dropped = []
        for v in scored:
            if v["id"] in survivor_ids:
                continue
            delete_variant(conn, v["id"])
            dropped.append(v["id"])
            append_history(
                conn,
                {
                    "type": "elimination",
                    "island": island_name,
                    "parent": None,
                    "child": v["id"],
                    "score": v["score"],
                    "cost": 0,
                    "details": None,
                    "timestamp": now_iso(),
                },
            )

        update_island(
            conn,
            island_name,
            last_elimination_at=state.step,
            last_updated=now_iso(),
        )
        logger.info("Eliminated %d variants: %s", len(dropped), dropped)
