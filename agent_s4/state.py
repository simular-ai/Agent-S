"""LoopState — the per-step context every Operator.apply() receives."""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from agent_s4.config.island import IslandConfig

if TYPE_CHECKING:
    from agent_s4.config.experiment import ExperimentConfig


@dataclass
class LoopState:
    conn: sqlite3.Connection
    current_island: str
    step: int
    rng: random.Random
    config: "ExperimentConfig"
    clone_dir: Path
    base_dir: Path
    eval_cache_dir: Path

    def current_config(self) -> IslandConfig:
        for i in self.config.islands:
            if i.name == self.current_island:
                return i
        raise KeyError(f"current_island {self.current_island!r} not in config.islands")

    def clone_dir_for(self, island: IslandConfig) -> Path:
        from agent_s4.benchmark_spec import BENCHMARKS

        template = BENCHMARKS[self.config.benchmark]["template"]
        return self.base_dir / f"{template}_{island.name}"
