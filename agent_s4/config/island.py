"""Per-island configuration. Pure data — no runtime logic.

In v2, the only per-island variation is the evaluator. Builder, seed,
and prompt are shared globally via ExperimentConfig.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IslandConfig:
    name: str
    evaluator_harness: str
    eval_model: str
