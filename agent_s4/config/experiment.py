"""ExperimentConfig — the single object the CLI produces and driver consumes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from agent_s4.benchmark_spec import BENCHMARKS
from agent_s4.config.island import IslandConfig
from agent_s4.config.scheduler import DeterministicScheduler, Scheduler

# (train, val, held-out) defaults per benchmark. Only the files that ship in
# eval_sets/ are named; the others must be passed explicitly.
_DEFAULT_SPLITS: dict[str, tuple[str | None, str | None, str | None]] = {
    "osworld_v2": (
        "osworld_v2_train_50.json",
        "osworld_v2_val_25.json",
        "eval_sets/osworld_v2_test.json",
    ),
    "scienceboard": (None, None, "eval_sets/scienceboard_test.json"),
    "osworld": (None, None, None),
}


@dataclass
class ExperimentConfig:
    islands: list[IslandConfig]

    # Shared across all islands in v2
    seed_dir: Path = Path("seeds/combined")
    prompt_template: Path = Path("prompts/combined.md")
    builder_model: str = "opus"
    builder_harness: str = "prime_agent"
    # Kept only so run configs written by older versions still round-trip
    # through `shadow-evol rerun`; the Prime builder ignores it.
    builder_claude_bin: str | None = None

    # Scheduler controls when each op fires
    scheduler: Scheduler = field(default_factory=DeterministicScheduler)
    elimination_count: int = 5
    # Validation maintenance runs after completed evolution steps. It does
    # not advance the scheduler, allocate an interface ID, or use a step.
    # Zero keeps existing experiments unchanged; opt in with an interval.
    reevaluation_interval: int = 0
    reevaluation_top_k: int = 3

    builder_timeout: int = 600
    # Prime Builder can run without a wall-clock deadline. Kept separate from
    # builder_timeout so saved configs state the policy explicitly.
    builder_no_time_limit: bool = False
    eval_timeout: int = 600
    # Per-episode agent budget. None preserves the evaluator's harness default.
    # no_time_limit disables both the in-episode wall clock and the evolve
    # subprocess's outer batch watchdog. Infrastructure setup/upload calls
    # retain their own bounded timeouts.
    eval_max_steps: int | None = None
    eval_no_time_limit: bool = False
    region: str = "us-east-1"

    scratch_vm_enabled: bool = True

    max_steps: int | None = None
    max_cost: float | None = None
    seed: int | None = None

    # Canonical task sets — no tiers
    # None is an internal sentinel so benchmark-aware defaults do not erase a
    # caller's explicit choice of the legacy filenames for a V2 diagnostic.
    train_set_file: str | None = None
    val_set_file: str | None = None
    held_out_tasks_path: Path | None = None

    selection_signal: Literal["val", "train"] = "val"

    # How mutation picks its parent.
    #
    # "tournament" (default): a binary tournament over the island's scored
    # population, ranked by selection_signal. This is a branching search --
    # with a population of 5 and k=2 the current best is not drawn 60% of the
    # time, so the lineage that comes out is a shallow tree rooted at the
    # seed, not a chain. The opus5xover25v15 run made the cost concrete:
    # seed was a parent in 7 of 10 steps and the deepest lineage was 3.
    #
    # "latest": always mutate the most recently produced scored variant, so
    # step N builds on step N-1's output whatever it scored. The lineage is a
    # strict chain of depth max_steps, which is what a linear-route baseline
    # needs; the trade is that one bad mutation carries forward instead of
    # being stepped around. Deliberate -- it removes the selection signal
    # from the search entirely, so the run measures what accumulating
    # mutations does without a val_score too noisy to rank on getting a vote.
    parent_selection: Literal["tournament", "latest"] = "tournament"

    # Migration semantics. "move" (default): phase 1 removes the picked
    # variant from its source island (today's behavior). "copy": phase 1
    # seeds the ring successor without removing the variant from its source,
    # so a known-good variant stays home. See migration.apply_phase1.
    migration_mode: Literal["move", "copy"] = "move"

    # Migration topology. "ring" (default): each island receives from its
    # single predecessor (today's behavior). "complete": all-to-all — each
    # island receives a migrant from EVERY other island, so its intra-island
    # crossover pool holds material from all islands at once. At N=3 this is
    # the bidirectional ring. See migration.sources_for / apply_phase2.
    migration_topology: Literal["ring", "complete"] = "ring"

    # Which benchmark suite the loop drives. Persisted in config.json so rerun
    # is benchmark-aware. See agent_s4/benchmark_spec.py.
    benchmark: Literal["osworld", "osworld_v2", "scienceboard"] = "osworld_v2"

    # Diagnostic escape hatch for deliberately smaller V2 train/val subsets.
    # Even when enabled, the subsets must be unique, disjoint, and contain no
    # canonical test IDs. Formal runs leave this false.
    allow_v2_noncanonical_splits: bool = False

    def __post_init__(self):
        if (
            type(self.reevaluation_interval) is not int
            or self.reevaluation_interval < 0
        ):
            raise ValueError("reevaluation_interval must be an integer >= 0")
        if type(self.reevaluation_top_k) is not int or self.reevaluation_top_k < 1:
            raise ValueError("reevaluation_top_k must be a positive integer")
        if self.reevaluation_interval and self.val_set_file == "none":
            raise ValueError("validation reevaluation requires a validation set")
        if self.builder_no_time_limit and self.builder_harness not in {
            "prime_agent",
            "opencode",
        }:
            raise ValueError(
                "builder_no_time_limit currently requires builder_harness="
                "'prime_agent' or 'opencode'"
            )
        if self.selection_signal not in ("val", "train"):
            raise ValueError(
                f"selection_signal must be 'val' or 'train', got "
                f"{self.selection_signal!r}"
            )
        if self.parent_selection not in ("tournament", "latest"):
            raise ValueError(
                f"parent_selection must be 'tournament' or 'latest', got "
                f"{self.parent_selection!r}"
            )
        if self.migration_mode not in ("move", "copy"):
            raise ValueError(
                f"migration_mode must be 'move' or 'copy', got "
                f"{self.migration_mode!r}"
            )
        if self.migration_topology not in ("ring", "complete"):
            raise ValueError(
                f"migration_topology must be 'ring' or 'complete', got "
                f"{self.migration_topology!r}"
            )
        if self.benchmark not in BENCHMARKS:
            raise ValueError(
                f"benchmark must be one of {sorted(BENCHMARKS)}, got "
                f"{self.benchmark!r}"
            )
        train_default, val_default, held_out_default = _DEFAULT_SPLITS[self.benchmark]
        if self.train_set_file is None:
            if train_default is None:
                raise ValueError(
                    f"benchmark {self.benchmark!r} ships no default train split; "
                    "pass train_set_file (--train-set)"
                )
            self.train_set_file = train_default
        if self.val_set_file is None:
            if val_default is None:
                raise ValueError(
                    f"benchmark {self.benchmark!r} ships no default val split; "
                    "pass val_set_file (--val-set), or 'none'"
                )
            self.val_set_file = val_default
        if self.held_out_tasks_path is None and held_out_default is not None:
            self.held_out_tasks_path = Path(held_out_default)
        if self.val_set_file == "none" and self.selection_signal == "val":
            raise ValueError("val_set_file=none requires selection_signal=train")

    def island_by_name(self, name: str) -> IslandConfig:
        for i in self.islands:
            if i.name == name:
                return i
        raise KeyError(
            f"no island named {name!r}; " f"known: {[i.name for i in self.islands]}"
        )

    def resolve_builder_model(self, island: IslandConfig) -> str:
        return self.builder_model  # uniform in v2

    def resolve_builder_bin(self, island: IslandConfig) -> str | None:
        return self.builder_claude_bin  # uniform in v2; unused by the Prime builder

    def resolve_eval_model(self, island: IslandConfig) -> str:
        return island.eval_model

    def resolve_builder_harness_key(self, island: IslandConfig) -> str:
        return self.builder_harness  # uniform in v2

    def resolve_evaluator_harness_key(self, island: IslandConfig) -> str:
        return island.evaluator_harness
