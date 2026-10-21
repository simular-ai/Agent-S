"""Command-line entry for shadow-evol.

Shape:
    shadow-evol run --island name=X,evaluator_harness=Y,eval_model=Z \
                    [--island ...] \
                    [--scheduler deterministic[:m,c,e] | probabilistic:k=w,...] \
                    [--seed-dir DIR --prompt-template FILE] \
                    [--builder-model X --builder-harness Y ...]

    shadow-evol rerun --config-path runs/<ts>/config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_s4.benchmark_spec import BENCHMARKS
from agent_s4.config.experiment import ExperimentConfig
from agent_s4.config.island import IslandConfig
from agent_s4.config.scheduler import (
    DeterministicScheduler,
    ProbabilisticScheduler,
    Scheduler,
)


def _parse_kv_list(arg: str) -> dict[str, str]:
    """Parse 'k1=v1,k2=v2' into {'k1': 'v1', 'k2': 'v2'}."""
    out: dict[str, str] = {}
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                f"expected k=v pairs separated by commas, got {part!r}"
            )
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _parse_island_arg(arg: str) -> dict[str, str]:
    """Parse + validate an --island argument at argparse time so missing
    keys fail with a clean usage message.

    v2 required keys: name, evaluator_harness, eval_model. Seed, prompt,
    builder model/harness are global on ExperimentConfig.
    """
    kv = _parse_kv_list(arg)
    required = {"name", "evaluator_harness", "eval_model"}
    missing = required - kv.keys()
    if missing:
        raise argparse.ArgumentTypeError(
            f"--island missing keys {sorted(missing)}; got {kv!r}"
        )
    return kv


def _parse_scheduler(spec: str) -> Scheduler:
    """Parse --scheduler argument.

    Accepted forms:
        "deterministic"
            → DeterministicScheduler() with defaults (5, 6, 10).
        "deterministic:M,C,E"
            → DeterministicScheduler(migration_cadence=M,
                                     crossover_cadence=C,
                                     elimination_cadence=E).
        "probabilistic:mutation=W,migration=W,crossover=W,elimination=W"
            → ProbabilisticScheduler({...}).
    """
    name, _, rest = spec.partition(":")
    name = name.strip()
    if name == "deterministic":
        if not rest:
            return DeterministicScheduler()
        try:
            cadences = [int(x.strip()) for x in rest.split(",") if x.strip()]
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"--scheduler deterministic cadences must be ints, got {rest!r}"
            ) from e
        if len(cadences) != 3:
            raise argparse.ArgumentTypeError(
                f"--scheduler deterministic expects 3 cadences "
                f"(migration,crossover,elimination), got {cadences!r}"
            )
        return DeterministicScheduler(
            migration_cadence=cadences[0],
            crossover_cadence=cadences[1],
            elimination_cadence=cadences[2],
        )
    if name == "probabilistic":
        if not rest:
            raise argparse.ArgumentTypeError(
                "--scheduler probabilistic requires kind=weight pairs"
            )
        kv = _parse_kv_list(rest)
        try:
            weights = {k: float(v) for k, v in kv.items()}
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"--scheduler probabilistic weights must be floats, got {kv!r}"
            ) from e
        return ProbabilisticScheduler(weights)
    raise argparse.ArgumentTypeError(
        f"unknown scheduler {spec!r}; expected 'deterministic[:M,C,E]' "
        f"or 'probabilistic:k=w,...'"
    )


def _build_island(kv: dict[str, str]) -> IslandConfig:
    required = {"name", "evaluator_harness", "eval_model"}
    missing = required - kv.keys()
    if missing:
        raise argparse.ArgumentTypeError(
            f"--island missing keys {sorted(missing)}; got {kv!r}"
        )
    return IslandConfig(
        name=kv["name"],
        evaluator_harness=kv["evaluator_harness"],
        eval_model=kv["eval_model"],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="shadow-evol")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Launch an experiment")
    run.add_argument(
        "--island",
        action="append",
        type=_parse_island_arg,
        required=True,
        help="Repeatable. Example: "
        "--island name=main,evaluator_harness=prime_agent,eval_model=openai/<deployment>",
    )
    run.add_argument(
        "--scheduler",
        type=_parse_scheduler,
        default=None,
        help="Scheduler spec. Examples: 'deterministic', "
        "'deterministic:5,6,10', "
        "'probabilistic:mutation=0.7,migration=0.15,"
        "crossover=0.1,elimination=0.05'. "
        "Default: DeterministicScheduler(5, 6, 10).",
    )
    run.add_argument(
        "--seed-dir", type=Path, help="Seed module directory shared by all islands."
    )
    run.add_argument(
        "--prompt-template", type=Path, help="Builder prompt template path."
    )
    run.add_argument("--builder-model")
    run.add_argument("--builder-harness")
    run.add_argument("--builder-timeout", type=int)
    run.add_argument(
        "--builder-no-time-limit",
        action="store_true",
        default=None,
        help="prime_agent Builder: disable the Builder wall-clock deadline.",
    )
    run.add_argument("--eval-timeout", type=int)
    run.add_argument(
        "--eval-max-steps",
        type=int,
        help="Maximum agent steps per task.",
    )
    run.add_argument(
        "--eval-no-time-limit",
        action="store_true",
        default=None,
        help="Disable both the per-task wall-clock "
        "limit and the outer evolve batch watchdog.",
    )
    run.add_argument("--max-steps", type=int)
    run.add_argument("--max-cost", type=float)
    run.add_argument("--region")
    run.add_argument("--seed", type=int)
    run.add_argument(
        "--elimination-count",
        type=int,
        help="How many bottom variants Elimination removes per fire.",
    )
    run.add_argument(
        "--reevaluation-interval",
        type=int,
        help="Reevaluate the top interfaces on validation after every N completed "
        "evolution steps (0 disables; default: 0). Does not consume steps.",
    )
    run.add_argument(
        "--reevaluation-top-k",
        type=int,
        help="Number of interfaces per island to reevaluate in parallel (default: 3).",
    )
    run.add_argument(
        "--no-scratch-vm",
        dest="scratch_vm_enabled",
        action="store_false",
        default=None,
        help="Ablation: run the builder without a per-mutation "
        "AWS scratch VM. Builder learns only from "
        "eval_results/ trajectories (no live app probing).",
    )
    run.add_argument(
        "--train-set",
        dest="train_set_file",
        help="Filename under eval_sets/ for the training task list. "
        "Default for osworld_v2: osworld_v2_train_50.json; required for "
        "the other benchmarks.",
    )
    run.add_argument(
        "--val-set",
        dest="val_set_file",
        help="Filename under eval_sets/ for the validation task list. "
        "Default for osworld_v2: osworld_v2_val_25.json; required for "
        "the other benchmarks. Pass 'none' to disable val "
        "entirely (requires --selection-signal train).",
    )
    run.add_argument(
        "--selection-signal",
        choices=("val", "train"),
        help="Which score drives tournament selection. "
        "Default: val. Set to 'train' for ablation runs that "
        "still record val_score but rank parents by train.",
    )
    run.add_argument(
        "--parent-selection",
        choices=("tournament", "latest"),
        help="How mutation picks its parent. Default: tournament "
        "(binary tournament over the scored population, "
        "ranked by --selection-signal). 'latest' always "
        "mutates the most recent scored variant, giving a "
        "strict linear chain of depth --max-steps and taking "
        "the selection signal out of the search.",
    )
    run.add_argument(
        "--migration-mode",
        choices=("move", "copy"),
        help="Migration semantics. Default: move (ship the picked "
        "variant off its source island). 'copy' keeps the "
        "variant on its source and only seeds the neighbor.",
    )
    run.add_argument(
        "--migration-topology",
        choices=("ring", "complete"),
        help="Migration topology. Default: ring (receive from the "
        "single predecessor). 'complete' is all-to-all — each "
        "island receives from every other island (= bidirectional "
        "ring at N=3).",
    )
    run.add_argument(
        "--benchmark",
        choices=sorted(BENCHMARKS),
        default="osworld_v2",
        help="Benchmark suite to drive (default: osworld_v2).",
    )
    run.add_argument(
        "--allow-v2-noncanonical-splits",
        action="store_true",
        default=None,
        help="Unsafe diagnostic escape hatch: permit non-canonical OSWorld V2 "
        "train/val subsets. Test IDs remain forbidden.",
    )

    rerun = sub.add_parser("rerun", help="Re-execute from a saved config.json")
    rerun.add_argument("--config-path", type=Path, required=True)

    return p.parse_args(argv)


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if args.command == "rerun":
        with open(args.config_path) as f:
            payload = json.load(f)
        return _config_from_dict(payload)

    islands = [_build_island(kv) for kv in args.island]

    kwargs: dict = {"islands": islands}
    for field in (
        "builder_model",
        "builder_harness",
        "builder_claude_bin",
        "builder_timeout",
        "builder_no_time_limit",
        "eval_timeout",
        "eval_max_steps",
        "eval_no_time_limit",
        "max_steps",
        "max_cost",
        "region",
        "seed",
        "scratch_vm_enabled",
        "train_set_file",
        "val_set_file",
        "selection_signal",
        "parent_selection",
        "migration_mode",
        "migration_topology",
        "elimination_count",
        "benchmark",
        "reevaluation_interval",
        "reevaluation_top_k",
        "allow_v2_noncanonical_splits",
    ):
        v = getattr(args, field, None)
        if v is not None:
            kwargs[field] = v
    for path_field in ("seed_dir", "prompt_template"):
        v = getattr(args, path_field, None)
        if v is not None:
            kwargs[path_field] = v
    if args.scheduler is not None:
        kwargs["scheduler"] = args.scheduler
    return ExperimentConfig(**kwargs)


def _scheduler_from_payload(payload: dict) -> Scheduler:
    """Reconstruct a Scheduler from its serialized form.

    Serialization shape (see agent_s4.orchestrator._config_to_dict):
        {"type": "DeterministicScheduler" | "ProbabilisticScheduler",
         "args": {...}}
    """
    sched_type = payload["type"]
    args = payload["args"]
    if sched_type == "DeterministicScheduler":
        return DeterministicScheduler(**args)
    if sched_type == "ProbabilisticScheduler":
        # ProbabilisticScheduler.__init__ takes a weights dict, but stores
        # kinds + probs. Reconstruct weights from those.
        weights = dict(zip(args["kinds"], args["probs"]))
        return ProbabilisticScheduler(weights)
    raise ValueError(f"unknown scheduler type {sched_type!r}")


def _config_from_dict(payload: dict) -> ExperimentConfig:
    """Round-trip deserialization for `shadow-evol rerun`. Must agree with
    agent_s4.orchestrator._config_to_dict.
    """
    islands = [
        IslandConfig(
            name=i["name"],
            evaluator_harness=i["evaluator_harness"],
            eval_model=i["eval_model"],
        )
        for i in payload["islands"]
    ]

    scalar_field_names = {
        "builder_model",
        "builder_harness",
        "builder_claude_bin",
        "builder_timeout",
        "builder_no_time_limit",
        "eval_timeout",
        "eval_max_steps",
        "eval_no_time_limit",
        "region",
        "scratch_vm_enabled",
        "max_steps",
        "max_cost",
        "seed",
        "train_set_file",
        "val_set_file",
        "selection_signal",
        "parent_selection",
        "migration_mode",
        "migration_topology",
        "elimination_count",
        "benchmark",
        "reevaluation_interval",
        "reevaluation_top_k",
        "allow_v2_noncanonical_splits",
    }
    kwargs: dict = {"islands": islands}
    for k in scalar_field_names:
        if k in payload and payload[k] is not None:
            kwargs[k] = payload[k]
    if "seed_dir" in payload and payload["seed_dir"] is not None:
        kwargs["seed_dir"] = Path(payload["seed_dir"])
    if "prompt_template" in payload and payload["prompt_template"] is not None:
        kwargs["prompt_template"] = Path(payload["prompt_template"])
    if "held_out_tasks_path" in payload and payload["held_out_tasks_path"] is not None:
        kwargs["held_out_tasks_path"] = Path(payload["held_out_tasks_path"])
    if "scheduler" in payload and payload["scheduler"] is not None:
        kwargs["scheduler"] = _scheduler_from_payload(payload["scheduler"])
    return ExperimentConfig(**kwargs)


def main() -> None:
    import logging

    # Surface INFO-level progress from orchestrator/runner/operators/harnesses.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    from agent_s4.paths import require_shared_root

    require_shared_root()
    config = build_config_from_args(args)
    from agent_s4.orchestrator import run

    run(config)


if __name__ == "__main__":
    main()
