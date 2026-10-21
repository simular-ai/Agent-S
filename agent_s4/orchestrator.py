"""Orchestrator: bootstrap clones, seed DB, deploy seeds, run initial seed
eval (parallel per-island), dump resolved config, then hand off to Driver.

Entry point:
    run(config: ExperimentConfig)            — programmatic
The CLI's `shadow-evol run ...` parses argv, builds an ExperimentConfig,
and calls run().
"""

from __future__ import annotations

import concurrent.futures
import json
import hashlib
import logging
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from agent_s4.benchmark_spec import BENCHMARKS, ns
from agent_s4.config.experiment import ExperimentConfig
from agent_s4.config.island import IslandConfig
from agent_s4.constants import DB_FILENAME, EVAL_CACHE_DIR, EVAL_SETS_DIR
from agent_s4.db import (
    connect as db_connect,
    get_variant,
    init_island,
    init_schema,
    insert_eval_result,
    insert_variant,
    load_val_set,
    recompute_score_from_train_set,
    recompute_val_score,
    update_variant,
    verify_val_set_unchanged,
)
from agent_s4.paths import PROJECT_ROOT, prompt_path
from agent_s4.harness import resolve_evaluator_harness
from agent_s4.tournament import _collect_per_task_records
from agent_s4.utils import _load_val_tasks

logger = logging.getLogger("agent_s4.orchestrator")

# Prompts directory (sibling of agent_s4/). Addenda here get appended
# to the base prompt at initialize_clone time when their config flag is on:
# a capability gated by a config flag is an addendum, not a parallel prompt
# file.
_PROJECT_ROOT = PROJECT_ROOT
_V2_PIN_PATH = _PROJECT_ROOT / "config" / "osworld_v2_release_pin.json"


def _load_unique_task_ids(path: Path, label: str) -> list[str]:
    try:
        values = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unreadable/invalid: {path}: {exc}") from exc
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, str) and value for value in values)
        or len(values) != len(set(values))
    ):
        raise RuntimeError(
            f"{label} must be a non-empty unique JSON string list: {path}"
        )
    return values


def _split_digest(task_ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(task_ids)) + "\n").encode()).hexdigest()


def validate_v2_run_splits(config: ExperimentConfig, base_dir: Path) -> None:
    """Bind the actual evolution config to the parent-pinned V2 split."""
    if config.benchmark != "osworld_v2":
        return
    try:
        pin = json.loads(_V2_PIN_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"OSWorld V2 parent release pin is invalid: {exc}") from exc

    canonical_names = {
        "train": "osworld_v2_train_50.json",
        "val": "osworld_v2_val_25.json",
        "test": "osworld_v2_test.json",
    }
    canonical = {}
    for label, name in canonical_names.items():
        path = base_dir / EVAL_SETS_DIR / name
        values = _load_unique_task_ids(path, f"canonical V2 {label} split")
        expected = pin.get("split_sha256", {}).get(name)
        if not expected or _split_digest(values) != expected:
            raise RuntimeError(
                f"canonical V2 {label} split differs from parent release pin: {path}"
            )
        canonical[label] = set(values)

    train_path = base_dir / EVAL_SETS_DIR / config.train_set_file
    train = set(_load_unique_task_ids(train_path, "configured V2 train split"))
    if config.val_set_file == "none":
        val = set()
    else:
        val_path = base_dir / EVAL_SETS_DIR / config.val_set_file
        val = set(_load_unique_task_ids(val_path, "configured V2 val split"))

    heldout_path = config.held_out_tasks_path
    if not heldout_path.is_absolute():
        heldout_path = base_dir / heldout_path
    heldout = set(_load_unique_task_ids(heldout_path, "configured V2 held-out split"))
    if heldout != canonical["test"]:
        raise RuntimeError(
            "configured V2 held-out split must exactly match the parent-pinned test set"
        )
    if train & val:
        raise RuntimeError("configured V2 train/val splits overlap")
    leaked = (train | val) & canonical["test"]
    if leaked:
        raise RuntimeError(
            f"configured V2 train/val splits contain held-out test IDs: {sorted(leaked)}"
        )
    if not config.allow_v2_noncanonical_splits:
        if train != canonical["train"] or val != canonical["val"]:
            raise RuntimeError(
                "formal OSWorld V2 runs require the parent-pinned 50/25 train/val "
                "splits; use --allow-v2-noncanonical-splits only for diagnostics"
            )


def validate_prime_agent_before_run(config: ExperimentConfig) -> None:
    """Fail before provisioning when Prime Agent's offline runtime is absent."""
    if os.environ.get("SHADOW_EVOL_DRYRUN") == "1":
        return
    uses_builder = config.builder_harness == "prime_agent"
    uses_evaluator = any(
        island.evaluator_harness == "prime_agent" for island in config.islands
    )
    if not (uses_builder or uses_evaluator):
        return
    from agent_s4.harness.prime.source import (
        validate_prime_agent_source,
        validate_prime_agent_runtime_cache,
    )

    validate_prime_agent_source()
    if uses_builder:
        from agent_s4.harness.prime.builder import resolve_prime_agent_binary

        resolve_prime_agent_binary()
    if uses_evaluator:
        from agent_s4.harness.prime.evaluator import _resolve_runtime_cache

        validate_prime_agent_runtime_cache(_resolve_runtime_cache())


def validate_v2_workspace_before_run(config: ExperimentConfig, base_dir: Path) -> None:
    """Apply the shared local V2 gate before DB writes or provider calls."""
    if config.benchmark != "osworld_v2":
        return
    from agent_s4.osworld_v2_preflight import validate_default_workspace

    required_harnesses = tuple(
        sorted(
            {
                config.builder_harness,
                *(island.evaluator_harness for island in config.islands),
            }
        )
    )
    # The proxy check covers only the tasks the driver runs (train + val), as
    # the launchers' own preflight does. An unreadable split falls back to the
    # conservative every-task check; the split validation reports it.
    split_files = [config.train_set_file]
    if config.val_set_file != "none":
        split_files.append(config.val_set_file)
    try:
        selected_task_ids = set().union(
            *(
                _load_unique_task_ids(base_dir / EVAL_SETS_DIR / name, name)
                for name in split_files
            )
        )
    except RuntimeError:
        selected_task_ids = None
    errors = validate_default_workspace(
        base_dir,
        required_harnesses=required_harnesses,
        selected_task_ids=selected_task_ids,
    )
    if errors:
        raise RuntimeError(
            "OSWorld V2 local preflight failed before experiment start:\n  - "
            + "\n  - ".join(errors)
        )


def _git_value(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(
            f"Could not resolve git {' '.join(args)} for template {repo}: "
            f"{proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def _template_provenance(
    template_dir: Path, config: ExperimentConfig
) -> dict[str, str]:
    """Return the source identity an island clone must preserve on resume."""
    provenance = {
        "benchmark": config.benchmark,
        "source_template_commit": _git_value(template_dir, "rev-parse", "HEAD"),
        "source_template_tag": _git_value(
            template_dir,
            "describe",
            "--tags",
            "--exact-match",
        ),
    }
    if config.benchmark == "osworld_v2":
        try:
            pin_bytes = _V2_PIN_PATH.read_bytes()
            pin = json.loads(pin_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"OSWorld V2 parent release pin is invalid: {exc}"
            ) from exc
        if provenance["source_template_commit"] != pin.get("code_commit"):
            raise RuntimeError(
                "OSWorld V2 template commit does not match parent release pin: "
                f"{provenance['source_template_commit']} != {pin.get('code_commit')}"
            )
        if provenance["source_template_tag"] != pin.get("code_tag"):
            raise RuntimeError(
                "OSWorld V2 template tag does not match parent release pin: "
                f"{provenance['source_template_tag']} != {pin.get('code_tag')}"
            )
        provenance.update(
            release=str(pin.get("release")),
            release_pin_sha256=hashlib.sha256(pin_bytes).hexdigest(),
        )
    return provenance


# A prompt template carrying this marker documents ungraded synthetic training
# tasks itself; the graded-task ScienceBoard eval-format addendum would then
# describe files the Builder never receives.
SB_UNGRADED_TRAIN_MARKER = "<!-- scienceboard: ungraded synthetic train -->"


def _read_init_marker(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def _assert_no_v2_gated_tasks(checkout: Path) -> None:
    """Keep evaluator task implementations out of every Builder-visible tree."""
    task_root = checkout / "evaluation_examples" / "task_class"
    leaked = sorted(task_root.glob("task_*.py")) if task_root.exists() else []
    if leaked:
        raise RuntimeError(
            f"OSWorld V2 Builder-visible checkout {checkout} contains "
            f"{len(leaked)} gated task definition(s); refusing to expose "
            "evaluator logic to the Builder"
        )


def _assert_no_scienceboard_task_tree(checkout: Path) -> None:
    """Keep official task JSONs (evaluate blocks, answers) out of the clone.

    ScienceBoard's template carries tasks/VM with every official task's
    ``evaluate`` block and ``ans`` field, validation and test included. The
    eval subprocess reads tasks from ScienceBoard_Base, never from the clone,
    so the Builder has no use for them and must not be able to read them --
    neither in the working tree nor in the template's git history.
    """
    if (checkout / "tasks").exists():
        raise RuntimeError(
            f"ScienceBoard Builder-visible checkout {checkout} contains the "
            "official task tree; refusing to expose evaluators and answers"
        )


def _scienceboard_history_leaks(clone_dir: Path) -> bool:
    """True when the clone's git history can still reach a tasks/ tree."""
    proc = subprocess.run(
        ["git", "log", "--all", "--format=%H", "--", "tasks"],
        cwd=str(clone_dir),
        capture_output=True,
        text=True,
    )
    return proc.returncode != 0 or bool(proc.stdout.strip())


def assemble_prompt_v2(config: ExperimentConfig) -> str:
    """Build the CLAUDE.md text from the shared prompt + config-gated addenda.

    Addenda are appended additively (never forked into parallel base
    prompts): the scratch_vm addendum when scratch_vm_enabled, and the
    ScienceBoard eval-format addendum when benchmark == "scienceboard".
    The two are independent and compose.

    v2: seed/prompt are shared across all islands (lived on IslandConfig
    in v1, now on ExperimentConfig). So this no longer takes an island
    argument — every clone gets the same CLAUDE.md.

    Each addendum is joined onto the base with a single blank line so the
    final CLAUDE.md has clean markdown section breaks regardless of
    whether the base ends with a trailing newline.
    """
    out = Path(config.prompt_template).read_text().rstrip() + "\n"
    if config.scratch_vm_enabled:
        # The scratch VM exposes a benchmark-specific surface (OSWorld has a
        # SetupController + browser/media ports; ScienceBoard launches the
        # scientific apps via execute_python_command), so the addendum differs.
        scratch_fname = (
            "scratch_vm_addendum_scienceboard.md"
            if config.benchmark == "scienceboard"
            else "scratch_vm_addendum.md"
        )
        addendum = prompt_path(scratch_fname).read_text().rstrip() + "\n"
        out = out + "\n" + addendum
    if config.benchmark == "scienceboard" and SB_UNGRADED_TRAIN_MARKER not in out:
        addendum = (
            prompt_path("scienceboard_eval_format_addendum.md").read_text().rstrip()
            + "\n"
        )
        out = out + "\n" + addendum
    return out


def initialize_clone_v2(
    template_dir: Path,
    clone_dir: Path,
    config: ExperimentConfig,
    island: IslandConfig,
    prompt_text: str,
) -> None:
    """Create (or no-op if present) an island clone at clone_dir.

    Rsyncs the OSWorld_Evol template, checks out the evol branch, writes
    the seed shadow_interface + CLAUDE.md + notebook.jsonl, commits and
    tags as evol/<name>/seed. Idempotent: if the clone directory already
    exists, this is a no-op (resume path).

    v2: seed_dir and prompt_template come from ExperimentConfig (shared
    across islands), not IslandConfig.
    """
    source_provenance = _template_provenance(template_dir, config)
    if config.benchmark == "osworld_v2":
        _assert_no_v2_gated_tasks(template_dir)
    if clone_dir.exists():
        if config.benchmark == "osworld_v2":
            _assert_no_v2_gated_tasks(clone_dir)
        if config.benchmark == "scienceboard":
            _assert_no_scienceboard_task_tree(clone_dir)
            if _scienceboard_history_leaks(clone_dir):
                raise RuntimeError(
                    f"ScienceBoard island clone {clone_dir} has git history "
                    "containing the official task tree; re-initialize it"
                )
        # Resume path — but flag drift. If the user's current
        # ExperimentConfig points at different seed_dir or prompt_template
        # than what the existing clone was initialized from, the new
        # values are silently ignored (we don't re-seed / re-write
        # CLAUDE.md on resume). Warn loudly so "my prompt edit didn't
        # take effect" doesn't become a silent debugging session.
        marker = clone_dir / ".shadow_evol_init"
        if marker.is_file():
            prior = _read_init_marker(marker)
            prior_seed = prior.get("seed_dir")
            prior_prompt = prior.get("prompt_template")
            if prior_seed and prior_seed != str(config.seed_dir):
                logger.warning(
                    "Clone %s was initialized with seed_dir=%s but "
                    "current config says %s. Resume path keeps the OLD "
                    "seed. `rm -rf %s` to re-init from the new seed.",
                    clone_dir,
                    prior_seed,
                    config.seed_dir,
                    clone_dir,
                )
            if prior_prompt and prior_prompt != str(config.prompt_template):
                logger.warning(
                    "Clone %s was initialized with prompt_template=%s but "
                    "current config says %s. Resume path keeps the OLD "
                    "prompt in CLAUDE.md. `rm -rf %s` to re-apply.",
                    clone_dir,
                    prior_prompt,
                    config.prompt_template,
                    clone_dir,
                )
            if config.benchmark == "osworld_v2":
                missing = sorted(set(source_provenance) - set(prior))
                mismatched = {
                    key: (prior.get(key), expected)
                    for key, expected in source_provenance.items()
                    if prior.get(key) != expected
                }
                if missing or mismatched:
                    raise RuntimeError(
                        f"OSWorld V2 island clone {clone_dir} has stale or missing "
                        f"release provenance: missing={missing}, "
                        f"mismatched={mismatched}. Re-initialize the island clone "
                        "from the parent-pinned template before resuming."
                    )
        elif config.benchmark == "osworld_v2":
            raise RuntimeError(
                f"OSWorld V2 island clone {clone_dir} has no .shadow_evol_init "
                "provenance marker; refusing an unpinned resume"
            )
        logger.info("Clone %s already exists — skipping init", clone_dir)
        return

    logger.info("Cloning %s → %s", template_dir, clone_dir)
    # ScienceBoard: the official task tree lives in the template's working
    # tree and history, so the clone takes neither -- only the code, under a
    # fresh repository whose root commit is the seed.
    scienceboard_excludes = (
        ["--exclude", "/tasks/", "--exclude", "/.git/", "--exclude", "/AGENTS.md"]
        if config.benchmark == "scienceboard"
        else []
    )
    subprocess.run(
        [
            "rsync",
            "-a",
            "--exclude",
            "results/",
            "--exclude",
            "cache/",
            "--exclude",
            "logs/",
            "--exclude",
            "__pycache__/",
            "--exclude",
            ".venv/",
            *scienceboard_excludes,
            f"{template_dir}/",
            f"{clone_dir}/",
        ],
        check=True,
    )
    if config.benchmark == "scienceboard":
        subprocess.run(
            ["git", "init", "-q", "-b", "evol"], cwd=str(clone_dir), check=True
        )
        _assert_no_scienceboard_task_tree(clone_dir)

    # Ensure clone is on its own 'evol' branch (not the template's default)
    cur = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=str(clone_dir),
        capture_output=True,
        text=True,
    )
    if cur.stdout.strip() != "evol":
        subprocess.run(
            ["git", "checkout", "-b", "evol"],
            cwd=str(clone_dir),
            check=True,
        )

    deploy_seed(clone_dir, config.seed_dir)
    (clone_dir / "CLAUDE.md").write_text(prompt_text)
    (clone_dir / "notebook.jsonl").touch()

    tag = f"evol/{island.name}/seed"
    subprocess.run(["git", "add", "-A", "."], cwd=str(clone_dir), check=True)
    subprocess.run(
        ["git", "commit", "-m", f"{tag}: initial seed interface"],
        cwd=str(clone_dir),
        check=True,
    )
    subprocess.run(["git", "tag", "-f", tag], cwd=str(clone_dir), check=True)

    # Record init provenance so future runs can detect stale clones.
    (clone_dir / ".shadow_evol_init").write_text(
        f"seed_dir={config.seed_dir}\n"
        f"prompt_template={config.prompt_template}\n"
        + "".join(f"{key}={value}\n" for key, value in source_provenance.items())
    )
    logger.info("Initialized %s with seed at %s", island.name, tag)


def deploy_seed(clone_dir: Path, seed_dir: Path) -> None:
    """Copy seed_dir into clone_dir/shadow_interface/ (overwriting)."""
    dst = clone_dir / "shadow_interface"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(str(seed_dir), str(dst))
    # Preserve executable bits for setup.sh if present
    setup = dst / "setup.sh"
    if setup.exists():
        os.chmod(setup, 0o755)


def _seed_was_eliminated(conn, island_name: str) -> bool:
    """True if the seed variant was ever removed from its island's pool.

    Two events both remove the seed and should suppress re-insertion:
      - 'elimination': tournament-driven cull (child=seed_id).
      - 'migration_out': seed shipped to another island via migration's
        "Move" semantics (child=seed_id; written by
        migration.apply_phase1 right after delete_variant).

    Either way the seed is gone from the gene pool; resuming the run
    should NOT re-create it (doing so would waste a full seed eval and
    re-inject a variant the population already moved past).
    """
    row = conn.execute(
        "SELECT 1 FROM history "
        "WHERE type IN ('elimination', 'migration_out') AND child=? "
        "LIMIT 1",
        (f"{island_name}-seed",),
    ).fetchone()
    return row is not None


def insert_seed_variant(conn, island_name: str) -> None:
    """Insert the seed variant row if missing and not previously eliminated.

    Resume semantics: if the seed was culled by a prior run's elimination
    operator, skip the insert. run_seed_eval_v2's own early-return on
    `seed is None` keeps the cost-$$ eval from running in that case.
    """
    if get_variant(conn, f"{island_name}-seed"):
        return
    if _seed_was_eliminated(conn, island_name):
        logger.info(
            "Seed %s-seed was eliminated in a prior run — not re-inserting "
            "(resume path; skips re-eval)",
            island_name,
        )
        return
    insert_variant(
        conn,
        {
            "id": f"{island_name}-seed",
            "island": island_name,
            "tag": f"evol/{island_name}/seed",
            "parent_id": None,
            "parent_b_id": None,
            "origin": "seed",
            "generation": 0,
            "score": None,
            "eval_level": 0,
            "builder_cost": 0.0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _seed_tasks_needing_eval(
    result_dir: Path,
    model: str,
    task_ids: list[str],
) -> list[str]:
    """Return tasks without a complete, finite cached OSWorld result.

    Seed evaluation is the only evolution phase that can be interrupted before
    any DB score exists.  Its task artifacts are already variant- and
    model-scoped, so valid ``result.txt`` files are safe checkpoints.  Resume
    only the missing/invalid tasks instead of paying to replay completed VM
    trajectories.
    """
    cache = result_dir / "shadow_eval" / "screenshot" / model
    pending = []
    for task_id in task_ids:
        path = cache / task_id / "result.txt"
        try:
            score = float(path.read_text().strip())
        except (OSError, ValueError):
            pending.append(task_id)
            continue
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            pending.append(task_id)
    return pending


def run_seed_eval_v2(
    conn, island: IslandConfig, config: ExperimentConfig, base_dir: Path
) -> None:
    """Score the seed under THIS island's evaluator on the canonical task set.

    v2: a single canonical train task set (config.train_set_file) replaces
    the tier cascade. The same seed code is scored independently under
    each island's own evaluator harness/model, so the three (or N)
    islands end up with different per-island seed scores even though the
    underlying interface is identical.
    """
    seed_id = f"{island.name}-seed"
    seed = get_variant(conn, seed_id)
    if seed is None:
        return
    if seed["score"] is not None:
        logger.info("Seed %s already scored — skipping seed eval", seed_id)
        return

    train_path = base_dir / EVAL_SETS_DIR / config.train_set_file
    if not train_path.is_file():
        raise FileNotFoundError(
            f"Seed eval requires {train_path} (config.train_set_file="
            f"{config.train_set_file!r}). Create that file or set "
            f"--train-set to one that exists."
        )
    with open(train_path) as f:
        train_tasks = json.load(f)

    val_tasks = _load_val_tasks(str(base_dir), config.val_set_file)
    all_tasks = train_tasks + val_tasks

    harness_key = config.resolve_evaluator_harness_key(island)
    harness = resolve_evaluator_harness(harness_key, config.benchmark)
    model = config.resolve_eval_model(island)
    template = BENCHMARKS[config.benchmark]["template"]
    clone_dir = base_dir / f"{template}_{island.name}"
    shadow_dir = clone_dir / "shadow_interface"
    result_dir = base_dir / ns(EVAL_CACHE_DIR, config.benchmark) / seed_id

    pending_tasks = _seed_tasks_needing_eval(result_dir, model, all_tasks)
    logger.info(
        "Seed eval for %s: %d train + %d val = %d tasks; %d cached, %d pending "
        "(model=%s, harness=%s)",
        seed_id,
        len(train_tasks),
        len(val_tasks),
        len(all_tasks),
        len(all_tasks) - len(pending_tasks),
        len(pending_tasks),
        model,
        harness_key,
    )

    if pending_tasks:
        out = harness.run_tasks(
            clone_dir=clone_dir,
            shadow_dir=shadow_dir,
            result_dir=result_dir,
            task_ids=pending_tasks,
            model=model,
            region=config.region,
            timeout=config.eval_timeout,
            max_steps=config.eval_max_steps,
            no_time_limit=config.eval_no_time_limit,
            harness=harness_key,
            benchmark=config.benchmark,
        )

        if out.timed_out:
            raise RuntimeError(
                f"Seed eval timed out for {seed_id}; refusing to turn unfinished "
                "infrastructure work into zero fitness"
            )
    else:
        logger.info("Seed %s has complete task cache — skipping VM evaluation", seed_id)

    now = datetime.now(timezone.utc).isoformat()

    # Read per-task results from disk (result.txt files written by the eval
    # harness, thresholded to 0/1 by _collect_per_task). This is the same
    # source-of-truth that Mutation.apply uses, and it avoids the field-name
    # mismatch with latest_summary.json: that file's `results` array uses
    # the key "score" (not "passed"), so the prior `entry.get("passed", 0)`
    # silently defaulted every row to 0 — the per-variant `score` column
    # was correct only because it read out.passed (the top-level aggregate)
    # directly. Once recompute_score started joining eval_results (val-set
    # rollout), that latent bug surfaced as score=0 across the board.
    results = _collect_per_task_records(
        str(clone_dir),
        seed_id,
        all_tasks,
        model,
        benchmark=config.benchmark,
    )
    for tid, record in results.items():
        insert_eval_result(
            conn,
            seed_id,
            tid,
            record["passed"],
            now,
            raw_score=record["raw_score"],
            fitness_score=record["fitness_score"],
            details_json=record["details_json"],
            failure_type=record["failure_type"],
        )

    # Score split: out.passed is total over all tasks (train + val); we want
    # them scored separately against their respective sets via the DB.
    score = recompute_score_from_train_set(conn, seed_id, train_tasks)
    val_score = recompute_val_score(conn, seed_id)
    update_variant(conn, seed_id, score=score, val_score=val_score, step=0)
    logger.info(
        "Seed %s scored %.3f/%d (val: %.3f/%d)",
        seed_id,
        score,
        len(train_tasks),
        val_score,
        len(val_tasks),
    )


def dump_config(config: ExperimentConfig, runs_dir: Path) -> Path:
    """Dump the resolved config to runs/<UTC-timestamp>/config.json."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = runs_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.json"
    payload = _config_to_dict(config)
    with open(path, "w") as f:
        json.dump(payload, f, default=str, indent=2)
    logger.info("Config dumped to %s", path)
    return path


def _config_to_dict(config: ExperimentConfig) -> dict:
    """Serialize ExperimentConfig → a plain dict round-trippable via the
    CLI's rerun path.

    v2: operators/triggers/tiers are gone. The scheduler picks ops per
    step, so it's the only piece of strategy state that needs to
    round-trip alongside the bare config knobs.
    """
    sched = config.scheduler
    sched_payload = {
        "type": type(sched).__name__,
        "args": {k: v for k, v in sched.__dict__.items()},
    }
    return {
        "islands": [
            {
                "name": i.name,
                "evaluator_harness": i.evaluator_harness,
                "eval_model": i.eval_model,
            }
            for i in config.islands
        ],
        "scheduler": sched_payload,
        "seed_dir": str(config.seed_dir),
        "prompt_template": str(config.prompt_template),
        "builder_model": config.builder_model,
        "builder_harness": config.builder_harness,
        "builder_claude_bin": config.builder_claude_bin,
        "builder_timeout": config.builder_timeout,
        "builder_no_time_limit": config.builder_no_time_limit,
        "eval_timeout": config.eval_timeout,
        "eval_max_steps": config.eval_max_steps,
        "eval_no_time_limit": config.eval_no_time_limit,
        "region": config.region,
        "scratch_vm_enabled": config.scratch_vm_enabled,
        "elimination_count": config.elimination_count,
        "reevaluation_interval": config.reevaluation_interval,
        "reevaluation_top_k": config.reevaluation_top_k,
        "max_steps": config.max_steps,
        "max_cost": config.max_cost,
        "seed": config.seed,
        "train_set_file": config.train_set_file,
        "val_set_file": config.val_set_file,
        "held_out_tasks_path": (
            str(config.held_out_tasks_path)
            if config.held_out_tasks_path is not None
            else None
        ),
        "selection_signal": config.selection_signal,
        "parent_selection": config.parent_selection,
        "migration_mode": config.migration_mode,
        "migration_topology": config.migration_topology,
        "benchmark": config.benchmark,
        "allow_v2_noncanonical_splits": config.allow_v2_noncanonical_splits,
    }


def run(config: ExperimentConfig) -> None:
    """v2 orchestrator: bootstraps clones, runs per-island seed eval in
    parallel (each under its own evaluator), then hands off to the global
    driver. No more tmux panes — one process drives everything."""
    base_dir = Path.cwd()
    validate_prime_agent_before_run(config)
    validate_v2_workspace_before_run(config, base_dir)
    validate_v2_run_splits(config, base_dir)
    template = BENCHMARKS[config.benchmark]["template"]
    template_dir = base_dir / template

    db_path = base_dir / ns(DB_FILENAME, config.benchmark)
    conn = db_connect(str(db_path))
    init_schema(conn)

    # Load val set (idempotent). On resume, verify_val_set_unchanged
    # guards against silently swapping the val IDs out from under
    # already-recorded val_score values.
    if config.val_set_file and config.val_set_file != "none":
        val_path = base_dir / EVAL_SETS_DIR / config.val_set_file
        if val_path.is_file():
            with open(val_path) as f:
                val_task_ids = json.load(f)
            verify_val_set_unchanged(conn, val_task_ids)
            load_val_set(conn, val_task_ids)
            logger.info(
                "Loaded %d val tasks from %s",
                len(val_task_ids),
                val_path,
            )
        else:
            logger.warning(
                "val_set_file %s not found at %s — val_set table left empty "
                "for this run",
                config.val_set_file,
                val_path,
            )

    # Per-island bootstrap. Prompt is shared across islands in v2.
    prompt_text = assemble_prompt_v2(config)
    for island in config.islands:
        clone_dir = base_dir / f"{template}_{island.name}"
        initialize_clone_v2(template_dir, clone_dir, config, island, prompt_text)
        init_island(conn, island.name)
        insert_seed_variant(conn, island.name)

    conn.close()
    db_path_str = str(db_path)

    # Parallel per-island seed eval — each island scores the same seed
    # code under its own evaluator. Each worker opens its own DB
    # connection (SQLite's WAL mode allows concurrent readers + one
    # writer at a time; per-island writes don't contend since they
    # target different variant rows).
    def _seed_eval_worker(island: IslandConfig) -> None:
        thread_conn = db_connect(db_path_str)
        try:
            run_seed_eval_v2(thread_conn, island, config, base_dir)
        finally:
            thread_conn.close()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(len(config.islands), 1)
    ) as executor:
        # list() forces exceptions to surface (executor.map is lazy)
        list(executor.map(_seed_eval_worker, config.islands))

    # Dump resolved config for reproducibility
    runs_dir = base_dir / "runs"
    config_path = dump_config(config, runs_dir)
    logger.info("Config dumped to %s", config_path)

    # Hand off to driver — single-process replacement for the old tmux
    # fan-out. Driver opens its own DB connection.
    from agent_s4.driver import Driver

    driver = Driver(config=config, base_dir=base_dir)
    driver.run()
