"""Small utility helpers used across agent_s4.

Moved out of island_common.py during the framework refactor.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shutil

from agent_s4.benchmark_spec import ns
from agent_s4.constants import EVAL_CACHE_DIR, EVAL_SETS_DIR

_PRIME_MESSAGE_UPDATE_LINE = re.compile(
    rb'^\s*\{\s*"type"\s*:\s*"message_update"\s*[,}]',
)


def compact_prime_trajectory_jsonl(data: bytes) -> bytes:
    """Drop Prime's redundant streaming snapshots from trajectory JSONL.

    ``message_update`` records repeatedly contain the growing in-progress
    message and can make a single trajectory exceed a gigabyte.  Complete
    messages are independently recorded by ``message_end``; tool and terminal
    events are separate records.  Preserve every other line verbatim,
    including malformed lines, so artifact validation can still diagnose
    genuine corruption.
    """
    return b"".join(
        line
        for line in data.splitlines(keepends=True)
        if not _PRIME_MESSAGE_UPDATE_LINE.match(line)
    )


def now_iso() -> str:
    """Current local time as ISO 8601 (no timezone). Matches the format
    used throughout the variants/history tables."""
    return datetime.datetime.now().isoformat()


def get_next_variant_num(island_data: dict, clone_dir: str | None = None) -> int:
    """Thin shim around git_utils.next_variant_num_from_population_and_tags
    that matches the old island_common call shape.

    island_data is expected to have a 'population' key → list of variant dicts.
    """
    from agent_s4.git_utils import next_variant_num_from_population_and_tags

    return next_variant_num_from_population_and_tags(
        island_data["population"],
        clone_dir,
    )


def _load_val_tasks(base_dir: str, val_set_file: str | None) -> list[str]:
    """Read the val task list from eval_sets/<val_set_file>.

    Returns [] when val is disabled (val_set_file is None or "none").
    Otherwise reads the JSON file and returns its task_ids list. Raises
    FileNotFoundError if the file is missing — callers should validate
    config-time existence to surface this early.
    """
    if val_set_file is None or val_set_file == "none":
        return []
    path = os.path.join(base_dir, EVAL_SETS_DIR, val_set_file)
    with open(path) as f:
        return json.load(f)


def stage_results_for_builder(
    variant_id: str,
    base_dir: str,
    clone_dir: str,
    conn,
    *,
    benchmark: str = "osworld",
    dest_name: str = "eval_results",
) -> None:
    """Copy the parent's eval_cache content into clone_dir/<dest_name>/
    (default "eval_results"), omitting val task trajectories and the aggregate
    latest_summary.json. dest_name lets a caller stage more than one variant
    into distinct dirs (e.g. crossover stages both parents).

    SL-purity boundary: this is the single function through which eval_cache
    content reaches the builder. Other callers that need eval trajectories in
    the builder's working tree must go through here. The DB is the source of
    truth for which task_ids are val (no in-memory list lookup).

    Filter rules:
      1. Per-task subdirs whose name appears in val_set are skipped (val IDs
         are UUIDs, no realistic collision with intermediate dirs like
         shadow_eval / screenshot / haiku).
      2. latest_summary.json at the variant root is skipped — its results
         array would otherwise leak val task IDs and scores.
      3. Evaluator-local VM caches, recordings, and Prime's duplicate
         ``mini_stdout.log`` are not useful builder evidence and are skipped.
      4. Prime trajectory ``message_update`` snapshots are removed while all
         semantically complete events and malformed diagnostic lines remain.
    """
    from agent_s4.db import get_val_task_ids

    val_ids = set(get_val_task_ids(conn))

    src = os.path.join(base_dir, ns(EVAL_CACHE_DIR, benchmark), variant_id)
    dst = os.path.join(clone_dir, dest_name)

    if os.path.exists(dst):
        shutil.rmtree(dst)

    if not os.path.exists(src):
        os.makedirs(dst)
        return

    src_abs = os.path.abspath(src)

    def ignore(directory, names):
        skip = set()
        # latest_summary.json only at the variant root level
        if os.path.abspath(directory) == src_abs and "latest_summary.json" in names:
            skip.add("latest_summary.json")
        # Per-task dirs whose name is in val_ids, anywhere in the tree
        for name in names:
            if name == "_task_cache" and os.path.isdir(os.path.join(directory, name)):
                skip.add(name)
            if name in {"mini_stdout.log", "recording.mp4"}:
                skip.add(name)
            if name in val_ids:
                full = os.path.join(directory, name)
                if os.path.isdir(full):
                    skip.add(name)
        return skip

    def copy_for_builder(src_path, dst_path, *, follow_symlinks=True):
        basename = os.path.basename(src_path)
        if basename.startswith("claude_trajectory") and basename.endswith(".jsonl"):
            with open(src_path, "rb") as source, open(dst_path, "wb") as target:
                for line in source:
                    if not _PRIME_MESSAGE_UPDATE_LINE.match(line):
                        target.write(line)
            shutil.copystat(src_path, dst_path, follow_symlinks=follow_symlinks)
            return dst_path
        return shutil.copy2(
            src_path,
            dst_path,
            follow_symlinks=follow_symlinks,
        )

    shutil.copytree(src, dst, ignore=ignore, copy_function=copy_for_builder)
