#!/usr/bin/env python3
"""
Create train / val / test split for ScienceBoard.

Walks <tasks-root>/<App>/*.json (default ScienceBoard_Base/tasks/VM, all VM
tasks across the apps) and writes three disjoint stratified splits to
--out-dir (default eval_sets/):

    scienceboard_train_<TRAIN>.json   (default 50)
    scienceboard_val_<VAL>.json       (default 25)
    scienceboard_test.json            (everything left)

Each split file is a JSON LIST of task-id strings of the form "App:stem"
(e.g. "ChimeraX:A-02"), matching how the ScienceBoard eval module's
load_tasks builds ids and how it filters them via --task_ids.

Stratification: each split samples proportionally across (app × difficulty),
where difficulty is the letter prefix (A/B/C/D) in each task's filename.
Train/val/test are disjoint — val is drawn from the post-train remainder,
test gets whatever's left.

No filtering is applied: all task types (including those whose evaluators
require a ScienceBoard Primitive emit, e.g. {"type":"stop","value":"ANS"})
are kept in the pool. Tasks agent_s4's Claude can't pass simply score 0,
which is still useful evolutionary signal.

Usage:
    python scripts/create_scienceboard_eval_sets.py
    python scripts/create_scienceboard_eval_sets.py --train 50 --val 25 --seed 42
    python scripts/create_scienceboard_eval_sets.py --apps ChimeraX,Lean
    python scripts/create_scienceboard_eval_sets.py \\
        --tasks-root ScienceBoard_Base/tasks/VM --out-dir eval_sets
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS_ROOT = REPO_ROOT / "ScienceBoard_Base" / "tasks" / "VM"
DEFAULT_OUT_DIR = REPO_ROOT / "eval_sets"

# Filename convention is <difficulty-letter>-<NN>.json, e.g. "A-02.json".
_DIFF_RE = re.compile(r"^([A-Z])-\d+")


def _difficulty(stem: str) -> str:
    m = _DIFF_RE.match(stem)
    return m.group(1) if m else "?"


def _walk_tasks(root: Path, apps: set[str] | None) -> list[dict]:
    """Return list of {task_id, app, difficulty} for all VM tasks under root,
    optionally filtered to the given app subset. No EARLY_STOP filtering —
    we keep every task in the pool."""
    out: list[dict] = []
    for app_dir in sorted(root.iterdir()):
        if not app_dir.is_dir():
            continue
        if apps and app_dir.name not in apps:
            continue
        for f in sorted(app_dir.glob("*.json")):
            out.append({
                "task_id": f"{app_dir.name}:{f.stem}",
                "app": app_dir.name,
                "difficulty": _difficulty(f.stem),
            })
    print(f"  loaded {len(out)} VM tasks")
    return out


def stratified_sample(tasks: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample n tasks stratified by (app × difficulty) bucket.

    Per-bucket allocation is proportional to bucket size, rounded; rounding
    error is corrected by tweaking the largest/most-flexible buckets.
    Within each bucket the selection is a deterministic random shuffle.
    """
    if n >= len(tasks):
        return list(tasks)
    by_bucket: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in tasks:
        by_bucket[(t["app"], t["difficulty"])].append(t)

    total = len(tasks)
    alloc = {k: max(1, round(n * len(v) / total)) for k, v in by_bucket.items()}
    while sum(alloc.values()) > n:
        k = max(alloc, key=lambda x: alloc[x] / max(1, len(by_bucket[x])))
        alloc[k] -= 1
    while sum(alloc.values()) < n:
        k = max(alloc, key=lambda x: len(by_bucket[x]) - alloc[x])
        alloc[k] += 1

    picked: list[dict] = []
    for bucket, k in alloc.items():
        pool = by_bucket[bucket][:]
        rng.shuffle(pool)
        picked.extend(pool[:k])
    return picked


def _summarize(name: str, tasks: list[dict]) -> None:
    by_app: dict[str, int] = defaultdict(int)
    by_diff: dict[str, int] = defaultdict(int)
    for t in tasks:
        by_app[t["app"]] += 1
        by_diff[t["difficulty"]] += 1
    apps_str = " ".join(f"{k}={v}" for k, v in sorted(by_app.items()))
    diff_str = " ".join(f"{k}={v}" for k, v in sorted(by_diff.items()))
    print(f"  {name:35s} ({len(tasks):3d}): apps[{apps_str}] diff[{diff_str}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT,
                    help=f"Default: {DEFAULT_TASKS_ROOT}")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"Default: {DEFAULT_OUT_DIR}")
    ap.add_argument("--apps", type=str, default=None,
                    help="Comma-separated subset (e.g. 'ChimeraX,Lean'). "
                         "Default: all apps under --tasks-root.")
    ap.add_argument("--train", type=int, default=50,
                    help="Train split size. Default: 50.")
    ap.add_argument("--val", type=int, default=25,
                    help="Validation split size. Default: 25. "
                         "Drawn from the post-train remainder.")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for deterministic stratification.")
    args = ap.parse_args()

    if not args.tasks_root.is_dir():
        print(f"ERROR: tasks-root not found: {args.tasks_root}", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    apps = set(args.apps.split(",")) if args.apps else None

    print(f"Walking {args.tasks_root}...")
    all_tasks = _walk_tasks(args.tasks_root, apps)
    if not all_tasks:
        print("No tasks found.", file=sys.stderr)
        return 1
    if args.train + args.val > len(all_tasks):
        print(
            f"ERROR: train ({args.train}) + val ({args.val}) "
            f"> available ({len(all_tasks)}); test would be empty.",
            file=sys.stderr,
        )
        return 1
    _summarize("all VM tasks", all_tasks)

    rng = random.Random(args.seed)

    # Disjoint splits: stratified-sample train; then val from remainder;
    # then test = whatever's left. Sort each split deterministically by
    # (app, difficulty, task_id) so file output is reproducible.
    train = stratified_sample(all_tasks, args.train, rng)
    train_ids = {t["task_id"] for t in train}
    remainder_after_train = [t for t in all_tasks if t["task_id"] not in train_ids]

    val = stratified_sample(remainder_after_train, args.val, rng)
    val_ids = {t["task_id"] for t in val}
    test = [t for t in remainder_after_train if t["task_id"] not in val_ids]

    splits = [
        (f"scienceboard_train_{args.train}.json", train),
        (f"scienceboard_val_{args.val}.json",     val),
        ("scienceboard_test.json",               test),
    ]

    written: list[Path] = []
    for fname, items in splits:
        items.sort(key=lambda t: (t["app"], t["difficulty"], t["task_id"]))
        out_path = args.out_dir / fname
        with open(out_path, "w") as f:
            json.dump([t["task_id"] for t in items], f, indent=2)
        _summarize(fname, items)
        written.append(out_path)

    print()
    print(f"Wrote {len(written)} files to {args.out_dir}:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
