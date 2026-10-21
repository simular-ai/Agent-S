#!/usr/bin/env python3
"""
Create train / val / test splits over OSWorld V2's native task suite.

V2 ships two corpora under evaluation_examples/. This script covers the
V2-native one: the 108 BaseTask classes in task_class/task_<NNN>.py, listed
by id in test_v2.json. (For the v1-style UUID corpus that both generations
ship, use create_eval_sets.py instead.)

Writes three disjoint stratified splits to --out-dir (default eval_sets/):

    osworld_v2_train_<TRAIN>.json   (default 50)
    osworld_v2_val_<VAL>.json       (default 25)
    osworld_v2_test.json            (everything left)

Each file is a JSON LIST of task-id strings ("001", "002", ...), which is
what agent_s4.eval.osworld's --task_ids expects and what
build_task_index() keys the task_class/ files by.

Stratification: V2 labels tasks by capability (dynamic_environment,
human_in_the_loop, ...) in per-category manifests, and a task usually carries
several labels. Each task is bucketed by its RAREST category so scarce
capabilities — human_in_the_loop and streaming_interaction have 6 tasks each
— get spread across the splits instead of landing in one. Ties break
alphabetically, so the bucketing is deterministic.

Usage:
    python scripts/eval/create_osworld_v2_eval_sets.py
    python scripts/eval/create_osworld_v2_eval_sets.py --train 50 --val 25 --seed 42
    python scripts/eval/create_osworld_v2_eval_sets.py \\
        --examples-root OSWorldV2_Base/evaluation_examples --out-dir eval_sets
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXAMPLES_ROOT = REPO_ROOT / "OSWorldV2_Base" / "evaluation_examples"
DEFAULT_OUT_DIR = REPO_ROOT / "eval_sets"

# Manifests that list tasks but aren't capability categories.
_NON_CATEGORY = {"test_v2.json", "test_v2_small.json", "test_all.json",
                 "test_builder.json"}


def _load_manifest_ids(path: Path) -> list[str]:
    """Task ids from test_v2.json ({"tasks": [...]})."""
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [str(t) for t in data]
    return [str(t) for t in data.get("tasks", [])]


def _load_categories(examples_root: Path) -> dict[str, list[str]]:
    """category -> [task_id]. One JSON per capability at the examples root,
    each shaped {"<category>": [ids]}."""
    cats: dict[str, list[str]] = {}
    for f in sorted(examples_root.glob("*.json")):
        if f.name in _NON_CATEGORY:
            continue
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        for name, ids in data.items():
            if isinstance(ids, list) and all(isinstance(i, str) for i in ids):
                cats[name] = ids
    return cats


def _bucket_by_rarest(ids: list[str],
                      cats: dict[str, list[str]]) -> dict[str, str]:
    """task_id -> bucket, where the bucket is the task's rarest category.

    Tasks in no manifest fall into 'uncategorized' so they still get sampled.
    """
    size = {name: len(members) for name, members in cats.items()}
    membership: dict[str, list[str]] = defaultdict(list)
    for name, members in cats.items():
        for tid in members:
            membership[tid].append(name)
    out = {}
    for tid in ids:
        names = membership.get(tid)
        if not names:
            out[tid] = "uncategorized"
            continue
        out[tid] = min(sorted(names), key=lambda n: size[n])
    return out


def stratified_sample(ids: list[str], buckets: dict[str, str], n: int,
                      rng: random.Random) -> list[str]:
    """Sample n ids stratified by bucket.

    Per-bucket allocation is proportional to bucket size, rounded; rounding
    error is corrected against the most/least over-drawn buckets.
    """
    if n >= len(ids):
        return list(ids)
    by_bucket: dict[str, list[str]] = defaultdict(list)
    for tid in ids:
        by_bucket[buckets[tid]].append(tid)

    total = len(ids)
    alloc = {k: max(1, round(n * len(v) / total)) for k, v in by_bucket.items()}
    # max(1, ...) can overshoot n when there are many small buckets. Trim the
    # most over-drawn bucket first, but only from buckets holding more than one
    # slot: taking a scarce bucket's single slot would drop that capability
    # from the split entirely, which is the exact failure this stratification
    # exists to prevent. Only when every bucket is down to one slot (n is
    # smaller than the bucket count) do we start zeroing them out.
    while sum(alloc.values()) > n:
        trimmable = ([k for k in alloc if alloc[k] > 1]
                     or [k for k in alloc if alloc[k] > 0])
        k = max(trimmable, key=lambda x: alloc[x] / max(1, len(by_bucket[x])))
        alloc[k] -= 1
    while sum(alloc.values()) < n:
        k = max(alloc, key=lambda x: len(by_bucket[x]) - alloc[x])
        alloc[k] += 1

    picked: list[str] = []
    for bucket, k in alloc.items():
        pool = by_bucket[bucket][:]
        rng.shuffle(pool)
        picked.extend(pool[:k])
    return picked


def _summarize(name: str, ids: list[str], buckets: dict[str, str]) -> None:
    counts: dict[str, int] = defaultdict(int)
    for tid in ids:
        counts[buckets[tid]] += 1
    detail = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"  {name:28s} ({len(ids):3d}): {detail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples-root", type=Path, default=DEFAULT_EXAMPLES_ROOT,
                    help=f"Default: {DEFAULT_EXAMPLES_ROOT}")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"Default: {DEFAULT_OUT_DIR}")
    ap.add_argument("--train", type=int, default=50,
                    help="Train split size. Default: 50.")
    ap.add_argument("--val", type=int, default=25,
                    help="Validation split size. Default: 25. "
                         "Drawn from the post-train remainder.")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for deterministic stratification.")
    args = ap.parse_args()

    manifest = args.examples_root / "test_v2.json"
    if not manifest.is_file():
        print(f"ERROR: {manifest} not found — point --examples-root at an "
              f"OSWorld V2 evaluation_examples/ tree.", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_ids = _load_manifest_ids(manifest)
    task_class = args.examples_root / "task_class"
    on_disk = {p.stem[len("task_"):] for p in task_class.glob("task_*.py")}
    missing = [t for t in all_ids if t not in on_disk]
    if missing:
        print(f"WARNING: {len(missing)} manifest ids have no task_class file: "
              f"{missing[:5]}", file=sys.stderr)
        all_ids = [t for t in all_ids if t in on_disk]

    cats = _load_categories(args.examples_root)
    buckets = _bucket_by_rarest(all_ids, cats)
    print(f"Loaded {len(all_ids)} V2 tasks, {len(cats)} categories "
          f"from {args.examples_root}")
    _summarize("all V2 tasks", all_ids, buckets)

    if args.train + args.val > len(all_ids):
        print(f"ERROR: train ({args.train}) + val ({args.val}) > available "
              f"({len(all_ids)}); test would be empty.", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    train = stratified_sample(all_ids, buckets, args.train, rng)
    remainder = [t for t in all_ids if t not in set(train)]
    val = stratified_sample(remainder, buckets, args.val, rng)
    test = [t for t in remainder if t not in set(val)]

    splits = [
        (f"osworld_v2_train_{args.train}.json", train),
        (f"osworld_v2_val_{args.val}.json",     val),
        ("osworld_v2_test.json",                test),
    ]
    for fname, ids in splits:
        ids = sorted(ids)
        (args.out_dir / fname).write_text(json.dumps(ids, indent=2) + "\n")
        _summarize(fname, ids, buckets)

    # Disjointness is the property that matters most here — a leak between
    # train and val silently inflates the selection signal.
    assert not (set(train) & set(val)), "train/val overlap"
    assert not (set(train) & set(test)), "train/test overlap"
    assert not (set(val) & set(test)), "val/test overlap"
    print(f"\nWrote 3 splits to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
