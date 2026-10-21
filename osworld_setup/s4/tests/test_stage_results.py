"""SL-purity boundary tests for stage_results_for_builder.

stage_results_for_builder is the single point at which val trajectories could
leak to the operator. These tests pin the actual nested layout of eval_cache/
and verify the filter is exact.

Real layout under eval_cache/<variant>/:
    latest_summary.json                                       ← LEAK VECTOR
    shadow_eval/screenshot/haiku/<task_id>/{six per-task files}
"""
import json
from pathlib import Path


from agent_s4.utils import stage_results_for_builder
from agent_s4.db import load_val_set


PER_TASK_FILES = [
    "claude_stderr.log",
    "claude_trajectory.jsonl",
    "exit_code.txt",
    "instruction.txt",
    "result.txt",
    "traj.jsonl",
]


def _build_eval_cache(base_dir: Path, variant_id: str,
                      train_ids: list[str], val_ids: list[str],
                      include_summary: bool = True) -> Path:
    """Reproduce the real eval_cache layout for testing."""
    cache_root = base_dir / "eval_cache" / variant_id
    haiku_root = cache_root / "shadow_eval" / "screenshot" / "haiku"
    haiku_root.mkdir(parents=True)
    for tid in train_ids + val_ids:
        td = haiku_root / tid
        td.mkdir()
        for fname in PER_TASK_FILES:
            (td / fname).write_text(f"content:{tid}:{fname}")
    if include_summary:
        results = [{"task_id": tid, "score": 0} for tid in train_ids + val_ids]
        (cache_root / "latest_summary.json").write_text(
            json.dumps({"total": len(results), "passed": 0, "results": results})
        )
    return cache_root


def test_layout_fidelity_train_kept_val_dropped(tmp_db, tmp_path):
    """Train task subdirs (with all six per-task files) are copied; val absent."""
    train_ids = ["train_aaa", "train_bbb"]
    val_ids = ["val_xxx", "val_yyy"]
    _build_eval_cache(tmp_path, "v1", train_ids, val_ids)
    load_val_set(tmp_db, val_ids)

    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    stage_results_for_builder("v1", str(tmp_path), str(clone_dir), tmp_db)

    haiku_dst = clone_dir / "eval_results" / "shadow_eval" / "screenshot" / "haiku"
    for tid in train_ids:
        for fname in PER_TASK_FILES:
            assert (haiku_dst / tid / fname).read_text() == f"content:{tid}:{fname}"
    for tid in val_ids:
        assert not (haiku_dst / tid).exists()


def test_intermediate_dirs_preserved(tmp_db, tmp_path):
    """The shadow_eval/screenshot/haiku dirs are present (not over-aggressive)."""
    _build_eval_cache(tmp_path, "v1", ["train_a"], ["val_x"])
    load_val_set(tmp_db, ["val_x"])

    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    stage_results_for_builder("v1", str(tmp_path), str(clone_dir), tmp_db)

    eval_results = clone_dir / "eval_results"
    assert (eval_results / "shadow_eval").is_dir()
    assert (eval_results / "shadow_eval" / "screenshot").is_dir()
    assert (eval_results / "shadow_eval" / "screenshot" / "haiku").is_dir()


def test_latest_summary_json_not_staged(tmp_db, tmp_path):
    """Top-level latest_summary.json is dropped — would otherwise leak val IDs."""
    _build_eval_cache(tmp_path, "v1", ["train_a"], ["val_x"], include_summary=True)
    load_val_set(tmp_db, ["val_x"])

    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    stage_results_for_builder("v1", str(tmp_path), str(clone_dir), tmp_db)

    assert not (clone_dir / "eval_results" / "latest_summary.json").exists()


def test_empty_val_set_stages_all_tasks(tmp_db, tmp_path):
    """When val is disabled, all task subdirs copy through (regression for --val-set none)."""
    _build_eval_cache(tmp_path, "v1", ["a", "b", "c"], [], include_summary=False)
    # val_set table is empty (no load_val_set call).

    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    stage_results_for_builder("v1", str(tmp_path), str(clone_dir), tmp_db)

    haiku_dst = clone_dir / "eval_results" / "shadow_eval" / "screenshot" / "haiku"
    for tid in ["a", "b", "c"]:
        assert (haiku_dst / tid / "result.txt").exists()


def test_empty_src_creates_empty_dst(tmp_db, tmp_path):
    """Missing eval_cache dir → dst created as empty, no crash."""
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    # No eval_cache/v1/ exists.
    stage_results_for_builder("v1", str(tmp_path), str(clone_dir), tmp_db)
    assert (clone_dir / "eval_results").is_dir()
    assert list((clone_dir / "eval_results").iterdir()) == []


def test_existing_dst_replaced(tmp_db, tmp_path):
    """Stale eval_results in clone_dir is wiped before staging."""
    _build_eval_cache(tmp_path, "v1", ["new_train"], [])

    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    stale_dst = clone_dir / "eval_results"
    stale_dst.mkdir()
    (stale_dst / "stale_file.txt").write_text("old content")

    stage_results_for_builder("v1", str(tmp_path), str(clone_dir), tmp_db)

    assert not (stale_dst / "stale_file.txt").exists()
    assert (
        stale_dst / "shadow_eval" / "screenshot" / "haiku" / "new_train" / "result.txt"
    ).exists()


def test_db_driven_filter_is_source_of_truth(tmp_db, tmp_path):
    """Adding task_ids to val_set after build still filters them at stage time."""
    train_ids = ["t_a", "t_b"]
    val_ids = ["t_c", "t_d", "t_e"]
    _build_eval_cache(tmp_path, "v1", train_ids, val_ids, include_summary=False)
    load_val_set(tmp_db, val_ids)

    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    stage_results_for_builder("v1", str(tmp_path), str(clone_dir), tmp_db)

    haiku_dst = clone_dir / "eval_results" / "shadow_eval" / "screenshot" / "haiku"
    for tid in train_ids:
        assert (haiku_dst / tid).exists()
    for tid in val_ids:
        assert not (haiku_dst / tid).exists()


def test_content_byte_preservation_for_train(tmp_db, tmp_path):
    """For staged train task dirs, every file's bytes match source exactly."""
    train_ids = ["only_train"]
    _build_eval_cache(tmp_path, "v1", train_ids, [], include_summary=False)
    load_val_set(tmp_db, [])

    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    stage_results_for_builder("v1", str(tmp_path), str(clone_dir), tmp_db)

    src_root = (
        tmp_path / "eval_cache" / "v1" / "shadow_eval" / "screenshot" / "haiku"
        / "only_train"
    )
    dst_root = (
        clone_dir / "eval_results" / "shadow_eval" / "screenshot" / "haiku"
        / "only_train"
    )
    for fname in PER_TASK_FILES:
        assert (dst_root / fname).read_bytes() == (src_root / fname).read_bytes()


def test_dest_name_routes_to_custom_dir(tmp_db, tmp_path):
    """dest_name stages into clone_dir/<dest_name> (not the default
    eval_results/), and the val filter still applies. Used by crossover to
    stage both parents into eval_results_parent_a / _b."""
    train_ids = ["train_a"]
    val_ids = ["val_x"]
    _build_eval_cache(tmp_path, "v1", train_ids, val_ids, include_summary=False)
    load_val_set(tmp_db, val_ids)

    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    stage_results_for_builder(
        "v1", str(tmp_path), str(clone_dir), tmp_db,
        dest_name="eval_results_parent_a",
    )

    custom = clone_dir / "eval_results_parent_a"
    assert custom.is_dir()
    assert not (clone_dir / "eval_results").exists()  # default dir untouched
    haiku_dst = custom / "shadow_eval" / "screenshot" / "haiku"
    assert (haiku_dst / "train_a" / "result.txt").exists()
    assert not (haiku_dst / "val_x").exists()  # val filter still enforced


def test_dest_name_defaults_to_eval_results(tmp_db, tmp_path):
    """Omitting dest_name preserves the original eval_results/ destination
    (mutation's contract is unchanged)."""
    _build_eval_cache(tmp_path, "v1", ["train_a"], [], include_summary=False)
    load_val_set(tmp_db, [])

    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    stage_results_for_builder("v1", str(tmp_path), str(clone_dir), tmp_db)

    assert (clone_dir / "eval_results").is_dir()


def test_builder_stage_drops_runtime_cache_recording_and_duplicate_stdout(
        tmp_db, tmp_path):
    cache = _build_eval_cache(tmp_path, "v1", ["train_a"], [])
    task = cache / "shadow_eval" / "screenshot" / "haiku" / "train_a"
    (task / "mini_stdout.log").write_bytes(b"duplicate trajectory")
    (task / "recording.mp4").write_bytes(b"video")
    runtime_cache = cache / "_task_cache" / "train_a"
    runtime_cache.mkdir(parents=True)
    (runtime_cache / "disk.img").write_bytes(b"large runtime cache")

    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    stage_results_for_builder("v1", str(tmp_path), str(clone_dir), tmp_db)

    staged = clone_dir / "eval_results"
    staged_task = staged / "shadow_eval" / "screenshot" / "haiku" / "train_a"
    assert not (staged_task / "mini_stdout.log").exists()
    assert not (staged_task / "recording.mp4").exists()
    assert not (staged / "_task_cache").exists()


def test_builder_stage_compacts_prime_stream_updates_only(tmp_db, tmp_path):
    cache = _build_eval_cache(tmp_path, "v1", ["train_a"], [])
    task = cache / "shadow_eval" / "screenshot" / "haiku" / "train_a"
    trajectory = (
        b'{"type":"agent_start"}\n'
        b'{"type":"message_update","message":{"text":"partial"}}\n'
        b'{"type":"message_end","message":{"text":"complete"}}\n'
        b'not-json-diagnostic\n'
        b'{"type":"agent_end"}\n'
    )
    (task / "claude_trajectory.jsonl").write_bytes(trajectory)

    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    stage_results_for_builder("v1", str(tmp_path), str(clone_dir), tmp_db)

    staged = (
        clone_dir / "eval_results" / "shadow_eval" / "screenshot" / "haiku"
        / "train_a" / "claude_trajectory.jsonl"
    ).read_bytes()
    assert b"message_update" not in staged
    assert staged == (
        b'{"type":"agent_start"}\n'
        b'{"type":"message_end","message":{"text":"complete"}}\n'
        b'not-json-diagnostic\n'
        b'{"type":"agent_end"}\n'
    )
