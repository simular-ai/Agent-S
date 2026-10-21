"""Regression test: `_collect_per_task` reads from
`{base_dir}/eval_cache/<variant_id>/shadow_eval/screenshot/<eval_model>/<task>/result.txt`.

Until April 2026 the model segment was hardcoded to the literal 'haiku',
silently returning all zeros for any eval configured with a non-haiku
model (sonnet, qwen, etc.). The eval writer (agent_s4.eval.osworld)
always uses the model flag it was invoked with, so reader and writer
must agree on the path."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_s4.tournament import _collect_per_task, _collect_per_task_records


def _stage_result(base_dir: Path, variant_id: str, model: str, task_id: str, content: str):
    """Create a per-task result.txt at the path the eval writer uses."""
    d = base_dir / "eval_cache" / variant_id / "shadow_eval" / "screenshot" / model / task_id
    d.mkdir(parents=True)
    (d / "result.txt").write_text(content)


def test_reads_from_model_specific_path(tmp_path):
    # Clone lives at tmp_path/OSWorld_Evol_gui so base_dir (parent) is tmp_path
    clone_dir = tmp_path / "OSWorld_Evol_gui"
    clone_dir.mkdir()

    # Stage haiku + sonnet + qwen results for the same variant
    for model, result_value in [("haiku", "1.0"), ("sonnet", "0"), ("qwen", "1")]:
        _stage_result(tmp_path, "gui-if-001", model, f"task_{model}", result_value)

    # Collect against each model; only the matching path's result should surface
    out_haiku = _collect_per_task(
        str(clone_dir), "gui-if-001", ["task_haiku"], "haiku",
    )
    assert out_haiku == {"task_haiku": 1}, \
        "haiku reader should see haiku result.txt, not sonnet's"

    out_sonnet = _collect_per_task(
        str(clone_dir), "gui-if-001", ["task_sonnet"], "sonnet",
    )
    assert out_sonnet == {"task_sonnet": 0}, \
        "sonnet reader must not fall through to haiku path"

    out_qwen = _collect_per_task(
        str(clone_dir), "gui-if-001", ["task_qwen"], "qwen",
    )
    assert out_qwen == {"task_qwen": 1}


def test_missing_result_txt_is_infrastructure_failure(tmp_path):
    clone_dir = tmp_path / "OSWorld_Evol_gui"
    clone_dir.mkdir()
    # No result.txt files staged anywhere
    with pytest.raises(RuntimeError, match="refusing.*zero fitness"):
        _collect_per_task(str(clone_dir), "gui-if-001", ["t1", "t2"], "haiku")


def test_malformed_result_txt_is_infrastructure_failure(tmp_path):
    clone_dir = tmp_path / "OSWorld_Evol_gui"
    clone_dir.mkdir()
    _stage_result(tmp_path, "gui-if-001", "haiku", "t1", "garbage")
    with pytest.raises(RuntimeError, match="invalid"):
        _collect_per_task(str(clone_dir), "gui-if-001", ["t1"], "haiku")


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-0.01", "1.0001", "999"])
def test_nonfinite_result_is_infrastructure_failure(tmp_path, value):
    clone_dir = tmp_path / "OSWorld_Evol_gui"
    clone_dir.mkdir()
    _stage_result(tmp_path, "gui-if-001", "haiku", "t1", value)
    with pytest.raises(RuntimeError, match="invalid"):
        _collect_per_task_records(
            str(clone_dir), "gui-if-001", ["t1"], "haiku",
        )


def test_collect_per_task_reads_namespaced_cache(tmp_path):
    clone = tmp_path / "ScienceBoard_Evol_i1"
    clone.mkdir()
    leaf = (tmp_path / "eval_cache_scienceboard" / "v1" /
            "shadow_eval" / "screenshot" / "sonnet" / "ChimeraX:A-02")
    leaf.mkdir(parents=True)
    (leaf / "result.txt").write_text("1.0\n")
    got = _collect_per_task(str(clone), "v1", ["ChimeraX:A-02"], "sonnet",
                            benchmark="scienceboard")
    # got maps task_id -> pass/score (0 or 1); the SB-cache task was found.
    assert got["ChimeraX:A-02"] >= 0.9


def test_fractional_fitness_is_v2_only(tmp_path):
    clone = tmp_path / "OSWorld_Evol_gui"
    clone.mkdir()
    _stage_result(tmp_path, "v1", "haiku", "task", "0.75")
    v1 = _collect_per_task_records(
        str(clone), "v1", ["task"], "haiku", benchmark="osworld",
    )["task"]
    assert v1["raw_score"] == 0.75
    assert v1["passed"] == 1
    assert v1["fitness_score"] == 1.0

    leaf = (tmp_path / "eval_cache_osworld_v2" / "v2" /
            "shadow_eval" / "screenshot" / "haiku" / "task")
    leaf.mkdir(parents=True)
    (leaf / "result.txt").write_text("0.75")
    v2 = _collect_per_task_records(
        str(clone), "v2", ["task"], "haiku", benchmark="osworld_v2",
    )["task"]
    assert v2["raw_score"] == 0.75
    assert v2["passed"] == 0
    assert v2["fitness_score"] == 0.75


def test_v2_normalizes_only_float_dust_before_strict_binary(tmp_path):
    clone = tmp_path / "OSWorld_Evol_gui"
    clone.mkdir()
    leaf = (tmp_path / "eval_cache_osworld_v2" / "v2" /
            "shadow_eval" / "screenshot" / "haiku")
    for task_id, value in {
        "dust": "0.9999999999999999",
        "exact": "1.0",
        "real-partial": "0.9979",
    }.items():
        task_leaf = leaf / task_id
        task_leaf.mkdir(parents=True)
        (task_leaf / "result.txt").write_text(value)

    records = _collect_per_task_records(
        str(clone), "v2", ["dust", "exact", "real-partial"], "haiku",
        benchmark="osworld_v2",
    )
    assert records["dust"]["raw_score"] == 1.0
    assert records["dust"]["passed"] == 1
    assert records["exact"]["passed"] == 1
    assert records["real-partial"]["raw_score"] == 0.9979
    assert records["real-partial"]["passed"] == 0
    assert records["real-partial"]["fitness_score"] == 0.9979
