from pathlib import Path

import pytest

from agent_s4.config.experiment import ExperimentConfig
from agent_s4.config.island import IslandConfig
from agent_s4 import orchestrator


def _config(tmp_path: Path) -> ExperimentConfig:
    seed = tmp_path / "seed"
    seed.mkdir()
    prompt = tmp_path / "prompt.md"
    prompt.write_text("prompt")
    return ExperimentConfig(
        islands=[IslandConfig(
            name="haiku", evaluator_harness="claude_code", eval_model="haiku",
        )],
        benchmark="osworld_v2", seed_dir=seed, prompt_template=prompt,
    )


def _provenance():
    return {
        "benchmark": "osworld_v2",
        "source_template_commit": "abc123",
        "source_template_tag": "v2026.08.08",
        "release": "osworld-v2-2026.08.08",
        "release_pin_sha256": "pin-hash",
    }


def _write_marker(clone: Path, config: ExperimentConfig, values: dict):
    payload = {
        "seed_dir": str(config.seed_dir),
        "prompt_template": str(config.prompt_template),
        **values,
    }
    (clone / ".shadow_evol_init").write_text(
        "".join(f"{key}={value}\n" for key, value in payload.items())
    )


def test_v2_resume_accepts_exact_source_provenance(tmp_path, monkeypatch):
    config = _config(tmp_path)
    clone = tmp_path / "OSWorldV2_Evol_haiku"
    clone.mkdir()
    _write_marker(clone, config, _provenance())
    monkeypatch.setattr(orchestrator, "_template_provenance", lambda *args: _provenance())
    orchestrator.initialize_clone_v2(
        tmp_path / "OSWorldV2_Evol", clone, config, config.islands[0], "prompt",
    )


@pytest.mark.parametrize("mutation", [
    {"source_template_commit": "old"},
    {"source_template_tag": "v-old"},
    {"release": "old-release"},
])
def test_v2_resume_rejects_stale_source_provenance(
        tmp_path, monkeypatch, mutation):
    config = _config(tmp_path)
    clone = tmp_path / "OSWorldV2_Evol_haiku"
    clone.mkdir()
    values = {**_provenance(), **mutation}
    _write_marker(clone, config, values)
    monkeypatch.setattr(orchestrator, "_template_provenance", lambda *args: _provenance())
    with pytest.raises(RuntimeError, match="stale or missing release provenance"):
        orchestrator.initialize_clone_v2(
            tmp_path / "OSWorldV2_Evol", clone, config,
            config.islands[0], "prompt",
        )


def test_v2_resume_rejects_missing_marker(tmp_path, monkeypatch):
    config = _config(tmp_path)
    clone = tmp_path / "OSWorldV2_Evol_haiku"
    clone.mkdir()
    monkeypatch.setattr(orchestrator, "_template_provenance", lambda *args: _provenance())
    with pytest.raises(RuntimeError, match="no .shadow_evol_init"):
        orchestrator.initialize_clone_v2(
            tmp_path / "OSWorldV2_Evol", clone, config,
            config.islands[0], "prompt",
        )


def test_v2_resume_rejects_gated_task_leak(tmp_path, monkeypatch):
    config = _config(tmp_path)
    template = tmp_path / "OSWorldV2_Evol"
    template.mkdir()
    clone = tmp_path / "OSWorldV2_Evol_haiku"
    task_root = clone / "evaluation_examples" / "task_class"
    task_root.mkdir(parents=True)
    (task_root / "task_001.py").write_text("gold = True\n")
    _write_marker(clone, config, _provenance())
    monkeypatch.setattr(orchestrator, "_template_provenance", lambda *args: _provenance())
    with pytest.raises(RuntimeError, match="gated task definition"):
        orchestrator.initialize_clone_v2(
            template, clone, config, config.islands[0], "prompt",
        )
