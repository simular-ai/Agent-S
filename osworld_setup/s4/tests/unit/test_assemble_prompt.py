"""Tests for orchestrator.assemble_prompt_v2 — additive composition of
base prompt + config-gated addenda.

v2: seed_dir and prompt_template live on ExperimentConfig (shared across
all islands), so assemble_prompt_v2 takes only the config.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_s4 import paths
from agent_s4.config.experiment import ExperimentConfig
from agent_s4.config.island import IslandConfig
from agent_s4.orchestrator import assemble_prompt_v2


REPO_ROOT = Path(__file__).resolve().parents[2]
SB_ROOT = REPO_ROOT.parents[1] / "scienceboard_setup" / "s4"


@pytest.fixture(autouse=True)
def _scienceboard_project_root(monkeypatch):
    """Run as the ScienceBoard launchers do: the SB addenda come from the SB
    project root, the generic ones fall back to the shared root."""
    monkeypatch.setattr(paths, "PROJECT_ROOT", SB_ROOT)


def _island(name: str = "combined") -> IslandConfig:
    return IslandConfig(
        name=name,
        evaluator_harness="prime_agent",
        eval_model="openai/qwen",
    )


def _config(scratch_vm_enabled: bool,
            prompt_template: Path | None = None,
            benchmark: str = "osworld") -> ExperimentConfig:
    return ExperimentConfig(
        islands=[_island()],
        prompt_template=prompt_template or (REPO_ROOT / "prompts" / "combined.md"),
        seed_dir=REPO_ROOT / "seeds" / "combined",
        scratch_vm_enabled=scratch_vm_enabled,
        benchmark=benchmark,
        train_set_file="train.json",
        val_set_file="val.json",
    )


def test_base_prompt_has_no_scratch_vm_section():
    """The base prompt should not contain the Scratch VM section — it
    lives only in the addendum."""
    for name in ("combined",):
        body = (REPO_ROOT / "prompts" / f"{name}.md").read_text()
        assert "## Scratch VM" not in body, (
            f"prompts/{name}.md still has a Scratch VM section — it should "
            f"live only in prompts/scratch_vm_addendum.md"
        )


def test_assemble_without_scratch_vm_returns_base_only():
    out = assemble_prompt_v2(_config(scratch_vm_enabled=False))
    assert "## Scratch VM" not in out
    assert "## Eval Results" in out       # base still present
    assert "## Constraints" in out
    assert out.endswith("\n")             # clean trailing newline


def test_assemble_with_scratch_vm_appends_addendum():
    out = assemble_prompt_v2(_config(scratch_vm_enabled=True))
    assert "## Scratch VM" in out
    # Must come after Constraints/Goal — i.e. appended at the end, not
    # injected mid-document.
    assert out.index("## Scratch VM") > out.index("## Constraints")
    # The addendum's key capability markers should all be present.
    for marker in ("from scratch_vm import", "redeploy_interface",
                   "ScratchVmUnavailable"):
        assert marker in out, f"addendum missing marker {marker!r}"


def test_assemble_joins_base_and_addendum_with_single_blank_line():
    """No duplicate blank lines at the join point."""
    out = assemble_prompt_v2(_config(scratch_vm_enabled=True))
    # Locate the addendum header and check the two lines before it
    idx = out.index("## Scratch VM")
    prefix = out[:idx]
    # Should end with exactly two newlines (one from base's trailing \n,
    # one blank separator). Never three (would be a double blank).
    assert prefix.endswith("\n\n")
    assert not prefix.endswith("\n\n\n"), (
        "assemble_prompt_v2 produced a double blank line at the join"
    )


def test_sb_addendum_included_only_for_scienceboard():
    """The ScienceBoard eval-format addendum is appended iff
    benchmark == 'scienceboard'. Anchor on the distinctive filename the
    addendum instructs the builder to read."""
    ANCHOR = "scienceboard_evaluation.json"
    sci = assemble_prompt_v2(_config(scratch_vm_enabled=False,
                                     benchmark="scienceboard"))
    osw = assemble_prompt_v2(_config(scratch_vm_enabled=False,
                                     benchmark="osworld"))
    assert ANCHOR in sci
    assert ANCHOR not in osw
    # Appended at the end (after the base's Constraints section), not
    # injected mid-document.
    assert sci.index(ANCHOR) > sci.index("## Constraints")


def test_sb_addendum_composes_with_scratch_vm():
    """Both addenda are independently gated and must compose — neither is
    mutually exclusive with the other."""
    out = assemble_prompt_v2(_config(scratch_vm_enabled=True,
                                     benchmark="scienceboard"))
    assert "## Scratch VM" in out
    assert "scienceboard_evaluation.json" in out


def test_scratch_vm_addendum_is_benchmark_specific():
    """The appended Scratch VM addendum differs by benchmark: ScienceBoard
    names its apps and launches them via execute_python_command (no
    SetupController); OSWorld exposes get_setup_controller. Both keep the
    common builder API (get_controller/redeploy_interface)."""
    sci = assemble_prompt_v2(_config(scratch_vm_enabled=True,
                                     benchmark="scienceboard"))
    osw = assemble_prompt_v2(_config(scratch_vm_enabled=True,
                                     benchmark="osworld"))
    # ScienceBoard-specific
    assert "ChimeraX" in sci
    assert "execute_python_command" in sci
    assert "get_setup_controller" not in sci   # SB has no SetupController
    # OSWorld-specific
    assert "get_setup_controller" in osw
    assert "ChimeraX" not in osw
    # Common builder API in both
    for marker in ("from scratch_vm import", "redeploy_interface",
                   "ScratchVmUnavailable"):
        assert marker in sci, f"SB addendum missing {marker!r}"
        assert marker in osw, f"OSWorld addendum missing {marker!r}"


def test_osworld_prompt_unchanged_by_sb_gating():
    """Byte-for-byte: the OSWorld prompt (benchmark=osworld) must equal the
    raw base prompt + trailing newline — no SB text leaks in."""
    base = (REPO_ROOT / "prompts" / "combined.md").read_text().rstrip() + "\n"
    out = assemble_prompt_v2(_config(scratch_vm_enabled=False,
                                     benchmark="osworld"))
    assert out == base
