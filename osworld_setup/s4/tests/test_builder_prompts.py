"""Snapshot tests for the per-island builder prompt files under prompts/.

Scratch VM instructions live in prompts/scratch_vm_addendum.md now,
appended by orchestrator.assemble_prompt when scratch_vm_enabled=True.
The per-island strategy hint still lives in the base prompt. See
tests/unit/test_assemble_prompt.py for the assembly-time coverage.
"""
from pathlib import Path

import pytest

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
# Files that aren't per-island builder prompts and shouldn't be iterated
# as such.
_NON_ISLAND_PROMPT_STEMS = {
    "crossover", "scratch_vm_addendum", "scratch_vm_addendum_scienceboard",
    "scienceboard_eval_format_addendum",
}


@pytest.fixture
def prompts():
    return {
        p.stem: p.read_text()
        for p in PROMPTS_DIR.glob("*.md")
        if p.stem not in _NON_ISLAND_PROMPT_STEMS
    }


def test_base_prompts_do_not_contain_scratch_vm_section(prompts):
    """Regression: base prompts must not carry the Scratch VM section;
    it lives in the addendum and is appended only when the config flag
    is on. Otherwise --no-scratch-vm runs would burn tokens on
    instructions for a capability they don't have."""
    for island, prompt in prompts.items():
        assert "## Scratch VM" not in prompt, \
            f"prompts/{island}.md still carries '## Scratch VM' section"


def test_scratch_vm_addendum_has_expected_content():
    addendum = (PROMPTS_DIR / "scratch_vm_addendum.md").read_text()
    assert "## Scratch VM" in addendum
    assert "from scratch_vm import" in addendum
    assert "redeploy_interface" in addendum
    assert "ScratchVmUnavailable" in addendum
