from unittest.mock import call, patch

import pytest

from agent_s4.git_utils import git_tag_variant


def test_git_tag_variant_stages_only_evolved_paths():
    with patch("agent_s4.git_utils.git", return_value=("", 0)) as git:
        git_tag_variant("/island", "evol/x/if-001", "mutation")

    assert git.call_args_list == [
        call(
            ["add", "-A", "--", "shadow_interface", "notebook.jsonl"],
            cwd="/island",
        ),
        call(["commit", "-m", "mutation", "--allow-empty"], cwd="/island"),
        call(["tag", "-f", "evol/x/if-001"], cwd="/island"),
    ]


def test_git_tag_variant_refuses_to_tag_when_the_commit_fails():
    """A tag pinned onto the parent's commit is undetectable downstream.

    This is what happened to prime25v15r4-if-007: the clone had no git
    identity to inherit, ``commit`` failed, and ``tag -f`` then pinned
    if-007 onto the seed commit. The held-out runner materializes variants
    from their tags, so it would have scored the seed and called it if-007.
    """
    def fake_git(args, cwd=None):
        return ("", 128 if args[0] == "commit" else 0)

    with patch("agent_s4.git_utils.git", side_effect=fake_git) as git:
        with pytest.raises(RuntimeError, match="commit did not land"):
            git_tag_variant("/island", "evol/x/if-007", "mutation")

    # The tag must never be attempted once the commit has failed.
    assert [c.args[0][0] for c in git.call_args_list] == ["add", "commit"]


@pytest.mark.parametrize("failing", ["add", "commit", "tag"])
def test_git_tag_variant_raises_on_any_failing_step(failing):
    def fake_git(args, cwd=None):
        return ("", 1 if args[0] == failing else 0)

    with patch("agent_s4.git_utils.git", side_effect=fake_git):
        with pytest.raises(RuntimeError, match=f"git {failing} failed"):
            git_tag_variant("/island", "evol/x/if-007", "mutation")
