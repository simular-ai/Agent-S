import subprocess

import pytest

from agent_s4.git_utils import git_archive_variant


@pytest.mark.parametrize(
    "source_name,dest_name,tag",
    [
        ("source", "destination", "evol/x/if-001"),
        ("source with spaces", "destination", "evol/x/if-001"),
        ("source", "destination with spaces", "evol/x/if-001"),
        ("source's island", "destination's copy", "evol/x/if-001"),
    ],
)
def test_git_archive_variant_preserves_paths(tmp_path, source_name, dest_name, tag):
    """Migration and crossover can extract an interface from any workspace path."""
    source = tmp_path / source_name
    inner = source / "shadow_interface"
    inner.mkdir(parents=True)
    subprocess.run(["git", "init", str(inner)], check=True, capture_output=True)
    interface = inner / "tool.py"
    interface.write_text("print('tagged interface')\n")
    subprocess.run(
        ["git", "-C", str(inner), "add", "tool.py"], check=True, capture_output=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(inner),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "interface",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(inner), "tag", tag], check=True, capture_output=True
    )
    interface.write_text("print('uncommitted interface')\n")

    destination = tmp_path / dest_name
    git_archive_variant(str(source), tag, str(destination))

    assert (
        destination / "shadow_interface" / "tool.py"
    ).read_text() == "print('tagged interface')\n"
    assert interface.read_text() == "print('uncommitted interface')\n"
