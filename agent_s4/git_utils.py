"""Git helpers for inner shadow_interface/ repos and outer island clones.

Each per-island clone (OSWorld_Evol_<name>) has TWO git repos:
- the outer clone repo (tracks the whole fork), and
- the inner shadow_interface/ repo (tracks just the evolved code,
  tagged per-variant as evol/<island>/if-NNN).

These helpers operate on the inner repo unless noted otherwise.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess

logger = logging.getLogger("agent_s4.git_utils")


def git(args_list: list[str], cwd: str | None = None) -> tuple[str, int]:
    """Run `git <args>` in cwd. Returns (stdout, returncode). Logs non-
    zero returncodes except the 'nothing to commit' case."""
    result = subprocess.run(
        ["git"] + args_list,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0 and "nothing to commit" not in result.stdout:
        logger.warning(
            "git %s failed in %s: %s",
            " ".join(args_list),
            cwd,
            result.stderr.strip()[:200],
        )
    return result.stdout.strip(), result.returncode


def shadow_path(clone_dir: str) -> str:
    """Path to the inner shadow_interface git repo for a clone."""
    return os.path.join(clone_dir, "shadow_interface")


def git_tag_variant(clone_dir: str, tag_name: str, message: str) -> None:
    """Commit only evolved state and tag it in the island repository.

    Some V2 checkouts do not have a nested ``shadow_interface/.git``. Running
    ``git add -A`` from that subdirectory still scans/stages the entire outer
    repository, including multi-GB ``eval_results`` artifacts. Use explicit
    root-relative pathspecs so evidence caches can never enter a variant tag.

    Every step raises rather than warning. ``git`` itself is tolerant by
    contract -- ``status`` and ``tag`` callers read a non-zero return as an
    answer -- but here a failure is silent data loss: if the commit does not
    land, ``tag -f`` still succeeds and pins the child's name onto the
    parent's commit. Nothing downstream can tell. The held-out runner
    materializes variants from their tags, so it would score the parent's
    interface and label it the child, and the next step's checkout would
    reset the Builder's work out of the tree. Failing here instead leaves
    that work staged and recoverable, and costs only the step.
    """
    for args in (
        ["add", "-A", "--", "shadow_interface", "notebook.jsonl"],
        ["commit", "-m", message, "--allow-empty"],
        ["tag", "-f", tag_name],
    ):
        _, returncode = git(args, cwd=clone_dir)
        if returncode != 0:
            raise RuntimeError(
                f"git {args[0]} failed in {clone_dir} while tagging "
                f"{tag_name}; refusing to tag a variant whose commit did not "
                "land. The Builder's output is still staged."
            )


def git_checkout_variant(clone_dir: str, tag: str) -> None:
    """Checkout a variant tag in the inner shadow_interface git repo.

    Force-cleans dirty state first. Every call site (mutation start,
    crossover start, fill_eval) expects a clean tree — any uncommitted
    modifications or untracked files at this point are stale artifacts
    from a killed prior run and should be discarded.
    """
    inner = shadow_path(clone_dir)

    status_out, _ = git(["status", "--porcelain"], cwd=inner)
    if status_out.strip():
        dirty_lines = status_out.strip().splitlines()
        logger.warning(
            "Discarding stale state in %s before checkout of %s: %s",
            inner,
            tag,
            dirty_lines[:5],
        )

    _, rc1 = git(["reset", "--hard", tag], cwd=inner)
    _, rc2 = git(["clean", "-fdx"], cwd=inner)
    if rc1 != 0 or rc2 != 0:
        raise RuntimeError(f"Failed to checkout {tag} in {inner}")


def git_archive_variant(source_clone: str, tag: str, dest_dir: str) -> None:
    """Extract a tag's shadow_interface content into dest_dir/shadow_interface/.

    Reads from the source's inner git (.git inside shadow_interface/) — safe
    even if the source's working tree is mid-mutation.
    """
    dest_shadow = os.path.join(dest_dir, "shadow_interface")
    os.makedirs(dest_shadow, exist_ok=True)
    inner = shadow_path(source_clone)
    proc = subprocess.run(
        f"git -C {inner} archive {tag} | tar -x -C {dest_shadow}",
        shell=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git archive failed: {proc.stderr}")


def revert_out_of_scope(clone_dir: str) -> list[str]:
    """Revert any builder modifications to files OUTSIDE shadow_interface/
    and notebook.jsonl. Uses the outer clone's git to reset tracked files.
    Returns list of reverted paths for logging."""
    protected = [
        "scripts/",
        "mm_agents/",
        "desktop_env/",
        "evaluation_examples/",
        "CLAUDE.md",
        "lib_run_single.py",
    ]
    out, _ = git(["status", "--porcelain"] + protected, cwd=clone_dir)
    modified = [line.split()[-1] for line in out.splitlines() if line.strip()]
    if modified:
        logger.warning(
            "Builder modified out-of-scope files in %s: %s",
            os.path.basename(clone_dir),
            modified,
        )
        git(["checkout", "HEAD", "--"] + protected, cwd=clone_dir)
    return modified


def next_variant_num_from_population_and_tags(
    population: list[dict],
    clone_dir: str | None = None,
) -> int:
    """Return the next variant number, avoiding collisions with existing
    inner git tags.

    Checks BOTH the population AND inner git tags (if clone_dir given)
    so eliminated variants' numbers don't get reused.
    """
    nums: list[int] = []
    for v in population:
        if "-if-" in v["id"]:
            try:
                nums.append(int(v["id"].split("-if-")[1]))
            except ValueError:
                pass
    if clone_dir:
        inner = shadow_path(clone_dir)
        try:
            out, rc = git(["tag"], cwd=inner)
            if rc == 0:
                for t in out.strip().split("\n"):
                    m = re.search(r"/if-(\d+)$", t)
                    if m:
                        nums.append(int(m.group(1)))
        except Exception:
            pass
    return max(nums, default=0) + 1
