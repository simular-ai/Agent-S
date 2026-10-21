"""Benchmark descriptor + workspace-artifact namespacing.

Pure data + one pure function. Imports nothing from the package, so it can be
imported anywhere without risking an import cycle. The benchmark->evaluator-class
mapping deliberately lives in agent_s4/harness/__init__.py, not here.
"""

from __future__ import annotations

import os
import re

# template / base are distinct benchmark checkout directory names taken
# verbatim.  OSWorld v1 currently uses registered git submodules; OSWorld V2
# and ScienceBoard use release-pinned local clones that are intentionally
# ignored by the parent repository.  Callers must not infer git ownership from
# these names.
# tag is the suffix ns() applies to the four bare run-workspace constants in
# constants.py (db, eval_cache, builder_logs, migrate_transit). osworld == ""
# so every osworld path is the current one (back-compat).
BENCHMARKS = {
    "osworld": {"template": "OSWorld_Evol", "base": "OSWorld_Base", "tag": ""},
    "scienceboard": {
        "template": "ScienceBoard_Evol",
        "base": "ScienceBoard_Base",
        "tag": "_scienceboard",
    },
    # OSWorld V2 (xlang-ai/OSWorld-V2). Its DesktopEnv kwargs are a superset of
    # the v1 fork's, so eval/osworld.py drives it unchanged; only the template /
    # base directory names and the artifact tag differ.
    "osworld_v2": {
        "template": "OSWorldV2_Evol",
        "base": "OSWorldV2_Base",
        "tag": "_osworld_v2",
    },
}


def ns(name: str, benchmark: str) -> str:
    """Namespace a run-workspace artifact name by benchmark so OSWorld and
    ScienceBoard never share workspace state. osworld -> bare name; others ->
    tag inserted before any extension ('shadow_evolution.db' ->
    'shadow_evolution_scienceboard.db', 'eval_cache' -> 'eval_cache_scienceboard')."""
    tag = BENCHMARKS[benchmark]["tag"]
    run_namespace = os.environ.get("SHADOW_EVOL_RUN_NAMESPACE", "").strip()
    if run_namespace:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_namespace):
            raise ValueError(
                "SHADOW_EVOL_RUN_NAMESPACE must contain only letters, digits, "
                f"underscores, or hyphens, got {run_namespace!r}"
            )
        tag += f"_{run_namespace}"
    if not tag:
        return name
    stem, dot, ext = name.partition(".")
    return f"{stem}{tag}{dot}{ext}"
