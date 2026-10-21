"""Filesystem anchors for Agent S4.

The Python package lives at ``agent_s4/``. Everything else a run reads
or writes is split between two directories:

- the shared root (``osworld_setup/s4/`` by default, ``SHADOW_EVOL_SHARED_ROOT``
  to override) holds what every benchmark uses: the vendored Prime runtime
  (``agent_harness/``), the VM asset cache (``vm_assets/``), the seed
  interfaces and the generic Builder prompts;
- the project root (``SHADOW_EVOL_PROJECT_ROOT``, default: the shared root) is
  the run workspace and holds the benchmark's own prompts, task splits, data,
  release pin, benchmark checkouts and every run artefact. The ScienceBoard
  launchers point it at ``scienceboard_setup/s4/``.
"""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SHARED_ROOT = Path(
    os.environ.get("SHADOW_EVOL_SHARED_ROOT")
    or PACKAGE_ROOT.parent / "osworld_setup" / "s4"
).resolve()
PROJECT_ROOT = Path(os.environ.get("SHADOW_EVOL_PROJECT_ROOT") or SHARED_ROOT).resolve()


def require_shared_root() -> None:
    """Stop with setup instructions when the shared root is missing.

    A pip-installed agent-s4 carries only the code; the shared root comes
    from an Agent-S checkout."""
    if (SHARED_ROOT / "prompts").is_dir():
        return
    raise SystemExit(
        f"Agent S4 setup files not found at {SHARED_ROOT}.\n"
        "The agent-s4 package contains only the code. Clone "
        "https://github.com/simular-ai/Agent-S and set "
        "SHADOW_EVOL_SHARED_ROOT=<clone>/osworld_setup/s4, or install the "
        "package from the clone with `pip install -e ./agent_s4`."
    )


def prompt_path(name: str) -> Path:
    """A Builder prompt file: the project root's copy if it has one, else the
    shared root's."""
    own = PROJECT_ROOT / "prompts" / name
    return own if own.is_file() else SHARED_ROOT / "prompts" / name
