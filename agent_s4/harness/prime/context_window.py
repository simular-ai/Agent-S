"""Which context window a Prime Agent run measures against.

The evaluator and the Builder both read ``context_window_tokens`` from an
interface's ``context_compaction.json`` so a variant can evolve the window it
reasons in. That plumbing landed after some runs had already started, and the
interfaces those runs carry hold a value that never reached Prime: they were
scored at whatever window Prime's own model catalog advertises. Resuming such
a run with the field now honoured would score its later steps at the
interface's window while their parents were measured at the catalog window --
the same mid-flight configuration swap that invalidated the 128K run's
selection signal, and worth more tasks than the evolution effect being
measured.

``SHADOW_PRIME_CONTEXT_WINDOW_SOURCE`` pins which behaviour a run uses:

    interface   honour the interface's field. The default, and what every run
                started after the plumbing landed wants.
    catalog     ignore the field and keep Prime's catalog window, so a run
                started before it was plumbed can be resumed without changing
                what it measures.

An unset variable means ``interface``, so nothing changes for runs that never
set it.
"""

from __future__ import annotations

import os

ENV_VAR = "SHADOW_PRIME_CONTEXT_WINDOW_SOURCE"
SOURCES = ("interface", "catalog")


def context_window_source() -> str:
    """Return the pinned source, rejecting a typo rather than guessing."""
    raw = os.environ.get(ENV_VAR, "").strip().lower() or "interface"
    if raw not in SOURCES:
        raise ValueError(f"{ENV_VAR} must be one of {list(SOURCES)}, got {raw!r}")
    return raw


def interface_window_is_honoured() -> bool:
    """False when the run is pinned to Prime's own model-catalog window."""
    return context_window_source() == "interface"
