"""Shared score semantics for evaluator output and evolution bookkeeping."""

from __future__ import annotations

import math

FULL_CREDIT_ABS_TOL = 1e-12
LEGACY_PASS_THRESHOLD = 0.5


def canonicalize_full_credit(score: float) -> float:
    """Snap floating-point dust around full credit to exact ``1.0``.

    Weighted evaluators commonly produce ``0.9999999999999999`` when every
    component is satisfied.  The tolerance is deliberately much smaller than
    the precision of real OSWorld V2 partial rewards, so values such as
    0.9979 and 0.99 remain partial credit.
    """
    value = float(score)
    if math.isclose(
        value,
        1.0,
        rel_tol=0.0,
        abs_tol=FULL_CREDIT_ABS_TOL,
    ):
        return 1.0
    return value


def is_full_credit(score: float) -> bool:
    """Return whether ``score`` is semantically exact full credit."""
    return canonicalize_full_credit(score) == 1.0


def is_binary_success(score: float, benchmark: str) -> bool:
    """Apply the benchmark's binary-success contract.

    OSWorld V2 reports strict task completion: only normalized full credit is
    a binary success.  Older benchmarks retain their historical >= 0.5 pass
    threshold for backwards compatibility.
    """
    if benchmark == "osworld_v2":
        return is_full_credit(score)
    return float(score) >= LEGACY_PASS_THRESHOLD
