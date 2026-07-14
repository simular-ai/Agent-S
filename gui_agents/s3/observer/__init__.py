"""Observation-only Agent S3 runtime.

This package deliberately contains no desktop action executor.  It can capture a
screen and ask models for a typed proposal, but it cannot click or type.
"""

from .actions import ActionCall, ActionParseError, ActionProposal, parse_action_call

__all__ = [
    "ActionCall",
    "ActionParseError",
    "ActionProposal",
    "parse_action_call",
]
