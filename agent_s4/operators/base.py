"""Operator ABC — the pluggable unit of the v2 driver."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from agent_s4.state import LoopState


class Operator(ABC):
    kind: ClassVar[
        str
    ]  # "mutation" | "migration" | "crossover" | "elimination" | "reevaluation"

    @abstractmethod
    def apply(self, state: "LoopState") -> None: ...
