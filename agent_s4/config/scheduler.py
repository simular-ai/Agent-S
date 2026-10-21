"""Scheduler — picks one operator per global step.

The Scheduler is the only point of pluggability in the driver's dispatch
loop. DeterministicScheduler (cadence-based with rarest-wins priority) is
the default. ProbabilisticScheduler samples per step from a fixed
weight distribution. Custom subclasses can implement any other scheme
(adaptive, threshold-based, etc.).

A cadence of 0 means "never fire" — useful for ablations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_s4.state import LoopState


class Scheduler(ABC):
    @abstractmethod
    def choose_op(self, step: int, state: "LoopState | None") -> str:
        """Return one of: 'mutation', 'migration', 'crossover', 'elimination'."""
        ...


class DeterministicScheduler(Scheduler):
    """Cadence-based with rarest-wins priority on collisions.

    Default cadences match today's intent: migration=5, crossover=6,
    elimination=10. Mutation is the fallback when no other cadence fires.
    Cadence of 0 means "never fire".

    When more than one cadence divides the same step (a collision), the
    *rarest* operator wins — the one with the largest cadence, since it
    fires least often and is the harder slot to give up. Equal cadences
    fall back to a fixed tie-break order: elimination > crossover >
    migration. (For the default 10/6/5 cadences this is identical to a
    fixed elim>crossover>migration priority, because crossover and
    migration only ever collide on a step that elimination also claims;
    the rarity sort only diverges once you flip the usual ordering, e.g.
    crossover fires more often than migration.)
    """

    def __init__(
        self,
        migration_cadence: int = 5,
        crossover_cadence: int = 6,
        elimination_cadence: int = 10,
    ):
        self.migration_cadence = migration_cadence
        self.crossover_cadence = crossover_cadence
        self.elimination_cadence = elimination_cadence

    def choose_op(self, step, state):
        if step <= 0:
            return "mutation"
        # (cadence, tie_break_rank, op) for every operator firing this step.
        # Lower tie_break_rank wins an equal-cadence tie.
        firing = [
            (cadence, rank, op)
            for cadence, rank, op in (
                (self.elimination_cadence, 0, "elimination"),
                (self.crossover_cadence, 1, "crossover"),
                (self.migration_cadence, 2, "migration"),
            )
            if cadence > 0 and step % cadence == 0
        ]
        if not firing:
            return "mutation"
        # Rarest wins: largest cadence first, then the tie-break rank.
        firing.sort(key=lambda c: (-c[0], c[1]))
        return firing[0][2]


class ProbabilisticScheduler(Scheduler):
    """Weighted random sample per step. Weights are normalized internally;
    any positive-valued weights are accepted. State.rng provides the
    randomness so seeded runs are reproducible.
    """

    def __init__(self, weights: dict[str, float]):
        total = sum(weights.values())
        if total <= 0:
            raise ValueError(f"weights must sum to > 0, got {total}")
        self.kinds = list(weights)
        self.probs = [weights[k] / total for k in self.kinds]

    def choose_op(self, step, state):
        if state is None or not hasattr(state, "rng"):
            raise ValueError("ProbabilisticScheduler requires a LoopState with .rng")
        return state.rng.choices(self.kinds, weights=self.probs)[0]
