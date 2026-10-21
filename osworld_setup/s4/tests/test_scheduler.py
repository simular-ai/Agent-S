import pytest
from agent_s4.config.scheduler import DeterministicScheduler


@pytest.mark.parametrize("step,mig,cross,elim,expected", [
    (0,  5, 6, 10, "mutation"),
    (1,  5, 6, 10, "mutation"),
    (5,  5, 6, 10, "migration"),
    (6,  5, 6, 10, "crossover"),
    (10, 5, 6, 10, "elimination"),     # rarest wins over migration
    (12, 5, 6, 10, "crossover"),
    (15, 5, 6, 10, "migration"),
    (18, 5, 6, 10, "crossover"),
    (20, 5, 6, 10, "elimination"),     # rarest wins over migration
    (30, 5, 6, 10, "elimination"),     # rarest wins over migration AND crossover
    (5,  0, 6, 10, "mutation"),        # mig disabled by 0
    (10, 0, 6, 0,  "mutation"),        # both disabled
    (6,  0, 6, 0,  "crossover"),
    # Flipped cadences (crossover fires MORE often than migration): the
    # crossover/migration collisions at lcm(3,4)=12,24 must go to the
    # RARER op, migration — not crossover. This is the case the old
    # fixed-priority code got wrong.
    (12, 4, 3, 10, "migration"),       # rarer (cadence 4) beats crossover (3)
    (24, 4, 3, 10, "migration"),
    (3,  4, 3, 10, "crossover"),       # crossover-only step
    (4,  4, 3, 10, "migration"),       # migration-only step
    (20, 4, 3, 10, "elimination"),     # elim (rarest) beats migration
    (30, 4, 3, 10, "elimination"),     # elim beats crossover
    # Equal-cadence tie falls back to elim > crossover > migration.
    (12, 4, 4, 10, "crossover"),       # mig & cross both cadence 4 -> crossover
    (20, 10, 10, 10, "elimination"),   # all three tie -> elimination
])
def test_deterministic_choose_op(step, mig, cross, elim, expected):
    sched = DeterministicScheduler(
        migration_cadence=mig,
        crossover_cadence=cross,
        elimination_cadence=elim,
    )
    assert sched.choose_op(step, state=None) == expected


import random
from agent_s4.config.scheduler import ProbabilisticScheduler


class _FakeState:
    def __init__(self, seed):
        self.rng = random.Random(seed)


def test_probabilistic_choose_op_uses_weights():
    sched = ProbabilisticScheduler({
        "mutation": 0.7, "migration": 0.15,
        "crossover": 0.1, "elimination": 0.05,
    })
    state = _FakeState(seed=42)
    # Sample many; observed proportions should roughly match weights
    counts = {"mutation": 0, "migration": 0, "crossover": 0, "elimination": 0}
    for step in range(2000):
        op = sched.choose_op(step, state)
        counts[op] += 1
    # Loose bounds (sampling variance); mutation should dominate
    assert counts["mutation"] > 1200
    assert counts["elimination"] < 200


def test_probabilistic_normalizes_weights():
    sched = ProbabilisticScheduler({"a": 2, "b": 2})  # un-normalized
    assert abs(sum(sched.probs) - 1.0) < 1e-9
    assert sched.kinds == ["a", "b"]
