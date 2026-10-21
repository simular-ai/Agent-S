"""Unit tests for LoopState.

v2: LoopState is a thin per-step context. The runtime fan-out lives in
the driver, so peers() / last_fire / record_fire are gone. What's left
is current_config() and clone_dir_for() — both pure helpers.
"""
import random
import sqlite3
from dataclasses import dataclass, field

from agent_s4.config.island import IslandConfig
from agent_s4.state import LoopState


@dataclass
class _FakeExperimentConfig:
    islands: list[IslandConfig] = field(default_factory=list)
    benchmark: str = "osworld"


def _make_state(tmp_path, current_island, islands, benchmark="osworld"):
    conn = sqlite3.connect(":memory:")
    cfg = _FakeExperimentConfig(islands=islands, benchmark=benchmark)
    return LoopState(
        conn=conn,
        current_island=current_island,
        step=0,
        rng=random.Random(42),
        config=cfg,
        clone_dir=tmp_path / f"OSWorld_Evol_{current_island}",
        base_dir=tmp_path,
        eval_cache_dir=tmp_path / "eval_cache",
    )


def _mk_island(name):
    return IslandConfig(
        name=name,
        evaluator_harness="claude_code",
        eval_model="haiku",
    )


def test_current_config_returns_current_island(tmp_path):
    islands = [_mk_island("gui"), _mk_island("cli")]
    state = _make_state(tmp_path, "cli", islands)
    assert state.current_config().name == "cli"


def test_clone_dir_for_peer(tmp_path):
    islands = [_mk_island("gui"), _mk_island("cli")]
    state = _make_state(tmp_path, "gui", islands)
    peer = [i for i in state.config.islands if i.name == "cli"][0]
    expected = tmp_path / "OSWorld_Evol_cli"
    assert state.clone_dir_for(peer) == expected


def test_clone_dir_osworld(tmp_path):
    islands = [_mk_island("i1")]
    state = _make_state(tmp_path, "i1", islands, benchmark="osworld")
    island = state.config.islands[0]
    assert state.clone_dir_for(island) == tmp_path / "OSWorld_Evol_i1"


def test_clone_dir_scienceboard(tmp_path):
    islands = [_mk_island("i1")]
    state = _make_state(tmp_path, "i1", islands, benchmark="scienceboard")
    island = state.config.islands[0]
    assert state.clone_dir_for(island) == tmp_path / "ScienceBoard_Evol_i1"
