"""Tests for v2 migration operator behavior.

Verifies the two load-bearing changes from the island-evolution-v2 plan
Task 8:

1. Ring topology — successor_of / predecessor_of are pure helpers that
   define the round-robin push order over islands[i] -> islands[(i+1) % N].
2. Phase 1 order of operations — the source-side phase MUST write the
   transit dir (atomic tmp+rename) BEFORE deleting the source variant
   from the DB. If the loop crashes between those two side-effects, the
   presence of the transit dir is the marker that says "phase 1 done for
   this island" and the resume dispatch will skip the pick.

The tests intentionally mock every external dependency (git, DB,
tournament, harnesses). They assert observable ordering and topology,
not internal data flow.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. Ring topology
# ---------------------------------------------------------------------------


def test_migration_ring_topology_3_islands():
    """With islands [haiku, sonnet, qwen], A->successor mapping is:
    haiku->sonnet, sonnet->qwen, qwen->haiku."""
    from agent_s4.operators.migration import successor_of
    islands = ["haiku", "sonnet", "qwen"]
    assert successor_of("haiku", islands) == "sonnet"
    assert successor_of("sonnet", islands) == "qwen"
    assert successor_of("qwen", islands) == "haiku"


def test_migration_ring_predecessor_3_islands():
    """Predecessor is the inverse of successor on the same ring."""
    from agent_s4.operators.migration import predecessor_of
    islands = ["haiku", "sonnet", "qwen"]
    assert predecessor_of("haiku", islands) == "qwen"
    assert predecessor_of("sonnet", islands) == "haiku"
    assert predecessor_of("qwen", islands) == "sonnet"


def test_migration_ring_2_islands():
    """N=2 degenerates to swap: successor and predecessor coincide."""
    from agent_s4.operators.migration import predecessor_of, successor_of
    islands = ["a", "b"]
    assert successor_of("a", islands) == "b"
    assert successor_of("b", islands) == "a"
    assert predecessor_of("a", islands) == "b"
    assert predecessor_of("b", islands) == "a"


# ---------------------------------------------------------------------------
# 2. Phase 1 ordering: transit written BEFORE delete_variant
# ---------------------------------------------------------------------------


def _make_phase1_state(tmp_path: Path):
    """Build a MagicMock LoopState that exercises Migration.apply_phase1
    without needing a real DB, clone dir, or config object."""
    state = MagicMock()
    state.step = 5
    state.current_island = "haiku"
    state.clone_dir = tmp_path / "clone_haiku"
    state.base_dir = tmp_path

    cfg = state.config
    cfg.benchmark = "osworld"
    cfg.selection_signal = "val"
    cfg.migration_mode = "move"
    cfg.eval_timeout = 60
    cfg.region = "us-east-1"
    cfg.resolve_eval_model.return_value = "haiku"

    state.current_config.return_value = MagicMock(name="island_cfg")
    state.clone_dir.mkdir(parents=True, exist_ok=True)
    return state


def test_migration_phase1_writes_transit_before_delete(tmp_path):
    """Order of operations matters for crash recovery.

    Phase 1 must:
      1. Write the transit dir at base_dir/_migrate_transit/<island>/
         via tmp+rename so a partial write is never observable.
      2. THEN call delete_variant on the picked source row.

    If we crash between (1) and (2), the next dispatch finds the transit
    dir already present and skips the tournament-pick. If we crashed
    BEFORE (1), the source row is still there and gets re-picked. Either
    way the migration is idempotent.

    This test records the existence of the transit dir at the moment
    delete_variant is called; it must be True.
    """
    from agent_s4.operators.migration import Migration
    from agent_s4.constants import MIGRATE_TRANSIT_DIR

    state = _make_phase1_state(tmp_path)
    transit_dir = tmp_path / MIGRATE_TRANSIT_DIR / "haiku"

    picked = {
        "id": "haiku-if-007",
        "tag": "evol/haiku/if-007",
        "score": 3,
        "val_score": 2,
        "eval_level": 0,
        "generation": 1,
        "island": "haiku",
    }

    # Fake git_archive: simulate the real call by actually creating
    # the dest_dir tree (the implementation writes to a .tmp dir then
    # renames). We only need the test's transit_dir to exist after
    # rename — the implementation does the rename, we just mock the
    # archive step itself.
    def fake_git_archive(source_clone, tag, dest_dir):
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        (Path(dest_dir) / "shadow_interface").mkdir(exist_ok=True)
        # also drop a placeholder file so the dir isn't empty
        (Path(dest_dir) / "shadow_interface" / "tools.py").write_text("# stub\n")

    delete_observed_transit = []

    def fake_delete(conn, variant_id):
        # At the moment delete_variant is called, the transit dir for
        # this source island MUST already exist.
        delete_observed_transit.append(transit_dir.exists())

    patches = {
        "get_population": MagicMock(return_value=[picked]),
        "tournament": MagicMock(return_value=[picked]),
        "git_archive_variant": MagicMock(side_effect=fake_git_archive),
        "delete_variant": MagicMock(side_effect=fake_delete),
    }
    started = [
        patch(f"agent_s4.operators.migration.{name}", val)
        for name, val in patches.items()
    ]
    for p in started:
        p.start()
    try:
        Migration().apply_phase1(state)
    finally:
        for p in started:
            p.stop()

    # delete_variant must have been called exactly once, and the transit
    # dir must have existed at that moment.
    assert patches["delete_variant"].call_count == 1, (
        f"expected exactly one delete_variant call, got "
        f"{patches['delete_variant'].call_count}"
    )
    assert delete_observed_transit == [True], (
        "transit dir must exist BEFORE delete_variant is called "
        f"(observed: {delete_observed_transit})"
    )

    # After phase 1 completes, the transit dir survives the call (it's
    # consumed by phase 2 on the destination side).
    assert transit_dir.exists(), (
        "transit dir must still exist after phase 1 returns"
    )
    # And the metadata.json carries the source variant identity.
    meta = json.loads((transit_dir / "metadata.json").read_text())
    assert meta == {"source_id": "haiku-if-007", "source_tag": "evol/haiku/if-007"}


def test_migration_phase1_idempotent_when_transit_exists(tmp_path):
    """If the transit dir already exists (from a prior crashed attempt),
    phase 1 must skip the tournament pick entirely — no DB writes, no
    re-archive."""
    from agent_s4.operators.migration import Migration
    from agent_s4.constants import MIGRATE_TRANSIT_DIR

    state = _make_phase1_state(tmp_path)
    transit_dir = tmp_path / MIGRATE_TRANSIT_DIR / "haiku"
    transit_dir.mkdir(parents=True)
    (transit_dir / "shadow_interface").mkdir()
    (transit_dir / "metadata.json").write_text(
        json.dumps({"source_id": "haiku-if-006", "source_tag": "evol/haiku/if-006"})
    )

    patches = {
        "get_population": MagicMock(return_value=[
            {
                "id": "haiku-if-007", "tag": "evol/haiku/if-007",
                "score": 3, "val_score": 2, "eval_level": 0,
                "generation": 1, "island": "haiku",
            }
        ]),
        "tournament": MagicMock(),
        "git_archive_variant": MagicMock(),
        "delete_variant": MagicMock(),
    }
    started = [
        patch(f"agent_s4.operators.migration.{name}", val)
        for name, val in patches.items()
    ]
    for p in started:
        p.start()
    try:
        Migration().apply_phase1(state)
    finally:
        for p in started:
            p.stop()

    # Neither tournament nor delete_variant should have been invoked.
    assert patches["tournament"].call_count == 0
    assert patches["delete_variant"].call_count == 0
    assert patches["git_archive_variant"].call_count == 0


def test_migration_phase1_empty_population_returns_quietly(tmp_path):
    """If the source island has no scored variants, phase 1 is a no-op
    (no transit dir created, no DB writes)."""
    from agent_s4.operators.migration import Migration
    from agent_s4.constants import MIGRATE_TRANSIT_DIR

    state = _make_phase1_state(tmp_path)

    patches = {
        "get_population": MagicMock(return_value=[]),
        "tournament": MagicMock(),
        "git_archive_variant": MagicMock(),
        "delete_variant": MagicMock(),
    }
    started = [
        patch(f"agent_s4.operators.migration.{name}", val)
        for name, val in patches.items()
    ]
    for p in started:
        p.start()
    try:
        Migration().apply_phase1(state)
    finally:
        for p in started:
            p.stop()

    assert patches["tournament"].call_count == 0
    assert patches["delete_variant"].call_count == 0
    assert patches["git_archive_variant"].call_count == 0
    transit_dir = tmp_path / MIGRATE_TRANSIT_DIR / "haiku"
    assert not transit_dir.exists()


# ---------------------------------------------------------------------------
# 3. Copy-migration mode (migration_mode="copy")
#
# In copy mode phase 1 seeds the neighbor WITHOUT removing the variant from
# its source island: no delete_variant, and the source-side history event is
# 'migration_copy_out' (a distinct type) rather than 'migration_out'. Phase 2
# is identical to move mode and is not re-tested here.
# ---------------------------------------------------------------------------


def _run_phase1_capturing_history(state, picked):
    """Run apply_phase1 with the external deps mocked, returning the
    (delete_variant, append_history) mocks for inspection. git_archive is
    faked to actually materialize the transit tree so the atomic rename in
    _atomic_write_transit succeeds."""
    from agent_s4.operators.migration import Migration

    def fake_git_archive(source_clone, tag, dest_dir):
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        (Path(dest_dir) / "shadow_interface").mkdir(exist_ok=True)
        (Path(dest_dir) / "shadow_interface" / "tools.py").write_text("# stub\n")

    patches = {
        "get_population": MagicMock(return_value=[picked]),
        "tournament": MagicMock(return_value=[picked]),
        "git_archive_variant": MagicMock(side_effect=fake_git_archive),
        "delete_variant": MagicMock(),
        "append_history": MagicMock(),
    }
    started = [
        patch(f"agent_s4.operators.migration.{name}", val)
        for name, val in patches.items()
    ]
    for p in started:
        p.start()
    try:
        Migration().apply_phase1(state)
    finally:
        for p in started:
            p.stop()
    return patches["delete_variant"], patches["append_history"]


_PICKED = {
    "id": "haiku-if-007", "tag": "evol/haiku/if-007",
    "score": 3, "val_score": 2, "eval_level": 0,
    "generation": 1, "island": "haiku",
}


def test_migration_phase1_copy_mode_does_not_delete_source(tmp_path):
    """Copy mode keeps the picked variant on its source island: phase 1
    must NOT call delete_variant."""
    state = _make_phase1_state(tmp_path)
    state.config.migration_mode = "copy"

    delete_variant, _ = _run_phase1_capturing_history(state, dict(_PICKED))

    assert delete_variant.call_count == 0, (
        "copy mode must not remove the variant from its source island"
    )


def test_migration_phase1_copy_mode_logs_copy_out_event(tmp_path):
    """Copy mode records the source-side event as 'migration_copy_out',
    NOT 'migration_out' (the latter signals the variant left the pool)."""
    state = _make_phase1_state(tmp_path)
    state.config.migration_mode = "copy"

    _, append_history = _run_phase1_capturing_history(state, dict(_PICKED))

    assert append_history.call_count == 1
    entry = append_history.call_args.args[1]
    assert entry["type"] == "migration_copy_out"
    assert entry["child"] == "haiku-if-007"


def test_migration_phase1_move_mode_logs_migration_out_event(tmp_path):
    """Move mode (the default) deletes the source variant and records a
    'migration_out' event. Regression guard around the mode branch."""
    state = _make_phase1_state(tmp_path)
    state.config.migration_mode = "move"

    delete_variant, append_history = _run_phase1_capturing_history(
        state, dict(_PICKED))

    assert delete_variant.call_count == 1
    entry = append_history.call_args.args[1]
    assert entry["type"] == "migration_out"


def test_migration_copy_out_does_not_trip_seed_elimination(tmp_db):
    """The resume guard _seed_was_eliminated must treat a copied-out seed as
    STILL PRESENT. 'migration_copy_out' must not be read as removal, while
    'migration_out' and 'elimination' still are."""
    from agent_s4.orchestrator import _seed_was_eliminated
    from agent_s4.db import append_history

    def _hist(event_type):
        return {
            "type": event_type, "island": "haiku",
            "parent": "haiku-seed", "child": "haiku-seed",
            "score": 1, "cost": 0, "details": None,
            "timestamp": "2026-06-02T00:00:00Z",
        }

    append_history(tmp_db, _hist("migration_copy_out"))
    assert _seed_was_eliminated(tmp_db, "haiku") is False

    append_history(tmp_db, _hist("migration_out"))
    assert _seed_was_eliminated(tmp_db, "haiku") is True


# ---------------------------------------------------------------------------
# 3b. Bidirectional / complete migration topology
#
# ring:     each island receives from its single predecessor (today).
# complete: each island receives from ALL other islands (all-to-all). At N=3
#           this is the bidirectional ring. apply_phase2 dispatches one
#           _receive_from_source per present source transit; it must NOT
#           delete transits (a source is read by multiple destinations).
# ---------------------------------------------------------------------------


def test_sources_for_ring_is_predecessor():
    from agent_s4.operators.migration import sources_for
    names = ["haiku", "sonnet", "qwen"]
    assert sources_for("haiku", names, "ring") == ["qwen"]
    assert sources_for("sonnet", names, "ring") == ["haiku"]
    assert sources_for("qwen", names, "ring") == ["sonnet"]


def test_sources_for_complete_is_all_other_islands():
    from agent_s4.operators.migration import sources_for
    names = ["haiku", "sonnet", "qwen"]
    assert set(sources_for("haiku", names, "complete")) == {"sonnet", "qwen"}
    assert set(sources_for("sonnet", names, "complete")) == {"haiku", "qwen"}
    assert "haiku" not in sources_for("haiku", names, "complete")


def _make_phase2_state(tmp_path: Path, topology: str, current="haiku",
                       islands=("haiku", "sonnet", "qwen")):
    state = MagicMock()
    state.current_island = current
    state.base_dir = tmp_path
    isls = []
    for n in islands:
        m = MagicMock()
        m.name = n
        isls.append(m)
    state.config.islands = isls
    state.config.migration_topology = topology
    state.config.benchmark = "osworld"  # ns() -> bare paths; matches _make_transit
    return state


def _make_transit(tmp_path: Path, source: str):
    from agent_s4.constants import MIGRATE_TRANSIT_DIR
    d = tmp_path / MIGRATE_TRANSIT_DIR / source
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps(
        {"source_id": f"{source}-if-001", "source_tag": f"evol/{source}/if-001"}))
    return d


def test_phase2_complete_receives_from_all_other_islands(tmp_path):
    from agent_s4.operators.migration import Migration
    state = _make_phase2_state(tmp_path, "complete")  # current=haiku
    _make_transit(tmp_path, "sonnet")
    _make_transit(tmp_path, "qwen")
    with patch.object(Migration, "_receive_from_source") as recv:
        Migration().apply_phase2(state)
    got = sorted(c.args[1] for c in recv.call_args_list)
    assert got == ["qwen", "sonnet"], "complete topology must receive from both other islands"


def test_phase2_ring_receives_only_from_predecessor(tmp_path):
    from agent_s4.operators.migration import Migration
    state = _make_phase2_state(tmp_path, "ring")  # current=haiku, predecessor=qwen
    _make_transit(tmp_path, "qwen")    # predecessor
    _make_transit(tmp_path, "sonnet")  # NOT the predecessor — must be ignored
    with patch.object(Migration, "_receive_from_source") as recv:
        Migration().apply_phase2(state)
    got = [c.args[1] for c in recv.call_args_list]
    assert got == ["qwen"], "ring topology must receive only from the predecessor"


def test_phase2_skips_sources_with_no_transit(tmp_path):
    from agent_s4.operators.migration import Migration
    state = _make_phase2_state(tmp_path, "complete")  # current=haiku
    _make_transit(tmp_path, "sonnet")  # qwen transit absent
    with patch.object(Migration, "_receive_from_source") as recv:
        Migration().apply_phase2(state)
    got = [c.args[1] for c in recv.call_args_list]
    assert got == ["sonnet"], "a source with no transit dir must be skipped"


# ---------------------------------------------------------------------------
# 4. Migration.apply is intentionally unimplemented in v2.
# ---------------------------------------------------------------------------


def test_migration_apply_raises_not_implemented():
    """The v2 driver invokes apply_phase1 and apply_phase2 directly,
    with a barrier between. The legacy single-island apply() is not
    used and must explicitly NotImplementedError so any accidental call
    fails loudly rather than silently doing the wrong thing."""
    from agent_s4.operators.migration import Migration
    with pytest.raises(NotImplementedError):
        Migration().apply(MagicMock())
