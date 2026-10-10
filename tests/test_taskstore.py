# tests/test_taskstore.py
"""TaskStore — memória procedural Agent-S3 (sqlite3 stdlib)."""
import os
import tempfile
import unittest

from gui_agents.s3 import taskstore


class TestTaskStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db = self._tmp.name
        os.environ["AGENT_S_USE_MEMORY"] = "1"

    def tearDown(self):
        try:
            os.unlink(self.db)
        except OSError:
            pass

    def _tracks(self):
        return [{"name": "KICK IN", "file": "/a/KICK IN.wav"},
                {"name": "SNARE TOP", "file": "/a/SNARE TOP.wav"}]

    def test_record_perfect_then_lookup_returns_sequence(self):
        sig = taskstore.signature(120, "1/16", self._tracks())
        seq = [{"name": "smart_quantize", "args": {"grid": "1/16"}},
               {"name": "correct_off_grid", "args": {}}]
        taskstore.record_run("r1", "bateria", sig, 1.0, 2.0, 1, 2, 2, 0.01,
                             True, {"perfect": False}, {"perfect": True}, seq, db_path=self.db)
        got = taskstore.lookup_winning_sequence(sig, db_path=self.db)
        self.assertIsNotNone(got)
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0]["name"], "smart_quantize")

    def test_record_non_perfect_ignored_by_lookup(self):
        sig = taskstore.signature(120, "1/16", self._tracks())
        seq = [{"name": "smart_quantize", "args": {}}]
        taskstore.record_run("r2", "bateria", sig, 1.0, 2.0, 3, 5, 3, 0.02,
                             False, {"perfect": False}, {"perfect": False}, seq, db_path=self.db)
        self.assertIsNone(taskstore.lookup_winning_sequence(sig, db_path=self.db))

    def test_collision_latest_perfect_wins(self):
        sig = taskstore.signature(120, "1/16", self._tracks())
        old = [{"name": "old_tool", "args": {}}]
        new = [{"name": "new_tool", "args": {}}]
        taskstore.record_run("a", "bateria", sig, 1.0, 2.0, 1, 1, 1, 0.01, True, {}, {}, old, db_path=self.db)
        taskstore.record_run("b", "bateria", sig, 3.0, 4.0, 1, 1, 1, 0.01, True, {}, {}, new, db_path=self.db)
        got = taskstore.lookup_winning_sequence(sig, db_path=self.db)
        self.assertEqual(got[0]["name"], "new_tool")

    def test_different_signature_no_match(self):
        sig1 = taskstore.signature(120, "1/16", self._tracks())
        sig2 = taskstore.signature(140, "1/16", self._tracks())  # bpm diferente
        taskstore.record_run("c", "bateria", sig1, 1.0, 2.0, 1, 1, 1, 0.01, True, {}, {},
                             [{"name": "x", "args": {}}], db_path=self.db)
        self.assertIsNone(taskstore.lookup_winning_sequence(sig2, db_path=self.db))

    def test_disabled_returns_none(self):
        os.environ["AGENT_S_USE_MEMORY"] = "0"
        try:
            sig = taskstore.signature(120, "1/16", self._tracks())
            taskstore.record_run("d", "bateria", sig, 1.0, 2.0, 1, 1, 1, 0.01, True, {}, {},
                                 [{"name": "x", "args": {}}], db_path=self.db)
            self.assertIsNone(taskstore.lookup_winning_sequence(sig, db_path=self.db))
        finally:
            os.environ["AGENT_S_USE_MEMORY"] = "1"

    def test_signature_stable_regardless_of_track_order(self):
        t1 = [{"name": "KICK"}, {"name": "SNARE"}]
        t2 = [{"name": "SNARE"}, {"name": "KICK"}]
        self.assertEqual(taskstore.signature(120, "1/16", t1),
                         taskstore.signature(120, "1/16", t2))

    def test_list_recent(self):
        sig = taskstore.signature(120, "1/16", self._tracks())
        taskstore.record_run("e", "bateria", sig, 1.0, 2.0, 1, 1, 1, 0.01, True, {}, {},
                             [{"name": "x", "args": {}}], db_path=self.db)
        rows = taskstore.list_recent(10, db_path=self.db)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_id"], "e")
        self.assertEqual(rows[0]["perfect"], 1)


if __name__ == "__main__":
    unittest.main()