import os
import tempfile
import time
import unittest

from gui_agents.s3.coordination.peer_lock import PeerLock


class TestPeerLock(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.remove(self.db_path)
        self.lock = PeerLock(self.db_path)

    def tearDown(self):
        self.lock.close()
        for suffix in ("", "-wal", "-shm"):
            path = self.db_path + suffix
            if os.path.exists(path):
                os.remove(path)

    def test_acquire_succeeds_when_free(self):
        result = self.lock.acquire("agent_s", "task-1", timeout_ms=1000)
        self.assertTrue(result["ok"])

    def test_second_acquire_times_out_while_held(self):
        self.lock.acquire("agent_s", "task-1", timeout_ms=1000)
        result = self.lock.acquire("cubeflow", "task-2", timeout_ms=300)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "timeout")

    def test_release_lets_another_peer_acquire(self):
        self.lock.acquire("agent_s", "task-1", timeout_ms=1000)
        self.assertTrue(self.lock.release("agent_s"))
        result = self.lock.acquire("cubeflow", "task-2", timeout_ms=1000)
        self.assertTrue(result["ok"])

    def test_release_by_non_holder_is_noop(self):
        self.lock.acquire("agent_s", "task-1", timeout_ms=1000)
        self.assertFalse(self.lock.release("cubeflow"))

    def test_heartbeat_updates_only_for_holder(self):
        self.lock.acquire("agent_s", "task-1", timeout_ms=1000)
        self.assertTrue(self.lock.heartbeat("agent_s"))
        self.assertFalse(self.lock.heartbeat("cubeflow"))

    def test_stale_heartbeat_lets_another_peer_steal(self):
        self.lock.acquire("agent_s", "task-1", timeout_ms=1000)
        stale = int(time.time() * 1000) - 31_000
        self.lock.conn.execute(
            "UPDATE driver_lock SET heartbeat_at = ? WHERE id = 1", (stale,)
        )
        result = self.lock.acquire("cubeflow", "task-2", timeout_ms=1000)
        self.assertTrue(result["ok"])
        events = self.lock.events_since(0)
        self.assertTrue(
            any(e["action"] == "lock_stolen" and e["peer"] == "cubeflow" for e in events)
        )

    def test_log_event_and_events_since_roundtrip(self):
        self.lock.log_event("agent_s", "task-1", "clicked", {"x": 10, "y": 20}, True)
        events = self.lock.events_since(0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["action"], "clicked")
        self.assertEqual(events[0]["payload"], {"x": 10, "y": 20})
        self.assertTrue(events[0]["success"])

    def test_events_since_respects_cursor(self):
        first_id = self.lock.log_event("agent_s", "t1", "a", {}, True)
        self.lock.log_event("agent_s", "t1", "b", {}, True)
        events = self.lock.events_since(first_id)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["action"], "b")

    def test_default_db_path_requires_env_var(self):
        os.environ.pop("CUBEFLOW_COORDINATION_DB", None)
        with self.assertRaises(RuntimeError):
            PeerLock()


if __name__ == "__main__":
    unittest.main()
