# tests/test_peer_lock_other.py
import os, tempfile, unittest
from gui_agents.s3.coordination.peer_lock import PeerLock

class TestIsHeldByOther(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd); os.remove(self.db_path)
        self.lock = PeerLock(self.db_path)
    def tearDown(self):
        self.lock.close()
        for s in ("", "-wal", "-shm"):
            p = self.db_path + s
            if os.path.exists(p): os.remove(p)

    def test_free_returns_false(self):
        self.assertFalse(self.lock.is_held_by_other("agent_s"))

    def test_held_by_self_returns_false(self):
        self.lock.acquire("agent_s", "t1", timeout_ms=1000)
        self.assertFalse(self.lock.is_held_by_other("agent_s"))

    def test_held_by_other_returns_true(self):
        self.lock.acquire("cubeflow", "t1", timeout_ms=1000)
        self.assertTrue(self.lock.is_held_by_other("agent_s"))

if __name__ == "__main__":
    unittest.main()