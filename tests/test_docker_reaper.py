# tests/test_docker_reaper.py
"""DockerExecutor.reap_orphans — reaper de contêineres órfãos por label.

Cobre o caminho de SIGKILL/OOM: o ``finally`` que remove o contêiner não roda
→ o reaper identifica órfãos pelo label ``agent_s3.owner=pid<PID>`` e remove
apenas os cujo owner NÃO está vivo. Mock do client docker (sem daemon, sem SDK).
"""
import types
import unittest
from unittest import mock

from gui_agents.s3.execution import docker_executor as de


def _fake_container(owner_pid: str, managed: bool = True):
    """Contêiner fake: labels + remove() contável."""
    labels = {"agent_s3.managed": "1"} if managed else {}
    if owner_pid is not None:
        labels["agent_s3.owner"] = owner_pid
    c = types.SimpleNamespace(labels=labels, removed=False)

    def _remove(force=True):
        c.removed = True

    c.remove = _remove
    return c


class _FakeContainers:
    def __init__(self, items):
        self._items = items

    def list(self, all=True, filters=None):
        # Devolve todos; o filtro por label é responsabilidade do docker real.
        # Aqui simulamos: só devolve os marcados managed=1 (o reaper filtra no
        # client real; no mock devolvemos o conjunto que o filtro entregaria).
        return [c for c in self._items if c.labels.get("agent_s3.managed") == "1"]


class _FakeClient:
    def __init__(self, items):
        self.containers = _FakeContainers(items)


class TestReapOrphans(unittest.TestCase):
    def setUp(self):
        # Guarda estado original p/ restaurar.
        self._orig_has = de._HAS_DOCKER
        self._orig_req = de._require_docker
        self._orig_alive = de._alive_pids
        self._orig_reap_dirs = de._reap_temp_dirs

    def tearDown(self):
        de._HAS_DOCKER = self._orig_has
        de._require_docker = self._orig_req
        de._alive_pids = self._orig_alive
        de._reap_temp_dirs = self._orig_reap_dirs

    def _patch(self, client, alive_pids):
        de._HAS_DOCKER = True
        de._require_docker = mock.Mock(return_value=client)
        de._alive_pids = mock.Mock(return_value=alive_pids)
        # Evita tocar no filesystem no teste do reaper de contêineres.
        de._reap_temp_dirs = mock.Mock(return_value=0)

    def test_dead_owner_reaped_live_owner_kept(self):
        """Owner morto (pid99999) é removido; owner vivo (pid12345) permanece."""
        dead = _fake_container("pid99999")
        live = _fake_container("pid12345")
        client = _FakeClient([dead, live])
        self._patch(client, alive_pids=[12345])

        ex = de.DockerExecutor.__new__(de.DockerExecutor)  # bypassa __init__
        n = ex.reap_orphans()

        self.assertEqual(n, 1, "apenas o contêiner órfão deve ser removido")
        self.assertTrue(dead.removed, "contêiner do PID morto deve ser removido")
        self.assertFalse(live.removed, "contêiner do PID vivo NÃO deve ser removido")

    def test_all_dead_all_reaped(self):
        """Dois órfãos (ambos PIDs mortos) → ambos removidos, retorna 2."""
        dead1 = _fake_container("pid11111")
        dead2 = _fake_container("pid22222")
        client = _FakeClient([dead1, dead2])
        self._patch(client, alive_pids=[99999])

        ex = de.DockerExecutor.__new__(de.DockerExecutor)
        n = ex.reap_orphans()

        self.assertEqual(n, 2)
        self.assertTrue(dead1.removed and dead2.removed)

    def test_no_managed_containers_returns_zero(self):
        """Nenhum contêiner marcado managed=1 → retorna 0, nada removido."""
        unmanaged = _fake_container("pid99999", managed=False)
        client = _FakeClient([unmanaged])
        self._patch(client, alive_pids=[12345])

        ex = de.DockerExecutor.__new__(de.DockerExecutor)
        n = ex.reap_orphans()

        self.assertEqual(n, 0)
        self.assertFalse(unmanaged.removed)

    def test_no_docker_sdk_returns_zero(self):
        """Sem SDK (_HAS_DOCKER=False) → short-circuit, retorna 0."""
        de._HAS_DOCKER = False
        ex = de.DockerExecutor.__new__(de.DockerExecutor)
        n = ex.reap_orphans()
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()