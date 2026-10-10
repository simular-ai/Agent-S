# tests/test_taskstore_adversarial.py
"""TaskStore — testes adversariais / edge-case da memória procedural.

Expõe: signature com projeto vazio vs None (trailing-colon), nomes unicode,
muitas tracks (perf), run_id duplicado (REPLACE), sequence vazia (None),
record_run com before/after None (json.dumps default=str), tracks com
name None, lookup quando memória desabilitada mid-flight.
"""
import os
import tempfile
import unittest

from gui_agents.s3 import taskstore


class TestTaskStoreAdversarial(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db = self._tmp.name
        self._prev_mem = os.environ.get("AGENT_S_USE_MEMORY")
        os.environ["AGENT_S_USE_MEMORY"] = "1"

    def tearDown(self):
        try:
            os.unlink(self.db)
        except OSError:
            pass
        if self._prev_mem is None:
            os.environ.pop("AGENT_S_USE_MEMORY", None)
        else:
            os.environ["AGENT_S_USE_MEMORY"] = self._prev_mem

    def test_signature_project_none_vs_empty_string_differ(self):
        # project=None NÃO adiciona sufixo (caller legado). project="" (string
        # vazia) é truthy-enough p/ passar `if project is None` → adiciona ":"
        # trailing. São signatures DIFERENTES p/ "sem projeto" — documenta o
        # edge: callers não devem passar project="" (passem None).
        s_none = taskstore.signature(120, "1/16", [], project=None)
        s_empty = taskstore.signature(120, "1/16", [], project="")
        self.assertNotEqual(s_none, s_empty)
        self.assertFalse(s_none.endswith(":"), "project=None não deixa trailing colon")
        self.assertTrue(s_empty.endswith(":"), "project='' gera trailing colon (edge)")

    def test_signature_unicode_names(self):
        # nomes com acento/espaço não quebram e são estáveis.
        t = [{"name": "BUMBO IN"}, {"name": "CAIXA®"}]
        s1 = taskstore.signature(120, "1/16", t, project="/p/drums.cpr")
        s2 = taskstore.signature(120, "1/16", list(reversed(t)), project="/p/drums.cpr")
        self.assertEqual(s1, s2, "ordem não importa (sorted)")

    def test_signature_many_tracks_does_not_crash(self):
        # 5000 tracks — sorted() + repr da lista deve ser rápido, não explode.
        t = [{"name": f"track{i}"} for i in range(5000)]
        s = taskstore.signature(140, "1/16", t, project="/p.cpr")
        self.assertIsInstance(s, str)
        self.assertIn("5000", s)

    def test_duplicate_run_id_replaces_not_duplicates(self):
        # INSERT OR REPLACE: mesmo run_id sobrescreve. list_recent não cresce.
        sig = taskstore.signature(120, "1/16", [{"name": "K"}])
        taskstore.record_run("dup", "t", sig, 1, 2, 1, 1, 1, 0.01, True, {}, {},
                             [{"name": "old", "args": {}}], db_path=self.db)
        taskstore.record_run("dup", "t", sig, 3, 4, 2, 2, 2, 0.02, True, {}, {},
                             [{"name": "new", "args": {}}], db_path=self.db)
        rows = taskstore.list_recent(10, db_path=self.db)
        self.assertEqual(len(rows), 1, "run_id duplicado substitui, não duplica")
        got = taskstore.lookup_winning_sequence(sig, db_path=self.db)
        self.assertEqual(got[0]["name"], "new", "REPLACE pegou a versão mais nova")

    def test_lookup_empty_sequence_returns_none(self):
        # sequence gravada como [] → lookup retorna None (não lista vazia que
        # viraria replay de nada). Evita replay de sequência vazia como se
        # fosse vencedora.
        sig = taskstore.signature(120, "1/16", [{"name": "K"}])
        taskstore.record_run("e", "t", sig, 1, 2, 1, 1, 1, 0.01, True, {}, {},
                             [], db_path=self.db)
        self.assertIsNone(taskstore.lookup_winning_sequence(sig, db_path=self.db))

    def test_record_run_with_none_before_after_persists(self):
        # before/after None (ex.: abort de discovery) → json.dumps(None)="null".
        # Não deve quebrar nem virar "None" literal Python (default=str só age
        # em tipos não-serializáveis).
        sig = taskstore.signature(120, "1/16", [{"name": "K"}])
        taskstore.record_run("n", "t", sig, 1, 2, 1, 0, 0, 0.0, False,
                             None, None, [{"name": "x", "args": {}}],
                             db_path=self.db)
        rows = taskstore.list_recent(5, db_path=self.db)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["perfect"], 0)

    def test_signature_track_missing_name_defaults_empty(self):
        # track sem "name" (só "file") → t.get("name","") = "" → não crash.
        s = taskstore.signature(120, "1/16",
                                [{"file": "/a.wav"}, {"name": "KICK"}])
        self.assertIsInstance(s, str)
        self.assertIn(":2:", s)  # 2 tracks

    def test_lookup_when_disabled_returns_none_even_with_data(self):
        sig = taskstore.signature(120, "1/16", [{"name": "K"}])
        taskstore.record_run("d", "t", sig, 1, 2, 1, 1, 1, 0.01, True, {}, {},
                             [{"name": "x", "args": {}}], db_path=self.db)
        os.environ["AGENT_S_USE_MEMORY"] = "0"
        try:
            self.assertIsNone(taskstore.lookup_winning_sequence(sig, db_path=self.db))
        finally:
            os.environ["AGENT_S_USE_MEMORY"] = "1"

    def test_init_idempotent(self):
        # init() chamado 2x não cria schemas duplicados nem erro.
        taskstore.init(self.db)
        taskstore.init(self.db)
        rows = taskstore.list_recent(10, db_path=self.db)
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()