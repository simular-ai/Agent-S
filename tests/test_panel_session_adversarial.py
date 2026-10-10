# tests/test_panel_session_adversarial.py
"""Panel server — _build_session adversarial / edge-case.

_build_session é um merge fino de overrides sobre DEFAULT_SESSION (design
C3.2: só mapeia chaves presentes e não-None). Estes testes DOCUMENTAM o
comportamento de borda — tipos malformados passam direto (validação é
responsabilidade downstream, não do merge; Eng#2 proíbe defesa especulativa
aqui). Confirmam: nenhum crash, defaults preservados, chaves desconhecidas
ignoradas.
"""
import unittest

from panel.server import _build_session, DEFAULT_SESSION, _SESSION_OVERRIDE_KEYS


class TestBuildSessionAdversarial(unittest.TestCase):
    def test_unknown_keys_ignored(self):
        # chaves fora de _SESSION_OVERRIDE_KEYS não entram na session.
        s = _build_session({"bpm": 120, "evil_key": "x", "tracks": "hax"})
        self.assertEqual(s["bpm"], 120)
        self.assertNotIn("evil_key", s)

    def test_empty_dict_returns_defaults_copy(self):
        s = _build_session({})
        self.assertEqual(s, DEFAULT_SESSION)
        # mutar a cópia não afeta o default (dict() copia)
        s["bpm"] = 999
        self.assertIsNone(DEFAULT_SESSION["bpm"])

    def test_bpm_as_string_passes_through(self):
        # DOCUMENTA: tipo fraco — "120" (string) passa direto. _build_session
        # não valida tipos (merge fino). Downstream (orchestrator/JS) decide.
        s = _build_session({"bpm": "120"})
        self.assertEqual(s["bpm"], "120")

    def test_tracks_non_list_passes_through(self):
        # DOCUMENTA: string em vez de lista passa. Validar tipo seria defesa
        # especulativa no merge (Eng#2). Frontend legítimo envia lista.
        s = _build_session({"tracks": "notalist"})
        self.assertEqual(s["tracks"], "notalist")

    def test_negative_tolerance_ms_passes(self):
        s = _build_session({"toleranceMs": -5})
        self.assertEqual(s["toleranceMs"], -5)

    def test_phase_pairs_and_tolerance_db_overridable(self):
        s = _build_session({"phasePairs": [[0, 1]], "toleranceDb": 6})
        self.assertEqual(s["phasePairs"], [[0, 1]])
        self.assertEqual(s["toleranceDb"], 6)

    def test_zero_is_not_clobbered(self):
        # 0 é válido (não None) → sobrescreve default. Diferente de None.
        s = _build_session({"bpm": 0, "toleranceMs": 0})
        self.assertEqual(s["bpm"], 0)
        self.assertEqual(s["toleranceMs"], 0)

    def test_all_override_keys_recognized(self):
        # sanity: toda chave documentada é aceita.
        for k in _SESSION_OVERRIDE_KEYS:
            self.assertIn(k, _build_session({k: 1}))


if __name__ == "__main__":
    unittest.main()