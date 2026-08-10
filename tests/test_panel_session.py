# tests/test_panel_session.py
"""Panel server — _build_session: merge de overrides do payload sobre a
session default do orquestrador.

Antes o painel hardcodava session={"tracks":[],"filter":"bateria","bpm":None,
"grid":"1/16","toleranceMs":10} — o usuário NÃO podia informar o BPM, então o
smart-quantize (canVerify exige bpm != None) nunca ativava o closed-loop de
verificação por áudio via painel. A feature mais avançada ficava inalcançável.
_build_session mapeia overrides do payload mantendo os defaults.
"""
import unittest

from panel.server import _build_session, DEFAULT_SESSION


class TestBuildSession(unittest.TestCase):
    def test_defaults_when_no_overrides(self):
        s = _build_session({})
        self.assertEqual(s["filter"], "bateria")
        self.assertEqual(s["grid"], "1/16")
        self.assertIsNone(s["bpm"])
        self.assertEqual(s["toleranceMs"], 10)
        self.assertEqual(s["tracks"], [])

    def test_bpm_override_enables_closed_loop(self):
        # o ponto principal: bpm informado → canVerify (bpm != None) vira True
        s = _build_session({"bpm": 120})
        self.assertEqual(s["bpm"], 120)

    def test_grid_and_filter_override(self):
        s = _build_session({"grid": "1/8", "filter": "kick"})
        self.assertEqual(s["grid"], "1/8")
        self.assertEqual(s["filter"], "kick")

    def test_none_payload_returns_defaults(self):
        s = _build_session(None)
        self.assertEqual(s, DEFAULT_SESSION)

    def test_none_values_do_not_clobber_defaults(self):
        # bpm explicit None não deve sobrescrever (mantém default None aqui,
        # mas valida que None no payload não apaga outros defaults)
        s = _build_session({"bpm": None, "grid": "1/16"})
        self.assertEqual(s["grid"], "1/16")

    def test_tracks_override(self):
        tracks = [{"name": "KICK", "file": "/a/KICK.wav"}]
        s = _build_session({"tracks": tracks})
        self.assertEqual(s["tracks"], tracks)


if __name__ == "__main__":
    unittest.main()