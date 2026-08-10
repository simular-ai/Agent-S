"""CriticAgent — testes das heurísticas estáticas e dataclasses (sem API key).

Cobre o que não depende de LLM/web: dataclasses SearchResult/ReviewResult,
_parse_json (incl. caso H3 — } dentro de string), _strip_fences,
_needs_research, _build_query, e ChatCLI._detect_language. Os métodos que
chamam OpenAI/Anthropic/DDGS ficam fora de escopo (env-gated, precisam key).
"""
import sys
import unittest

from gui_agents.s3.cli.chat_cli import ChatCLI
from gui_agents.s3.cognition.critic_agent import (
    CriticAgent,
    ReviewResult,
    SearchResult,
)


class TestSearchResult(unittest.TestCase):
    def test_fields_assigned(self):
        r = SearchResult(title="T", url="https://x", snippet="S")
        self.assertEqual(r.title, "T")
        self.assertEqual(r.url, "https://x")
        self.assertEqual(r.snippet, "S")


class TestReviewResult(unittest.TestCase):
    def test_defaults(self):
        r = ReviewResult(corrected_code="print('hi')")
        self.assertEqual(r.corrected_code, "print('hi')")
        self.assertEqual(r.issues, [])
        self.assertFalse(r.changed)
        self.assertEqual(r.raw_response, "")

    def test_explicit(self):
        r = ReviewResult(
            corrected_code="x = 1",
            issues=["missing import"],
            changed=True,
            raw_response="{}",
        )
        self.assertEqual(r.issues, ["missing import"])
        self.assertTrue(r.changed)


class TestParseJson(unittest.TestCase):
    def test_valid_object(self):
        out = CriticAgent._parse_json('{"issues": [], "changed": false}')
        self.assertEqual(out, {"issues": [], "changed": False})

    def test_fenced_json(self):
        out = CriticAgent._parse_json("```json\n{\"changed\": true}\n```")
        self.assertEqual(out, {"changed": True})

    def test_prose_around_object(self):
        out = CriticAgent._parse_json("Here is the review:\n{\"issues\": [\"x\"]}\nThanks.")
        self.assertEqual(out, {"issues": ["x"]})

    def test_brace_inside_string(self):
        # H3: brace-counter quebra com } dentro de string. raw_decode é um
        # parser JSON real então tolera. O valor contém "click({x:100})".
        raw = '{"corrected_code": "btn.click({x:100})", "changed": true}'
        out = CriticAgent._parse_json(raw)
        self.assertEqual(out["corrected_code"], "btn.click({x:100})")
        self.assertTrue(out["changed"])

    def test_invalid_returns_none(self):
        self.assertIsNone(CriticAgent._parse_json("not json at all"))

    def test_empty_returns_none(self):
        self.assertIsNone(CriticAgent._parse_json(""))

    def test_none_input(self):
        self.assertIsNone(CriticAgent._parse_json(None))  # type: ignore[arg-type]

    def test_picks_first_valid_object(self):
        # dois objetos na string — raw_decode devolve o primeiro válido.
        out = CriticAgent._parse_json('{"a": 1} garbage {"b": 2}')
        self.assertEqual(out, {"a": 1})


class TestStripFences(unittest.TestCase):
    def test_python_fences(self):
        self.assertEqual(
            CriticAgent._strip_fences("```python\nprint(1)\n```"),
            "print(1)",
        )

    def test_bare_fences(self):
        self.assertEqual(CriticAgent._strip_fences("```\nx = 2\n```"), "x = 2")

    def test_no_fences_passthrough(self):
        self.assertEqual(CriticAgent._strip_fences("print(3)"), "print(3)")

    def test_multiline_inside_fences(self):
        raw = "```python\nimport os\n\nprint(os.getcwd())\n```"
        self.assertEqual(
            CriticAgent._strip_fences(raw),
            "import os\n\nprint(os.getcwd())",
        )


class TestNeedsResearch(unittest.TestCase):
    def test_library_keyword(self):
        self.assertTrue(CriticAgent._needs_research("parse PDF with pdfplumber library"))

    def test_how_to(self):
        self.assertTrue(CriticAgent._needs_research("how to quantize audio"))

    def test_version_pattern(self):
        self.assertTrue(CriticAgent._needs_research("upgrade React 18 hooks"))

    def test_pip_install(self):
        self.assertTrue(CriticAgent._needs_research("pip install requests"))

    def test_plain_task_skips(self):
        self.assertFalse(CriticAgent._needs_research("quantize the drum loop"))

    def test_empty_skips(self):
        self.assertFalse(CriticAgent._needs_research(""))


class TestBuildQuery(unittest.TestCase):
    def test_strips_version_numbers(self):
        # "pdfplumber 0.11" → "pdfplumber" (número removido, nome da lib mantido)
        q = CriticAgent._build_query("parse PDF with pdfplumber 0.11 and Python 3.12")
        self.assertNotIn("0.11", q)
        self.assertNotIn("3.12", q)
        self.assertIn("pdfplumber", q)
        self.assertIn("Python", q)  # _VERSION_RE mantém o nome da lib (Python)

    def test_strips_filler_words(self):
        q = CriticAgent._build_query("how to use the requests library with api")
        self.assertNotIn("how", q)
        self.assertNotIn("the", q)
        self.assertNotIn("library", q)
        self.assertIn("requests", q)
        self.assertIn("api", q)

    def test_caps_at_six_words(self):
        q = CriticAgent._build_query("alpha beta gamma delta epsilon zeta eta theta")
        self.assertEqual(len(q.split()), 6)

    def test_drops_single_char_tokens(self):
        q = CriticAgent._build_query("parse a csv with pandas")
        # "a" (len 1) removido
        self.assertNotIn("a", q.split())


class TestDetectLanguage(unittest.TestCase):
    def test_python_default(self):
        self.assertEqual(ChatCLI._detect_language("quantize the drum loop"), "python")

    def test_bash_hint(self):
        self.assertEqual(ChatCLI._detect_language("grep -r foo ."), "bash")

    def test_bash_keyword(self):
        self.assertEqual(ChatCLI._detect_language("run a bash script"), "bash")

    def test_apt(self):
        self.assertEqual(ChatCLI._detect_language("apt install ffmpeg"), "bash")


class TestParselineSlashRouting(unittest.TestCase):
    """``/exit`` etc. devem rotear para ``do_*``, não virar tarefa (default)."""

    def setUp(self):
        # __init__ imprime o banner (rich) — pytest captura stdout, ok.
        import io
        self._saved_stdout = sys.stdout
        sys.stdout = io.StringIO()
        self.cli = ChatCLI()

    def tearDown(self):
        sys.stdout = self._saved_stdout

    def test_slash_exit_routes_to_exit(self):
        cmd, arg, line = self.cli.parseline("/exit")
        self.assertEqual(cmd, "exit")

    def test_slash_help_routes_to_help(self):
        cmd, _arg, _line = self.cli.parseline("/help")
        self.assertEqual(cmd, "help")

    def test_slash_memory_routes_to_memory(self):
        cmd, _arg, _line = self.cli.parseline("/memory")
        self.assertEqual(cmd, "memory")

    def test_bare_exit_still_works(self):
        cmd, _arg, _line = self.cli.parseline("exit")
        self.assertEqual(cmd, "exit")

    def test_path_starting_slash_falls_back_to_default(self):
        # "/tmp/foo.pdf" → do_tmp não existe → não roubeia, cai em default.
        cmd, _arg, _line = self.cli.parseline("/tmp/foo.pdf")
        self.assertEqual(cmd, "")

    def test_plain_task_has_no_do_method(self):
        # parseline devolve a 1ª palavra como cmd; roteamento p/ default
        # acontece no onecmd (do_quantize não existe). Verificamos isso.
        cmd, _arg, _line = self.cli.parseline("quantize the drum loop")
        self.assertEqual(cmd, "quantize")
        self.assertFalse(hasattr(self.cli, "do_" + cmd))


if __name__ == "__main__":
    unittest.main()