import json
import locale
import os
import subprocess
import unittest
from unittest.mock import patch

from gui_agents.s3.utils.local_env import LocalController


class TestBashTimeoutOutput(unittest.TestCase):
    def test_timeout_output_remains_json_serializable(self):
        error = subprocess.TimeoutExpired(
            "bash", 0.1, output=b"partial stdout", stderr=b"partial stderr"
        )
        with patch("subprocess.run", side_effect=error):
            result = LocalController().run_bash_script("unused", timeout=1)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["returncode"], -1)
        self.assertEqual(result["output"], "partial stdoutpartial stderr")
        self.assertIn("TimeoutExpired", result["error"])
        json.dumps(result)

    def test_timeout_decodes_using_the_subprocess_text_encoding(self):
        error = subprocess.TimeoutExpired("bash", 1, output=b"caf\xe9")
        with patch("subprocess.run", side_effect=error), patch.object(
            locale, "getpreferredencoding", return_value="cp1252"
        ):
            result = LocalController().run_bash_script("unused")
        self.assertEqual(result["output"], "café")
        json.dumps(result)

    def test_partial_invalid_output_does_not_mask_timeout(self):
        error = subprocess.TimeoutExpired("bash", 1, output=b"partial\xff")
        with patch("subprocess.run", side_effect=error), patch.object(
            locale, "getpreferredencoding", return_value="utf-8"
        ):
            result = LocalController().run_bash_script("unused")
        self.assertEqual(result["output"], "partial\ufffd")
        self.assertIn("TimeoutExpired", result["error"])

    def test_timeout_also_preserves_text_or_missing_output(self):
        for stdout, stderr, expected in [
            (None, None, ""),
            ("stdout", "stderr", "stdoutstderr"),
            (None, b"stderr", "stderr"),
        ]:
            with self.subTest(stdout=stdout, stderr=stderr):
                error = subprocess.TimeoutExpired(
                    "bash", 1, output=stdout, stderr=stderr
                )
                with patch("subprocess.run", side_effect=error):
                    result = LocalController().run_bash_script("unused")
                self.assertEqual(result["output"], expected)
                json.dumps(result)

    @unittest.skipUnless(os.path.exists("/bin/bash"), "requires bash")
    def test_real_timeout_keeps_partial_output_as_text(self):
        result = LocalController().run_bash_script(
            "printf 'stdout'; printf 'stderr' >&2; while :; do :; done", timeout=0.5
        )
        self.assertEqual(result["status"], "error")
        # A login shell may also print output from the user's profile.
        self.assertIsInstance(result["output"], str)
        self.assertIn("stdout", result["output"])
        self.assertIn("stderr", result["output"])
        json.dumps(result)

    def test_success_and_nonzero_exit_remain_text(self):
        for code in (0, 2):
            with self.subTest(code=code):
                completed = subprocess.CompletedProcess(
                    [], code, stdout="stdout", stderr="stderr"
                )
                with patch("subprocess.run", return_value=completed):
                    result = LocalController().run_bash_script("unused")
                self.assertEqual(result["output"], "stdoutstderr")
                self.assertEqual(result["returncode"], code)
                self.assertEqual(result["status"], "ok" if code == 0 else "error")
                json.dumps(result)
