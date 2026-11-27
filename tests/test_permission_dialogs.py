import ast
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


CLI_APPS = (
    "gui_agents/s1/cli_app.py",
    "gui_agents/s2/cli_app.py",
    "gui_agents/s2_5/cli_app.py",
    "gui_agents/s3/cli_app.py",
)


def load_permission_dialog(path: str):
    """Load only the dialog helper so CLI module side effects are not executed."""
    source_path = Path(path)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=path)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "show_permission_dialog"
    )
    module = SimpleNamespace(
        platform=SimpleNamespace(system=Mock()),
        subprocess=SimpleNamespace(run=Mock()),
    )
    namespace = {"platform": module.platform, "subprocess": module.subprocess}
    exec(compile(ast.Module(body=[function], type_ignores=[]), path, "exec"), namespace)
    return namespace["show_permission_dialog"], module


class PermissionDialogTests(unittest.TestCase):
    def test_macos_passes_model_text_as_osascript_argv(self):
        model_text = '"; do shell script "touch /tmp/pwned"; "'

        for path in CLI_APPS:
            with self.subTest(path=path):
                show_permission_dialog, module = load_permission_dialog(path)
                module.platform.system.return_value = "Darwin"
                module.subprocess.run.return_value = subprocess.CompletedProcess([], 0)

                self.assertTrue(show_permission_dialog(model_text, "open settings"))

                argv = module.subprocess.run.call_args.args[0]
                self.assertEqual(argv[:2], ["osascript", "-e"])
                self.assertIn("argv item 1", argv[2])
                self.assertNotIn(model_text, argv[2])
                self.assertIn(model_text, argv[3])
                self.assertEqual(module.subprocess.run.call_args.kwargs, {"check": False})

    def test_linux_passes_model_text_as_zenity_argv(self):
        model_text = '$(touch /tmp/pwned); `id`; "quoted"'

        for path in CLI_APPS:
            with self.subTest(path=path):
                show_permission_dialog, module = load_permission_dialog(path)
                module.platform.system.return_value = "Linux"
                module.subprocess.run.return_value = subprocess.CompletedProcess([], 1)

                self.assertFalse(show_permission_dialog(model_text, "open settings"))

                argv = module.subprocess.run.call_args.args[0]
                self.assertEqual(argv[0], "zenity")
                self.assertEqual(argv[argv.index("--text") + 1].count(model_text), 1)
                self.assertEqual(module.subprocess.run.call_args.kwargs, {"check": False})


if __name__ == "__main__":
    unittest.main()
