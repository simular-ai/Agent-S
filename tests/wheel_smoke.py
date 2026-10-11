"""Build/install a wheel in isolation and serve its packaged frontend."""

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="agent-s-wheel-") as directory:
        directory = Path(directory)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                str(root),
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(directory),
            ],
            check=True,
        )
        wheel = next(directory.glob("*.whl"))
        with zipfile.ZipFile(wheel) as archive:
            for asset in (
                "index.html",
                "static/main.js",
                "static/api.js",
                "static/configPanel.js",
                "static/runPanel.js",
                "static/types.js",
                "static/styles.css",
            ):
                assert "gui_agents/s3/ui/" + asset in archive.namelist(), asset
        target = directory / "install"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(target),
                str(wheel),
            ],
            check=True,
        )
        code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from gui_agents.s3 import ui_server
assert Path(ui_server.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
from fastapi.testclient import TestClient
app = ui_server.create_app(config_path=sys.argv[2], token='wheel-smoke', allowed_hosts=['testserver'])
with TestClient(app) as client:
    assert client.get('/').status_code == 200
    assert client.get('/static/main.js').status_code == 200
    assert client.get('/api/config', headers={'Authorization': 'Bearer wheel-smoke'}).status_code == 200
print('Isolated wheel frontend/import/API smoke check passed.')
"""
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                code,
                str(target),
                str(directory / "config.json"),
            ],
            cwd=directory,
            check=True,
        )


if __name__ == "__main__":
    main()
