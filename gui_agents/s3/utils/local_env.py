import os
import subprocess
import sys
from typing import Dict


def _docker_executor():
    """Lazy DockerExecutor (TIER 4 M1). None se SDK/daemon indisponíveis."""
    if not os.environ.get("AGENT_S3_USE_DOCKER"):
        return None
    try:
        from gui_agents.s3.execution.docker_executor import DockerExecutor

        return DockerExecutor()
    except Exception as e:  # noqa: BLE001 — fallback p/ subprocess host
        print(f"[docker] sandbox indisponível ({e}) — usando subprocess host")
        return None


class LocalController:
    """Minimal controller to execute bash and python code locally.

    WARNING: Executing arbitrary code is dangerous. Only enable/use this in trusted
    environments and with trusted inputs.

    TIER 4 M1: se ``AGENT_S3_USE_DOCKER=1``, executa num contêiner sandbox
    (DockerExecutor) em vez do host; fallback gracioso p/ subprocess se o
    daemon/SDK estiver indisponível.
    """

    def run_bash_script(self, code: str, timeout: int = 30) -> Dict:
        docker = _docker_executor()
        if docker is not None:
            r = docker.run_bash(code, timeout=float(timeout))
            output = r.stdout + (("\n" + r.stderr) if r.stderr else "")
            print("BASH OUTPUT =======================================")
            print(output)
            print("BASH OUTPUT =======================================")
            return {
                "status": "ok" if r.success else "error",
                "returncode": r.exit_code if r.exit_code is not None else -1,
                "output": output,
                "error": r.stderr,
            }
        try:
            proc = subprocess.run(
                ["/bin/bash", "-lc", code],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = (proc.stdout or "") + (proc.stderr or "")

            print("BASH OUTPUT =======================================")
            print(output)
            print("BASH OUTPUT =======================================")

            return {
                "status": "ok" if proc.returncode == 0 else "error",
                "returncode": proc.returncode,
                "output": output,
                "error": "",
            }
        except subprocess.TimeoutExpired as e:
            return {
                "status": "error",
                "returncode": -1,
                "output": e.stdout or "",
                "error": f"TimeoutExpired: {str(e)}",
            }
        except Exception as e:
            return {
                "status": "error",
                "returncode": -1,
                "output": "",
                "error": str(e),
            }

    def run_python_script(self, code: str) -> Dict:
        docker = _docker_executor()
        if docker is not None:
            r = docker.run_python(code)
            print("PYTHON OUTPUT =======================================")
            print(r.stdout)
            print("PYTHON OUTPUT =======================================")
            return {
                "status": "ok" if r.success else "error",
                "return_code": r.exit_code if r.exit_code is not None else -1,
                "output": r.stdout,
                "error": r.stderr,
            }
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
            )
            print("PYTHON OUTPUT =======================================")
            print(proc.stdout or "")
            print("PYTHON OUTPUT =======================================")
            return {
                "status": "ok" if proc.returncode == 0 else "error",
                "return_code": proc.returncode,
                "output": proc.stdout or "",
                "error": proc.stderr or "",
            }
        except Exception as e:
            return {
                "status": "error",
                "return_code": -1,
                "output": "",
                "error": str(e),
            }


class LocalEnv:
    """Simple environment that provides a controller compatible with CodeAgent."""

    def __init__(self):
        self.controller = LocalController()
