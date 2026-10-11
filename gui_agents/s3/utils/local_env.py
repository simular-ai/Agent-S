import subprocess
import sys
import time
from typing import Dict


class LocalController:
    """Minimal controller to execute bash and python code locally.

    WARNING: Executing arbitrary code is dangerous. Only enable/use this in trusted
    environments and with trusted inputs.
    """

    def run_bash_script(self, code: str, timeout: int = 30) -> Dict:
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
        if hasattr(self, "stop_event"):
            return self._supervised_python(code)
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

    def _supervised_python(self, code):
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + self.script_timeout
        try:
            while True:
                if self.stop_event.is_set() or time.monotonic() >= deadline:
                    process.kill()
                    out, err = process.communicate()
                    raise InterruptedError("Local script stopped or timed out")
                try:
                    out, err = process.communicate(timeout=0.2)
                    return {
                        "status": "ok" if process.returncode == 0 else "error",
                        "return_code": process.returncode,
                        "output": out[-65536:],
                        "error": err[-65536:],
                    }
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


class LocalEnv:
    """Simple environment that provides a controller compatible with CodeAgent."""

    def __init__(self):
        self.controller = LocalController()
