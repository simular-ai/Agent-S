import base64
import io
import shlex
import subprocess
from typing import Dict, List, Optional, Tuple


class DockerController:
    """Controller that runs bash/python inside a docker or podman container.

    The container is identified by name or id. Execution uses `docker exec`
    (or `podman exec` when `runtime="podman"`). The container must already be
    running and contain `bash` and a python interpreter on PATH.
    """

    def __init__(
        self,
        container: str,
        runtime: str = "docker",
        user: Optional[str] = None,
        workdir: Optional[str] = None,
        python_bin: str = "python3",
        env: Optional[Dict[str, str]] = None,
        runtime_path: Optional[str] = None,
    ):
        if runtime not in ("docker", "podman"):
            raise ValueError(f"runtime must be 'docker' or 'podman', got {runtime!r}")
        self.container = container
        self.runtime = runtime
        self.user = user
        self.workdir = workdir
        self.python_bin = python_bin
        self.env = env or {}
        self.runtime_path = runtime_path or runtime

    def _exec_argv(self, argv: List[str]) -> List[str]:
        cmd = [self.runtime_path, "exec"]
        if self.user:
            cmd += ["--user", self.user]
        if self.workdir:
            cmd += ["--workdir", self.workdir]
        for k, v in self.env.items():
            cmd += ["--env", f"{k}={v}"]
        cmd += [self.container]
        cmd += argv
        return cmd

    def _run(self, argv: List[str], timeout: int) -> Dict:
        try:
            proc = subprocess.run(
                self._exec_argv(argv),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "status": "ok" if proc.returncode == 0 else "error",
                "returncode": proc.returncode,
                "output": (proc.stdout or "") + (proc.stderr or ""),
                "error": "",
            }
        except subprocess.TimeoutExpired as e:
            return {
                "status": "error",
                "returncode": -1,
                "output": e.stdout or "",
                "error": f"TimeoutExpired: {e}",
            }
        except FileNotFoundError as e:
            return {
                "status": "error",
                "returncode": -1,
                "output": "",
                "error": f"runtime binary not found: {e}",
            }

    def run_bash_script(self, code: str, timeout: int = 30) -> Dict:
        result = self._run(["/bin/bash", "-lc", code], timeout)
        print("BASH OUTPUT =======================================")
        print(result["output"])
        print("BASH OUTPUT =======================================")
        return result

    def run_python_script(self, code: str, timeout: int = 60) -> Dict:
        argv = [self.python_bin, "-c", code]
        result = self._run(argv, timeout)
        print("PYTHON OUTPUT =======================================")
        print(result["output"])
        print("PYTHON OUTPUT =======================================")
        # Match LocalController shape (return_code, separate stderr).
        return {
            "status": result["status"],
            "return_code": result["returncode"],
            "output": result["output"],
            "error": result["error"],
        }


    def screen_size(self, display: str = ":1") -> Tuple[int, int]:
        """Return (width, height) of the container's X display via xdpyinfo or xwd header."""
        # Try xdpyinfo first.
        result = self._run(["bash", "-lc", f"DISPLAY={display} xdpyinfo 2>/dev/null | awk '/dimensions:/ {{print $2}}'"], timeout=10)
        out = (result["output"] or "").strip()
        if "x" in out:
            try:
                w, h = out.split()[0].split("x")
                return int(w), int(h)
            except Exception:
                pass
        # Fallback: parse xwd header (uses python in container).
        py = (
            "import subprocess, struct, sys; "
            f"d=subprocess.check_output(['xwd','-root','-silent','-display','{display}']); "
            "h=struct.unpack('>25I', d[:100]); "
            "print(h[4], h[5])"
        )
        result = self._run([self.python_bin, "-c", py], timeout=15)
        out = (result["output"] or "").strip().split()
        if len(out) >= 2:
            return int(out[0]), int(out[1])
        raise RuntimeError(f"could not determine screen size: {result}")

    def screenshot(self, display: str = ":1") -> bytes:
        """Grab a PNG screenshot from the container's X display.

        Uses xwd inside the container and converts to PNG locally with Pillow.
        Returns PNG bytes.
        """
        # Try Pillow ImageGrab inside the container first (cleaner) — falls
        # back to xwd if PIL is not present.
        py = (
            "import sys, base64\n"
            "try:\n"
            "    from PIL import ImageGrab\n"
            "    import io as _io\n"
            "    im = ImageGrab.grab()\n"
            "    buf = _io.BytesIO(); im.save(buf, format='PNG')\n"
            "    sys.stdout.write('PNG:'+base64.b64encode(buf.getvalue()).decode())\n"
            "except Exception as e:\n"
            "    import subprocess\n"
            f"    d = subprocess.check_output(['xwd','-root','-silent','-display','{display}'])\n"
            "    sys.stdout.write('XWD:'+base64.b64encode(d).decode())\n"
        )
        # Use exec+env to ensure DISPLAY is set.
        argv = ["bash", "-lc", f"DISPLAY={display} {shlex.quote(self.python_bin)} -c {shlex.quote(py)}"]
        result = self._run(argv, timeout=30)
        out = (result["output"] or "").strip()
        if out.startswith("PNG:"):
            return base64.b64decode(out[4:])
        if out.startswith("XWD:"):
            from PIL import Image
            xwd = base64.b64decode(out[4:])
            try:
                im = Image.open(io.BytesIO(xwd))
            except Exception:
                # PIL doesn't read xwd directly — parse minimal header and pixels.
                raise RuntimeError("xwd fallback hit; install Pillow inside the container")
            buf = io.BytesIO(); im.save(buf, format="PNG")
            return buf.getvalue()
        raise RuntimeError(f"screenshot failed: {result}")


class DockerEnv:
    """Environment that exposes a controller targeting a docker/podman container.

    Example:
        env = DockerEnv(container="webtop-src", runtime="podman", user="abc")
    """

    def __init__(
        self,
        container: str,
        runtime: str = "docker",
        user: Optional[str] = None,
        workdir: Optional[str] = None,
        python_bin: str = "python3",
        env: Optional[Dict[str, str]] = None,
        runtime_path: Optional[str] = None,
    ):
        self.controller = DockerController(
            container=container,
            runtime=runtime,
            user=user,
            workdir=workdir,
            python_bin=python_bin,
            env=env,
            runtime_path=runtime_path,
        )

    @staticmethod
    def container_running(container: str, runtime: str = "docker", runtime_path: Optional[str] = None) -> bool:
        rt = runtime_path or runtime
        try:
            out = subprocess.run(
                [rt, "inspect", "-f", "{{.State.Running}}", container],
                capture_output=True, text=True, timeout=10,
            )
            return out.returncode == 0 and out.stdout.strip() == "true"
        except Exception:
            return False
