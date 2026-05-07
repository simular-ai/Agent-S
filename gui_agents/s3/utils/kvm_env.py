import shlex
import subprocess
from typing import Dict, List, Mapping, Optional


class KvmController:
    """Controller that runs bash/python inside a KVM/libvirt guest.

    Two transports are supported:

    - transport="ssh"   (recommended): execute over `ssh user@host CMD`. Most
      reliable; assumes guest sshd is reachable and key auth is set up.
    - transport="qga"   : execute via `virsh qemu-agent-command DOMAIN ...`
      using the QEMU guest agent (`guest-exec` + `guest-exec-status`). No
      networking needed but requires qemu-guest-agent in the guest.
    """

    def __init__(
        self,
        transport: str = "ssh",
        # ssh transport
        ssh_host: Optional[str] = None,
        ssh_user: Optional[str] = None,
        ssh_port: int = 22,
        ssh_key: Optional[str] = None,
        ssh_options: Optional[List[str]] = None,
        # qga transport
        domain: Optional[str] = None,
        virsh_uri: Optional[str] = None,
        # common
        python_bin: str = "python3",
        display: str = ":0",
        xauthority: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        if transport not in ("ssh", "qga"):
            raise ValueError(f"transport must be 'ssh' or 'qga', got {transport!r}")
        if transport == "ssh" and not (ssh_host and ssh_user):
            raise ValueError("ssh transport requires ssh_host and ssh_user")
        if transport == "qga" and not domain:
            raise ValueError("qga transport requires domain")
        self.transport = transport
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.ssh_port = ssh_port
        self.ssh_key = ssh_key
        self.ssh_options = ssh_options or [
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
        ]
        self.domain = domain
        self.virsh_uri = virsh_uri
        self.python_bin = python_bin
        self.display = display
        # Build the env prefix used for remote bash/python execution so the
        # generated pyautogui code runs against the guest's X session.
        self._remote_env = dict(env or {})
        if display:
            self._remote_env.setdefault("DISPLAY", display)
        if xauthority:
            self._remote_env.setdefault("XAUTHORITY", xauthority)

    # ---- ssh ----
    def _ssh_argv(self, remote_cmd: str) -> List[str]:
        cmd = ["ssh", "-p", str(self.ssh_port)]
        if self.ssh_key:
            cmd += ["-i", self.ssh_key]
        cmd += list(self.ssh_options)
        cmd += [f"{self.ssh_user}@{self.ssh_host}", remote_cmd]
        return cmd

    def _ssh_run(self, remote_cmd: str, timeout: int) -> Dict:
        try:
            proc = subprocess.run(
                self._ssh_argv(remote_cmd),
                capture_output=True, text=True, timeout=timeout,
            )
            return {
                "status": "ok" if proc.returncode == 0 else "error",
                "returncode": proc.returncode,
                "output": (proc.stdout or "") + (proc.stderr or ""),
                "error": "",
            }
        except subprocess.TimeoutExpired as e:
            return {"status": "error", "returncode": -1, "output": e.stdout or "", "error": f"TimeoutExpired: {e}"}

    # ---- qga ----
    def _virsh(self, *args: str) -> List[str]:
        cmd = ["virsh"]
        if self.virsh_uri:
            cmd += ["-c", self.virsh_uri]
        cmd += list(args)
        return cmd

    def _qga_run(self, argv: List[str], timeout: int) -> Dict:
        import json, time

        exec_payload = json.dumps({
            "execute": "guest-exec",
            "arguments": {
                "path": argv[0],
                "arg": argv[1:],
                "capture-output": True,
            },
        })
        try:
            proc = subprocess.run(
                self._virsh("qemu-agent-command", self.domain, exec_payload),
                capture_output=True, text=True, timeout=timeout,
            )
            if proc.returncode != 0:
                return {"status": "error", "returncode": -1, "output": proc.stdout, "error": proc.stderr}
            pid = json.loads(proc.stdout)["return"]["pid"]
            deadline = time.monotonic() + timeout
            while True:
                status_payload = json.dumps({
                    "execute": "guest-exec-status",
                    "arguments": {"pid": pid},
                })
                sp = subprocess.run(
                    self._virsh("qemu-agent-command", self.domain, status_payload),
                    capture_output=True, text=True, timeout=10,
                )
                if sp.returncode != 0:
                    return {"status": "error", "returncode": -1, "output": sp.stdout, "error": sp.stderr}
                ret = json.loads(sp.stdout)["return"]
                if ret.get("exited"):
                    import base64
                    out = base64.b64decode(ret.get("out-data", "")).decode("utf-8", "replace") if ret.get("out-data") else ""
                    err = base64.b64decode(ret.get("err-data", "")).decode("utf-8", "replace") if ret.get("err-data") else ""
                    code = int(ret.get("exitcode", -1))
                    return {
                        "status": "ok" if code == 0 else "error",
                        "returncode": code,
                        "output": out + err,
                        "error": "",
                    }
                if time.monotonic() >= deadline:
                    return {"status": "error", "returncode": -1, "output": "", "error": "qga exec timeout"}
                time.sleep(0.25)
        except subprocess.TimeoutExpired as e:
            return {"status": "error", "returncode": -1, "output": "", "error": f"TimeoutExpired: {e}"}

    def _env_prefix(self) -> str:
        if not self._remote_env:
            return ""
        return " ".join(f"{k}={shlex.quote(v)}" for k, v in self._remote_env.items()) + " "

    # ---- public ----
    def run_bash_script(self, code: str, timeout: int = 30) -> Dict:
        prefix = self._env_prefix()
        if self.transport == "ssh":
            result = self._ssh_run(f"{prefix}bash -lc {shlex.quote(code)}", timeout)
        else:
            result = self._qga_run(["/bin/bash", "-lc", code], timeout)
        print("BASH OUTPUT =======================================")
        print(result["output"])
        print("BASH OUTPUT =======================================")
        return result

    def run_python_script(self, code: str, timeout: int = 60) -> Dict:
        prefix = self._env_prefix()
        if self.transport == "ssh":
            remote = f"{prefix}{shlex.quote(self.python_bin)} -c {shlex.quote(code)}"
            result = self._ssh_run(remote, timeout)
        else:
            result = self._qga_run([self.python_bin, "-c", code], timeout)
        print("PYTHON OUTPUT =======================================")
        print(result["output"])
        print("PYTHON OUTPUT =======================================")
        return {
            "status": result["status"],
            "return_code": result["returncode"],
            "output": result["output"],
            "error": result["error"],
        }


    def screen_size(self, display: Optional[str] = None) -> "tuple[int, int]":
        display = display or self.display
        prefix = self._env_prefix()
        cmd = f"{prefix}xdpyinfo 2>/dev/null | awk '/dimensions:/ {{print $2}}'"
        if self.transport == "ssh":
            res = self._ssh_run(cmd, timeout=10)
        else:
            res = self._qga_run(["/bin/bash", "-lc", cmd], timeout=10)
        out = (res["output"] or "").strip()
        if "x" in out:
            try:
                w, h = out.split()[0].split("x")
                return int(w), int(h)
            except Exception:
                pass
        raise RuntimeError(f"could not determine screen size: {res}")

    def screenshot(self, display: Optional[str] = None) -> bytes:
        import base64
        prefix = self._env_prefix()
        py = (
            "import sys, base64, io\n"
            "try:\n"
            "    from PIL import ImageGrab\n"
            "except Exception:\n"
            "    import os; os.environ.setdefault('PYTHONPATH', '/home/user/.local/lib/python3.10/site-packages')\n"
            "    sys.path.insert(0, '/home/user/.local/lib/python3.10/site-packages')\n"
            "    from PIL import ImageGrab\n"
            "im = ImageGrab.grab()\n"
            "buf = io.BytesIO(); im.save(buf, format='PNG')\n"
            "sys.stdout.write('PNG:'+base64.b64encode(buf.getvalue()).decode())\n"
        )
        if self.transport == "ssh":
            import shlex as _sh
            remote = f"{prefix}{_sh.quote(self.python_bin)} -c {_sh.quote(py)}"
            res = self._ssh_run(remote, timeout=30)
        else:
            res = self._qga_run(["/bin/bash", "-lc", f"{prefix}{self.python_bin} -c {shlex.quote(py)}"], timeout=30)
        out = (res["output"] or "").strip()
        if out.startswith("PNG:"):
            return base64.b64decode(out[4:])
        raise RuntimeError(f"screenshot failed: {res}")


class KvmEnv:
    """Environment exposing a controller targeting a KVM/libvirt guest.

    Example (ssh):
        env = KvmEnv(transport="ssh", ssh_host="192.168.122.5", ssh_user="ubuntu")

    Example (qga):
        env = KvmEnv(transport="qga", domain="webtop-vm")
    """

    def __init__(self, **kwargs):
        self.controller = KvmController(**kwargs)
