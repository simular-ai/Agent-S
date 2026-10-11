"""Optional desktop narration in an isolated Tk process.

No Tk object crosses a thread or process boundary. The window remains withdrawn
from screenshot capture through action execution, including approval transitions.
"""

import json
import queue
import subprocess
import sys
import threading
import uuid


class OverlayController:
    def __init__(self):
        self.process = None
        self.responses = queue.Queue()
        self.lock = threading.RLock()
        self.visible = False

    def _read(self):
        for line in self.process.stdout:
            try:
                self.responses.put(json.loads(line))
            except ValueError:
                pass

    def _request(self, command, **data):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError("Overlay process is unavailable")
            token = uuid.uuid4().hex
            try:
                self.process.stdin.write(
                    json.dumps({"command": command, "token": token, **data}) + "\n"
                )
                self.process.stdin.flush()
                while True:
                    response = self.responses.get(timeout=3)
                    if response.get("token") == token:
                        if response.get("error"):
                            raise RuntimeError(response["error"])
                        return response
            except (OSError, queue.Empty, RuntimeError):
                # Timeout must not leave a queued hide/show against a live window.
                self.close(force=True)
                raise RuntimeError("Overlay acknowledgment failed; window closed")

    def show(self, task, models, monitor):
        self.close()
        self.responses = queue.Queue()
        self.process = subprocess.Popen(
            [sys.executable, "-m", "gui_agents.s3.overlay", "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        threading.Thread(target=self._read, daemon=True).start()
        try:
            self._request("init", task=task[:160], models=models[:90], monitor=monitor)
            self.restore()
            return "ok"
        except RuntimeError as exc:
            self.close(force=True)
            return f"unavailable: {exc}"

    def update(self, step, action=""):
        if self.process is not None:
            try:
                self._request("update", step=step[:160], action=action[:400])
            except RuntimeError:
                pass

    def withdraw(self):
        if self.process is None:
            return
        try:
            self._request("hide")
            self.visible = False
        except RuntimeError:
            self.close(force=True)

    def restore(self):
        if self.process is None:
            return
        try:
            self._request("show")
            self.visible = True
        except RuntimeError:
            self.close(force=True)

    def close(self, force=False):
        with self.lock:
            process = self.process
            if process is None:
                return
            if not force and process.poll() is None:
                try:
                    self._request("close")
                except RuntimeError:
                    pass
            self.process = None
            self.visible = False
            try:
                if force and process.poll() is None:
                    process.kill()
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            finally:
                process.stdin.close()
                process.stdout.close()


def _window_main():
    from gui_agents.s3.screens import enable_physical_coordinates

    enable_physical_coordinates()
    import tkinter as tk

    first = json.loads(sys.stdin.readline())
    token = first["token"]

    def reply(token, error=None):
        print(json.dumps({"token": token, "error": error}), flush=True)

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.configure(bg="#14181f")
        monitor = first["monitor"]
        left = monitor["left"] + max(0, monitor["width"] - 416)
        top = monitor["top"] + max(0, monitor["height"] - 256)
        root.geometry(f"400x240{left:+d}{top:+d}")
        for text, color in (
            ("AGENT S3", "#4da3ff"),
            (first["task"], "#e8ecf1"),
            (first["models"], "#9aa4b2"),
        ):
            tk.Label(
                root,
                text=text,
                bg="#14181f",
                fg=color,
                anchor="w",
                justify="left",
                wraplength=380,
            ).pack(fill="x", padx=10, pady=4)
        step = tk.Label(root, text="Starting", bg="#14181f", fg="#e0a63c", anchor="w")
        step.pack(fill="x", padx=10)
        action = tk.Label(
            root,
            bg="#14181f",
            fg="#c9d1d9",
            anchor="nw",
            justify="left",
            wraplength=380,
        )
        action.pack(fill="both", expand=True, padx=10, pady=8)
        root.update_idletasks()
        if sys.platform == "win32":
            import ctypes

            user32 = ctypes.windll.user32
            hwnd = user32.GetParent(root.winfo_id())
            style = user32.GetWindowLongW(hwnd, -20)
            user32.SetWindowLongW(hwnd, -20, style | 0x08000000 | 0x00000020)
            user32.SetWindowPos(hwnd, -1, left, top, 400, 240, 0x0010)
        pending = queue.Queue()

        def read_commands():
            for line in sys.stdin:
                try:
                    pending.put(json.loads(line))
                except ValueError:
                    pass
            pending.put({"command": "close", "token": "eof"})

        threading.Thread(target=read_commands, daemon=True).start()

        def poll():
            while True:
                try:
                    item = pending.get_nowait()
                except queue.Empty:
                    break
                command = item["command"]
                if command == "close":
                    reply(item["token"])
                    root.quit()
                    return
                if command == "hide":
                    root.withdraw()
                    root.update_idletasks()
                elif command == "show":
                    root.deiconify()
                elif command == "update":
                    step.configure(text=item["step"])
                    action.configure(text=item["action"])
                reply(item["token"])
            root.after(20, poll)

        reply(token)
        root.after(20, poll)
        root.mainloop()
    except Exception as exc:
        reply(token, str(exc))
    finally:
        if root is not None:
            root.destroy()


if __name__ == "__main__":
    _window_main()
