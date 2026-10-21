"""Accessibility tree primitives."""
import re
import subprocess

_ATSPI_ADDR = None


def _atspi_address():
    """Resolve and cache the AT-SPI bus address. Returns None if unavailable."""
    global _ATSPI_ADDR
    if _ATSPI_ADDR:
        return _ATSPI_ADDR
    r = subprocess.run(
        ["gdbus", "call", "--session",
         "--dest", "org.a11y.Bus",
         "--object-path", "/org/a11y/bus",
         "--method", "org.a11y.Bus.GetAddress"],
        capture_output=True, text=True, timeout=5,
    )
    if r.returncode != 0:
        return None
    m = re.match(r"^\('([^']+)'", r.stdout.strip())
    _ATSPI_ADDR = m.group(1) if m else None
    return _ATSPI_ADDR


def get_accessibility_tree():
    """Return the AT-SPI accessibility tree's root children via gdbus.
    Result is {"stdout", "stderr", "returncode"} — caller parses the gdbus output."""
    addr = _atspi_address()
    if not addr:
        return {"stdout": "", "stderr": "AT-SPI bus address unavailable", "returncode": 2}
    cmd = ["gdbus", "call", "--address", addr,
           "--dest", "org.a11y.atspi.Registry",
           "--object-path", "/org/a11y/atspi/accessible/root",
           "--method", "org.a11y.atspi.Accessible.GetChildren"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    return {"stdout": r.stdout, "stderr": r.stderr, "returncode": r.returncode}


def get_window_list():
    """List X11 windows via `wmctrl -l -p`. Returns the raw stdout as a string
    (one line per window: window-id, desktop, pid, host, title)."""
    r = subprocess.run("wmctrl -l -p", shell=True, capture_output=True, text=True, timeout=5)
    return r.stdout


def get_focused_element():
    """Return the AT-SPI role of the currently focused element as a string
    (whatever gdbus prints — typically a tuple-ish wrapper around a role name)."""
    addr = _atspi_address()
    if not addr:
        return ""
    cmd = ["gdbus", "call", "--address", addr,
           "--dest", "org.a11y.atspi.Registry",
           "--object-path", "/org/a11y/atspi/accessible/root",
           "--method", "org.a11y.atspi.Accessible.GetRole"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    return r.stdout.strip()
