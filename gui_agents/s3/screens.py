"""Physical monitor geometry and capture for the supervised desktop worker."""

import io
import os


def enable_physical_coordinates():
    if os.name == "nt":
        import ctypes

        # Set this before importing GUI/input libraries in the worker process.
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))


def monitors():
    import mss

    with mss.mss() as capture:
        return [
            {"id": i, **dict(monitor)}
            for i, monitor in enumerate(capture.monitors)
            if i
        ]


def selected_monitor(index):
    choices = monitors()
    for monitor in choices:
        if monitor["id"] == index:
            return monitor
    raise ValueError(f"Monitor {index} is unavailable")


def capture_screenshot_png(monitor, max_dimension=2400):
    import mss
    from PIL import Image

    with mss.mss() as capture:
        shot = capture.grab({k: monitor[k] for k in ("left", "top", "width", "height")})
        image = Image.frombytes("RGB", shot.size, shot.rgb)
    scale = min(1, max_dimension / image.width, max_dimension / image.height)
    if scale < 1:
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def to_ui_jpeg(png):
    from PIL import Image

    image = Image.open(io.BytesIO(png)).convert("RGB")
    image.thumbnail((960, 960))
    output = io.BytesIO()
    image.save(output, "JPEG", quality=70)
    return output.getvalue()


def foreground_window():
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = wintypes.HWND
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    rect = wintypes.RECT()
    title = ctypes.create_unicode_buffer(1024)
    user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect))
    user32.GetWindowTextW(wintypes.HWND(hwnd), title, len(title))
    return {
        "hwnd": hwnd,
        "title": title.value,
        "rect": [rect.left, rect.top, rect.right, rect.bottom],
    }


def restore_foreground_window(snapshot):
    if snapshot is None:
        return
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    hwnd = wintypes.HWND(snapshot["hwnd"])
    if not user32.IsWindow(hwnd):
        raise InterruptedError("The approved target window was closed")
    rect = wintypes.RECT()
    title = ctypes.create_unicode_buffer(1024)
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    user32.GetWindowTextW(hwnd, title, len(title))
    if [rect.left, rect.top, rect.right, rect.bottom] != snapshot[
        "rect"
    ] or title.value != snapshot["title"]:
        raise InterruptedError(
            "The target window changed during approval; replan the task"
        )
    from pywinauto.controls.hwndwrapper import HwndWrapper

    HwndWrapper(snapshot["hwnd"]).set_focus()
    user32.GetForegroundWindow.restype = wintypes.HWND
    if user32.GetForegroundWindow() != snapshot["hwnd"]:
        raise InterruptedError("Could not restore the approved target window's focus")
