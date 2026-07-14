"""Single-monitor screenshot capture for the isolated VM."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from datetime import datetime, timezone

import mss
from PIL import Image


@dataclass(frozen=True)
class CapturedObservation:
    png: bytes
    width: int
    height: int
    captured_at: str
    sha256: str

    def metadata(self) -> dict[str, str | int]:
        return {
            "width": self.width,
            "height": self.height,
            "captured_at": self.captured_at,
            "screenshot_sha256": self.sha256,
        }


class MSSCapture:
    """Capture exactly one 1920x1080 VM monitor."""

    def __init__(self, expected_width: int = 1920, expected_height: int = 1080):
        self.expected_width = expected_width
        self.expected_height = expected_height

    def capture(self) -> CapturedObservation:
        with mss.mss() as grabber:
            physical_monitors = grabber.monitors[1:]
            if len(physical_monitors) != 1:
                raise RuntimeError(
                    f"Observer requires exactly one monitor; found {len(physical_monitors)}"
                )
            monitor = physical_monitors[0]
            width = int(monitor["width"])
            height = int(monitor["height"])
            if (width, height) != (self.expected_width, self.expected_height):
                raise RuntimeError(
                    "Observer requires a "
                    f"{self.expected_width}x{self.expected_height} display; "
                    f"found {width}x{height}"
                )
            raw = grabber.grab(monitor)
        image = Image.frombytes("RGB", raw.size, raw.rgb)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        png = buffer.getvalue()
        return CapturedObservation(
            png=png,
            width=width,
            height=height,
            captured_at=datetime.now(timezone.utc).isoformat(),
            sha256=hashlib.sha256(png).hexdigest(),
        )
