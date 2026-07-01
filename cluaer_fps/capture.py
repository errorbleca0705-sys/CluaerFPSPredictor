from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import mss
import numpy as np


@dataclass(slots=True)
class MonitorInfo:
    index: int
    left: int
    top: int
    width: int
    height: int

    @property
    def label(self) -> str:
        return f"Monitor {self.index} ({self.width}x{self.height})"


class ScreenCapture:
    def list_monitors(self) -> list[MonitorInfo]:
        with mss.mss() as sct:
            monitors = []
            for index, monitor in enumerate(sct.monitors):
                if index == 0:
                    continue
                monitors.append(
                    MonitorInfo(
                        index=index,
                        left=int(monitor["left"]),
                        top=int(monitor["top"]),
                        width=int(monitor["width"]),
                        height=int(monitor["height"]),
                    )
                )
            return monitors

    def grab(self, monitor_index: int = 1) -> np.ndarray:
        with mss.mss() as sct:
            monitors: list[dict[str, Any]] = sct.monitors
            if monitor_index < 1 or monitor_index >= len(monitors):
                monitor_index = 1

            raw = np.array(sct.grab(monitors[monitor_index]), dtype=np.uint8)
            return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

    def session(self) -> "ScreenCaptureSession":
        return ScreenCaptureSession()


class ScreenCaptureSession:
    def __init__(self) -> None:
        self.sct: mss.mss | None = None

    def __enter__(self) -> "ScreenCaptureSession":
        self.sct = mss.mss()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.sct is not None:
            self.sct.close()
            self.sct = None

    def grab(self, monitor_index: int = 1) -> np.ndarray:
        if self.sct is None:
            raise RuntimeError("ScreenCaptureSession is not open.")

        monitors: list[dict[str, Any]] = self.sct.monitors
        if monitor_index < 1 or monitor_index >= len(monitors):
            monitor_index = 1

        raw = np.array(self.sct.grab(monitors[monitor_index]), dtype=np.uint8)
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
