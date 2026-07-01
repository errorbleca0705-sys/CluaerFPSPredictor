from __future__ import annotations

import cv2
import numpy as np

from .model import ACTOR_LABELS, HOSTILE_LABELS, DeepLearningDetector
from .types import VisualSignals


class FrameAnalyzer:
    def __init__(self, detector: DeepLearningDetector) -> None:
        self.detector = detector
        self.previous_gray: np.ndarray | None = None

    def analyze(self, frame_bgr: np.ndarray) -> VisualSignals:
        height, width = frame_bgr.shape[:2]
        center = self._center_crop(frame_bgr, 0.45)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        detections = self.detector.detect(frame_bgr)
        visible_actor_count = sum(1 for det in detections if det.label.lower() in ACTOR_LABELS)
        visible_hostile_count = sum(
            1 for det in detections if det.label.lower() in HOSTILE_LABELS
        )

        red_flash_score = self._red_flash_score(frame_bgr)
        whiteout_score = self._whiteout_score(frame_bgr)
        smoke_score = self._smoke_score(hsv)
        edge_density = self._edge_density(center)
        motion_score = self._motion_score(gray)
        danger_direction = self._danger_direction(detections, width)

        return VisualSignals(
            red_flash_score=round(red_flash_score, 3),
            whiteout_score=round(whiteout_score, 3),
            smoke_score=round(smoke_score, 3),
            center_edge_density=round(edge_density, 3),
            motion_score=round(motion_score, 3),
            visible_actor_count=visible_actor_count,
            visible_hostile_count=visible_hostile_count,
            detections=detections,
            danger_direction=danger_direction,
            detector_status=self.detector.status,
        )

    def _center_crop(self, frame_bgr: np.ndarray, scale: float) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        crop_w = int(width * scale)
        crop_h = int(height * scale)
        x1 = max(0, (width - crop_w) // 2)
        y1 = max(0, (height - crop_h) // 2)
        return frame_bgr[y1 : y1 + crop_h, x1 : x1 + crop_w]

    def _red_flash_score(self, frame_bgr: np.ndarray) -> float:
        height, width = frame_bgr.shape[:2]
        margin_x = max(1, width // 8)
        margin_y = max(1, height // 8)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[:margin_y, :] = 1
        mask[-margin_y:, :] = 1
        mask[:, :margin_x] = 1
        mask[:, -margin_x:] = 1

        b, g, r = cv2.split(frame_bgr)
        red_dominant = (r.astype(np.int16) - np.maximum(b, g).astype(np.int16)) > 45
        red_pixels = np.logical_and(red_dominant, mask == 1)
        return float(np.mean(red_pixels))

    def _whiteout_score(self, frame_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray > 235))

    def _smoke_score(self, hsv: np.ndarray) -> float:
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        smoke_like = np.logical_and(saturation < 35, np.logical_and(value > 70, value < 220))
        return float(np.mean(smoke_like))

    def _edge_density(self, frame_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        return float(np.mean(edges > 0))

    def _motion_score(self, gray: np.ndarray) -> float:
        small = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
        if self.previous_gray is None:
            self.previous_gray = small
            return 0.0

        diff = cv2.absdiff(small, self.previous_gray)
        self.previous_gray = small
        return float(np.mean(diff) / 255.0)

    def _danger_direction(self, detections: list, width: int) -> str:
        if not detections:
            return "확인 필요"

        primary = max(detections, key=lambda det: det.confidence)
        third = width / 3
        if primary.center_x < third:
            return "좌측 화면 영역에서 감지됨"
        if primary.center_x > third * 2:
            return "우측 화면 영역에서 감지됨"
        return "화면 중앙 근처에서 감지됨"
