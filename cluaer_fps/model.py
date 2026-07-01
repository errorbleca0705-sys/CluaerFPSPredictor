from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .types import Detection


HOSTILE_LABELS = {"enemy", "opponent", "hostile", "person"}
ACTOR_LABELS = {"enemy", "ally", "opponent", "hostile", "person", "player"}


class DeepLearningDetector:
    def __init__(
        self,
        model_path: str = "",
        labels_path: str = "models/labels.txt",
        backend: str = "auto",
        confidence: float = 0.35,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.labels_path = Path(labels_path) if labels_path else None
        self.backend = "disabled"
        self.confidence = confidence
        self.model: Any = None
        self.net: Any = None
        self.labels = self._load_labels()

        if not self.model_path or not self.model_path.exists():
            return

        suffix = self.model_path.suffix.lower()
        if backend in {"auto", "ultralytics"} and suffix == ".pt":
            self._try_load_ultralytics()
        elif backend in {"auto", "opencv-onnx"} and suffix == ".onnx":
            self._try_load_opencv_onnx()

    @property
    def status(self) -> str:
        if self.backend == "disabled":
            return "disabled"
        return f"{self.backend}:{self.model_path.name if self.model_path else ''}"

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if self.backend == "ultralytics":
            return self._detect_ultralytics(frame_bgr)
        if self.backend == "opencv-onnx":
            return self._detect_opencv_onnx(frame_bgr)
        return []

    def _load_labels(self) -> list[str]:
        if not self.labels_path or not self.labels_path.exists():
            return []
        with self.labels_path.open("r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    def _try_load_ultralytics(self) -> None:
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))
            self.backend = "ultralytics"
        except Exception:
            self.model = None
            self.backend = "disabled"

    def _try_load_opencv_onnx(self) -> None:
        try:
            self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
            self.backend = "opencv-onnx"
        except Exception:
            self.net = None
            self.backend = "disabled"

    def _detect_ultralytics(self, frame_bgr: np.ndarray) -> list[Detection]:
        if self.model is None:
            return []

        results = self.model.predict(
            source=frame_bgr,
            conf=self.confidence,
            verbose=False,
            imgsz=640,
        )
        detections: list[Detection] = []
        for result in results:
            names = getattr(result, "names", {}) or getattr(self.model, "names", {})
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < self.confidence:
                    continue
                class_id = int(box.cls[0])
                label = str(names.get(class_id, class_id))
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                detections.append(Detection(label, confidence, x1, y1, x2, y2))
        return detections

    def _detect_opencv_onnx(self, frame_bgr: np.ndarray) -> list[Detection]:
        if self.net is None:
            return []

        height, width = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame_bgr,
            scalefactor=1 / 255.0,
            size=(640, 640),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        output = self.net.forward()
        rows = self._normalize_yolo_output(output)

        detections: list[Detection] = []
        for row in rows:
            if len(row) < 6:
                continue
            box_confidence = float(row[4])
            class_scores = row[5:]
            class_id = int(np.argmax(class_scores)) if len(class_scores) else 0
            class_confidence = float(class_scores[class_id]) if len(class_scores) else 1.0
            confidence = box_confidence * class_confidence
            if confidence < self.confidence:
                continue

            cx, cy, bw, bh = [float(value) for value in row[:4]]
            x1 = int((cx - bw / 2) * width / 640)
            y1 = int((cy - bh / 2) * height / 640)
            x2 = int((cx + bw / 2) * width / 640)
            y2 = int((cy + bh / 2) * height / 640)
            label = self.labels[class_id] if class_id < len(self.labels) else str(class_id)
            detections.append(Detection(label, confidence, x1, y1, x2, y2))

        return self._nms(detections)

    def _normalize_yolo_output(self, output: np.ndarray) -> np.ndarray:
        data = np.squeeze(output)
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        if data.ndim == 2 and data.shape[0] < data.shape[1] and data.shape[0] < 100:
            data = data.T
        return data

    def _nms(self, detections: list[Detection]) -> list[Detection]:
        if not detections:
            return []

        boxes = [
            [det.x1, det.y1, max(0, det.x2 - det.x1), max(0, det.y2 - det.y1)]
            for det in detections
        ]
        scores = [det.confidence for det in detections]
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence, 0.45)
        if len(indices) == 0:
            return []
        flat_indices = np.array(indices).flatten().tolist()
        return [detections[index] for index in flat_indices]
