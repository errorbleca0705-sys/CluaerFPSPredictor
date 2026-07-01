from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AppConfig:
    capture_fps: int = 8
    analysis_fps: int = 2
    monitor_index: int = 1
    preview_width: int = 720
    situation_model_path: str = "models/fps_situation_model.json"
    model_backend: str = "auto"
    model_path: str = ""
    labels_path: str = "models/labels.txt"
    detection_confidence: float = 0.35
    auto_log_interval_sec: int = 10
    data_path: str = "data/predictions.jsonl"

    @classmethod
    def load(cls, path: str | Path = "config.json") -> "AppConfig":
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with config_path.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = json.load(handle)

        valid_keys = set(cls.__dataclass_fields__.keys())
        values = {key: value for key, value in raw.items() if key in valid_keys}
        return cls(**values)

    def save(self, path: str | Path = "config.json") -> None:
        config_path = Path(path)
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, ensure_ascii=False, indent=2)
