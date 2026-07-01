from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PredictionStorage:
    def __init__(self, path: str = "data/predictions.jsonl") -> None:
        self.path = Path(path)
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        with self.lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def append_feedback(self, record: dict[str, Any], actual_result: str) -> None:
        payload = dict(record)
        payload["actual_result"] = actual_result
        payload["accuracy_checked"] = True
        payload["learning_note"] = self._learning_note(record, actual_result)
        self.append(payload)

    def _learning_note(self, record: dict[str, Any], actual_result: str) -> str:
        main_prediction = record.get("main_prediction", "")
        if not main_prediction:
            return f"실제 결과 피드백 저장: {actual_result}"
        return f"예측 '{main_prediction}' 이후 실제 결과: {actual_result}"
