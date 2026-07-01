from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


POSITIVE_KEYWORDS = [
    "승",
    "성공",
    "생존",
    "클러치",
    "이김",
    "win",
    "won",
    "success",
    "survive",
    "survived",
    "clutch",
]
NEGATIVE_KEYWORDS = [
    "패",
    "실패",
    "사망",
    "죽",
    "탈락",
    "loss",
    "lost",
    "fail",
    "failed",
    "died",
    "dead",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the baseline situation model from saved feedback JSONL."
    )
    parser.add_argument("--data", default="data/predictions.jsonl")
    parser.add_argument("--model", default="models/fps_situation_model.json")
    parser.add_argument("--out", default="models/fps_situation_model.json")
    args = parser.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model)
    out_path = Path(args.out)

    if not data_path.exists():
        raise SystemExit(f"Feedback data not found: {data_path}")
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    records = list(load_feedback_records(data_path))
    if len(records) < 5:
        raise SystemExit("Need at least 5 feedback records before calibration.")

    positive_count = 0
    probability_sum = 0.0
    probability_count = 0

    for record in records:
        actual = str(record.get("actual_result", ""))
        positive_count += int(classify_actual_result(actual))
        probability = first_prediction_probability(record)
        if probability is not None:
            probability_sum += probability
            probability_count += 1

    positive_rate = positive_count / len(records)
    avg_probability = probability_sum / probability_count if probability_count else 0.5
    calibration_error = positive_rate - avg_probability

    with model_path.open("r", encoding="utf-8") as handle:
        model: dict[str, Any] = json.load(handle)

    model["version"] = int(model.get("version", 1)) + 1
    model["calibration"] = {
        "feedback_records": len(records),
        "positive_rate": round(positive_rate, 4),
        "average_main_probability": round(avg_probability, 4),
        "calibration_error": round(calibration_error, 4),
    }

    advantage = model.setdefault("advantage", {})
    base_score = float(advantage.get("base_score", 50))
    advantage["base_score"] = round(max(35, min(65, base_score + calibration_error * 10)), 2)

    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(model, handle, ensure_ascii=False, indent=2)

    print(f"Calibrated {out_path}")
    print(f"feedback_records={len(records)}")
    print(f"positive_rate={positive_rate:.2%}")
    print(f"avg_main_probability={avg_probability:.2%}")
    print(f"new_base_score={advantage['base_score']}")


def load_feedback_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("accuracy_checked") and record.get("actual_result"):
                records.append(record)
    return records


def classify_actual_result(actual_result: str) -> bool:
    text = actual_result.lower()
    positive = any(keyword in text for keyword in POSITIVE_KEYWORDS)
    negative = any(keyword in text for keyword in NEGATIVE_KEYWORDS)
    if positive and not negative:
        return True
    if negative and not positive:
        return False
    return positive


def first_prediction_probability(record: dict[str, Any]) -> float | None:
    predictions = record.get("prediction")
    if not isinstance(predictions, list) or not predictions:
        return None

    probability = predictions[0].get("probability")
    if not isinstance(probability, str):
        return None

    clean = probability.strip().replace("%", "")
    try:
        return float(clean) / 100.0
    except ValueError:
        return None


if __name__ == "__main__":
    main()
