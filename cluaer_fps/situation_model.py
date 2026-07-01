from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import ManualContext, VisualSignals


DEFAULT_MODEL: dict[str, Any] = {
    "model_name": "cluaerAI FPS baseline situation model",
    "model_type": "visible_screen_heuristic_v1",
    "version": 1,
    "advantage": {
        "base_score": 50,
        "health_center": 50,
        "health_scale": 0.25,
        "team_delta_weight": 8,
        "ammo_low_threshold": 5,
        "ammo_low_penalty": -10,
        "ammo_ready_threshold": 20,
        "ammo_ready_bonus": 4,
        "utility_each_bonus": 2,
        "utility_bonus_cap": 8,
        "visible_hostile_penalty_each": -5,
        "visible_hostile_penalty_cap": -15,
        "red_flash_threshold": 0.08,
        "red_flash_penalty": -12,
        "whiteout_threshold": 0.25,
        "whiteout_penalty": -15,
        "smoke_threshold": 0.55,
        "smoke_penalty": -6,
        "center_edge_threshold": 0.13,
        "center_edge_bonus": 3,
        "motion_threshold": 0.12,
        "motion_penalty": -5,
        "min_score": 5,
        "max_score": 95,
    },
    "confidence": {
        "base": 35,
        "known_field_bonus": 4,
        "detector_bonus": 10,
        "visible_actor_bonus": 8,
        "visual_signal_bonus": 5,
        "min": 20,
        "max": 90,
    },
    "probabilities": {
        "valorant": {
            "win_min": 8,
            "win_max": 92,
            "fight_loss_base": 100,
            "fight_loss_visible_hostile_bonus": 8,
            "fight_loss_min": 5,
            "fight_loss_max": 88,
            "info_gain_base": 38,
            "info_gain_utility_bonus": 5,
            "info_gain_min": 10,
            "info_gain_max": 75,
        },
        "pubg": {
            "survive_offset": 5,
            "survive_min": 8,
            "survive_max": 94,
            "spotted_base": 55,
            "spotted_motion_scale": 60,
            "spotted_min": 8,
            "spotted_max": 85,
            "rotate_offset": -5,
            "rotate_min": 10,
            "rotate_max": 88,
        },
        "generic": {
            "win_min": 8,
            "win_max": 92,
            "survive_offset": 3,
            "survive_min": 8,
            "survive_max": 94,
            "reposition_base": 45,
            "reposition_utility_bonus": 4,
            "reposition_motion_scale": -20,
            "reposition_min": 8,
            "reposition_max": 85,
        },
    },
}


class SituationModel:
    def __init__(self, data: dict[str, Any] | None = None, source_path: str = "") -> None:
        self.data = self._merge(DEFAULT_MODEL, data or {})
        self.source_path = source_path

    @classmethod
    def load(cls, path: str = "models/fps_situation_model.json") -> "SituationModel":
        model_path = Path(path)
        if not model_path.exists():
            return cls(source_path="")

        with model_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(data=data, source_path=str(model_path))

    @property
    def status(self) -> str:
        name = self.data.get("model_name", "baseline")
        version = self.data.get("version", "unknown")
        if self.source_path:
            return f"{name} v{version} ({self.source_path})"
        return f"{name} v{version} (defaults)"

    def advantage_score(self, context: ManualContext, signals: VisualSignals) -> int:
        model = self.data["advantage"]
        score = float(model["base_score"])

        health = self._parse_int(context.health)
        if health is not None:
            score += (health - float(model["health_center"])) * float(model["health_scale"])

        allies = self._parse_int(context.alive_allies)
        enemies = self._parse_int(context.alive_enemies)
        if allies is not None and enemies is not None:
            score += (allies - enemies) * float(model["team_delta_weight"])

        ammo = self._parse_first_int(context.ammo)
        if ammo is not None:
            if ammo <= int(model["ammo_low_threshold"]):
                score += float(model["ammo_low_penalty"])
            elif ammo >= int(model["ammo_ready_threshold"]):
                score += float(model["ammo_ready_bonus"])

        utility_bonus = len(context.utility_list()) * float(model["utility_each_bonus"])
        score += min(float(model["utility_bonus_cap"]), utility_bonus)

        hostile_penalty = signals.visible_hostile_count * float(model["visible_hostile_penalty_each"])
        score += max(float(model["visible_hostile_penalty_cap"]), hostile_penalty)

        if signals.red_flash_score > float(model["red_flash_threshold"]):
            score += float(model["red_flash_penalty"])
        if signals.whiteout_score > float(model["whiteout_threshold"]):
            score += float(model["whiteout_penalty"])
        if signals.smoke_score > float(model["smoke_threshold"]):
            score += float(model["smoke_penalty"])
        if signals.center_edge_density > float(model["center_edge_threshold"]):
            score += float(model["center_edge_bonus"])
        if signals.motion_score > float(model["motion_threshold"]):
            score += float(model["motion_penalty"])

        return self.clamp(score, int(model["min_score"]), int(model["max_score"]))

    def confidence(self, context: ManualContext, signals: VisualSignals) -> int:
        model = self.data["confidence"]
        known_fields = [
            context.game,
            context.health,
            context.weapon,
            context.ammo,
            context.alive_allies,
            context.alive_enemies,
            context.position,
            context.remaining_time,
            context.minimap_info,
            context.kill_log,
        ]
        score = float(model["base"])
        score += sum(float(model["known_field_bonus"]) for value in known_fields if value.strip())
        if signals.detector_status != "disabled":
            score += float(model["detector_bonus"])
        if signals.visible_actor_count:
            score += float(model["visible_actor_bonus"])
        if signals.red_flash_score or signals.whiteout_score or signals.motion_score:
            score += float(model["visual_signal_bonus"])
        return self.clamp(score, int(model["min"]), int(model["max"]))

    def game_probabilities(self, game: str) -> dict[str, Any]:
        probabilities = self.data["probabilities"]
        if game in probabilities:
            return probabilities[game]
        return probabilities["generic"]

    def clamp(self, value: float, low: int, high: int) -> int:
        return max(low, min(high, int(round(value))))

    def _parse_int(self, value: str) -> int | None:
        value = value.strip()
        if not value:
            return None
        digits = "".join(ch for ch in value if ch.isdigit())
        return int(digits) if digits else None

    def _parse_first_int(self, value: str) -> int | None:
        chunk = ""
        for char in value:
            if char.isdigit():
                chunk += char
            elif chunk:
                break
        return int(chunk) if chunk else None

    def _merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge(merged[key], value)
            else:
                merged[key] = value
        return merged
