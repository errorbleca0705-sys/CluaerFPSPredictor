from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ManualContext:
    game: str = "auto"
    map_name: str = ""
    mode: str = ""
    score: str = ""
    remaining_time: str = ""
    health: str = ""
    armor: str = ""
    weapon: str = ""
    ammo: str = ""
    utility: str = ""
    alive_allies: str = ""
    alive_enemies: str = ""
    position: str = ""
    minimap_info: str = ""
    kill_log: str = ""

    def utility_list(self) -> list[str]:
        return [item.strip() for item in self.utility.split(",") if item.strip()]


@dataclass(slots=True)
class VisualSignals:
    red_flash_score: float = 0.0
    whiteout_score: float = 0.0
    smoke_score: float = 0.0
    center_edge_density: float = 0.0
    motion_score: float = 0.0
    visible_actor_count: int = 0
    visible_hostile_count: int = 0
    detections: list[Detection] = field(default_factory=list)
    danger_direction: str = "확인 필요"
    detector_status: str = "disabled"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["detections"] = [d.to_dict() for d in self.detections]
        return data


@dataclass(slots=True)
class PredictionItem:
    result: str
    probability: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class PredictionReport:
    screen_summary: str
    state_judgement: str
    predictions: list[PredictionItem]
    risks: list[str]
    safe_action: str
    aggressive_action: str
    team_action: str
    avoid_action: str
    confidence: int
    visual_signals: VisualSignals
    manual_context: ManualContext
    main_prediction: str

    def learning_data(self) -> dict[str, Any]:
        ctx = self.manual_context
        return {
            "game": ctx.game,
            "map": ctx.map_name,
            "mode": ctx.mode,
            "round_or_phase": ctx.score,
            "screen_summary": self.screen_summary,
            "player_state": {
                "health": ctx.health,
                "armor": ctx.armor,
                "weapon": ctx.weapon,
                "ammo": ctx.ammo,
                "utility": ctx.utility_list(),
            },
            "team_state": {
                "alive_allies": ctx.alive_allies,
                "alive_enemies": ctx.alive_enemies,
                "advantage": self.state_judgement,
            },
            "environment_state": {
                "position": ctx.position,
                "cover": "화면 기반 추정 필요",
                "danger_direction": self.visual_signals.danger_direction,
                "time_pressure": ctx.remaining_time,
            },
            "prediction": [item.to_dict() for item in self.predictions],
            "main_prediction": self.main_prediction,
            "confidence": f"{self.confidence}%",
            "visual_signals": self.visual_signals.to_dict(),
            "actual_result": None,
            "accuracy_checked": False,
            "learning_note": "",
        }
