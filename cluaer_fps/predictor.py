from __future__ import annotations

from .situation_model import SituationModel
from .types import ManualContext, PredictionItem, PredictionReport, VisualSignals


class PredictionEngine:
    def __init__(self, situation_model: SituationModel | None = None) -> None:
        self.situation_model = situation_model or SituationModel()

    def predict(self, context: ManualContext, signals: VisualSignals) -> PredictionReport:
        game = context.game.lower()
        score = self._estimate_advantage_score(context, signals)
        confidence = self._confidence(context, signals)
        judgement = self._judgement(score)
        summary = self._summary(context, signals)

        if game == "valorant":
            predictions = self._valorant_predictions(context, signals, score)
            risks = self._valorant_risks(context, signals)
        elif game in {"pubg", "battlegrounds", "배틀그라운드"}:
            predictions = self._pubg_predictions(context, signals, score)
            risks = self._pubg_risks(context, signals)
        else:
            predictions = self._generic_predictions(context, signals, score)
            risks = self._generic_risks(context, signals)

        main_prediction = predictions[0].result if predictions else "확인 필요"
        safe_action, aggressive_action, team_action, avoid_action = self._actions(
            context, signals, score
        )

        return PredictionReport(
            screen_summary=summary,
            state_judgement=judgement,
            predictions=predictions,
            risks=risks,
            safe_action=safe_action,
            aggressive_action=aggressive_action,
            team_action=team_action,
            avoid_action=avoid_action,
            confidence=confidence,
            visual_signals=signals,
            manual_context=context,
            main_prediction=main_prediction,
        )

    def format_report(self, report: PredictionReport) -> str:
        ctx = report.manual_context
        sig = report.visual_signals
        risks = "\n".join(f"- {risk}" for risk in report.risks)
        predictions = "\n\n".join(
            (
                f"{index}. 결과: {item.result}\n"
                f"   - 확률: {item.probability}\n"
                f"   - 근거: {item.reason}"
            )
            for index, item in enumerate(report.predictions, start=1)
        )

        return (
            "## cluaerAI FPS 예측 결과\n\n"
            "### 1. 현재 상황 요약\n"
            f"{report.screen_summary}\n\n"
            "### 2. 감지된 핵심 정보\n"
            f"- 게임: {ctx.game or '확인 필요'}\n"
            f"- 맵: {ctx.map_name or '확인 필요'}\n"
            f"- 모드: {ctx.mode or '확인 필요'}\n"
            f"- 현재 점수: {ctx.score or '확인 필요'}\n"
            f"- 남은 시간: {ctx.remaining_time or '확인 필요'}\n"
            f"- 체력: {ctx.health or '확인 필요'}\n"
            f"- 방어구: {ctx.armor or '확인 필요'}\n"
            f"- 무기: {ctx.weapon or '확인 필요'}\n"
            f"- 탄약: {ctx.ammo or '확인 필요'}\n"
            f"- 스킬/아이템: {ctx.utility or '확인 필요'}\n"
            f"- 아군 상태: {ctx.alive_allies or '확인 필요'}\n"
            f"- 적 상태: {ctx.alive_enemies or '확인 필요'}\n"
            f"- 위치: {ctx.position or '확인 필요'}\n"
            f"- 미니맵 정보: {ctx.minimap_info or '확인 필요'}\n"
            f"- 킬로그: {ctx.kill_log or '확인 필요'}\n"
            f"- 위험 방향: {sig.danger_direction}\n"
            f"- 화면 기반 신호: red={sig.red_flash_score}, flash={sig.whiteout_score}, "
            f"smoke={sig.smoke_score}, motion={sig.motion_score}\n"
            f"- 딥러닝 감지: {sig.detector_status}, visible={sig.visible_actor_count}, "
            f"hostile_estimate={sig.visible_hostile_count}\n\n"
            "### 3. 현재 상태 판단\n"
            f"{report.state_judgement}\n\n"
            "### 4. 가능한 결과 예측\n"
            f"{predictions}\n\n"
            "### 5. 가장 위험한 요소\n"
            f"{risks}\n\n"
            "### 6. 추천 행동\n"
            f"- 안전한 선택: {report.safe_action}\n"
            f"- 공격적인 선택: {report.aggressive_action}\n"
            f"- 팀 플레이 선택: {report.team_action}\n"
            f"- 피해야 할 행동: {report.avoid_action}\n\n"
            "### 7. 예측 신뢰도\n"
            f"현재 정보 기준 예측 신뢰도: {report.confidence}%\n\n"
            "### 8. 저장할 학습 데이터\n"
            "앱 내부에서 JSONL로 저장됩니다.\n\n"
            "### 9. 추가로 확인하면 좋은 정보\n"
            "- 정확한 체력, 탄약, 아군/적 생존 수\n"
            "- 미니맵에 보이는 마지막 적 위치\n"
            "- 스파이크/자기장/궁극기/투척물 상태\n"
            "- 실제 결과 피드백"
        )

    def _estimate_advantage_score(self, context: ManualContext, signals: VisualSignals) -> int:
        return self.situation_model.advantage_score(context, signals)

    def _confidence(self, context: ManualContext, signals: VisualSignals) -> int:
        return self.situation_model.confidence(context, signals)

    def _judgement(self, score: int) -> str:
        if score >= 62:
            return "현재 상황은 유리한 편입니다. 다만 화면 기반 추정이므로 보이지 않는 적 위치는 확인이 필요합니다."
        if score <= 42:
            return "현재 상황은 불리한 편입니다. 정보 부족, 피격 신호, 인원 열세 또는 시야 방해가 위험 요소입니다."
        return "현재 상황은 중립에 가깝습니다. 추가 정보와 첫 교전 결과에 따라 크게 흔들릴 수 있습니다."

    def _summary(self, context: ManualContext, signals: VisualSignals) -> str:
        game = context.game if context.game else "FPS 게임"
        position = context.position if context.position else "현재 위치 확인 필요"
        visible = (
            f"화면에서 감지된 대상 {signals.visible_actor_count}개"
            if signals.visible_actor_count
            else "화면에서 확정 감지된 대상 없음"
        )
        return (
            f"{game} 화면을 실시간 캡처 중입니다. 위치는 {position}이며, {visible}입니다. "
            f"위험 방향은 {signals.danger_direction}으로 추정됩니다."
        )

    def _valorant_predictions(
        self, context: ManualContext, signals: VisualSignals, score: int
    ) -> list[PredictionItem]:
        model = self.situation_model.game_probabilities("valorant")
        win = self.situation_model.clamp(score, int(model["win_min"]), int(model["win_max"]))
        fight_loss = self.situation_model.clamp(
            float(model["fight_loss_base"])
            - score
            + (float(model["fight_loss_visible_hostile_bonus"]) if signals.visible_hostile_count else 0),
            int(model["fight_loss_min"]),
            int(model["fight_loss_max"]),
        )
        info_gain = self.situation_model.clamp(
            float(model["info_gain_base"])
            + len(context.utility_list()) * float(model["info_gain_utility_bonus"]),
            int(model["info_gain_min"]),
            int(model["info_gain_max"]),
        )
        return [
            PredictionItem(
                "현재 라운드 승리 가능",
                f"{win}%",
                "체력, 인원 수, 탄약, 스킬 보유량, 화면 위험 신호를 종합한 점수입니다.",
            ),
            PredictionItem(
                "다음 교전에서 먼저 손해를 볼 가능성",
                f"{fight_loss}%",
                "확인된 적 정보가 부족하거나 화면 중앙 위험, 피격/시야 방해 신호가 있으면 상승합니다.",
            ),
            PredictionItem(
                "스킬 또는 팀 동시 진입으로 정보 우위 확보",
                f"{info_gain}%",
                "스킬/아이템 입력값과 현재 위험 방향 추정을 기반으로 계산했습니다.",
            ),
        ]

    def _pubg_predictions(
        self, context: ManualContext, signals: VisualSignals, score: int
    ) -> list[PredictionItem]:
        model = self.situation_model.game_probabilities("pubg")
        survive = self.situation_model.clamp(
            score + float(model["survive_offset"]),
            int(model["survive_min"]),
            int(model["survive_max"]),
        )
        spotted = self.situation_model.clamp(
            float(model["spotted_base"]) - score + signals.motion_score * float(model["spotted_motion_scale"]),
            int(model["spotted_min"]),
            int(model["spotted_max"]),
        )
        rotate = self.situation_model.clamp(
            score + float(model["rotate_offset"]),
            int(model["rotate_min"]),
            int(model["rotate_max"]),
        )
        return [
            PredictionItem(
                "다음 1분 생존",
                f"{survive}%",
                "체력, 아이템, 노출 신호, 화면 움직임, 감지 대상을 합산했습니다.",
            ),
            PredictionItem(
                "이동 중 적에게 먼저 발견될 가능성",
                f"{spotted}%",
                "화면 움직임이 크거나 엄폐/위치 정보가 부족하면 위험도가 올라갑니다.",
            ),
            PredictionItem(
                "안전 지점으로 이동 성공",
                f"{rotate}%",
                "현재 위치와 아이템 정보가 부족하면 보수적으로 낮게 잡습니다.",
            ),
        ]

    def _generic_predictions(
        self, context: ManualContext, signals: VisualSignals, score: int
    ) -> list[PredictionItem]:
        model = self.situation_model.game_probabilities("generic")
        win = self.situation_model.clamp(score, int(model["win_min"]), int(model["win_max"]))
        survive = self.situation_model.clamp(
            score + float(model["survive_offset"]),
            int(model["survive_min"]),
            int(model["survive_max"]),
        )
        reposition = self.situation_model.clamp(
            float(model["reposition_base"])
            + len(context.utility_list()) * float(model["reposition_utility_bonus"])
            + signals.motion_score * float(model["reposition_motion_scale"]),
            int(model["reposition_min"]),
            int(model["reposition_max"]),
        )
        return [
            PredictionItem(
                "현재 교전 또는 다음 교전에서 우세",
                f"{win}%",
                "입력된 상태 정보와 화면 기반 위험 신호를 합산했습니다.",
            ),
            PredictionItem(
                "다음 30초 생존",
                f"{survive}%",
                "피격 신호, 시야 방해, 화면 움직임, 감지 대상을 반영했습니다.",
            ),
            PredictionItem(
                "위치 재정비 후 정보 우위 확보",
                f"{reposition}%",
                "스킬/아이템 보유량과 현재 위험 방향이 핵심 변수입니다.",
            ),
        ]

    def _valorant_risks(self, context: ManualContext, signals: VisualSignals) -> list[str]:
        risks = self._generic_risks(context, signals)
        if not context.remaining_time:
            risks.append("라운드 시간과 스파이크 상태가 입력되지 않아 시간 압박 판단이 제한됨")
        return risks[:5]

    def _pubg_risks(self, context: ManualContext, signals: VisualSignals) -> list[str]:
        risks = self._generic_risks(context, signals)
        if not context.position:
            risks.append("자기장 기준 현재 위치와 이동 경로 정보가 부족함")
        return risks[:5]

    def _generic_risks(self, context: ManualContext, signals: VisualSignals) -> list[str]:
        risks: list[str] = []
        if signals.red_flash_score > 0.08:
            risks.append("화면 가장자리의 붉은 피격 신호가 높아 교전 손실 가능성 있음")
        if signals.whiteout_score > 0.25:
            risks.append("섬광 또는 과노출 상태로 시야 확보가 어렵다고 추정됨")
        if signals.smoke_score > 0.55:
            risks.append("연막 또는 낮은 대비 화면으로 적 위치 확인이 제한됨")
        if signals.visible_hostile_count:
            risks.append("화면에 적 또는 상대 플레이어로 추정되는 대상이 감지됨")
        if not context.alive_allies or not context.alive_enemies:
            risks.append("아군/적 생존 수 정보가 부족해 승률 판단이 흔들릴 수 있음")
        if not risks:
            risks.append("확정 위험 신호는 적지만, 보이지 않는 각도와 미니맵 정보는 확인 필요")
        return risks[:5]

    def _actions(
        self, context: ManualContext, signals: VisualSignals, score: int
    ) -> tuple[str, str, str, str]:
        if score <= 42:
            safe = "엄폐를 우선하고, 장전/회복/스킬 준비 후 한 각도씩 확인하세요."
            aggressive = "적 위치가 하나라도 확인된 뒤에만 짧게 피킹하세요."
        elif score >= 62:
            safe = "현재 유리함을 유지하며 무리한 추격보다 교차각과 정보 유지에 집중하세요."
            aggressive = "팀원과 타이밍을 맞춰 확인된 위험 방향을 압박하세요."
        else:
            safe = "소리와 미니맵 정보를 더 모은 뒤, 위험 방향 반대편 엄폐로 재정비하세요."
            aggressive = "스킬이나 투척물로 먼저 시야를 흔든 뒤 짧은 교전을 선택하세요."

        team = "혼자 진입하지 말고 핑/브리핑으로 위험 방향을 공유한 뒤 동시에 움직이세요."
        avoid = "확인되지 않은 벽 뒤 적 위치를 확정한 것처럼 행동하거나, 탄약 부족 상태로 긴 교전을 열지 마세요."
        return safe, aggressive, team, avoid
