"""ScoreRanker — 4축 점수에서 결정적으로 priority_order를 도출.

모델이 출력한 scores와 priority_order가 모순될 수 있으므로(채점은 맞는데 순서가 어긋나는
사례가 검증에서 반복 확인됨), 점수 가중합으로 순서를 재계산해 score↔order 일관성을 보장한다.

가중치 구조는 발표 자료의 Feature Generator(긴급 0.5 / 중요 0.3 / 마감 0.1 / 정렬보너스 0.1)를
4축 스키마에 맞게 옮긴 것이다. 마감 임박도는 v2/v3 스키마에서 urgency에 흡수되어 있고,
대신 dependency 축이 추가되었다.
"""
from __future__ import annotations

from dataclasses import dataclass

from .data.schema import ScheduleResponse, ScoreItem


@dataclass(frozen=True)
class RankWeights:
    urgency: float = 0.40
    importance: float = 0.25
    dependency: float = 0.20
    time_constraint: float = 0.05
    alignment_bonus: float = 0.10  # 아이젠하워 1사분면 강조: urgency_norm × importance_norm


DEFAULT_WEIGHTS = RankWeights()


def _norm(score: int) -> float:
    """1-5점 → 0.0-1.0 정규화: (x-1)/4."""
    return (score - 1) / 4


def task_score(s: ScoreItem, w: RankWeights = DEFAULT_WEIGHTS) -> float:
    u, i = _norm(s.urgency), _norm(s.importance)
    return (
        w.urgency * u
        + w.importance * i
        + w.dependency * _norm(s.dependency)
        + w.time_constraint * _norm(s.time_constraint)
        + w.alignment_bonus * (u * i)
    )


def rerank(
    resp: ScheduleResponse,
    weights: RankWeights = DEFAULT_WEIGHTS,
) -> ScheduleResponse:
    """scores 가중합 기준으로 priority_order를 재계산한 새 ScheduleResponse 반환.

    동점은 모델이 제시한 priority_order(없으면 입력 순서)를 유지하는 안정 정렬.
    scores가 없는 태스크는 0점 처리되어 후순위로 밀린다.
    """
    if not resp.tasks or not resp.scores:
        return resp

    score_map = {s.task_id: s for s in resp.scores}
    model_rank = {tid: r for r, tid in enumerate(resp.priority_order)}
    task_ids = [t.id for t in resp.tasks]

    def _key(tid: int) -> tuple[float, int]:
        s = score_map.get(tid)
        val = task_score(s, weights) if s else 0.0
        return (-val, model_rank.get(tid, len(model_rank)))

    new_order = sorted(task_ids, key=_key)
    return resp.model_copy(update={"priority_order": new_order})


def rerank_guard(resp: ScheduleResponse) -> ScheduleResponse:
    """가드레일 재정렬 — 모델 순서를 유지하되 명백한 모순만 교정.

    held-out 평가에서 전면 rerank(가중합 정렬)는 4축 점수가 표현하지 못하는 정보
    (같은 날 내 시각 순서, 체인 단계 순서)를 파괴해 통과율을 떨어뜨렸다.
    이 함수는 v3 프롬프트 규칙상 '지난 일정' 시그니처(urgency≤1 AND time_constraint≤1)인
    태스크만 상대 순서를 보존하며 최하위로 보낸다.
    """
    if not resp.tasks or not resp.scores:
        return resp
    score_map = {s.task_id: s for s in resp.scores}

    def _is_past(tid: int) -> bool:
        s = score_map.get(tid)
        return s is not None and s.urgency <= 1 and s.time_constraint <= 1

    live = [t for t in resp.priority_order if not _is_past(t)]
    past = [t for t in resp.priority_order if _is_past(t)]
    return resp.model_copy(update={"priority_order": live + past})


def order_consistency(resp: ScheduleResponse, weights: RankWeights = DEFAULT_WEIGHTS) -> float:
    """모델 priority_order와 점수 기반 순서의 일치율 (같은 위치 비율, 0.0-1.0)."""
    if not resp.priority_order:
        return 1.0
    reranked = rerank(resp, weights).priority_order
    same = sum(1 for a, b in zip(resp.priority_order, reranked) if a == b)
    return same / len(resp.priority_order)
