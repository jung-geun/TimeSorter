import json

import pytest

from timesorter.data.schema import ScheduleResponse
from timesorter.feedback import FeedbackRecord, to_dpo_pair
from timesorter.rank import DEFAULT_WEIGHTS, order_consistency, rerank, rerank_guard, task_score


def _resp(scores: list[tuple[int, int, int, int, int]], order: list[int]) -> ScheduleResponse:
    """scores: (task_id, urgency, importance, dependency, time_constraint)."""
    return ScheduleResponse.model_validate({
        "tasks": [{"id": tid, "text": f"task{tid}"} for tid, *_ in scores],
        "priority_order": order,
        "scores": [
            {"task_id": tid, "urgency": u, "importance": i, "dependency": d,
             "time_constraint": t, "reason": ""}
            for tid, u, i, d, t in scores
        ],
    })


def test_rerank_orders_by_weighted_score():
    resp = _resp([(1, 1, 1, 1, 1), (2, 5, 5, 5, 5), (3, 3, 3, 3, 3)], order=[1, 2, 3])
    out = rerank(resp)
    assert out.priority_order == [2, 3, 1]
    # 원본은 불변
    assert resp.priority_order == [1, 2, 3]


def test_rerank_tie_keeps_model_order():
    resp = _resp([(1, 3, 3, 3, 3), (2, 3, 3, 3, 3)], order=[2, 1])
    assert rerank(resp).priority_order == [2, 1]


def test_task_score_range():
    resp = _resp([(1, 5, 5, 5, 5)], order=[1])
    assert task_score(resp.scores[0]) == pytest.approx(
        DEFAULT_WEIGHTS.urgency + DEFAULT_WEIGHTS.importance + DEFAULT_WEIGHTS.dependency
        + DEFAULT_WEIGHTS.time_constraint + DEFAULT_WEIGHTS.alignment_bonus
    )
    resp_low = _resp([(1, 1, 1, 1, 1)], order=[1])
    assert task_score(resp_low.scores[0]) == 0.0


def test_rerank_guard_demotes_past_only():
    # task 1 = 지난 일정 시그니처(u1,t1)가 1위, task 2·3은 모델 순서 유지돼야 함
    resp = _resp([(1, 1, 3, 1, 1), (2, 5, 3, 1, 5), (3, 4, 5, 1, 3)], order=[1, 3, 2])
    out = rerank_guard(resp)
    assert out.priority_order == [3, 2, 1]  # 3,2 상대 순서 보존 + 1만 최하위


def test_rerank_guard_no_past_is_noop():
    resp = _resp([(1, 5, 3, 1, 5), (2, 2, 2, 1, 1)], order=[2, 1])
    assert rerank_guard(resp).priority_order == [2, 1]


def test_order_consistency():
    consistent = _resp([(1, 5, 5, 5, 5), (2, 1, 1, 1, 1)], order=[1, 2])
    assert order_consistency(consistent) == 1.0
    inconsistent = _resp([(1, 5, 5, 5, 5), (2, 1, 1, 1, 1)], order=[2, 1])
    assert order_consistency(inconsistent) == 0.0


def test_feedback_to_dpo_pair():
    resp = _resp([(1, 3, 3, 1, 1), (2, 4, 4, 1, 1)], order=[1, 2])
    rec = FeedbackRecord(
        prompt="- task1\n- task2",
        model_output=resp.model_dump_json(),
        corrected_order=[2, 1],
        today="2026-06-10",
    )
    pair = to_dpo_pair(rec)
    assert pair is not None
    assert json.loads(pair["chosen"])["priority_order"] == [2, 1]
    assert json.loads(pair["rejected"])["priority_order"] == [1, 2]
    assert pair["category"] == "user_feedback"
    assert pair["today"] == "2026-06-10"


def test_feedback_no_change_returns_none():
    resp = _resp([(1, 3, 3, 1, 1)], order=[1])
    rec = FeedbackRecord(prompt="- t", model_output=resp.model_dump_json(), corrected_order=[1])
    assert to_dpo_pair(rec) is None


def test_feedback_invalid_order_raises():
    resp = _resp([(1, 3, 3, 1, 1)], order=[1])
    rec = FeedbackRecord(prompt="- t", model_output=resp.model_dump_json(), corrected_order=[1, 9])
    with pytest.raises(ValueError):
        to_dpo_pair(rec)
