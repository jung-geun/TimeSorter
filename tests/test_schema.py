"""src/drl/data/schema.py 단위 테스트."""
from __future__ import annotations

import json

import pytest

from drl.data.schema import (
    ScheduleResponse,
    format_for_sft,
    parse_lenient,
    parse_or_repair,
    parse_strict,
    response_to_text,
)

# ── 정상 케이스 ───────────────────────────────────────────────────────────────

_VALID_JSON = json.dumps({
    "tasks": [{"id": 1, "text": "보고서 작성"}, {"id": 2, "text": "팀 미팅"}],
    "priority_order": [2, 1],
    "scores": [
        {"task_id": 1, "urgency": 3, "importance": 4, "dependency": 2, "time_constraint": 1,
         "reason": "내일까지 마감"},
        {"task_id": 2, "urgency": 5, "importance": 5, "dependency": 3, "time_constraint": 5,
         "reason": "오후 2시 고정"},
    ],
}, ensure_ascii=False)


def test_parse_strict_valid():
    resp = parse_strict(_VALID_JSON)
    assert len(resp.tasks) == 2
    assert resp.priority_order == [2, 1]
    assert resp.scores[0].urgency == 3


def test_parse_lenient_with_fence():
    fenced = f"```json\n{_VALID_JSON}\n```"
    resp = parse_lenient(fenced)
    assert resp is not None
    assert len(resp.tasks) == 2


def test_parse_lenient_with_preamble():
    text = f"네, 결과입니다:\n{_VALID_JSON}\n감사합니다."
    resp = parse_lenient(text)
    assert resp is not None


def test_parse_lenient_returns_none_on_garbage():
    assert parse_lenient("이건 JSON이 아닙니다") is None


def test_parse_or_repair_fallback():
    result = parse_or_repair("1) 보고서 작성 - 긴급\n2) 팀 미팅 - 중요")
    assert isinstance(result, ScheduleResponse)
    assert len(result.tasks) == 1  # 자유 텍스트 폴백


def test_parse_or_repair_valid_json():
    result = parse_or_repair(_VALID_JSON)
    assert len(result.tasks) == 2


# ── 검증 케이스 ───────────────────────────────────────────────────────────────

def test_score_out_of_range():
    bad = json.dumps({
        "tasks": [{"id": 1, "text": "태스크"}],
        "priority_order": [1],
        "scores": [{"task_id": 1, "urgency": 6, "importance": 1,
                    "dependency": 1, "time_constraint": 1}],
    })
    with pytest.raises(Exception):
        parse_strict(bad)


def test_priority_order_unknown_id():
    bad = json.dumps({
        "tasks": [{"id": 1, "text": "태스크"}],
        "priority_order": [99],
        "scores": [],
    })
    with pytest.raises(Exception):
        parse_strict(bad)


# ── format_for_sft ────────────────────────────────────────────────────────────

def test_format_for_sft_roundtrip():
    resp = parse_strict(_VALID_JSON)
    sft_text = format_for_sft(resp)
    resp2 = parse_strict(sft_text)
    assert resp2.tasks == resp.tasks
    assert resp2.priority_order == resp.priority_order


# ── response_to_text ──────────────────────────────────────────────────────────

def test_response_to_text_contains_task_names():
    resp = parse_strict(_VALID_JSON)
    text = response_to_text(resp)
    assert "보고서 작성" in text
    assert "팀 미팅" in text


def test_response_to_text_refusal():
    resp = ScheduleResponse(refusal_reason="할 일 목록이 아닙니다")
    assert "거부" in response_to_text(resp)
