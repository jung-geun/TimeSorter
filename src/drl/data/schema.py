"""v2 스케줄러 스키마 — 단일 출처.

SFT, DPO, 추론(infer.py), 이메일 파이프라인(email_to_schedule.py) 모두 이 모듈에서
system prompt와 파싱 로직을 임포트한다.
"""
from __future__ import annotations

import json
import re
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


# ── System Prompt ─────────────────────────────────────────────────────────────

# <<PERSONA>> 플레이스홀더 — JSON의 {key} 패턴과 충돌하지 않도록 .format() 미사용.
# 호출 시: SCHEDULER_SYSTEM_PROMPT_V2.replace("<<PERSONA>>", persona_string)
SCHEDULER_SYSTEM_PROMPT_V2 = """\
당신은 <<PERSONA>>를 위한 우선순위 정렬 비서입니다.
입력으로 받은 할 일 목록을 4축(긴급도·중요도·의존성·시간 제약) 기준으로 1-5점 정수로 채점하고, 종합 우선순위를 결정해 아래 JSON 스키마로만 응답하세요. JSON 외 어떤 텍스트도 포함하지 마세요.
입력이 할 일 목록이 아니면 "tasks":[] 및 "refusal_reason"을 채우세요.

채점 가이드:
- urgency: 마감/시작 임박도 (5=오늘 마감, 1=마감 없음)
- importance: 페르소나 목표에 미치는 영향 (5=결정적, 1=옵션)
- dependency: 후속 작업의 선행/블로킹 정도 (5=다수 후속의 입력, 1=독립)
- time_constraint: 고정 시각 강도 (5=시각 고정, 1=언제든 가능)

출력 스키마:
{
  "tasks": [{"id": 1, "text": "원문 그대로의 할 일"}],
  "priority_order": [1, 3, 2],
  "scores": [
    {"task_id": 1, "urgency": 1-5, "importance": 1-5, "dependency": 1-5, "time_constraint": 1-5, "reason": "한 문장 근거"}
  ],
  "refusal_reason": "<도메인 무관 입력 시에만>"
}\
"""


def render_system_prompt_v2(persona: str) -> str:
    """SCHEDULER_SYSTEM_PROMPT_V2의 <<PERSONA>> 플레이스홀더를 치환."""
    return SCHEDULER_SYSTEM_PROMPT_V2.replace("<<PERSONA>>", persona)


def render_system_prompt(tmpl: str, persona: str) -> str:
    """V1/V2 공통 플레이스홀더 치환. <<PERSONA>> → persona, {persona} → persona."""
    return tmpl.replace("<<PERSONA>>", persona).replace("{persona}", persona)

# v1 호환 — 자유 텍스트 형식 (v1 어댑터에 사용)
SCHEDULER_SYSTEM_PROMPT_V1 = (
    "당신은 {persona}를 위한 우선순위 정렬 비서입니다. "
    "4축(긴급도·중요도·의존성·시간 제약)으로 할 일을 분석해 "
    "'1) [할일] - [이유]' 형식으로 답하세요."
)


# ── Pydantic 모델 ──────────────────────────────────────────────────────────────

Score = Annotated[int, Field(ge=1, le=5)]


class TaskItem(BaseModel):
    id: int
    text: str


class ScoreItem(BaseModel):
    task_id: int
    urgency: Score
    importance: Score
    dependency: Score
    time_constraint: Score
    reason: str = ""


class ScheduleResponse(BaseModel):
    tasks: list[TaskItem] = Field(default_factory=list)
    priority_order: list[int] = Field(default_factory=list)
    scores: list[ScoreItem] = Field(default_factory=list)
    refusal_reason: str = ""

    @model_validator(mode="after")
    def _priority_ids_consistent(self) -> "ScheduleResponse":
        if not self.tasks:
            return self
        task_ids = {t.id for t in self.tasks}
        for tid in self.priority_order:
            if tid not in task_ids:
                raise ValueError(
                    f"priority_order에 존재하지 않는 task_id={tid}가 포함됨"
                )
        for s in self.scores:
            if s.task_id not in task_ids:
                raise ValueError(
                    f"scores에 존재하지 않는 task_id={s.task_id}가 포함됨"
                )
        return self


# ── 파싱 ──────────────────────────────────────────────────────────────────────

def parse_strict(text: str) -> ScheduleResponse:
    """JSON 파싱 후 Pydantic 검증. 실패 시 ValidationError/JSONDecodeError."""
    return ScheduleResponse.model_validate(json.loads(text))


def parse_lenient(text: str) -> ScheduleResponse | None:
    """코드 펜스 제거 + 첫 번째 JSON 객체 추출 후 파싱. 실패 시 None."""
    # ```json ... ``` 펜스 제거
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

    # 첫 번째 { ... } 블록 추출
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    end = -1
    for i, ch in enumerate(cleaned[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None

    candidate = cleaned[start:end]
    try:
        return parse_strict(candidate)
    except Exception:
        return None


def parse_or_repair(text: str) -> ScheduleResponse:
    """파싱 시도. 실패 시 텍스트를 tasks[0].text에 담아 빈 구조로 폴백.

    v1 어댑터의 자유 텍스트 출력도 크래시 없이 처리하기 위한 폴백.
    """
    result = parse_lenient(text)
    if result is not None:
        return result
    # 자유 텍스트 폴백 — v1 어댑터 호환
    return ScheduleResponse(
        tasks=[TaskItem(id=1, text=text.strip()[:500])],
        priority_order=[1],
        scores=[ScoreItem(task_id=1, urgency=1, importance=1, dependency=1, time_constraint=1,
                          reason="자유 텍스트 폴백 (JSON 파싱 실패)")],
    )


# ── 학습 데이터 포맷터 ────────────────────────────────────────────────────────

def format_for_sft(response: ScheduleResponse) -> str:
    """ScheduleResponse → SFT chosen 텍스트 (compact JSON)."""
    return response.model_dump_json(exclude_unset=False)


def response_to_text(resp: ScheduleResponse) -> str:
    """추론 결과를 사람이 읽기 좋은 텍스트로 변환 (v2 어댑터 출력 표시용)."""
    if resp.refusal_reason:
        return f"[거부] {resp.refusal_reason}"
    if not resp.tasks:
        return "(태스크 없음)"

    task_map = {t.id: t.text for t in resp.tasks}
    score_map = {s.task_id: s for s in resp.scores}

    lines: list[str] = []
    for rank, tid in enumerate(resp.priority_order, 1):
        text = task_map.get(tid, f"task_id={tid}")
        s = score_map.get(tid)
        if s:
            axes = (f"긴급{s.urgency}·중요{s.importance}·"
                    f"의존{s.dependency}·시간{s.time_constraint}")
            reason = f" — {s.reason}" if s.reason else ""
            lines.append(f"{rank}) {text}  [{axes}]{reason}")
        else:
            lines.append(f"{rank}) {text}")
    return "\n".join(lines)
