"""v9 스케줄러 스키마 — JSON-in / JSON-out 전면 개편.

v3까지: 자연어 할 일 목록 → {tasks, priority_order, scores(1-5), refusal}.
v9: 구조화 JSON 입력(현재시각·페르소나·태스크[마감/소요시간/메모]) →
    구조화 JSON 출력(scheduled_tasks[priority_rank, scoring(0-10)+total_score,
    reasoning(summary+chaining_detail), recommended_schedule(start/end)]).

생성·검증·추론이 모두 이 모듈을 단일 출처로 사용한다.
total_score 가중치와 채점 곡선은 파일 상단 상수로 노출 — 튜닝 지점.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

KST = _dt.timezone(_dt.timedelta(hours=9))

# ── total_score 가중치 (튜닝 지점) ──────────────────────────────────────────────
# 합=1.0. 예시 샘플 숫자 역산이 아니라 일관된 가중치 스키마(사용자 튜닝용).
W_URGENCY = 0.30
W_DEADLINE = 0.25
W_IMPORTANCE = 0.30
W_CHAINING = 0.15

# 점수 허용 오차 (검증기에서 total_score 재계산 대조)
TOTAL_TOL = 0.05
# 시간블록 소요시간 허용 오차(분)
DURATION_TOL_MIN = 5

Score = Annotated[float, Field(ge=0.0, le=10.0)]


# ── System Prompt (v9) ─────────────────────────────────────────────────────────
SCHEDULER_SYSTEM_PROMPT_V9 = """\
당신은 사용자의 할 일을 우선순위화하고 하루 일정에 배치하는 스케줄링 비서입니다.
입력은 JSON 객체이며 `current_time`(현재 시각, ISO8601), `user_persona`(사용자 정보),
`tasks`(할 일 배열: task_id·title·memo·source·deadline·estimated_duration_minutes)를 담습니다.
아래 JSON 스키마로만 응답하세요. JSON 외 어떤 텍스트도 포함하지 마세요.

채점(각 0-10 실수):
- deadline_proximity: 마감 임박도. current_time 기준 마감이 가까울수록 높음(마감 없음=0).
- urgency: 지금 당장 착수해야 하는 정도. 마감까지 여유(slack)가 적을수록·불이익(에스컬레이션/
  위약금/클레임/법정기한)이 있을수록 높음.
- task_importance: 사용자 목표에 미치는 영향. 위험 문구가 있으면 9-10으로 상향.
- task_chaining: 후속 작업을 블로킹하는 정도. 다른 작업의 선행이면 높음(독립 작업=낮음).
- total_score: 가중 종합 = 0.30*urgency + 0.25*deadline_proximity + 0.30*task_importance
  + 0.15*task_chaining (소수 2자리).

우선순위·일정 규칙:
- priority_rank: 1부터 시작하는 실행 순서. total_score 내림차순을 기본으로 하되,
  선후관계(체인)는 선행 작업이 반드시 먼저 오도록 하고, 이미 지난 고정 일정은 최하위에 둡니다.
- recommended_schedule: 페르소나의 가용 시간대(availability) 안에서 current_time 이후로
  priority_rank 순서대로 시간블록을 배치합니다. 블록 길이 = estimated_duration_minutes,
  블록끼리 겹치지 않게, 가능하면 각 작업의 deadline 전에 끝나도록 합니다.
- reasoning.summary: 점수·우선순위 근거 한두 문장.
- reasoning.chaining_detail: 해당 작업이 체인(선후관계)에 속할 때만 채우고, 독립 작업이면 "".

출력 스키마:
{
  "scheduled_tasks": [
    {
      "task_id": "task_001",
      "title": "원문 그대로의 할 일 제목",
      "priority_rank": 1,
      "scoring": {"deadline_proximity": 0-10, "task_importance": 0-10,
                  "task_chaining": 0-10, "urgency": 0-10, "total_score": 0-10},
      "reasoning": {"summary": "...", "chaining_detail": "<체인일 때만>"},
      "recommended_schedule": {"start_time": "ISO8601", "end_time": "ISO8601"}
    }
  ]
}\
"""


def system_prompt_v9() -> str:
    """v9 시스템 프롬프트 (페르소나는 입력 JSON에 포함되므로 치환 불필요)."""
    return SCHEDULER_SYSTEM_PROMPT_V9


# ── System Prompt (v9 EN-US) — KR 프롬프트의 영어판 (다국어 확장) ──────────────────
SCHEDULER_SYSTEM_PROMPT_V9_EN = """\
You are a scheduling assistant that prioritizes the user's tasks and lays them out across the day.
The input is a JSON object with `current_time` (ISO8601), `user_persona` (user info), and
`tasks` (array: task_id, title, memo, source, deadline, estimated_duration_minutes).
Respond ONLY with the JSON schema below. Do not include any text outside the JSON.

Scoring (each a float 0-10):
- deadline_proximity: how close the deadline is relative to current_time (higher when nearer; no deadline = 0).
- urgency: how much it must be started right now. Higher when there is little slack until the deadline,
  or when there is a penalty (escalation / late fee / client claim / statutory deadline).
- task_importance: impact on the user's goals. Raise to 9-10 when a risk clause is present.
- task_chaining: how much it blocks downstream work. Higher when it is a prerequisite of other tasks
  (independent task = low).
- total_score: weighted sum = 0.30*urgency + 0.25*deadline_proximity + 0.30*task_importance
  + 0.15*task_chaining (rounded to 2 decimals).

Priority and scheduling rules:
- priority_rank: execution order starting at 1. Default to descending total_score, but ensure a chain
  predecessor always comes before its successor, and put already-past fixed events last.
- recommended_schedule: place time blocks in priority_rank order, after current_time, within the persona's
  availability window. Block length = estimated_duration_minutes, blocks must not overlap, and finish
  before each task's deadline when possible.
- reasoning.summary: one or two sentences justifying the score/priority.
- reasoning.chaining_detail: fill only when the task is in a chain; use "" for independent tasks.

Output schema:
{
  "scheduled_tasks": [
    {
      "task_id": "task_001",
      "title": "the task title, verbatim",
      "priority_rank": 1,
      "scoring": {"deadline_proximity": 0-10, "task_importance": 0-10,
                  "task_chaining": 0-10, "urgency": 0-10, "total_score": 0-10},
      "reasoning": {"summary": "...", "chaining_detail": "<only when in a chain>"},
      "recommended_schedule": {"start_time": "ISO8601", "end_time": "ISO8601"}
    }
  ]
}\
"""


# ── Pydantic 모델 ──────────────────────────────────────────────────────────────

class Location(BaseModel):
    country: str = "South Korea"
    city: str = ""


class UserPersona(BaseModel):
    occupations: list[str] = Field(default_factory=list)
    detailed_status: str = ""
    age: int = 0
    gender: str = ""
    location: Location = Field(default_factory=Location)
    bio: str = ""
    availability: str = "09:00-18:00"  # 가용 시간창 "HH:MM-HH:MM"


class InputTask(BaseModel):
    task_id: str
    title: str
    memo: str = ""
    source: str = ""
    deadline: str | None = None  # ISO8601 또는 None(마감 없음)
    estimated_duration_minutes: int = 60


class ScheduleInput(BaseModel):
    current_time: str  # ISO8601 (+09:00)
    user_persona: UserPersona
    tasks: list[InputTask] = Field(default_factory=list)


class Scoring(BaseModel):
    deadline_proximity: Score
    task_importance: Score
    task_chaining: Score
    urgency: Score
    total_score: Score


class Reasoning(BaseModel):
    summary: str = ""
    chaining_detail: str = ""


class RecommendedSchedule(BaseModel):
    start_time: str
    end_time: str


class ScheduledTask(BaseModel):
    task_id: str
    title: str
    priority_rank: int
    scoring: Scoring
    reasoning: Reasoning = Field(default_factory=Reasoning)
    recommended_schedule: RecommendedSchedule


class ScheduleResponseV9(BaseModel):
    scheduled_tasks: list[ScheduledTask] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ranks_unique(self) -> "ScheduleResponseV9":
        ranks = [t.priority_rank for t in self.scheduled_tasks]
        if ranks and sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise ValueError(f"priority_rank가 1..N 순열이 아님: {sorted(ranks)}")
        return self


# ── total_score 공식 ────────────────────────────────────────────────────────────

def compute_total_score(deadline_proximity: float, task_importance: float,
                        task_chaining: float, urgency: float) -> float:
    """가중 종합 점수 (소수 2자리)."""
    return round(
        W_URGENCY * urgency + W_DEADLINE * deadline_proximity
        + W_IMPORTANCE * task_importance + W_CHAINING * task_chaining,
        2,
    )


def recompute_total_scores(resp: "ScheduleResponseV9") -> "ScheduleResponseV9":
    """모델 출력의 4축으로 total_score를 결정적으로 재계산(후처리).

    LLM은 가중합 산술이 부정확하므로 배포·평가 시 모델이 낸 축 점수만 신뢰하고
    total_score는 공식으로 다시 채운다.
    """
    for t in resp.scheduled_tasks:
        sc = t.scoring
        sc.total_score = compute_total_score(
            sc.deadline_proximity, sc.task_importance, sc.task_chaining, sc.urgency)
    return resp


# ── 결정적 채점 (skeleton fact 기반) ────────────────────────────────────────────

def deadline_proximity_score(now: _dt.datetime, deadline: _dt.datetime | None) -> float:
    """마감 임박도 0-10. 마감 없음=0, 지남=낮음, 가까울수록 높음."""
    if deadline is None:
        return 0.0
    hrs = (deadline - now).total_seconds() / 3600.0
    if hrs <= 0:
        return 1.0  # 이미 지남 (지난 일정은 별도 처리되지만 안전망)
    for limit, score in [(3, 10.0), (6, 9.5), (12, 9.0), (24, 8.0),
                         (48, 6.5), (72, 5.0), (168, 3.5)]:
        if hrs <= limit:
            return score
    return 2.0


def urgency_score(now: _dt.datetime, deadline: _dt.datetime | None,
                 duration_min: int, risk: bool) -> float:
    """지금 착수 필요도 0-10. slack(마감-현재-소요)이 적을수록·위험할수록 높음."""
    if deadline is None:
        base = 1.5
    else:
        slack_h = ((deadline - now).total_seconds() / 60.0 - duration_min) / 60.0
        if slack_h <= 0:
            base = 10.0
        elif slack_h <= 3:
            base = 9.0
        elif slack_h <= 8:
            base = 7.5
        elif slack_h <= 24:
            base = 6.0
        elif slack_h <= 72:
            base = 4.0
        else:
            base = 2.5
    if risk:
        base = min(10.0, base + 1.5)
    return round(base, 1)


def importance_score(kind: str, risk: bool, in_chain: bool) -> float:
    """중요도 0-10. 위험 문구=9-10, 체인 작업=7, kind별 기본."""
    if risk:
        return 9.5
    if in_chain:
        return 7.0
    return {"today": 6.0, "future": 5.0, "none": 3.0, "past": 2.0,
            "chain": 7.0, "risk": 9.5}.get(kind, 5.0)


def chaining_score(in_chain: bool, is_final: bool) -> float:
    """블로킹 정도 0-10. 비최종 체인(후속 블로킹)=9, 최종 체인=6, 독립=2."""
    if not in_chain:
        return 2.0
    return 6.0 if is_final else 9.0


# ── 시간블록 스케줄러 ────────────────────────────────────────────────────────────

def _parse_window(availability: str) -> tuple[_dt.time, _dt.time]:
    """'09:00-18:00' → (time(9,0), time(18,0)). 파싱 실패 시 기본창."""
    try:
        a, b = availability.split("-")
        sh, sm = (int(x) for x in a.strip().split(":"))
        eh, em = (int(x) for x in b.strip().split(":"))
        return _dt.time(sh, sm), _dt.time(eh, em)
    except Exception:
        return _dt.time(9, 0), _dt.time(18, 0)


def _advance_into_window(cursor: _dt.datetime, win_start: _dt.time,
                        win_end: _dt.time, dur: _dt.timedelta) -> _dt.datetime:
    """cursor를 가용창 안에서 dur가 들어갈 수 있는 가장 이른 시각으로 이동."""
    for _ in range(14):  # 최대 2주 롤오버 방지
        day_start = cursor.replace(hour=win_start.hour, minute=win_start.minute,
                                   second=0, microsecond=0)
        day_end = cursor.replace(hour=win_end.hour, minute=win_end.minute,
                                 second=0, microsecond=0)
        if cursor < day_start:
            cursor = day_start
        if cursor + dur <= day_end:
            return cursor
        # 오늘 창에 안 들어감 → 다음 날 창 시작
        cursor = (cursor + _dt.timedelta(days=1)).replace(
            hour=win_start.hour, minute=win_start.minute, second=0, microsecond=0)
    return cursor


def build_schedule(now: _dt.datetime, availability: str,
                  ranked: list[tuple[str, int]]) -> dict[str, tuple[_dt.datetime, _dt.datetime]]:
    """priority_rank 순 (task_id, duration_min) 리스트 → {task_id: (start, end)}.

    가용창 안에서 current_time 이후로 겹치지 않게 그리디 배치.
    """
    win_start, win_end = _parse_window(availability)
    cursor = now
    out: dict[str, tuple[_dt.datetime, _dt.datetime]] = {}
    for tid, dur_min in ranked:
        dur = _dt.timedelta(minutes=dur_min)
        cursor = _advance_into_window(cursor, win_start, win_end, dur)
        start = cursor
        end = start + dur
        out[tid] = (start, end)
        cursor = end
    return out


def iso(dt: _dt.datetime) -> str:
    """KST aware datetime → ISO8601 +09:00 문자열."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    return dt.isoformat()


def parse_dt(s: str) -> _dt.datetime:
    """ISO8601 파싱. tz 없으면 KST 부여. 'YYYY-MM-DD HH:MM'도 허용."""
    s = s.strip()
    try:
        d = _dt.datetime.fromisoformat(s)
    except ValueError:
        d = _dt.datetime.fromisoformat(s.replace(" ", "T"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=KST)
    return d


# ── 파싱 / 포맷 ──────────────────────────────────────────────────────────────────

def parse_lenient_v9(text: str) -> ScheduleResponseV9 | None:
    """코드펜스 제거 + 첫 JSON 객체 추출 후 파싱. 실패 시 None."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    start = cleaned.find("{")
    if start == -1:
        return None
    depth, end = 0, -1
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
    try:
        return ScheduleResponseV9.model_validate_json(cleaned[start:end])
    except Exception:
        return None


def format_for_sft_v9(resp: ScheduleResponseV9) -> str:
    """ScheduleResponseV9 → SFT chosen 텍스트 (compact JSON, 한글 보존)."""
    return json.dumps(resp.model_dump(), ensure_ascii=False, separators=(",", ":"))


def format_input_v9(inp: ScheduleInput) -> str:
    """ScheduleInput → prompt 텍스트 (pretty JSON, 한글 보존)."""
    return json.dumps(inp.model_dump(), ensure_ascii=False, indent=2)


# ── 검증기 (생성 게이트) ──────────────────────────────────────────────────────────

def verify_chosen_v9(inp: ScheduleInput, out: ScheduleResponseV9,
                    chain_pairs: list[tuple[str, str]] | None = None) -> list[str]:
    """입력 대비 출력 위반 목록 반환 (빈 리스트=통과).

    chain_pairs: (선행 task_id, 후행 task_id) 쌍 목록 — 체인 의존성 순서 검사용.
    """
    errors: list[str] = []
    in_ids = [t.task_id for t in inp.tasks]
    out_ids = [s.task_id for s in out.scheduled_tasks]

    # 1) 커버리지 & 순열
    if sorted(out_ids) != sorted(in_ids):
        errors.append(f"task_id 집합 불일치: 입력 {sorted(in_ids)} vs 출력 {sorted(out_ids)}")
        return errors
    ranks = sorted(s.priority_rank for s in out.scheduled_tasks)
    if ranks != list(range(1, len(out_ids) + 1)):
        errors.append(f"priority_rank가 1..N 순열 아님: {ranks}")

    now = parse_dt(inp.current_time)
    dur_map = {t.task_id: t.estimated_duration_minutes for t in inp.tasks}
    dl_map = {t.task_id: (parse_dt(t.deadline) if t.deadline else None) for t in inp.tasks}

    blocks: list[tuple[_dt.datetime, _dt.datetime, str]] = []
    for s in out.scheduled_tasks:
        sc = s.scoring
        # 2) 점수 범위
        for name, v in [("deadline_proximity", sc.deadline_proximity),
                        ("task_importance", sc.task_importance),
                        ("task_chaining", sc.task_chaining), ("urgency", sc.urgency)]:
            if not (0.0 <= v <= 10.0):
                errors.append(f"{s.task_id}.{name}={v} 범위[0,10] 벗어남")
        # 3) total_score 재계산 일치
        expected = compute_total_score(sc.deadline_proximity, sc.task_importance,
                                       sc.task_chaining, sc.urgency)
        if abs(sc.total_score - expected) > TOTAL_TOL:
            errors.append(f"{s.task_id}.total_score={sc.total_score} ≠ 재계산 {expected}")
        # 4) 시간블록: start ≥ now, duration 일치
        try:
            st = parse_dt(s.recommended_schedule.start_time)
            en = parse_dt(s.recommended_schedule.end_time)
        except Exception:
            errors.append(f"{s.task_id} recommended_schedule 시각 파싱 실패")
            continue
        if st < now - _dt.timedelta(minutes=1):
            errors.append(f"{s.task_id} start_time이 current_time 이전")
        block_min = (en - st).total_seconds() / 60.0
        if abs(block_min - dur_map.get(s.task_id, 60)) > DURATION_TOL_MIN:
            errors.append(f"{s.task_id} 블록 길이 {block_min:.0f}분 ≠ 소요 {dur_map.get(s.task_id)}분")
        # 4b) 마감 실현가능성: 마감 있는 태스크는 블록이 마감 전에 끝나야 함
        dl = dl_map.get(s.task_id)
        if dl is not None and en > dl + _dt.timedelta(minutes=DURATION_TOL_MIN):
            errors.append(f"{s.task_id} 일정 종료 {en.isoformat()}가 마감 {dl.isoformat()} 초과(실현불가)")
        blocks.append((st, en, s.task_id))
        # 5) chaining_detail 조건부
        is_chain = chain_pairs is not None and any(
            s.task_id in (a, b) for a, b in chain_pairs)
        if is_chain and not s.reasoning.chaining_detail.strip():
            errors.append(f"{s.task_id} 체인인데 chaining_detail 비어있음")
        if not is_chain and s.reasoning.chaining_detail.strip():
            errors.append(f"{s.task_id} 독립인데 chaining_detail 채워짐")

    # 6) 블록 중복 없음
    blocks.sort()
    for i in range(1, len(blocks)):
        if blocks[i][0] < blocks[i - 1][1]:
            errors.append(f"시간블록 중복: {blocks[i-1][2]} ↔ {blocks[i][2]}")

    # 7) 체인 선후 순서 (priority_rank 기준)
    rank_map = {s.task_id: s.priority_rank for s in out.scheduled_tasks}
    for a, b in (chain_pairs or []):
        if a in rank_map and b in rank_map and rank_map[a] >= rank_map[b]:
            errors.append(f"체인 순서 위반: 선행 {a}(rank {rank_map[a]}) ≥ 후행 {b}(rank {rank_map[b]})")

    return errors
