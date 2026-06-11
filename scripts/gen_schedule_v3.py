#!/usr/bin/env python
"""v3 스케줄 데이터 생성 — 날짜 추론·마감 세분화·의존성·리스크 시나리오.

v2 검증에서 확인된 3대 실패(날짜 혼동, 오전/오후 미반영, 의존성 분산)를 겨냥한 데이터.
프로그램이 시나리오 골격(태스크별 마감·과거 여부·체인·리스크)을 먼저 결정하고,
GPT는 (1) 골격에 맞는 자연어 태스크 텍스트, (2) v3 system prompt 하의 chosen JSON만 생성한다.
chosen은 골격 기준으로 프로그래매틱 검증(과거 일정 후순위, 마감 시각 순서, 체인 연속성,
리스크 importance)을 통과해야 저장된다.

시나리오:
  dated_mixed      35% — 오늘 ≠ 태스크 날짜. 지난 일정 1-2개 + 오늘 오전/오후 + 미래 + 무마감
  intraday         20% — 같은 날 서로 다른 시각 마감 4-5개
  dependency_chain 20% — 작성→검토→발송류 3단계 체인 + 독립 태스크
  risk             15% — 에스컬레이션·위약금·클레임 등 리스크 신호 포함
  relative         10% — "내일"·"모레"·"이번 주 금요일" 상대 날짜 표현

사용:
  uv run python scripts/gen_schedule_v3.py --total 20 --verify     # dry-run
  uv run python scripts/gen_schedule_v3.py --total 2000
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from timesorter.data.schema import (
    SCHEDULER_SYSTEM_PROMPT_V3,
    format_for_sft,
    parse_lenient,
    render_system_prompt,
)

# gen_schedule_v2의 공용 헬퍼 재사용 (캐시·체크포인트·API 호출)
from gen_schedule_v2 import (
    _APICache,
    _append_checkpoint_async,
    _call_api,
    _load_checkpoint,
    _NEMOTRON_PATH,
)

load_dotenv()

_TEXT_MODEL = "gpt-5.4-mini"     # 태스크 텍스트 생성
_CHOSEN_MODEL = "gpt-5.4"        # chosen JSON 생성 (정책: OpenAI는 gpt-5.4 계열만 사용)
_DEFAULT_CONCURRENCY = 12

_SCENARIOS = [
    ("dated_mixed", 0.35),
    ("intraday", 0.20),
    ("dependency_chain", 0.20),
    ("risk", 0.15),
    ("relative", 0.10),
]

# v4 증강 시나리오 (--scenarios "name:count,..." 로 지정)
#   past_split    : today 기준 지난/안 지난 스케줄 다수 혼합 — 깨끗한 분리 학습
#   no_today      : 오늘 날짜 미상 — 절대 날짜의 상대 정렬만 수행, '지남' 판단 금지
#   am_escalation : 오전 마감 + 에스컬레이션 → urgency=5 최우선 (v2 검증 PR 실패 대응)

_RISK_CLAUSES = [
    "미처리 시 팀장에게 자동 에스컬레이션",
    "기한 초과 시 위약금 발생",
    "지연 시 고객사 클레임 예상",
    "법정 제출 기한 — 연장 불가",
    "미응답 시 계약 자동 해지 조항 발동",
]

_WEEKDAYS_KO = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]


# ── 시나리오 골격 ─────────────────────────────────────────────────────────────

@dataclass
class TaskSpec:
    idx: int                      # 1-based, 입력 순서 = task id
    kind: str                     # past | today | future | none | chain | risk
    deadline: str | None = None   # ISO "YYYY-MM-DD HH:MM" 또는 None
    is_past: bool = False
    chain_group: int = 0          # 0=독립, 1+=체인 id
    chain_pos: int = 0            # 체인 내 순서 (1부터)
    risk: bool = False
    risk_clause: str = ""
    rel_expr: str = ""            # 상대 날짜 표현 (텍스트에 이 표현을 사용)


@dataclass
class Skeleton:
    scenario: str
    today: str                    # ISO date
    specs: list[TaskSpec] = field(default_factory=list)


def _dt(today: datetime.date, day_offset: int, hour: int, minute: int = 0) -> str:
    d = today + datetime.timedelta(days=day_offset)
    return f"{d.isoformat()} {hour:02d}:{minute:02d}"


def _shuffle_reindex(specs: list[TaskSpec], rng: random.Random) -> list[TaskSpec]:
    """입력 순서를 섞고 idx를 재부여 (체인 내부 상대 순서는 유지)."""
    rng.shuffle(specs)
    for i, s in enumerate(specs, 1):
        s.idx = i
    return specs


def build_skeleton(scenario: str, today: datetime.date, rng: random.Random) -> Skeleton:
    specs: list[TaskSpec] = []

    if scenario == "dated_mixed":
        n_past = rng.choice([1, 1, 2])
        for _ in range(n_past):
            off = -rng.randint(1, 3)
            specs.append(TaskSpec(0, "past", _dt(today, off, rng.randint(10, 18)), is_past=True))
        specs.append(TaskSpec(0, "today", _dt(today, 0, rng.randint(9, 11), rng.choice([0, 30]))))
        specs.append(TaskSpec(0, "today", _dt(today, 0, rng.randint(14, 18))))
        specs.append(TaskSpec(0, "future", _dt(today, rng.randint(2, 7), rng.randint(9, 18))))
        specs.append(TaskSpec(0, "none"))

    elif scenario == "intraday":
        hours = rng.sample([9, 10, 11, 13, 14, 16, 17, 19], k=rng.randint(4, 5))
        for h in hours:
            specs.append(TaskSpec(0, "today", _dt(today, 0, h)))
        if rng.random() < 0.5:
            specs.append(TaskSpec(0, "none"))

    elif scenario == "dependency_chain":
        chain_deadline_off = rng.choice([0, 1, 1, 2])
        end_hour = rng.randint(14, 18)
        for pos in range(1, 4):
            specs.append(TaskSpec(
                0, "chain",
                _dt(today, chain_deadline_off, end_hour) if pos == 3 else None,
                chain_group=1, chain_pos=pos,
            ))
        specs.append(TaskSpec(0, "today", _dt(today, 0, rng.randint(9, 17))))
        specs.append(TaskSpec(0, "none"))

    elif scenario == "risk":
        n = rng.randint(4, 5)
        risk_idx = rng.sample(range(n), k=rng.choice([1, 2]))
        for i in range(n):
            if i in risk_idx:
                specs.append(TaskSpec(
                    0, "risk", _dt(today, rng.choice([0, 1]), rng.randint(9, 18)),
                    risk=True, risk_clause=rng.choice(_RISK_CLAUSES),
                ))
            elif rng.random() < 0.5:
                specs.append(TaskSpec(0, "future", _dt(today, rng.randint(2, 7), rng.randint(9, 18))))
            else:
                specs.append(TaskSpec(0, "none"))

    elif scenario == "relative":
        choices = [
            ("오늘 17시까지", 0, 17), ("내일 오전까지", 1, 11), ("내일까지", 1, 18),
            ("모레까지", 2, 18), ("이번 주 금요일까지", (4 - today.weekday()) % 7 or 7, 18),
            ("다음 주 화요일까지", ((1 - today.weekday()) % 7) + 7, 18),
        ]
        for expr, off, hour in rng.sample(choices, k=4):
            specs.append(TaskSpec(0, "today" if off == 0 else "future",
                                  _dt(today, off, hour), rel_expr=expr))
        specs.append(TaskSpec(0, "none"))

    elif scenario == "past_split":
        # 지난 일정 3-4개 + 유효 일정 3-4개 혼합 — 깨끗한 과거/유효 분리가 목표
        for _ in range(rng.randint(3, 4)):
            off = -rng.randint(1, 5)
            specs.append(TaskSpec(0, "past", _dt(today, off, rng.randint(9, 19)), is_past=True))
        specs.append(TaskSpec(0, "today", _dt(today, 0, rng.randint(9, 18))))
        for _ in range(rng.randint(1, 2)):
            specs.append(TaskSpec(0, "future", _dt(today, rng.randint(1, 6), rng.randint(9, 18))))
        if rng.random() < 0.5:
            specs.append(TaskSpec(0, "none"))

    elif scenario == "no_today":
        # 오늘 날짜 미상 — 절대 날짜들 간 상대 정렬만 가능. '지남' 판단은 환각.
        base = today  # 기준일은 내부 생성용일 뿐, 시스템 프롬프트에는 미주입
        offs = rng.sample(range(0, 14), k=rng.randint(4, 5))
        for off in offs:
            specs.append(TaskSpec(0, "dated", _dt(base, off, rng.randint(9, 18))))
        if rng.random() < 0.6:
            specs.append(TaskSpec(0, "none"))

    elif scenario == "am_escalation":
        # 오전 마감 + 에스컬레이션(긴급5) vs 같은 날 오후 마감 고중요 — 오전이 1위여야 함
        specs.append(TaskSpec(
            0, "risk", _dt(today, 0, rng.randint(9, 11), rng.choice([0, 30])),
            risk=True, risk_clause=rng.choice(_RISK_CLAUSES),
        ))
        specs.append(TaskSpec(0, "today", _dt(today, 0, rng.randint(14, 18))))
        specs.append(TaskSpec(0, "future", _dt(today, rng.randint(2, 7), rng.randint(9, 18))))
        if rng.random() < 0.7:
            specs.append(TaskSpec(0, "none"))

    else:
        raise ValueError(f"unknown scenario: {scenario}")

    specs = _shuffle_reindex(specs, rng)
    skel_today = "" if scenario == "no_today" else today.isoformat()
    return Skeleton(scenario=scenario, today=skel_today, specs=specs)


# ── GPT 프롬프트 ──────────────────────────────────────────────────────────────

_TEXT_GEN_SYSTEM = """\
당신은 한국어 업무/일상 할 일 텍스트 작성기입니다. 페르소나와 태스크 골격이 주어지면
각 골격에 맞는 현실적인 할 일 한 줄씩을 생성하세요.

규칙:
- 골격에 마감 일시가 있으면 그 일시를 텍스트에 자연스럽게 포함 (예: "(5/24 17:00까지)", "5월 24일 오전 11시 마감")
- '상대 표현 사용' 지시가 있으면 절대 날짜 대신 반드시 그 표현을 사용 (예: "내일 오전까지")
- 지난 일정이라도 텍스트에 '지났다', '어제' 같은 단서를 절대 쓰지 말 것 — 날짜만 그대로 표기
- 리스크 문구가 지정된 태스크는 그 문구를 텍스트에 포함
- 체인 태스크는 같은 업무의 연속 단계로 작성 (예: 보고서 작성 → 보고서 검토 반영 → 보고서 발송)
- 페르소나의 직업·상황에 맞는 소재 사용, 태스크 간 소재 중복 금지
- 출력: {"tasks": ["...", ...]} — 골격과 같은 개수·같은 순서"""

_TEXT_GEN_USER = """\
[페르소나] {persona}
[오늘] {today}

[태스크 골격]
{spec_lines}

골격과 같은 순서로 {n}개의 할 일 텍스트를 JSON으로 생성하세요."""

_CHOSEN_HINT_HEADER = (
    "\n\n(채점 참고 사실 — 출력 JSON에는 이 블록을 포함하지 말 것:\n{facts}\n)"
)


def _spec_to_gen_line(s: TaskSpec, today: str) -> str:
    parts = [f"{s.idx}."]
    if s.kind == "chain":
        parts.append(f"체인 {s.chain_group}의 {s.chain_pos}단계.")
        parts.append(f"마감 {s.deadline}" if s.deadline else "마감 표기 없음")
    elif s.kind == "none":
        parts.append("마감 없음 (날짜·시각 표기 금지)")
    elif s.rel_expr:
        parts.append(f"마감을 상대 표현 '{s.rel_expr}'로 표기 (절대 날짜 금지)")
    else:
        parts.append(f"마감 {s.deadline} (텍스트에 일시 포함)")
    if s.risk:
        parts.append(f"+ 리스크 문구 포함: \"{s.risk_clause}\"")
    return " ".join(parts)


def _spec_to_fact(s: TaskSpec, today: str) -> str:
    if s.kind == "none":
        return f"- 태스크 {s.idx}: 마감 없음"
    if s.kind == "chain":
        dl = f", 최종 마감 {s.deadline}" if s.deadline else ""
        return f"- 태스크 {s.idx}: 연쇄 업무 {s.chain_pos}단계 (선행 완료 필요{dl})"
    state = ""
    if s.is_past:
        state = f" — 오늘({today}) 기준 이미 지남"
    rel = f" (표현 '{s.rel_expr}')" if s.rel_expr else ""
    risk = f" — 리스크: {s.risk_clause}" if s.risk else ""
    return f"- 태스크 {s.idx}: 마감 {s.deadline}{rel}{state}{risk}"


# ── chosen 검증 ───────────────────────────────────────────────────────────────

def verify_chosen(skel: Skeleton, chosen_json: str) -> list[str]:
    """골격 대비 chosen의 위반 목록 반환 (빈 리스트 = 통과)."""
    resp = parse_lenient(chosen_json)
    if resp is None or not resp.tasks:
        return ["JSON 파싱 실패"]

    errors: list[str] = []
    n = len(skel.specs)
    if len(resp.tasks) != n:
        errors.append(f"태스크 수 불일치: 입력 {n} vs 출력 {len(resp.tasks)}")
        return errors
    if sorted(resp.priority_order) != sorted(t.id for t in resp.tasks):
        errors.append("priority_order가 전체 태스크를 정확히 1회씩 포함하지 않음")
        return errors

    pos = {tid: i for i, tid in enumerate(resp.priority_order)}
    smap = {s.task_id: s for s in resp.scores}
    spec = {s.idx: s for s in skel.specs}

    # no_today: 오늘 미상 — 절대 날짜의 상대 정렬만 검증, '지남' 판단은 환각으로 간주
    if skel.today == "":
        dated = sorted((s for s in skel.specs if s.deadline), key=lambda s: s.deadline)
        for a, b in zip(dated, dated[1:]):
            if a.deadline < b.deadline and pos[a.idx] > pos[b.idx]:
                errors.append(f"마감 {a.deadline}(태스크 {a.idx})이 "
                              f"{b.deadline}(태스크 {b.idx})보다 후순위")
        for s in resp.scores:
            if "이미 지" in (s.reason or "") or "지난 일정" in (s.reason or ""):
                errors.append(f"오늘 미상인데 태스크 {s.task_id}의 reason이 '지남'을 단정함")
        none_ids = [s.idx for s in skel.specs if s.kind == "none"]
        if none_ids and resp.priority_order and resp.priority_order[0] in none_ids:
            errors.append("마감 없는 태스크가 1위에 배치됨")
        return errors

    # 1) 지난 일정은 모든 미지남 태스크보다 후순위 + 낮은 urgency/time_constraint
    past_ids = [s.idx for s in skel.specs if s.is_past]
    live_ids = [s.idx for s in skel.specs if not s.is_past]
    for p in past_ids:
        if any(pos[p] < pos[live] for live in live_ids):
            errors.append(f"지난 일정 태스크 {p}가 유효 태스크보다 상위에 배치됨")
        sc = smap.get(p)
        if sc and (sc.urgency > 2 or sc.time_constraint > 2):
            errors.append(f"지난 일정 태스크 {p}의 urgency/time_constraint가 2 초과")

    # 2) 같은 날(오늘) 마감 간 시각 순서
    today_specs = sorted(
        (s for s in skel.specs if s.kind in ("today", "risk") and s.deadline
         and s.deadline.startswith(skel.today)),
        key=lambda s: s.deadline,
    )
    for a, b in zip(today_specs, today_specs[1:]):
        risk_pair = spec[a.idx].risk != spec[b.idx].risk
        if not risk_pair and a.deadline < b.deadline and pos[a.idx] > pos[b.idx]:
            errors.append(f"오늘 {a.deadline[-5:]} 마감(태스크 {a.idx})이 "
                          f"{b.deadline[-5:]} 마감(태스크 {b.idx})보다 후순위")

    # 3) 체인: 순서 보존 + 연속 배치 + dependency 점수
    chains: dict[int, list[TaskSpec]] = {}
    for s in skel.specs:
        if s.chain_group:
            chains.setdefault(s.chain_group, []).append(s)
    for members in chains.values():
        members.sort(key=lambda s: s.chain_pos)
        ranks = [pos[m.idx] for m in members]
        if ranks != sorted(ranks):
            errors.append("체인 태스크의 순서가 단계 순서와 다름")
        if max(ranks) - min(ranks) != len(ranks) - 1:
            errors.append("체인 태스크가 연속으로 배치되지 않음")
        for m in members[:-1]:
            sc = smap.get(m.idx)
            if sc and sc.dependency < 4:
                errors.append(f"체인 선행 태스크 {m.idx}의 dependency가 4 미만")

    # 4) 리스크 태스크 importance
    for s in skel.specs:
        if s.risk:
            sc = smap.get(s.idx)
            if sc and sc.importance < 4:
                errors.append(f"리스크 태스크 {s.idx}의 importance가 4 미만")

    # 5) 무마감 태스크가 1위면 의심
    none_ids = [s.idx for s in skel.specs if s.kind == "none"]
    if none_ids and resp.priority_order and resp.priority_order[0] in none_ids:
        errors.append("마감 없는 태스크가 1위에 배치됨")

    # 6) am_escalation: 오전 마감+에스컬레이션은 urgency=5 + 같은 날 오후 마감보다 상위
    if skel.scenario == "am_escalation":
        am = next((s for s in skel.specs if s.risk), None)
        pm = next((s for s in skel.specs if s.kind == "today"), None)
        if am:
            sc = smap.get(am.idx)
            if sc and sc.urgency < 5:
                errors.append(f"오전 마감+에스컬레이션 태스크 {am.idx}의 urgency가 5 미만")
            if pm and pos[am.idx] > pos[pm.idx]:
                errors.append(f"오전 마감 태스크 {am.idx}가 오후 마감 태스크 {pm.idx}보다 후순위")

    return errors


# ── 행 생성 ───────────────────────────────────────────────────────────────────

async def gen_row(
    sample_id: int,
    persona_label: str,
    persona_name: str,
    scenario: str,
    client: AsyncOpenAI,
    cache: _APICache,
    cache_lock: asyncio.Lock,
    text_model: str,
    chosen_model: str,
    rng: random.Random,
) -> dict | None:
    today = datetime.date(2026, 1, 1) + datetime.timedelta(days=rng.randint(0, 360))
    skel = build_skeleton(scenario, today, rng)
    today_h = (f"{skel.today} ({_WEEKDAYS_KO[today.weekday()]})" if skel.today
               else "날짜 미상 (오늘 날짜를 알 수 없음)")

    # Step 1: 태스크 텍스트 생성
    spec_lines = "\n".join(_spec_to_gen_line(s, skel.today) for s in skel.specs)
    raw1 = await _call_api(
        client, text_model,
        [
            {"role": "system", "content": _TEXT_GEN_SYSTEM},
            {"role": "user", "content": _TEXT_GEN_USER.format(
                persona=persona_label, today=today_h,
                spec_lines=spec_lines, n=len(skel.specs))},
        ],
        cache, cache_lock, max_tokens=500, temperature=0.9,
        label=f"v3-text {sample_id}",
    )
    if not raw1:
        return None
    try:
        texts = json.loads(raw1).get("tasks", [])
        if len(texts) != len(skel.specs):
            return None
    except Exception:
        return None

    task_lines = "\n".join(f"- {t}" for t in texts)
    user_prompt = f"[{persona_name} 씨의 오늘의 할 일 목록]\n{task_lines}"

    # Step 2: chosen 생성 (+검증 실패 시 위반 사항을 알려주고 1회 재생성)
    system = render_system_prompt(SCHEDULER_SYSTEM_PROMPT_V3, persona_label, today=skel.today)
    facts = "\n".join(_spec_to_fact(s, skel.today) for s in skel.specs)
    if not skel.today:
        facts = ("- 오늘 날짜 미상: 어떤 일정도 '이미 지났다'고 단정하지 말 것. "
                 "절대 날짜 간 상대 비교(이른 마감 = 높은 urgency)만 수행.\n" + facts)
    gen_user = user_prompt + _CHOSEN_HINT_HEADER.format(facts=facts)

    feedback = ""
    for _attempt in range(2):
        raw2 = await _call_api(
            client, chosen_model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": gen_user + feedback},
            ],
            cache, cache_lock, max_tokens=3000, temperature=None,
            label=f"v3-chosen {sample_id}",
        )
        if not raw2:
            return None
        errors = verify_chosen(skel, raw2)
        if not errors:
            parsed = parse_lenient(raw2)
            return {
                "prompt": user_prompt,
                "chosen": format_for_sft(parsed),
                "persona": persona_label,
                "today": skel.today,
                "source": f"v3_{scenario}",
                "meta": json.dumps(asdict(skel), ensure_ascii=False),
            }
        feedback = "\n\n(이전 응답의 위반 사항 — 반드시 수정:\n" + "\n".join(f"- {e}" for e in errors) + "\n)"
    print(f"  [검증실패] {sample_id} ({scenario}): {errors[:3]}")
    return None


# ── 메인 ─────────────────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    nem_df = pd.read_parquet(_NEMOTRON_PATH, columns=["persona", "age", "occupation"])
    nem_df = nem_df.sample(args.total * 2, random_state=args.seed).reset_index(drop=True)
    print(f"[v3] 페르소나 {len(nem_df)}개 샘플링, 목표 {args.total}행")

    # 시나리오 배분 (--scenarios "name:count,..." 지정 시 그대로 사용)
    plan: list[str] = []
    if args.scenarios:
        for part in args.scenarios.split(","):
            name, _, cnt = part.strip().partition(":")
            plan.extend([name] * int(cnt))
        args.total = len(plan)
    else:
        for name, w in _SCENARIOS:
            plan.extend([name] * int(args.total * w))
        while len(plan) < args.total:
            plan.append("dated_mixed")
    rng.shuffle(plan)

    ckpt_path = Path(args.ckpt)
    existing_rows, done_keys = _load_checkpoint(ckpt_path)
    cache = _APICache(Path(args.cache))
    cache_lock = asyncio.Lock()
    ckpt_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    counter = {"ok": 0, "fail": 0}
    new_rows: list[dict] = []

    async def _process(i: int, scenario: str) -> None:
        key = f"v3-{args.seed}-{i}"
        if key in done_keys:
            return
        row = nem_df.iloc[i]
        persona_raw = str(row.get("persona", ""))
        persona_name = persona_raw.split(" 씨는")[0].strip() if " 씨는" in persona_raw else f"사용자{i}"
        persona_label = f"{persona_name} ({row.get('occupation', '직장인')}, {row.get('age', 30)}세)"
        local_rng = random.Random(args.seed * 100_000 + i)
        async with semaphore:
            result = await gen_row(
                i, persona_label, persona_name, scenario, client,
                cache, cache_lock, args.text_model, args.chosen_model, local_rng,
            )
        if result:
            await _append_checkpoint_async(ckpt_path, result, key, ckpt_lock)
            new_rows.append(result)
            counter["ok"] += 1
            if counter["ok"] % 50 == 0:
                print(f"  진행: {counter['ok']}개 완료 (실패 {counter['fail']})")
        else:
            counter["fail"] += 1

    await asyncio.gather(*[_process(i, sc) for i, sc in enumerate(plan)])

    all_rows = existing_rows + new_rows
    print(f"\n[완료] 총 {len(all_rows)}행 (신규 {counter['ok']}, 실패 {counter['fail']})")
    if not all_rows:
        sys.exit(1)

    if args.verify:
        ok = sum(1 for r in all_rows if not verify_chosen(
            Skeleton(**{**json.loads(r["meta"]),
                        "specs": [TaskSpec(**s) for s in json.loads(r["meta"])["specs"]]}),
            r["chosen"]))
        print(f"[검증] 골격 정합성 통과: {ok}/{len(all_rows)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_parquet(str(out), index=False)
    print(f"[저장] {out} ({len(all_rows)}행)")

    # 시나리오 분포 출력
    dist = pd.DataFrame(all_rows)["source"].value_counts()
    print(dist.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v3 스케줄 데이터 생성")
    parser.add_argument("--total", type=int, default=2000)
    parser.add_argument("--scenarios", default=None,
                        help='시나리오별 수량 지정 (예: "past_split:200,no_today:200,am_escalation:150")')
    parser.add_argument("--text-model", default=_TEXT_MODEL)
    parser.add_argument("--chosen-model", default=_CHOSEN_MODEL)
    parser.add_argument("--concurrency", type=int, default=_DEFAULT_CONCURRENCY)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--out", default="data/scheduler_v3.parquet")
    parser.add_argument("--ckpt", default="outputs/.ckpt_gen_v3.jsonl")
    parser.add_argument("--cache", default="outputs/.api_cache_v3.jsonl")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("[에러] OPENAI_API_KEY 환경변수가 필요합니다.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run(args))
