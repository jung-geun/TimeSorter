#!/usr/bin/env python
"""스케줄 결과를 대형 LLM으로 교차 검증.

검증 프로세스:
  Phase 1 — 판사 LLM이 원본 이메일만 보고 독자적으로 태스크 추출 + 우선순위 결정
  Phase 2 — 모델 출력 vs 판사 기준 비교: 태스크 커버리지, 순서 정확도, 추론 품질 채점

사용:
  # 기본 (gpt-5.5, outputs/schedule_result.json)
  uv run python scripts/validate_schedule.py

  # 결과 파일과 이메일 디렉토리 명시
  uv run python scripts/validate_schedule.py \\
      --result outputs/schedule_result.json \\
      --email-dir data/sample_emails

  # 판사 모델 변경
  uv run python scripts/validate_schedule.py --judge gpt-5.5

  # 결과 저장
  uv run python scripts/validate_schedule.py --out outputs/validation_result.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── 판사 프롬프트 ─────────────────────────────────────────────────────────────

_PHASE1_SYSTEM = """당신은 스케줄 관리 전문가입니다.
주어진 이메일들을 꼼꼼히 읽고, 수신자가 처리해야 할 모든 행동 항목을 누락 없이 추출한 뒤
4축 기준(긴급도·중요도·의존성·시간 제약)으로 최적 우선순위를 결정하십시오.

반드시 아래 JSON 형식으로만 응답하십시오:
{
  "tasks": [
    {"id": 1, "text": "태스크 설명 (마감 포함)", "source_email": "파일명"},
    ...
  ],
  "priority_order": [1, 3, 2, ...],
  "priority_reasoning": [
    {"rank": 1, "task_id": X, "reason": "이유"},
    ...
  ]
}"""

_PHASE1_USER = """페르소나: {persona}

=== 원본 이메일 목록 ===
{emails_block}

위 이메일들에서 수신자({persona})가 처리해야 할 모든 행동 항목과 최적 우선순위를 JSON으로 반환하세요."""


_PHASE2_SYSTEM = """당신은 AI 스케줄링 모델의 출력을 평가하는 전문 심사위원입니다.
기준 정답(판사 생성)과 모델 출력을 비교하여 객관적이고 엄정하게 평가하십시오.

채점 기준:
- task_coverage (1-5): 태스크 추출 완성도 (누락·환각 여부)
- priority_accuracy (1-5): 우선순위 순서의 합리성
- reasoning_quality (1-5): 각 항목에 붙은 이유의 논리성과 구체성
- score_consistency (1-5): 4축 점수(urgency/importance/dependency/time_constraint)와 priority_order 간 일관성. 모델 출력에 4축 점수가 없으면 3점으로 기본 처리.
- overall (1-5): 종합 점수

판정:
- PASS: overall ≥ 4
- PARTIAL: overall = 3
- FAIL: overall ≤ 2

반드시 아래 JSON 형식으로만 응답하십시오:
{
  "task_coverage": {
    "score": <1-5>,
    "found_correctly": ["모델이 올바르게 찾은 태스크", ...],
    "missed_by_model": ["모델이 놓친 태스크", ...],
    "hallucinated_by_model": ["모델이 없는 내용을 추가한 태스크", ...]
  },
  "priority_accuracy": {
    "score": <1-5>,
    "correct_items": ["순서가 맞는 항목", ...],
    "wrong_items": [{"task": "태스크", "model_rank": X, "expected_rank": Y, "reason": "이유"}, ...]
  },
  "reasoning_quality": {
    "score": <1-5>,
    "strengths": ["잘된 점", ...],
    "weaknesses": ["부족한 점", ...]
  },
  "score_consistency": {
    "score": <1-5>,
    "has_axis_scores": true/false,
    "inconsistencies": ["불일치 사례", ...]
  },
  "overall": <1-5>,
  "verdict": "PASS|PARTIAL|FAIL",
  "summary": "종합 평가 2-3문장"
}"""

_PHASE2_USER = """페르소나: {persona}

=== 원본 이메일 요약 ===
{emails_block}

=== 판사(기준) 태스크 목록 및 우선순위 ===
{reference_block}

=== 모델 출력 (평가 대상) ===
추출된 태스크:
{model_tasks_block}

생성된 스케줄:
{model_schedule}

위를 비교하여 JSON으로 평가 결과를 반환하세요."""


# ── 데이터 구조 ───────────────────────────────────────────────────────────────

@dataclass
class ReferenceTask:
    id: int
    text: str
    source_email: str = ""


@dataclass
class ReferenceResult:
    tasks: list[ReferenceTask]
    priority_order: list[int]
    priority_reasoning: list[dict]


@dataclass
class EvaluationResult:
    judge_model: str
    persona: str
    reference: dict
    evaluation: dict
    verdict: str
    overall_score: int


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def _load_emails(email_dir: Path) -> dict[str, str]:
    """파일명 → 본문 매핑."""
    result: dict[str, str] = {}
    for p in sorted(email_dir.iterdir()):
        if p.suffix in (".txt", ".eml") and not p.name.startswith("."):
            result[p.name] = p.read_text(encoding="utf-8", errors="replace")
    return result


def _build_emails_block(emails: dict[str, str]) -> str:
    parts = []
    for fname, content in emails.items():
        parts.append(f"--- [{fname}] ---\n{content.strip()}")
    return "\n\n".join(parts)


def _call_judge(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    label: str,
) -> dict:
    print(f"  [{label}] {model} 호출 중...")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=4000,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content.strip()
    return json.loads(raw)


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def phase1_generate_reference(
    emails: dict[str, str],
    persona: str,
    client: OpenAI,
    model: str,
) -> dict:
    emails_block = _build_emails_block(emails)
    user_msg = _PHASE1_USER.format(persona=persona, emails_block=emails_block)
    return _call_judge(client, model, _PHASE1_SYSTEM, user_msg, "Phase 1 독립 생성")


# ── Phase 2 ───────────────────────────────────────────────────────────────────

def phase2_evaluate(
    emails: dict[str, str],
    persona: str,
    reference: dict,
    model_tasks: list[str],
    model_schedule: str,
    client: OpenAI,
    model: str,
) -> dict:
    # 이메일 요약 (제목만)
    emails_summary = "\n".join(
        f"- {fname}: {content.splitlines()[3] if len(content.splitlines()) > 3 else content[:60]}"
        for fname, content in emails.items()
    )

    # 기준 블록
    ref_tasks = reference.get("tasks", [])
    priority_order = reference.get("priority_order", [])
    priority_reasoning = reference.get("priority_reasoning", [])

    ref_task_map = {t["id"]: t["text"] for t in ref_tasks}
    ref_ordered = [
        f"{rank+1}. [ID={tid}] {ref_task_map.get(tid, '?')}"
        for rank, tid in enumerate(priority_order)
    ]
    ref_reasons = [
        f"  → {r.get('reason', '')}"
        for r in priority_reasoning
    ]
    reference_block = "\n".join(
        line for pair in zip(ref_ordered, ref_reasons) for line in pair
    )
    if not reference_block:
        reference_block = json.dumps(reference, ensure_ascii=False, indent=2)

    model_tasks_block = "\n".join(f"- {t}" for t in model_tasks)

    user_msg = _PHASE2_USER.format(
        persona=persona,
        emails_block=emails_summary,
        reference_block=reference_block,
        model_tasks_block=model_tasks_block,
        model_schedule=model_schedule[:2000],  # 너무 길면 잘라냄
    )
    return _call_judge(client, model, _PHASE2_SYSTEM, user_msg, "Phase 2 비교 평가")


# ── 보고서 출력 ───────────────────────────────────────────────────────────────

_VERDICT_EMOJI = {"PASS": "✅", "PARTIAL": "⚠️", "FAIL": "❌"}
_SCORE_BAR = {1: "█░░░░", 2: "██░░░", 3: "███░░", 4: "████░", 5: "█████"}


def print_report(result: dict, judge_model: str) -> None:
    ev = result.get("evaluation", {})
    ref = result.get("reference", {})
    verdict = ev.get("verdict", "?")
    overall = ev.get("overall", 0)

    print(f"\n{'='*65}")
    print(f"  교차 검증 결과 ({judge_model})")
    print(f"{'='*65}")
    print(f"  종합 판정: {_VERDICT_EMOJI.get(verdict, '?')} {verdict}  "
          f"({overall}/5 {_SCORE_BAR.get(overall, '')})")
    print(f"{'─'*65}")

    # 태스크 커버리지
    tc = ev.get("task_coverage", {})
    print(f"\n📋 태스크 커버리지  {tc.get('score',0)}/5")
    if tc.get("found_correctly"):
        print("  ✔ 올바르게 추출:")
        for t in tc["found_correctly"]:
            print(f"      - {t}")
    if tc.get("missed_by_model"):
        print("  ✘ 누락:")
        for t in tc["missed_by_model"]:
            print(f"      - {t}")
    if tc.get("hallucinated_by_model"):
        print("  ⚠ 환각 (없는 내용):")
        for t in tc["hallucinated_by_model"]:
            print(f"      - {t}")

    # 우선순위 정확도
    pa = ev.get("priority_accuracy", {})
    print(f"\n📊 우선순위 정확도  {pa.get('score',0)}/5")
    if pa.get("correct_items"):
        print("  ✔ 순서 적절:")
        for t in pa["correct_items"]:
            print(f"      - {t}")
    if pa.get("wrong_items"):
        print("  ✘ 순서 오류:")
        for w in pa["wrong_items"]:
            if isinstance(w, dict):
                print(f"      - {w.get('task','?')}  "
                      f"(모델:{w.get('model_rank','?')}위 → 권장:{w.get('expected_rank','?')}위)  "
                      f"— {w.get('reason','')}")
            else:
                print(f"      - {w}")

    # 추론 품질
    rq = ev.get("reasoning_quality", {})
    print(f"\n💬 추론 품질  {rq.get('score',0)}/5")
    if rq.get("strengths"):
        print("  ✔ 강점:")
        for s in rq["strengths"]:
            print(f"      - {s}")
    if rq.get("weaknesses"):
        print("  ⚠ 약점:")
        for w in rq["weaknesses"]:
            print(f"      - {w}")

    # 4축 점수 일관성
    sc = ev.get("score_consistency", {})
    if sc:
        has_scores = sc.get("has_axis_scores", False)
        print(f"\n🎯 4축 점수 일관성  {sc.get('score',0)}/5"
              f"  ({'점수 있음' if has_scores else '점수 없음'})")
        if sc.get("inconsistencies"):
            print("  ⚠ 불일치:")
            for inc in sc["inconsistencies"]:
                print(f"      - {inc}")

    # 판사 기준 (독립 생성)
    ref_tasks = ref.get("tasks", [])
    ref_order = ref.get("priority_order", [])
    ref_map = {t["id"]: t["text"] for t in ref_tasks}
    print(f"\n🏛  판사 기준 우선순위 ({judge_model} 독립 생성)")
    for rank, tid in enumerate(ref_order, 1):
        print(f"  {rank}. {ref_map.get(tid, '?')}")

    # 종합 요약
    summary = ev.get("summary", "")
    if summary:
        print("\n📝 종합 평가")
        print(f"  {summary}")

    print(f"\n{'='*65}\n")


# ── 진입점 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="스케줄 결과 교차 검증")
    parser.add_argument(
        "--result", default="outputs/schedule_result.json",
        help="email_to_schedule.py 출력 JSON (기본: outputs/schedule_result.json)"
    )
    parser.add_argument(
        "--email-dir", default="data/sample_emails",
        help="원본 이메일 디렉토리 (기본: data/sample_emails)"
    )
    parser.add_argument(
        "--judge", default="gpt-5.5",
        help="판사 모델 (기본: gpt-5.5)"
    )
    parser.add_argument(
        "--out", default=None,
        help="검증 결과 JSON 저장 경로"
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[에러] OPENAI_API_KEY 환경변수가 필요합니다.", file=sys.stderr)
        sys.exit(1)

    result_path = Path(args.result)
    if not result_path.exists():
        print(f"[에러] 결과 파일 없음: {result_path}", file=sys.stderr)
        print("  'make email-pipeline' 또는 'uv run python scripts/email_to_schedule.py'를 먼저 실행하세요.")
        sys.exit(1)

    schedule_result = json.loads(result_path.read_text(encoding="utf-8"))
    email_dir = Path(args.email_dir)
    emails = _load_emails(email_dir)

    if not emails:
        print(f"[에러] {email_dir}에 이메일 파일이 없습니다.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    persona = schedule_result.get("persona", "직장인")
    model_tasks = schedule_result.get("tasks", [])
    model_schedule = schedule_result.get("schedule", "")

    print("\n[교차 검증 시작]")
    print(f"  판사 모델  : {args.judge}")
    print(f"  이메일 수  : {len(emails)}건")
    print(f"  모델 태스크: {len(model_tasks)}개")
    print(f"  페르소나   : {persona}\n")

    # Phase 1
    reference = phase1_generate_reference(emails, persona, client, args.judge)
    ref_tasks = reference.get("tasks", [])
    print(f"  → 판사 기준 태스크: {len(ref_tasks)}개 추출됨")

    # Phase 2
    evaluation = phase2_evaluate(
        emails=emails,
        persona=persona,
        reference=reference,
        model_tasks=model_tasks,
        model_schedule=model_schedule,
        client=client,
        model=args.judge,
    )

    verdict = evaluation.get("verdict", "?")
    overall = evaluation.get("overall", 0)
    print(f"  → 종합 판정: {verdict} ({overall}/5)\n")

    # 보고서 출력
    print_report(
        result={"reference": reference, "evaluation": evaluation},
        judge_model=args.judge,
    )

    # 저장
    final = {
        "judge_model": args.judge,
        "persona": persona,
        "email_count": len(emails),
        "model_task_count": len(model_tasks),
        "reference": reference,
        "evaluation": evaluation,
        "verdict": verdict,
        "overall_score": overall,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(final, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[저장] {out_path}")
