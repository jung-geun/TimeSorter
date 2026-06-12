#!/usr/bin/env python
"""v1/v2 레거시 데이터를 v3-v5 포맷 재가공용 JSONL 슬라이스로 export.

v1(자유 텍스트) + v2_schedule(persona_fit 낮음) 데이터에서 태스크 목록을 추출하고,
각 샘플에 시나리오 골격(today, 마감, 체인, 리스크)을 붙여 Claude 에이전트 지시문 JSONL 작성.
에이전트는 기존 태스크 텍스트를 개선 후 v3-v5 스키마 JSON을 생성한다.

사용:
  uv run python scripts/rework_legacy_prep.py --source v1 --max-rows 2000 \
      --slices 20 --out-prefix outputs/rework_v1
  uv run python scripts/rework_legacy_prep.py --source v2 --max-rows 2000 \
      --slices 20 --out-prefix outputs/rework_v2
"""
from __future__ import annotations

import argparse
import datetime
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from gen_schedule_v3 import (
    _RISK_CLAUSES,
    TaskSpec,
    Skeleton,
    _dt,
    _shuffle_reindex,
    _spec_to_gen_line,
    _spec_to_fact,
)


def extract_tasks(prompt: str) -> list[str]:
    """프롬프트에서 '-' 또는 번호로 시작하는 태스크 줄 추출."""
    lines = prompt.split("\n")
    tasks = []
    for line in lines:
        line = line.strip()
        # '- 태스크' 형식
        if line.startswith("- "):
            t = line[2:].strip()
            if t:
                tasks.append(t)
        # '1) 태스크' 또는 '1. 태스크' 형식
        elif re.match(r"^\d+[.)]\s+", line):
            t = re.sub(r"^\d+[.)]\s+", "", line).strip()
            # 점수 행 제외 (urgency: 같은 패턴)
            if t and "urgency" not in t.lower() and "importance" not in t.lower():
                tasks.append(t)
    return tasks


def build_rework_skeleton(n_tasks: int, today: datetime.date, rng: random.Random) -> Skeleton:
    """n개 태스크에 맞는 시나리오 골격 생성 (기존 build_skeleton 로직 활용)."""
    # 태스크 수에 따라 적절한 시나리오 선택
    scenario = rng.choice([
        "dated_mixed", "dated_mixed", "dated_mixed",
        "intraday", "dependency_chain",
        "risk", "past_split", "no_today", "am_escalation",
    ])

    specs: list[TaskSpec] = []

    if scenario == "dated_mixed":
        n_past = min(rng.choice([1, 1, 2]), n_tasks - 3) if n_tasks >= 4 else 0
        for _ in range(max(n_past, 0)):
            off = -rng.randint(1, 3)
            specs.append(TaskSpec(0, "past", _dt(today, off, rng.randint(10, 18)), is_past=True))
        specs.append(TaskSpec(0, "today", _dt(today, 0, rng.randint(9, 11), rng.choice([0, 30]))))
        specs.append(TaskSpec(0, "today", _dt(today, 0, rng.randint(14, 18))))
        remaining = n_tasks - len(specs)
        if remaining > 0:
            specs.append(TaskSpec(0, "future", _dt(today, rng.randint(2, 7), rng.randint(9, 18))))
            remaining -= 1
        for _ in range(remaining):
            specs.append(TaskSpec(0, "none"))

    elif scenario == "intraday":
        hours = rng.sample([9, 10, 11, 13, 14, 16, 17, 19], k=min(n_tasks, 8))
        for h in hours[:n_tasks]:
            specs.append(TaskSpec(0, "today", _dt(today, 0, h)))
        for _ in range(n_tasks - len(specs)):
            specs.append(TaskSpec(0, "none"))

    elif scenario == "dependency_chain":
        chain_len = min(3, n_tasks - 2) if n_tasks >= 3 else 2
        end_hour = rng.randint(14, 18)
        for pos in range(1, chain_len + 1):
            specs.append(TaskSpec(
                0, "chain",
                _dt(today, rng.choice([0, 1]), end_hour) if pos == chain_len else None,
                chain_group=1, chain_pos=pos,
            ))
        remaining = n_tasks - len(specs)
        if remaining > 0:
            specs.append(TaskSpec(0, "today", _dt(today, 0, rng.randint(9, 17))))
            remaining -= 1
        for _ in range(remaining):
            specs.append(TaskSpec(0, "none"))

    elif scenario == "risk":
        risk_idx = rng.sample(range(n_tasks), k=min(rng.choice([1, 2]), n_tasks))
        for i in range(n_tasks):
            if i in risk_idx:
                specs.append(TaskSpec(
                    0, "risk", _dt(today, rng.choice([0, 1]), rng.randint(9, 18)),
                    risk=True, risk_clause=rng.choice(_RISK_CLAUSES),
                ))
            elif rng.random() < 0.5:
                specs.append(TaskSpec(0, "future", _dt(today, rng.randint(2, 7), rng.randint(9, 18))))
            else:
                specs.append(TaskSpec(0, "none"))

    elif scenario == "past_split":
        n_past = min(rng.randint(1, 2), n_tasks - 2)
        for _ in range(max(n_past, 0)):
            specs.append(TaskSpec(0, "past", _dt(today, -rng.randint(1, 5), rng.randint(9, 19)), is_past=True))
        specs.append(TaskSpec(0, "today", _dt(today, 0, rng.randint(9, 18))))
        remaining = n_tasks - len(specs)
        for i in range(remaining):
            if i % 2 == 0:
                specs.append(TaskSpec(0, "future", _dt(today, rng.randint(1, 6), rng.randint(9, 18))))
            else:
                specs.append(TaskSpec(0, "none"))

    elif scenario == "no_today":
        offs = rng.sample(range(1, 15), k=min(n_tasks - 1, 4))
        for off in offs:
            specs.append(TaskSpec(0, "dated", _dt(today, off, rng.randint(9, 18))))
        for _ in range(n_tasks - len(specs)):
            specs.append(TaskSpec(0, "none"))

    elif scenario == "am_escalation":
        specs.append(TaskSpec(
            0, "risk", _dt(today, 0, rng.randint(9, 11), rng.choice([0, 30])),
            risk=True, risk_clause=rng.choice(_RISK_CLAUSES),
        ))
        specs.append(TaskSpec(0, "today", _dt(today, 0, rng.randint(14, 18))))
        remaining = n_tasks - 2
        for i in range(remaining):
            if i % 2 == 0:
                specs.append(TaskSpec(0, "future", _dt(today, rng.randint(2, 7), rng.randint(9, 18))))
            else:
                specs.append(TaskSpec(0, "none"))
    else:
        for _ in range(n_tasks):
            specs.append(TaskSpec(0, "none"))

    # 수 맞추기
    while len(specs) < n_tasks:
        specs.append(TaskSpec(0, "none"))
    specs = specs[:n_tasks]

    specs = _shuffle_reindex(specs, rng)
    skel_today = "" if scenario == "no_today" else today.isoformat()
    return Skeleton(scenario=scenario, today=skel_today, specs=specs)


_AGENT_INSTRUCTION = """\
당신은 한국어 일정 데이터 품질 개선 전문가입니다.

아래 태스크 목록을 v3-v5 포맷으로 재가공하세요.

[페르소나] {persona}
[오늘] {today_display}

[원본 태스크 목록]
{original_tasks}

[골격 (각 태스크에 맞춰 텍스트 개선)]
{spec_lines}

[채점 참고 사실]
{facts}

지시:
1. 각 태스크를 골격에 맞게 개선하세요:
   - 마감 일시가 있으면 태스크 텍스트에 자연스럽게 포함 (예: "보고서 제출 (6/15 17:00까지)")
   - 리스크 문구는 반드시 포함 (지정된 경우)
   - 체인 태스크는 같은 업무의 단계로 연결 (예: 작성→검토→제출)
   - 지난 일정도 텍스트에 '지났다' 표현 금지 — 날짜만 표기
   - 페르소나({persona_short}) 직업/상황에 맞는 구체적인 업무 내용 사용
2. v3 스키마 JSON으로 우선순위를 결정하세요:
   - 지난 일정: urgency=1, importance=1, time_constraint≤2
   - 리스크 태스크: importance≥4, urgency≥4
   - 체인 최하단 마감이 오늘/내일: 체인 전체 연속 배치
   - 오전 마감: 오후 마감보다 상위
   - 오늘 날짜 미상(today 빈칸): 지난/유효 판단 절대 금지

출력 형식 (JSON 한 줄):
{{"id": "{row_id}", "tasks": ["개선된 태스크1", "개선된 태스크2", ...], "chosen": {{"tasks": [...], "priority_order": [...], "scores": [...]}}}}

주의: tasks 배열은 원본 순서({n}개)와 동일하게 유지"""

_SYSTEM_V3_BRIEF = """\
응답은 반드시 JSON 형식으로만 하세요. 설명이나 마크다운 블록 없이 순수 JSON 한 줄만 출력."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["v1", "v2", "both"], default="both")
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--slices", type=int, default=20)
    parser.add_argument("--out-prefix", default="outputs/rework_v1v2")
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    today_base = datetime.date(2026, 1, 1)
    rows_to_process: list[dict] = []

    # ── v1 로드 ──────────────────────────────────────────────────────────────
    if args.source in ("v1", "both"):
        v1 = pd.read_parquet("data/scheduler_ko_combined.parquet")
        # 태스크 목록 형식인 것만 (할 일 목록 헤더 포함)
        v1 = v1[v1["prompt"].str.contains("할 일 목록|태스크|일정", na=False)]
        v1 = v1.sample(min(args.max_rows // 2, len(v1)), random_state=args.seed)
        print(f"v1 샘플: {len(v1)}행")
        for _, row in v1.iterrows():
            rows_to_process.append({
                "source_type": "v1",
                "prompt": row["prompt"],
                "persona": str(row.get("persona", "")),
            })

    # ── v2_schedule 미사용분 로드 ─────────────────────────────────────────────
    if args.source in ("v2", "both"):
        v2 = pd.read_parquet("data/scheduler_v2_combined.parquet")
        v4 = pd.read_parquet("data/sft_v4_train.parquet")
        used = set(v4["prompt"].tolist())
        v2_unused = v2[
            v2["prompt"].str.contains("할 일 목록|일정", na=False) &
            ~v2["prompt"].isin(used)
        ]
        # v2_offformat(orca/xlam) 제외: 스케줄 도메인 확인
        v2_unused = v2_unused[
            v2_unused["prompt"].str.contains(r"[-]\s+[가-힣]", na=False, regex=True)
        ]
        v2_unused = v2_unused.sample(min(args.max_rows // 2, len(v2_unused)), random_state=args.seed)
        print(f"v2_unused 샘플: {len(v2_unused)}행")
        for _, row in v2_unused.iterrows():
            rows_to_process.append({
                "source_type": "v2",
                "prompt": row["prompt"],
                "persona": str(row.get("persona", "")),
            })

    print(f"총 처리 대상: {len(rows_to_process)}행")
    rng.shuffle(rows_to_process)
    rows_to_process = rows_to_process[:args.max_rows]

    # ── JSONL 슬라이스 생성 ──────────────────────────────────────────────────
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    slice_size = max(1, len(rows_to_process) // args.slices)

    slices = []
    for s_idx in range(args.slices):
        batch = rows_to_process[s_idx * slice_size:(s_idx + 1) * slice_size]
        if not batch:
            break
        out_path = Path(f"{args.out_prefix}_part{s_idx:02d}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r_idx, row in enumerate(batch):
                tasks = extract_tasks(row["prompt"])
                if len(tasks) < 3:
                    continue  # 태스크 수 부족 — 건너뜀

                today = today_base + datetime.timedelta(days=rng.randint(0, 360))
                local_rng = random.Random(args.seed * 10000 + s_idx * 1000 + r_idx)
                skel = build_rework_skeleton(len(tasks), today, local_rng)

                spec_lines = "\n".join(
                    _spec_to_gen_line(s, skel.today or today.isoformat())
                    for s in skel.specs
                )
                facts = "\n".join(
                    _spec_to_fact(s, skel.today or today.isoformat())
                    for s in skel.specs
                )

                today_display = skel.today if skel.today else "(날짜 미상)"
                persona_short = row["persona"].split("씨")[0] if "씨" in row["persona"] else row["persona"][:20]
                original_tasks_text = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(tasks))

                instruction = _AGENT_INSTRUCTION.format(
                    persona=row["persona"],
                    today_display=today_display,
                    original_tasks=original_tasks_text,
                    spec_lines=spec_lines,
                    facts=facts,
                    persona_short=persona_short,
                    row_id=f"rework_{s_idx:02d}_{r_idx:04d}",
                    n=len(tasks),
                )

                record = {
                    "id": f"rework_{s_idx:02d}_{r_idx:04d}",
                    "source_type": row["source_type"],
                    "persona": row["persona"],
                    "today": skel.today,
                    "scenario": skel.scenario,
                    "n_tasks": len(tasks),
                    "original_tasks": tasks,
                    "meta": json.dumps({
                        "scenario": skel.scenario,
                        "today": skel.today,
                        "specs": [dict(
                            idx=s.idx, kind=s.kind, deadline=s.deadline,
                            is_past=s.is_past, chain_group=s.chain_group,
                            chain_pos=s.chain_pos, risk=s.risk,
                            risk_clause=s.risk_clause, rel_expr=s.rel_expr,
                        ) for s in skel.specs],
                    }),
                    "instruction": instruction,
                    "system": _SYSTEM_V3_BRIEF,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        n_written = sum(1 for _ in open(out_path, encoding="utf-8"))
        print(f"  {out_path}: {n_written}행")
        slices.append(str(out_path))

    # ── 슬라이스 목록 저장 ────────────────────────────────────────────────────
    manifest = Path(f"{args.out_prefix}_manifest.json")
    with open(manifest, "w") as f:
        json.dump({"slices": slices, "total": len(rows_to_process)}, f, indent=2)
    print(f"\n매니페스트: {manifest}")
    print(f"다음 단계: Agent 워크플로로 각 슬라이스 처리 후 assemble_rework_rows.py 실행")


if __name__ == "__main__":
    main()
