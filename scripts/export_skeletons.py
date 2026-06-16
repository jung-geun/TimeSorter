#!/usr/bin/env python
"""Claude Code 하위 에이전트용 시나리오 골격 export.

gen_schedule_v3.build_skeleton으로 골격을 만들고, 에이전트가 태스크 텍스트와 chosen JSON을
채울 수 있도록 행별 지시문(골격 라인·채점 참고 사실·시스템 프롬프트)을 JSONL 슬라이스로 저장.
에이전트 출력은 assemble_claude_rows.py 가 골격 규칙으로 검증·병합한다.

사용:
  uv run python scripts/export_skeletons.py \
      --scenarios "past_split:100,no_today:100,am_escalation:100,dated_mixed:100" \
      --seed 53 --slices 8 --out-prefix outputs/skeletons_v4
"""
from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from timesorter.data.schema import SCHEDULER_SYSTEM_PROMPT_V3, render_system_prompt

from gen_schedule_v3 import (
    _NEMOTRON_PATH,
    _WEEKDAYS_KO,
    _spec_to_fact,
    _spec_to_gen_line,
    build_skeleton,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", required=True, help='"name:count,..."')
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--slices", type=int, default=8)
    parser.add_argument("--out-prefix", default="outputs/skeletons_v4")
    parser.add_argument("--id-prefix", default="v4c", help="행 id 접두 (예: v7chain)")
    args = parser.parse_args()

    plan: list[str] = []
    for part in args.scenarios.split(","):
        name, _, cnt = part.strip().partition(":")
        plan.extend([name] * int(cnt))
    rng = random.Random(args.seed)
    rng.shuffle(plan)

    nem = pd.read_parquet(_NEMOTRON_PATH, columns=["persona", "age", "occupation"])
    nem = nem.sample(len(plan), random_state=args.seed).reset_index(drop=True)

    rows = []
    for i, scenario in enumerate(plan):
        local = random.Random(args.seed * 100_000 + i)
        today = datetime.date(2026, 1, 1) + datetime.timedelta(days=local.randint(0, 360))
        skel = build_skeleton(scenario, today, local)

        p = nem.iloc[i]
        praw = str(p.get("persona", ""))
        pname = praw.split(" 씨는")[0].strip() if " 씨는" in praw else f"사용자{i}"
        plabel = f"{pname} ({p.get('occupation', '직장인')}, {p.get('age', 30)}세)"

        today_h = (f"{skel.today} ({_WEEKDAYS_KO[today.weekday()]})" if skel.today
                   else "날짜 미상 (오늘 날짜를 알 수 없음)")
        facts = "\n".join(_spec_to_fact(s, skel.today) for s in skel.specs)
        if not skel.today:
            facts = ("- 오늘 날짜 미상: 어떤 일정도 '이미 지났다'고 단정하지 말 것. "
                     "절대 날짜 간 상대 비교만 수행.\n" + facts)

        rows.append({
            "id": f"{args.id_prefix}-{args.seed}-{i}",
            "scenario": scenario,
            "today": skel.today,
            "today_display": today_h,
            "persona_name": pname,
            "persona_label": plabel,
            "system_prompt": render_system_prompt(
                SCHEDULER_SYSTEM_PROMPT_V3, plabel, today=skel.today),
            "gen_lines": [_spec_to_gen_line(s, skel.today) for s in skel.specs],
            "facts": facts,
            "meta": json.dumps(asdict(skel), ensure_ascii=False),
        })

    per = (len(rows) + args.slices - 1) // args.slices
    for si in range(args.slices):
        chunk = rows[si * per:(si + 1) * per]
        if not chunk:
            break
        path = Path(f"{args.out_prefix}_part{si + 1}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in chunk:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{path}: {len(chunk)}행")
