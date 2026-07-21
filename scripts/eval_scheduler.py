#!/usr/bin/env python
"""프로그래매틱 평가 하니스 — 시나리오 골격 규칙으로 모델 출력을 자동 채점.

gpt-5.5 판사 검증은 이메일 5건 시나리오 1개(n=1)라 분산이 크고 비용이 든다.
이 스크립트는 골격(meta)이 있는 held-out 시나리오 N개에 대해 vLLM 서빙 모델을 호출해
verify_chosen()의 규칙 위반(지난 일정 순위·동일일 시각 순서·체인 연속성·리스크 importance·
무마감 1위)을 집계한다. 학습에 쓰지 않은 seed로 생성한 데이터를 입력해야 한다.

사용:
  # held-out 생성 (학습 seed 46과 다른 seed)
  uv run python scripts/gen_schedule_v3.py --total 150 --seed 47 \
      --out data/scheduler_v3_eval.parquet --ckpt outputs/.ckpt_gen_v3_eval.jsonl

  # 평가 (vLLM 서빙 필요: make serve-docker ADAPTER=... LORA_NAME=scheduler)
  uv run python scripts/eval_scheduler.py --input data/scheduler_v3_eval.parquet
  uv run python scripts/eval_scheduler.py --input data/scheduler_v3_eval.parquet --rerank
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from timesorter.data.schema import (
    SCHEDULER_SYSTEM_PROMPT_V3,
    ScheduleResponse,
    parse_or_repair,
    render_system_prompt,
)

from gen_schedule_v3 import Skeleton, TaskSpec, verify_chosen


def _violation_kind(msg: str) -> str:
    if "지난 일정" in msg:
        return "past_rank"
    if "마감(태스크" in msg or "후순위" in msg:
        return "intraday_order"
    if "체인" in msg:
        return "chain"
    if "리스크" in msg:
        return "risk_importance"
    if "마감 없는" in msg:
        return "none_first"
    if "urgency/time_constraint" in msg:
        return "past_score"
    return "parse_or_count"


def main() -> None:
    parser = argparse.ArgumentParser(description="골격 규칙 기반 자동 평가")
    parser.add_argument("--input", default="data/scheduler_v3_eval.parquet")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--model", default="scheduler")
    parser.add_argument("--rerank", action="store_true", help="ScoreRanker 후처리 적용 후 평가")
    parser.add_argument("--rerank-mode", default="full", choices=["full", "guard"],
                        help="full=가중합 전면 재정렬, guard=지난 일정만 최하위 강등")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--out", default=None, help="상세 결과 JSON 저장 경로")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    if args.limit:
        df = df.head(args.limit)
    print(f"[평가] {len(df)}개 시나리오 | rerank={args.rerank} | model={args.model}")

    client = OpenAI(api_key="EMPTY", base_url=f"{args.server_url}/v1")

    per_scenario: dict[str, Counter] = {}
    per_kind = Counter()
    n_clean = 0
    details = []

    for i, row in df.iterrows():
        meta = json.loads(row["meta"])
        skel = Skeleton(
            scenario=meta["scenario"], today=meta["today"],
            specs=[TaskSpec(**s) for s in meta["specs"]],
        )
        system = render_system_prompt(
            SCHEDULER_SYSTEM_PROMPT_V3, str(row["persona"]), today=str(row["today"]))
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": str(row["prompt"])}],
            max_tokens=args.max_tokens, temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = parse_or_repair(raw)
        if args.rerank:
            from timesorter.rank import rerank, rerank_guard
            parsed = rerank_guard(parsed) if args.rerank_mode == "guard" else rerank(parsed)
        out_json = parsed.model_dump_json()

        errors = verify_chosen(skel, out_json)
        sc = per_scenario.setdefault(skel.scenario, Counter())
        sc["n"] += 1
        if errors:
            sc["violated"] += 1
            for e in errors:
                per_kind[_violation_kind(e)] += 1
        else:
            n_clean += 1
        details.append({"idx": int(i), "scenario": skel.scenario, "errors": errors})
        if (len(details)) % 25 == 0:
            print(f"  진행 {len(details)}/{len(df)} (무위반 {n_clean})")

    total = len(df)
    print(f"\n=== 결과: 전 규칙 통과 {n_clean}/{total} ({n_clean/total*100:.1f}%) ===")
    print(f"{'시나리오':22s} {'통과율':>8s}")
    for name, c in sorted(per_scenario.items()):
        ok = c["n"] - c["violated"]
        print(f"{name:22s} {ok}/{c['n']} ({ok/c['n']*100:.0f}%)")
    print("\n위반 유형별 건수:")
    for kind, n in per_kind.most_common():
        print(f"  {kind:18s} {n}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "model": args.model, "rerank": args.rerank, "total": total, "clean": n_clean,
            "per_scenario": {k: dict(v) for k, v in per_scenario.items()},
            "per_kind": dict(per_kind), "details": details,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[저장] {args.out}")


if __name__ == "__main__":
    main()
