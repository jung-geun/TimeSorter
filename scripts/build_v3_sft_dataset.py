#!/usr/bin/env python
"""v3 SFT 통합 데이터셋 빌드.

구성:
  - data/scheduler_v3.parquet     : v3 신규 (dated/intraday/chain/risk/relative, today 포함)
  - data/scheduler_v2_combined.parquet 에서 replay 샘플:
      * refusal_v2/* 전부 (거부 동작 유지)
      * 나머지에서 랜덤 N개 (일반 분포 유지)
    replay 행은 절대 날짜가 없으므로 임의의 today를 부여해도 정합성이 깨지지 않는다.

출력: data/scheduler_v3_combined.parquet (prompt, chosen, persona, today, source)
"""
from __future__ import annotations

import argparse
import datetime
import random

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3", default="data/scheduler_v3.parquet")
    parser.add_argument("--v2", default="data/scheduler_v2_combined.parquet")
    parser.add_argument("--replay-n", type=int, default=2500, help="refusal 외 v2 replay 수")
    parser.add_argument("--out", default="data/scheduler_v3_combined.parquet")
    parser.add_argument("--seed", type=int, default=46)
    args = parser.parse_args()

    v3 = pd.read_parquet(args.v3)[["prompt", "chosen", "persona", "today", "source"]]
    print(f"[v3 신규] {len(v3)}행")

    v2 = pd.read_parquet(args.v2)
    refusal = v2[v2["source"].astype(str).str.startswith("refusal", na=False)]
    rest = v2.drop(refusal.index)
    replay = pd.concat([
        refusal,
        rest.sample(min(args.replay_n, len(rest)), random_state=args.seed),
    ]).copy()

    rng = random.Random(args.seed)
    replay["today"] = [
        (datetime.date(2026, 1, 1) + datetime.timedelta(days=rng.randint(0, 360))).isoformat()
        for _ in range(len(replay))
    ]
    replay = replay[["prompt", "chosen", "persona", "today", "source"]]
    print(f"[v2 replay] {len(replay)}행 (refusal {len(refusal)} + 일반 {len(replay) - len(refusal)})")

    out = pd.concat([v3, replay]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    out.to_parquet(args.out, index=False)
    print(f"[저장] {args.out} ({len(out)}행)")
    print(out["source"].value_counts().head(12).to_string())
