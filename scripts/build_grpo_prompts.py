#!/usr/bin/env python
"""GRPO 프롬프트 셋 빌드 — meta(골격) 있는 행에서 prompt/persona/today/meta만 추출.

held-out 실측에서 약한 시나리오(dependency_chain, dated_mixed/past 계열)를 과대표집해
보상 신호가 실제 실패 모드에 집중되도록 한다. 평가셋(scheduler_v3_eval)은 제외.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_WEAK = {"v3_dependency_chain": 3.0, "v3_dated_mixed": 2.0, "v3_past_split": 2.0,
         "v4_claude_past_split": 2.0, "v3_intraday": 1.5}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=[
        "data/scheduler_v3.parquet",
        "data/scheduler_v4_openai.parquet",
        "data/scheduler_v4_claude.parquet",
    ])
    parser.add_argument("--out", default="data/grpo_prompts_v4.parquet")
    parser.add_argument("--total", type=int, default=384)
    parser.add_argument("--seed", type=int, default=46)
    args = parser.parse_args()

    frames = [pd.read_parquet(p) for p in args.inputs if Path(p).exists()]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["meta"].notna()][["prompt", "persona", "today", "source", "meta"]]

    w = df["source"].map(_WEAK).fillna(1.0)
    sampled = df.sample(min(args.total, len(df)), weights=w, random_state=args.seed)
    sampled = sampled.reset_index(drop=True)
    sampled.to_parquet(args.out, index=False)
    print(f"[저장] {args.out} ({len(sampled)}행)")
    print(sampled["source"].value_counts().to_string())
