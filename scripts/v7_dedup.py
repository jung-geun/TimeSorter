#!/usr/bin/env python
"""v7 체인 데이터 dedup + train/eval 누출 검사 (advisor 요구).

1) train 내부: 정규화 태스크-셋이 동일한 행 제거(첫 행 유지).
2) train/eval 누출: eval 행의 태스크-셋이 train 행과 과도하게 겹치면(Jaccard≥임계 또는
   정규화 태스크 3개 이상 공유) 해당 eval 행 제거 — 학습 암기로 인한 평가 오염 방지.
3) 전역 고유 태스크-텍스트 비율 보고.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


def norm(t: str) -> str:
    t = re.sub(r"\([^)]*\)", "", t)         # 마감 괄호 제거
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def task_set(prompt: str) -> frozenset[str]:
    lines = [ln[2:] for ln in prompt.splitlines() if ln.startswith("- ")]
    return frozenset(norm(x) for x in lines)


def main() -> None:
    train = pd.read_parquet("data/scheduler_v7_chain.parquet")
    eval_ = pd.read_parquet("data/scheduler_v7_chain_eval.parquet")

    # 1) train 내부 dedup
    seen: set[frozenset[str]] = set()
    keep = []
    for _, r in train.iterrows():
        ts = task_set(r["prompt"])
        if ts in seen:
            continue
        seen.add(ts)
        keep.append(r)
    train2 = pd.DataFrame(keep)
    print(f"train 내부 dedup: {len(train)} → {len(train2)} ({len(train)-len(train2)} 제거)")

    # 2) train/eval 누출 검사
    train_sets = [task_set(p) for p in train2["prompt"]]
    drop_idx = []
    for i, r in eval_.reset_index(drop=True).iterrows():
        es = task_set(r["prompt"])
        leak = False
        for ts in train_sets:
            inter = len(es & ts)
            union = len(es | ts) or 1
            if inter >= 3 or inter / union >= 0.5:
                leak = True
                break
        if leak:
            drop_idx.append(i)
    eval2 = eval_.reset_index(drop=True).drop(index=drop_idx)
    print(f"train/eval 누출 제거: {len(eval_)} → {len(eval2)} ({len(drop_idx)} 제거)")

    # 3) 전역 고유 태스크-텍스트 비율
    alltasks = [norm(ln[2:]) for p in train2["prompt"] for ln in p.splitlines() if ln.startswith("- ")]
    uniq = len(set(alltasks)) / len(alltasks) * 100 if alltasks else 0
    print(f"train 전역 고유 태스크 비율: {uniq:.1f}% ({len(set(alltasks))}/{len(alltasks)})")

    train2.to_parquet("data/scheduler_v7_chain.parquet", index=False)
    eval2.to_parquet("data/scheduler_v7_chain_eval.parquet", index=False)
    print(f"[저장] train {len(train2)}행 · eval {len(eval2)}행")


if __name__ == "__main__":
    main()
