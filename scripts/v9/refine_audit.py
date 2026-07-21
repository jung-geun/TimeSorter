#!/usr/bin/env python
"""전수 검수 결과 적용 — 검수 탈락 행 제거 후 정제 SFT 저장.

입력: scheduler_v9.parquet(RangeIndex=idx) + 검수결과 JSON([{batch,model,rows:[{idx,pass,realism,issues}]}]).
게이트: pass=False 또는 realism < --min-realism 인 행 제거. 판정 없는 행은 유지(검증기 기통과).
출력: scheduler_v9.parquet(정제, 덮어쓰기) + hf_versioned/sft/v9.parquet. 원본은 .prefull 백업.

사용: uv run python scripts/v9/refine_audit.py --audit outputs/v9/audit_result.json --min-realism 3
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

SFT = Path("data/scheduler_v9.parquet")
HF = Path("data/hf_versioned/sft/v9.parquet")


def load_verdicts(path: str) -> tuple[dict, Counter]:
    raw = Path(path).read_text()
    arr = json.loads(raw)
    if isinstance(arr, str):
        arr = json.loads(arr)
    verdicts, bymodel = {}, Counter()
    for batch in arr:
        if not batch.get("rows"):
            continue
        bymodel[batch.get("model", "?")] += 1
        for v in batch["rows"]:
            verdicts[int(v["idx"])] = v
    return verdicts, bymodel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--min-realism", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_parquet(SFT).reset_index(drop=True)
    verdicts, bymodel = load_verdicts(args.audit)

    keep, drop_fail, drop_real, no_verdict = [], 0, 0, 0
    drop_reasons = []
    for idx in range(len(df)):
        v = verdicts.get(idx)
        if v is None:
            no_verdict += 1
            keep.append(idx)
            continue
        if not v.get("pass", True):
            drop_fail += 1
            drop_reasons.append((idx, v.get("issues", [])[:1]))
            continue
        if v.get("realism", 5) < args.min_realism:
            drop_real += 1
            continue
        keep.append(idx)

    refined = df.iloc[keep].reset_index(drop=True)
    if not Path(str(SFT) + ".prefull").exists():
        shutil.copy(SFT, str(SFT) + ".prefull")
    refined.to_parquet(SFT, index=False)
    refined.to_parquet(HF, index=False)

    print(f"검수 판정: {len(verdicts)}/{len(df)}행 · 모델배치 {dict(bymodel)}")
    print(f"정제: {len(df)} → {len(refined)}행 (제거 pass=False {drop_fail} · "
          f"realism<{args.min_realism} {drop_real} · 무판정유지 {no_verdict})")
    avg_real = sum(v.get("realism", 0) for v in verdicts.values()) / max(1, len(verdicts))
    print(f"평균 realism: {avg_real:.2f}")
    if drop_reasons:
        print("탈락 예시:", drop_reasons[:5])
    print(f"[saved] {SFT} · {HF}")


if __name__ == "__main__":
    main()
