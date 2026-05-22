#!/usr/bin/env python
"""여러 parquet 데이터셋을 하나의 scheduler_ko_combined.parquet으로 병합.

사용:
  uv run python scripts/merge_datasets.py
  uv run python scripts/merge_datasets.py --out data/scheduler_ko_combined.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SOURCES = [
    "data/scheduler_ko.parquet",           # Nemotron round 1 (500)
    "data/scheduler_nemotron_r2.parquet",  # Nemotron round 2 (500)
    "data/scheduler_nemotron_r3.parquet",  # Nemotron round 3 (500)
    "data/scheduler_nemotron_r4.parquet",  # Nemotron round 4 (500)
    "data/scheduler_generic.parquet",      # Generic 8 personas (최대 4000)
]

REQUIRED_COLS = {"prompt", "chosen"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/scheduler_ko_combined.parquet")
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    for path in SOURCES:
        p = Path(path)
        if not p.exists():
            print(f"[스킵] {path} — 없음")
            continue
        df = pd.read_parquet(path)
        if not REQUIRED_COLS.issubset(df.columns):
            print(f"[스킵] {path} — 필수 컬럼 누락 ({list(df.columns)})")
            continue
        # prompt를 문자열로 통일
        df["prompt"] = df["prompt"].astype(str)
        df["chosen"] = df["chosen"].astype(str)
        frames.append(df[["prompt", "chosen", "persona", "source"]] if "source" in df.columns
                      else df[["prompt", "chosen", "persona"]])
        print(f"[로드] {path}: {len(df)}개")

    if not frames:
        print("[에러] 로드된 데이터셋 없음")
        return

    combined = pd.concat(frames, ignore_index=True)

    # 중복 제거 (동일 prompt+chosen 쌍)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["prompt", "chosen"])
    after = len(combined)
    if before != after:
        print(f"[중복제거] {before - after}개 제거")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(str(out_path), index=False)
    print(f"\n[완료] 총 {len(combined)}개 → {out_path}")
    print("페르소나 분포 (상위 10):")
    if "persona" in combined.columns:
        print(combined["persona"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
