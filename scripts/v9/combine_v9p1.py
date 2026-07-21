#!/usr/bin/env python
"""v9p1 combined 생성 — v9combined + EN3 augmentation 결합.

assemble_v9.py로 게이트된(verify_chosen_v9 통과) EN3 데이터를 v9combined에 추가.
opus 없이 결정론적 검증만 사용하므로 combine_v9.py의 audit 인자 불필요.

사용:
  uv run python scripts/v9/combine_v9p1.py \
    --base data/scheduler_v9_combined.parquet \
    --new data/scheduler_v9_en4.parquet \
    --out data/scheduler_v9p1_combined.parquet \
    --out-hf data/hf_versioned/sft/v9p1.parquet
"""
from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path

import pandas as pd


def _norm(t: str) -> str:
    return re.sub(r"\s+", "", t).lower()


def titleset(chosen: str) -> frozenset:
    titles = re.findall(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)"', chosen)
    return frozenset(_norm(t) for t in titles)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/scheduler_v9_combined.parquet")
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default="data/scheduler_v9p1_combined.parquet")
    ap.add_argument("--out-hf", default="data/hf_versioned/sft/v9p1.parquet")
    args = ap.parse_args()

    base = pd.read_parquet(args.base).reset_index(drop=True)
    new_df = pd.read_parquet(args.new).reset_index(drop=True)

    print(f"Base: {len(base)} rows")
    print(f"  versions: {dict(collections.Counter(base['version'].tolist()))}")
    print(f"New:  {len(new_df)} rows")
    print(f"  versions: {dict(collections.Counter(new_df['version'].tolist()))}")

    # Dedup against base title sets
    base_sets = {titleset(c) for c in base["chosen"]}
    seen = set(base_sets)
    kept, dup = [], 0
    for _, r in new_df.iterrows():
        ts = titleset(r["chosen"])
        if ts in seen:
            dup += 1
            continue
        seen.add(ts)
        kept.append(r)

    new_clean = pd.DataFrame(kept) if kept else new_df.iloc[0:0]
    print(f"New after dedup: {len(new_clean)} rows (dropped {dup} duplicates)")

    final = pd.concat([base, new_clean], ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_hf).parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(args.out, index=False)
    final.to_parquet(args.out_hf, index=False)

    print(f"\nFinal: {len(final)} rows")
    print(f"  versions: {dict(collections.Counter(final['version'].tolist()))}")
    all_flat = [t for c in final["chosen"]
                for t in re.findall(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)"', c)]
    uniq = len(set(_norm(t) for t in all_flat))
    print(f"  title uniqueness: {uniq}/{len(all_flat)} ({100*uniq/len(all_flat):.1f}%)")
    print(f"[saved] {args.out} · {args.out_hf}")


if __name__ == "__main__":
    main()
