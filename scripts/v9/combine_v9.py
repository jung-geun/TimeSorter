#!/usr/bin/env python
"""정제본 + 검수 통과 보충분 결합 → 최종 v9 SFT.

- base: 1차 검수 통과분(scheduler_v9.parquet, 정제됨)
- topup: 보충 생성분(검수 결과 적용해 통과·realism>=min 만 채택)
- topup을 base와 제목집합 dedup 후 concat.

사용:
  uv run python scripts/v9/combine_v9.py --base data/scheduler_v9.parquet \\
    --topup data/scheduler_v9_topup.parquet --audit outputs/v9/audit2_result.json \\
    --min-realism 3 --out data/scheduler_v9.parquet --out-hf data/hf_versioned/sft/v9.parquet
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def titleset(chosen: str) -> frozenset:
    titles = re.findall(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)"', chosen)
    return frozenset(re.sub(r"\s+", "", t) for t in titles)


def load_verdicts(path: str) -> dict:
    arr = json.loads(Path(path).read_text())
    if isinstance(arr, str):
        arr = json.loads(arr)
    v = {}
    for b in arr:
        for r in (b.get("rows") or []):
            v[int(r["idx"])] = r
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-audit", help="base에도 검수 적용(없으면 base 그대로 유지)")
    ap.add_argument("--topup", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--min-realism", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-hf", required=True)
    args = ap.parse_args()

    base = pd.read_parquet(args.base).reset_index(drop=True)
    if args.base_audit:  # base(.prefull 등)에 검수 적용해 정제
        bverds = load_verdicts(args.base_audit)
        bkeep = [i for i in range(len(base))
                 if (v := bverds.get(i)) is None
                 or (v.get("pass", True) and v.get("realism", 5) >= args.min_realism)]
        base = base.iloc[bkeep].reset_index(drop=True)
        print(f"base 검수 적용: → {len(base)}행")
    topup = pd.read_parquet(args.topup).reset_index(drop=True)
    verds = load_verdicts(args.audit)

    # topup 검수 적용
    keep_idx, drop_fail, drop_real, no_v = [], 0, 0, 0
    for i in range(len(topup)):
        v = verds.get(i)
        if v is None:
            no_v += 1
            keep_idx.append(i)
        elif not v.get("pass", True):
            drop_fail += 1
        elif v.get("realism", 5) < args.min_realism:
            drop_real += 1
        else:
            keep_idx.append(i)
    topup_clean = topup.iloc[keep_idx].reset_index(drop=True)

    # base 제목집합과 dedup
    base_sets = {titleset(c) for c in base["chosen"]}
    seen = set(base_sets)
    rows, dup = [], 0
    for _, r in topup_clean.iterrows():
        ts = titleset(r["chosen"])
        if ts in seen:
            dup += 1
            continue
        seen.add(ts)
        rows.append(r)
    topup_dedup = pd.DataFrame(rows) if rows else topup_clean.iloc[0:0]

    final = pd.concat([base, topup_dedup], ignore_index=True)
    final.to_parquet(args.out, index=False)
    final.to_parquet(args.out_hf, index=False)

    import collections
    print(f"base {len(base)} + topup검수통과 {len(topup_clean)}"
          f"(탈락 fail {drop_fail}·realism {drop_real}·무판정유지 {no_v})"
          f" - base중복 {dup} = 최종 {len(final)}행")
    print("직업군 분포:", dict(collections.Counter(final["source"])))
    # 최종 제목 고유율
    allt = [t for c in final["chosen"] for t in titleset(c)]
    # titleset already normalized per-row; recompute flat
    flat = []
    for c in final["chosen"]:
        flat += [re.sub(r"\s+", "", t) for t in re.findall(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)"', c)]
    print(f"제목 고유율: {len(set(flat))}/{len(flat)} ({100*len(set(flat))/len(flat):.1f}%)")
    print(f"[saved] {args.out} · {args.out_hf}")


if __name__ == "__main__":
    main()
