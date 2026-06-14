#!/usr/bin/env python
"""Qwen3.5-4B 4-way 벤치마크 (base/no-sft/SFT/DPO) — benchmark_9b.run() 재사용.

9B와 동일 방식(schema-strict + content-level). 기본 n=150 (9B와 동일 표본).

사용:
  uv run python scripts/benchmark_q35_4way.py --target all --limit 150
"""
from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_9b import run  # 동일 채점 로직 재사용

TARGETS = {
    "base_no_rlhf": {"label": "Qwen3.5-4B-Base (no FT)",
                     "base_model": "Qwen/Qwen3.5-4B-Base", "adapter": None},
    "instruct_base": {"label": "Qwen3.5-4B (no adapter)",
                      "base_model": "Qwen/Qwen3.5-4B", "adapter": None},
    "sft_q35": {"label": "SFT v4 (Qwen3.5-4B)",
                "base_model": "Qwen/Qwen3.5-4B", "adapter": "outputs/sft_q35_4b_v4"},
    "dpo_q35": {"label": "DPO v5 (Qwen3.5-4B)",
                "base_model": "Qwen/Qwen3.5-4B", "adapter": "outputs/dpo_q35_4b_v5"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="all")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--eval-file", default="data/scheduler_v3_eval.parquet")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    keys = list(TARGETS) if args.target == "all" else [k.strip() for k in args.target.split(",")]
    for k in keys:
        cfg = TARGETS[k]
        if cfg["adapter"] and not Path(cfg["adapter"]).exists():
            print(f"[skip] {k}: adapter 없음"); continue
        out = f"{args.out_dir}/eval_q35_4way_{k}_n{args.limit}.json"
        if Path(out).exists() and not args.force:
            print(f"[skip] {out} 존재 (--force)"); continue
        print(f"\n=== {cfg['label']} ===")
        run(cfg, args.eval_file, args.limit, out)


if __name__ == "__main__":
    main()
