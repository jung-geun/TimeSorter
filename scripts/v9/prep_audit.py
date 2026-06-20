#!/usr/bin/env python
"""전수 검수용 배치 준비 — scheduler_v9.parquet → outputs/v9/audit/batch_NNN.json.

각 행: {idx(parquet 행번호), occ, chain_pairs, prompt(입력JSON), chosen(출력JSON)}.
검수 에이전트(sonnet/opus 혼합)가 이 파일을 읽어 행별 pass/realism/issues 판정.
스캐폴드와 분리된 디렉토리라 에이전트 부산물 파일이 생겨도 본 파이프라인 무해.

사용: uv run python scripts/v9/prep_audit.py --batch-size 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--sft", default="data/scheduler_v9.parquet")
    ap.add_argument("--out-dir", default="outputs/v9/audit")
    args = ap.parse_args()
    SFT = Path(args.sft)
    OUT = Path(args.out_dir)
    df = pd.read_parquet(SFT)
    OUT.mkdir(parents=True, exist_ok=True)
    # 기존 배치 파일 정리(재실행 대비)
    for f in OUT.glob("batch_*.json"):
        f.unlink()
    bs = args.batch_size
    n_batch = (len(df) + bs - 1) // bs
    for b in range(n_batch):
        chunk = df.iloc[b * bs:(b + 1) * bs]
        rows = []
        for idx, r in chunk.iterrows():
            meta = json.loads(r["meta"])
            rows.append({
                "idx": int(idx), "occ": r["source"],
                "chain_pairs": meta.get("chain_pairs", []),
                "prompt": r["prompt"], "chosen": r["chosen"],
            })
        (OUT / f"batch_{b:03d}.json").write_text(json.dumps(rows, ensure_ascii=False))
    (OUT / "manifest.json").write_text(json.dumps(
        {"n_rows": len(df), "n_batch": n_batch, "batch_size": bs}, ensure_ascii=False))
    print(f"[saved] {OUT}/ — {len(df)}행 / {n_batch}배치 (배치당 {bs})")


if __name__ == "__main__":
    main()
