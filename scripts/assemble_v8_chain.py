#!/usr/bin/env python
"""v8 체인 데이터 최종 조립 — 감사 통과 행 수집 + dedup + 기존 v7과 병합.

사용:
  # 생성된 행 수집 + dedup + 저장
  uv run python scripts/assemble_v8_chain.py \
      --gen-dir outputs/v8_chain/gen \
      --pass-ids outputs/v8_chain/pass_ids \
      --out-sft data/scheduler_v8_chain.parquet \
      --out-sft-full data/hf_versioned/sft/v8_selfcontained.parquet

  # DPO 생성도 연이어 실행 (--run-dpo)
  uv run python scripts/assemble_v8_chain.py ... --run-dpo
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_V7_CHAIN = "data/scheduler_v7_chain.parquet"
_V7_SELFCONTAINED = "data/hf_versioned/sft/v7_selfcontained.parquet"


def _norm(t: str) -> str:
    t = re.sub(r"\([^)]*\)", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _task_set(prompt: str) -> frozenset[str]:
    return frozenset(_norm(ln[2:]) for ln in prompt.splitlines() if ln.startswith("- "))


def load_gen_rows(gen_dir: str, pass_ids_dir: str) -> list[dict]:
    """감사 통과 ID만 gen JSONL에서 로드."""
    # pass IDs 수집
    passed: set[str] = set()
    for txt in sorted(glob.glob(f"{pass_ids_dir}/*.txt")):
        content = Path(txt).read_text(encoding="utf-8").strip()
        if content:
            passed.update(content.splitlines())
    print(f"[pass_ids] 통과 ID {len(passed)}개")

    # gen JSONL에서 해당 ID 수집
    rows: list[dict] = []
    for jsonl in sorted(glob.glob(f"{gen_dir}/*.jsonl")):
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("id") in passed:
                    rows.append(obj)
    print(f"[gen rows] 수집 {len(rows)}행 (통과 {len(passed)}개 중 매칭)")
    return rows


def dedup_against_existing(new_rows: list[dict], existing_df: pd.DataFrame) -> list[dict]:
    """기존 학습 데이터와 태스크-셋 중복 제거."""
    existing_sets = {_task_set(p) for p in existing_df["prompt"]}
    kept, dropped = [], 0
    seen: set[frozenset] = set()
    for r in new_rows:
        ts = _task_set(r.get("prompt", ""))
        if ts in existing_sets or ts in seen:
            dropped += 1
        else:
            kept.append(r)
            seen.add(ts)
    print(f"[dedup] {len(new_rows)}행 → {len(kept)}행 ({dropped}행 제거)")
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default="outputs/v8_chain/gen")
    ap.add_argument("--pass-ids", default="outputs/v8_chain/pass_ids")
    ap.add_argument("--out-sft", default="data/scheduler_v8_chain.parquet")
    ap.add_argument("--out-sft-full", default="data/hf_versioned/sft/v8_selfcontained.parquet")
    ap.add_argument("--run-dpo", action="store_true", help="조립 후 DPO 쌍 자동 생성")
    ap.add_argument("--dpo-out", default="data/dpo_pairs_v8_chain.parquet")
    args = ap.parse_args()

    # 1. 생성 행 로드
    rows = load_gen_rows(args.gen_dir, args.pass_ids)
    if not rows:
        print("[경고] 통과 행이 없습니다. 감사 파이프라인 완료 여부를 확인하세요.")
        sys.exit(1)

    # 2. 기존 v7 체인 데이터 대비 dedup
    v7_chain = pd.read_parquet(_V7_CHAIN) if Path(_V7_CHAIN).exists() else pd.DataFrame(columns=["prompt"])
    rows = dedup_against_existing(rows, v7_chain)

    # 3. 신규 v8 체인 parquet 저장
    req_cols = ["prompt", "chosen", "persona", "today", "source", "meta"]

    def _serialize(r: dict) -> dict:
        out = {c: r.get(c, "") for c in req_cols}
        for col in ("chosen", "meta"):
            if isinstance(out.get(col), (dict, list)):
                out[col] = json.dumps(out[col], ensure_ascii=False)
        return out

    new_df = pd.DataFrame([_serialize(r) for r in rows])
    new_df["version"] = "v8"
    Path(args.out_sft).parent.mkdir(parents=True, exist_ok=True)
    new_df.to_parquet(args.out_sft, index=False)
    print(f"[SFT 증분] {len(new_df)}행 → {args.out_sft}")

    # 4. v8 자립형 = v7_selfcontained + 신규 v8 체인
    if Path(_V7_SELFCONTAINED).exists():
        v7_full = pd.read_parquet(_V7_SELFCONTAINED)
        # v8 행에는 version 컬럼 맞춤
        merged_cols = [c for c in v7_full.columns if c in new_df.columns]
        v8_full = pd.concat([v7_full, new_df[merged_cols]], ignore_index=True)
        Path(args.out_sft_full).parent.mkdir(parents=True, exist_ok=True)
        v8_full.to_parquet(args.out_sft_full, index=False)
        print(f"[SFT 자립형] {len(v7_full)} + {len(new_df)} = {len(v8_full)}행 → {args.out_sft_full}")

    # 5. 전역 고유 태스크 비율
    all_prompts = list(v7_chain["prompt"]) + [r["prompt"] for r in rows]
    all_tasks = [_norm(ln[2:]) for p in all_prompts for ln in p.splitlines() if ln.startswith("- ")]
    uniq_ratio = len(set(all_tasks)) / len(all_tasks) * 100 if all_tasks else 0
    print(f"[품질] 전역 고유 태스크 비율: {uniq_ratio:.1f}% ({len(set(all_tasks))}/{len(all_tasks)})")

    # 6. DPO 쌍 자동 생성 (선택)
    if args.run_dpo:
        print(f"\n[DPO] {args.out_sft} → {args.dpo_out} 생성 중...")
        result = subprocess.run(
            [sys.executable, "scripts/gen_v8_dpo_chain.py",
             "--sft", args.out_sft, "--out", args.dpo_out],
            capture_output=False,
        )
        if result.returncode != 0:
            print("[DPO] 생성 실패. 수동 실행:")
            print(f"  uv run python scripts/gen_v8_dpo_chain.py --sft {args.out_sft} --out {args.dpo_out}")


if __name__ == "__main__":
    main()
