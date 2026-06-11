#!/usr/bin/env python
"""HF 릴리스 통합 빌드 — v1~v4 전체 데이터를 용도별(SFT/DPO/GRPO/eval)로 통합.

출력 (data/hf_release/):
  sft_train.parquet      v2+v3+v4 JSON 스키마 통합 (중복 prompt 제거, version 컬럼)
  sft_v1_text.parquet    v1 자유 텍스트 합본 (포맷이 달라 별도 config)
  dpo_train.parquet      v1~v4 DPO 쌍 통합 (중복 쌍 제거)
  grpo_train.parquet     골격(meta) 보유 행 전체 — GRPO 검증 가능 보상용
  eval_heldout.parquet   held-out 평가셋 (seed 47, 학습 미사용)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path("data/hf_release")
OUT.mkdir(parents=True, exist_ok=True)


def norm(df: pd.DataFrame, version: str, cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            d[c] = ""
    d["version"] = version
    return d[cols + ["version"]]


# ── SFT (JSON 스키마 v2~v4) ───────────────────────────────────────────────────
SFT_COLS = ["prompt", "chosen", "persona", "today", "source", "meta"]
sft = pd.concat([
    norm(pd.read_parquet("data/scheduler_v2_combined.parquet"), "v2", SFT_COLS),
    norm(pd.read_parquet("data/scheduler_v3.parquet"), "v3", SFT_COLS),
    norm(pd.read_parquet("data/scheduler_v4_extra.parquet"), "v4", SFT_COLS),
], ignore_index=True)
before = len(sft)
sft = sft.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
sft.to_parquet(OUT / "sft_train.parquet", index=False)
print(f"sft_train: {len(sft):,} (중복 제거 {before - len(sft)})")
print(sft["version"].value_counts().to_string(), "\n")

# ── SFT v1 (자유 텍스트 — 별도 포맷) ─────────────────────────────────────────
v1 = pd.read_parquet("data/scheduler_ko_combined.parquet")
v1 = norm(v1, "v1", ["prompt", "chosen", "persona", "source"])
v1.to_parquet(OUT / "sft_v1_text.parquet", index=False)
print(f"sft_v1_text: {len(v1):,}\n")

# ── DPO (v1~v4 쌍 통합) ──────────────────────────────────────────────────────
DPO_COLS = ["prompt", "chosen", "rejected", "persona", "today", "category", "source"]
d1 = pd.read_parquet("data/dpo_pairs.parquet").drop(columns=["pair"], errors="ignore")
d1["category"], d1["source"] = "v1_pair", "dpo_v1"
dpo = pd.concat([
    norm(d1, "v1", DPO_COLS),
    norm(pd.read_parquet("data/dpo_pairs_v2.parquet"), "v2", DPO_COLS),
    norm(pd.read_parquet("data/dpo_pairs_v3.parquet"), "v3", DPO_COLS),
    norm(pd.read_parquet("data/dpo_pairs_v4_extra.parquet"), "v4", DPO_COLS),
], ignore_index=True)
before = len(dpo)
dpo = dpo.drop_duplicates(subset=["prompt", "chosen", "rejected"]).reset_index(drop=True)


def _tier(row) -> str:
    """학습 권장 등급 — docs/DATASET_AUDIT.md 참고."""
    if row["version"] == "v1":
        return "legacy_text"          # 자유 텍스트 chosen — v3+ 학습 사용 금지
    src = str(row["source"])
    if src.startswith("refusal") or "refusal" in str(row["category"]):
        return "refusal"              # 거부 능력 유지 — 항상 소량 혼합
    if row["version"] in ("v3", "v4") and str(row["source"]).startswith(("v3_", "v4_")):
        return "hard"                 # 내용 오류 negative — 본 학습 권장
    return "easy_format"              # v2 형식·단순 negative — 길이 편향 주의


dpo["tier"] = dpo.apply(_tier, axis=1)
dpo.to_parquet(OUT / "dpo_train.parquet", index=False)
print(dpo["tier"].value_counts().to_string())
print(f"dpo_train: {len(dpo):,} (중복 제거 {before - len(dpo)})")
print(dpo["version"].value_counts().to_string(), "\n")

# ── GRPO (골격 meta 보유 — 검증 가능 보상) ───────────────────────────────────
GRPO_COLS = ["prompt", "persona", "today", "source", "meta"]
grpo = pd.concat([
    norm(pd.read_parquet("data/scheduler_v3.parquet"), "v3", GRPO_COLS),
    norm(pd.read_parquet("data/scheduler_v4_extra.parquet"), "v4", GRPO_COLS),
], ignore_index=True)
grpo = grpo[grpo["meta"].astype(str).str.len() > 2].drop_duplicates(subset=["prompt"])
grpo = grpo.reset_index(drop=True)
grpo.to_parquet(OUT / "grpo_train.parquet", index=False)
print(f"grpo_train: {len(grpo):,}\n")

# ── eval ─────────────────────────────────────────────────────────────────────
ev = pd.read_parquet("data/scheduler_v3_eval.parquet")
ev["version"] = "v3_eval"
ev.to_parquet(OUT / "eval_heldout.parquet", index=False)
print(f"eval_heldout: {len(ev):,}")
