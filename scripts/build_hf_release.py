#!/usr/bin/env python
"""HF 릴리스 통합 빌드 — v1~v5+ 전체 데이터를 용도별(SFT/DPO/GRPO/eval)로 통합.

출력 (data/hf_release/):
  sft_train.parquet      v2+v3+v4+v5+rework JSON 스키마 통합 (중복 prompt 제거, version 컬럼)
  sft_v1_text.parquet    v1 자유 텍스트 합본 (포맷이 달라 별도 config)
  dpo_train.parquet      v1~v5 DPO 쌍 통합 (중복 쌍 제거)
  grpo_train.parquet     골격(meta) 보유 행 전체 — GRPO 검증 가능 보상용
  eval_heldout.parquet   held-out 평가셋 (seed 47, 학습 미사용)

새 데이터셋 추가 시 이 파일만 수정 후 재실행:
  uv run python scripts/build_hf_release.py
  uv run python scripts/upload_hf_dataset.py
"""
from __future__ import annotations

import os
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


def load_if_exists(path: str, version: str, cols: list[str]) -> pd.DataFrame | None:
    """파일이 있을 때만 로드 (재가공 중인 데이터셋용)."""
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"  + {path}: {len(df):,}행")
        return norm(df, version, cols)
    return None


# ── SFT (JSON 스키마 v2~v5+rework) ───────────────────────────────────────────
SFT_COLS = ["prompt", "chosen", "persona", "today", "source", "meta"]
_sft_parts = [
    norm(pd.read_parquet("data/scheduler_v2_combined.parquet"), "v2", SFT_COLS),
    norm(pd.read_parquet("data/scheduler_v3.parquet"), "v3", SFT_COLS),
    norm(pd.read_parquet("data/scheduler_v4_extra.parquet"), "v4", SFT_COLS),
    norm(pd.read_parquet("data/scheduler_v5_claude.parquet"), "v5", SFT_COLS),
]
# v1/v2 재가공 데이터 — assemble_rework_rows.py 완료 시 자동 포함
_rework = load_if_exists("data/scheduler_rework_v1v2.parquet", "v6_rework", SFT_COLS)
if _rework is not None:
    _sft_parts.append(_rework)
sft = pd.concat(_sft_parts, ignore_index=True)
before = len(sft)
sft = sft.drop_duplicates(subset=["prompt"]).reset_index(drop=True)

# 학습 권장 등급 (opus 4.8 의미 감사 결과 — docs/DATASET_AUDIT.md)
_v2_src = pd.read_parquet("data/scheduler_v2_combined.parquet")
_v2_src_map = dict(zip(_v2_src["prompt"], _v2_src["source"]))


def _sft_tier(row) -> str:
    if row["version"] in ("v3", "v4", "v5"):
        return "curated"            # 골격 검증 + persona_fit 4.9-5.0 — 본 학습 권장
    if row["version"] == "v6_rework":
        return "rework"             # v1/v2 재가공 — persona_fit 3.55, 변별력 필터 통과
    src = str(_v2_src_map.get(row["prompt"], ""))
    if src.startswith("refusal"):
        return "v2_refusal"         # 거부 학습 — 항상 혼합 권장
    if "할 일 목록" in row["prompt"] or "일정" in row["prompt"]:
        return "v2_schedule"        # 스케줄 형식이나 persona_fit 3.3 — 소량 혼합만
    return "v2_offformat"           # orca/xlam 등 비스케줄 — 학습 비권장


sft["tier"] = sft.apply(_sft_tier, axis=1)
sft.to_parquet(OUT / "sft_train.parquet", index=False)
print(sft["tier"].value_counts().to_string())
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
    # v5 on-policy: sft_v4 모델이 실제로 위반한 출력을 rejected로 수집한 쌍
    norm(pd.read_parquet("data/dpo_pairs_v5_onpolicy.parquet"), "v5", DPO_COLS),
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
    if str(row["category"]) == "onpolicy":
        return "hard"                 # on-policy — 모델 실제 오류, 최우선 권장
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
    norm(pd.read_parquet("data/scheduler_v5_claude.parquet"), "v5", GRPO_COLS),
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
