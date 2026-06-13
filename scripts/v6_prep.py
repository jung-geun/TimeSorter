#!/usr/bin/env python
"""v6 통합 준비 — v2~v5 합치고, v2(미검증)를 opus 검수용 배치로 분할.

- v3/v4/v5(meta 보유): verify_chosen 자동검증 통과분만 직접 포함 (data/v6_audit/sft_curated.parquet)
- v2(meta 없음): 100행 배치 JSONL로 분할 → opus 하위에이전트 검수+수정
- DPO도 동일: v3/v5는 직접 포함, v2는 배치 분할

출력:
  data/v6_audit/sft/batch_NNN.jsonl     (v2 SFT 검수 대상)
  data/v6_audit/dpo/batch_NNN.jsonl     (v2 DPO 검수 대상)
  data/v6_audit/sft_curated.parquet     (v3-v5 검증 통과분, 직접 포함)
  data/v6_audit/dpo_curated.parquet     (v3/v5, 직접 포함)
  data/v6_audit/manifest.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from gen_schedule_v3 import Skeleton, TaskSpec, verify_chosen  # noqa

OUT = Path("data/v6_audit")
(OUT / "sft").mkdir(parents=True, exist_ok=True)
(OUT / "dpo").mkdir(parents=True, exist_ok=True)
BATCH = 100


def _verify_meta(df: pd.DataFrame) -> pd.DataFrame:
    """meta 보유 행 중 verify_chosen 통과분만 반환."""
    keep = []
    for _, r in df.iterrows():
        try:
            m = json.loads(r["meta"])
            skel = Skeleton(scenario=m["scenario"], today=m["today"],
                            specs=[TaskSpec(**s) for s in m["specs"]])
            if not verify_chosen(skel, str(r["chosen"])):
                keep.append(True); continue
        except Exception:
            pass
        keep.append(False)
    return df[pd.Series(keep, index=df.index)]


def _write_batches(df: pd.DataFrame, sub: str, cols: list[str]) -> int:
    n = 0
    for i in range(0, len(df), BATCH):
        chunk = df.iloc[i:i + BATCH]
        rows = []
        for gi, (_, r) in enumerate(chunk.iterrows()):
            rows.append({"idx": i + gi, **{c: ("" if pd.isna(r.get(c)) else str(r.get(c, ""))) for c in cols}})
        path = OUT / sub / f"batch_{n:03d}.jsonl"
        path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows))
        n += 1
    return n


# ── SFT ──────────────────────────────────────────────────────────────────────
SFT_SRC = [("v2", "data/scheduler_v2_combined.parquet"),
           ("v3", "data/scheduler_v3.parquet"),
           ("v4", "data/scheduler_v4_extra.parquet"),
           ("v5", "data/scheduler_v5_claude.parquet")]
sft_parts = []
for v, f in SFT_SRC:
    d = pd.read_parquet(f); d["version"] = v
    for c in ["today", "meta"]:
        if c not in d.columns: d[c] = ""
    sft_parts.append(d[["prompt", "chosen", "persona", "today", "source", "meta", "version"]])
sft = pd.concat(sft_parts, ignore_index=True).drop_duplicates(subset=["prompt"]).reset_index(drop=True)

sft_curated = sft[sft["version"].isin(["v3", "v4", "v5"])].copy()
sft_curated = _verify_meta(sft_curated)
sft_curated.to_parquet(OUT / "sft_curated.parquet", index=False)
sft_v2 = sft[sft["version"] == "v2"].reset_index(drop=True)
n_sft_batches = _write_batches(sft_v2, "sft", ["prompt", "chosen", "persona", "today"])

# ── DPO ──────────────────────────────────────────────────────────────────────
DPO_COLS = ["prompt", "chosen", "rejected", "persona", "today", "category", "source"]
dpo_v2 = pd.read_parquet("data/dpo_pairs_v2.parquet")
dpo_v3 = pd.read_parquet("data/dpo_pairs_v3.parquet")
dpo_v3 = dpo_v3[~dpo_v3["source"].astype(str).str.startswith("v2_replay")]
dpo_v5 = pd.read_parquet("data/dpo_pairs_v5_onpolicy.parquet")
for d in (dpo_v2, dpo_v3, dpo_v5):
    for c in DPO_COLS:
        if c not in d.columns: d[c] = ""

dpo_curated = pd.concat([dpo_v3[DPO_COLS].assign(version="v3"),
                         dpo_v5[DPO_COLS].assign(version="v5")], ignore_index=True)
dpo_curated = dpo_curated.drop_duplicates(subset=["prompt", "chosen", "rejected"]).reset_index(drop=True)
dpo_curated.to_parquet(OUT / "dpo_curated.parquet", index=False)
dpo_v2 = dpo_v2[DPO_COLS].drop_duplicates(subset=["prompt", "chosen", "rejected"]).reset_index(drop=True)
n_dpo_batches = _write_batches(dpo_v2, "dpo", ["prompt", "chosen", "rejected", "persona", "today", "category"])

manifest = {
    "batch_size": BATCH,
    "sft": {"v2_audit_rows": len(sft_v2), "v2_batches": n_sft_batches,
            "curated_keep": len(sft_curated), "total_candidate": len(sft)},
    "dpo": {"v2_audit_rows": len(dpo_v2), "v2_batches": n_dpo_batches,
            "curated_keep": len(dpo_curated)},
}
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print(json.dumps(manifest, ensure_ascii=False, indent=2))
