#!/usr/bin/env python
"""v6 조립 — 검수된 v2 + 큐레이션 v3-v5를 단일 자립형 v6 데이터셋으로 통합.

선행: scripts/v6_prep.py + v6 검수 workflow (data/v6_audit/{sft,dpo}/fixed_*.jsonl)

조립:
  v6 SFT = sft_curated(v3-v5 검증통과) + 검수·수정된 v2 SFT (drop 제외)
  v6 DPO = dpo_curated(v3/v5) + 검수·수정된 v2 DPO (drop 제외)
출력:
  data/hf_versioned/sft/v6.parquet
  data/hf_versioned/dpo/v6.parquet
  data/v6_audit/audit_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from timesorter.data.schema import parse_lenient  # noqa


def _to_str(v) -> str:
    """에이전트가 중첩 객체(dict/list)로 기록한 경우 JSON 문자열로 정규화."""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

AUD = Path("data/v6_audit")
OUTSFT = Path("data/hf_versioned/sft/v6.parquet")
OUTDPO = Path("data/hf_versioned/dpo/v6.parquet")


def _rebuild_sft_v2() -> pd.DataFrame:
    d = pd.read_parquet("data/scheduler_v2_combined.parquet")
    for c in ["today", "meta"]:
        if c not in d.columns: d[c] = ""
    d = d[["prompt", "chosen", "persona", "today", "source", "meta"]].copy()
    d["version"] = "v2"
    # v6_prep: 전체 concat 후 prompt dedup → v2만 추출. v2가 맨 앞이라 dedup이 v2 내부만 영향.
    full = d  # v3-v5는 prompt가 달라 v2 행 보존에 영향 없음(검증됨)
    return d.drop_duplicates(subset=["prompt"]).reset_index(drop=True)


def _rebuild_dpo_v2() -> pd.DataFrame:
    DPO_COLS = ["prompt", "chosen", "rejected", "persona", "today", "category", "source"]
    d = pd.read_parquet("data/dpo_pairs_v2.parquet")
    for c in DPO_COLS:
        if c not in d.columns: d[c] = ""
    return d[DPO_COLS].drop_duplicates(subset=["prompt", "chosen", "rejected"]).reset_index(drop=True)


def _load_fixed(sub: str) -> dict[int, dict]:
    out = {}
    for f in sorted((AUD / sub).glob("fixed_*.jsonl")):
        for line in f.read_text().strip().split("\n"):
            if not line.strip():
                continue
            o = json.loads(line)
            out[int(o["idx"])] = o
    return out


def assemble_sft() -> dict:
    base = _rebuild_sft_v2()
    fixed = _load_fixed("sft")
    rep = {"v2_rows": len(base), "fixed_found": len(fixed),
           "keep": 0, "fix": 0, "drop": 0, "missing": 0, "invalid_after": 0}
    rows = []
    for idx, r in base.iterrows():
        fx = fixed.get(idx)
        if fx is None:
            rep["missing"] += 1
            rows.append(r.to_dict()); continue
        act = fx["action"]; rep[act] = rep.get(act, 0) + 1
        if act == "drop":
            continue
        chosen = _to_str(fx["chosen"]) if act == "fix" else str(r["chosen"])
        if parse_lenient(chosen) is None:
            rep["invalid_after"] += 1
            continue
        nr = r.to_dict(); nr["chosen"] = chosen
        rows.append(nr)
    audited = pd.DataFrame(rows)

    curated = pd.read_parquet(AUD / "sft_curated.parquet")
    v6 = pd.concat([curated, audited], ignore_index=True)
    v6 = v6.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
    v6["version"] = "v6"
    v6.to_parquet(OUTSFT, index=False)
    rep["curated"] = len(curated); rep["audited_kept"] = len(audited); rep["v6_total"] = len(v6)
    return rep


def assemble_dpo() -> dict:
    base = _rebuild_dpo_v2()
    fixed = _load_fixed("dpo")
    rep = {"v2_rows": len(base), "fixed_found": len(fixed),
           "keep": 0, "fix": 0, "drop": 0, "missing": 0, "invalid_after": 0}
    rows = []
    for idx, r in base.iterrows():
        fx = fixed.get(idx)
        if fx is None:
            rep["missing"] += 1
            rows.append(r.to_dict()); continue
        act = fx["action"]; rep[act] = rep.get(act, 0) + 1
        if act == "drop":
            continue
        chosen = _to_str(fx["chosen"]) if act == "fix" else str(r["chosen"])
        rejected = _to_str(fx.get("rejected")) if act == "fix" else str(r["rejected"])
        # chosen은 반드시 유효(정답). rejected는 malformed 허용(잘못된 출력=정당한 DPO negative).
        if parse_lenient(chosen) is None:
            rep["invalid_after"] += 1
            continue
        if chosen == rejected:
            rep["same_pair"] = rep.get("same_pair", 0) + 1
            continue
        rep["rejected_malformed"] = rep.get("rejected_malformed", 0) + (parse_lenient(rejected) is None)
        nr = r.to_dict(); nr["chosen"] = chosen; nr["rejected"] = rejected
        rows.append(nr)
    audited = pd.DataFrame(rows)

    curated = pd.read_parquet(AUD / "dpo_curated.parquet")
    keep_cols = ["prompt", "chosen", "rejected", "persona", "today", "category", "source"]
    audited = audited[[c for c in keep_cols if c in audited.columns]]
    curated = curated[[c for c in keep_cols if c in curated.columns]]
    v6 = pd.concat([curated, audited], ignore_index=True)
    v6 = v6.drop_duplicates(subset=["prompt", "chosen", "rejected"]).reset_index(drop=True)
    v6["version"] = "v6"
    v6.to_parquet(OUTDPO, index=False)
    rep["curated"] = len(curated); rep["audited_kept"] = len(audited); rep["v6_total"] = len(v6)
    return rep


if __name__ == "__main__":
    report = {"sft": assemble_sft(), "dpo": assemble_dpo()}
    (AUD / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[saved] {OUTSFT}")
    print(f"[saved] {OUTDPO}")
