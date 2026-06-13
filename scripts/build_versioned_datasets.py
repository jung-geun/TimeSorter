#!/usr/bin/env python
"""버전별 증분(incremental) 데이터셋 빌드 — SFT/DPO 분리.

각 버전 vN = 그 버전에서 **새로 생성된 증분 데이터만** (누적 아님).
origin source 태그 기준으로 분리해 재현성을 보장한다.

출력:
  data/hf_versioned/sft/v1.parquet ... v5.parquet
  data/hf_versioned/dpo/v1.parquet, v2, v3, v5.parquet
  data/hf_versioned/VERSIONS.json   (버전별 메타: 행수·특징·증분목적)

버저닝 규칙 (앞으로):
  - 새 버전 vN은 직전까지와 겹치지 않는 **신규 생성 행만** vN 파일로 기록.
  - 학습 시 vN 모델 = v1..vN 증분을 누적해서 사용 (build_hf_release.py가 누적본 생성).
  - 업로드는 증분 단위(이 스크립트) — 무엇이 언제 추가됐는지 추적 가능.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path("data/hf_versioned")
(OUT / "sft").mkdir(parents=True, exist_ok=True)
(OUT / "dpo").mkdir(parents=True, exist_ok=True)

SFT_COLS = ["prompt", "chosen", "persona", "today", "source", "meta"]
DPO_COLS = ["prompt", "chosen", "rejected", "persona", "today", "category", "source"]


def norm(df: pd.DataFrame, cols: list[str], version: str) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            d[c] = ""
    d = d[cols].copy()
    d["version"] = version
    return d


# ── 버전 정의 (증분 = 그 버전 신규 생성분) ───────────────────────────────────
SFT_SPEC = {
    "v1": dict(file="data/scheduler_ko_combined.parquet",
               title="자유 텍스트 우선순위",
               purpose="events-scheduling(영문)→한국어 할 일 번역 + Nemotron 페르소나 주입. "
                       "'1) 할일 - 이유' 자유 텍스트 출력의 기본 셋.",
               feat="앱 파싱 불가(자유 텍스트), 페르소나 말투 반영"),
    "v2": dict(file="data/scheduler_v2_combined.parquet",
               title="4축 점수 JSON",
               purpose="v1 내용을 4축(긴급/중요/의존/시간) 점수 JSON으로 재생성. "
                       "구조화·앱 연동·설명가능성 확보. refusal·orca·xlam 혼합 포함.",
               feat="JSON 스키마 도입, 날짜 개념 없음(지난 일정 미구분)"),
    "v3": dict(file="data/scheduler_v3.parquet",
               title="골격 기반 + 오늘 날짜 주입",
               purpose="시나리오 골격(skeleton) 우선 생성 + 오늘 날짜·지난 일정 규칙 추가. "
                       "골격 검증(verify_chosen) 통과분만 수록, meta 보존.",
               feat="날짜 추론·지난 일정 강등, dated/intraday/chain/risk/relative 5종"),
    "v4": dict(file="data/scheduler_v4_extra.parquet",
               title="이중 생성 경로 + 신규 시나리오 3종",
               purpose="OpenAI(gpt-5.4)·Claude(Sonnet 4.6/Opus 4.8) 이중 생성으로 표현 다양화. "
                       "past_split·no_today·am_escalation 신규 시나리오 추가.",
               feat="이중 경로 품질 수렴, persona_fit 4.9-5.0 큐레이션"),
    "v5": dict(file="data/scheduler_v5_claude.parquet",
               title="의존성 체인 보강",
               purpose="잔여 약점(체인)을 타깃해 Claude가 4-5단계 긴 의존성 체인 시나리오 증량 생성.",
               feat="dependency_chain·past_split 비중 확대"),
}

DPO_SPEC = {
    "v1": dict(loader=lambda: _load_dpo_v1(),
               title="초기 선호쌍",
               purpose="v1 우선순위 출력의 선호/비선호 쌍.",
               feat="자유 텍스트 기반 (v3+ 학습 비권장)"),
    "v2": dict(loader=lambda: pd.read_parquet("data/dpo_pairs_v2.parquet"),
               title="4축 JSON 선호쌍 + refusal",
               purpose="v2 JSON 포맷 선호쌍 + 비일정 입력 거절(refusal) 쌍 대량 추가.",
               feat="형식 negative 위주(길이 편향 주의), refusal 도메인 30종"),
    "v3": dict(loader=lambda: _load_dpo_v3(),
               title="hard negative 5종",
               purpose="형식은 chosen과 동일하고 내용 한 곳만 틀린 hard negative 도입 "
                       "(date_confusion/granularity_swap/dependency_scatter/risk_ignore/order_score_mismatch). "
                       "v2_replay는 v2 재사용이므로 증분에서 제외.",
               feat="내용 오류 negative — 실제 실패 모드 벌점"),
    "v5": dict(loader=lambda: pd.read_parquet("data/dpo_pairs_v5_onpolicy.parquet"),
               title="on-policy 선호쌍",
               purpose="SFT v4 모델이 실제로 생성한 오답을 rejected로 수집(on-policy). "
                       "체인·dated 오류 집중.",
               feat="모델 실제 오류 기반 — 학습 밀도 최고"),
}
# v4는 SFT-only (GRPO 파일럿은 동률이라 선호 단계로 미포함) — DPO 증분 없음


def _load_dpo_v1() -> pd.DataFrame:
    d = pd.read_parquet("data/dpo_pairs.parquet").drop(columns=["pair"], errors="ignore")
    d["category"] = d.get("category", "v1_pair")
    d["source"] = d.get("source", "dpo_v1")
    return d


def _load_dpo_v3() -> pd.DataFrame:
    d = pd.read_parquet("data/dpo_pairs_v3.parquet")
    return d[~d["source"].astype(str).str.startswith("v2_replay")].copy()


def main():
    meta = {"sft": {}, "dpo": {}, "convention": (
        "각 vN은 해당 버전 신규 생성 증분만 포함(누적 아님). "
        "vN 모델 학습 = v1..vN 누적. 업로드는 증분 단위로 추적성 확보.")}

    print("=== SFT 증분 ===")
    for v, spec in SFT_SPEC.items():
        df = norm(pd.read_parquet(spec["file"]), SFT_COLS, v)
        df.to_parquet(OUT / "sft" / f"{v}.parquet", index=False)
        meta["sft"][v] = {"rows": len(df), "title": spec["title"],
                          "purpose": spec["purpose"], "feat": spec["feat"]}
        print(f"  sft/{v}: {len(df):,}행 — {spec['title']}")

    print("=== DPO 증분 ===")
    for v, spec in DPO_SPEC.items():
        df = norm(spec["loader"](), DPO_COLS, v)
        df.to_parquet(OUT / "dpo" / f"{v}.parquet", index=False)
        meta["dpo"][v] = {"rows": len(df), "title": spec["title"],
                          "purpose": spec["purpose"], "feat": spec["feat"]}
        print(f"  dpo/{v}: {len(df):,}행 — {spec['title']}")

    (OUT / "VERSIONS.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\n[saved] {OUT}/VERSIONS.json")


if __name__ == "__main__":
    main()
