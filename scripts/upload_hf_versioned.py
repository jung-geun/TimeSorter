#!/usr/bin/env python
"""버전별 증분 데이터셋을 HF에 SFT/DPO 2개 repo로 분리 업로드.

선행: uv run python scripts/build_versioned_datasets.py

repo:
  pieroot/timesorter-sft-ko   SFT 증분 v1~v5 (각 버전 = 신규 생성분)
  pieroot/timesorter-dpo-ko   DPO 증분 v1,v2,v3,v5

각 버전은 별도 config(v1..v5) — 무엇이 언제 추가됐는지 추적 가능.

사용:
  uv run python scripts/upload_hf_versioned.py
  uv run python scripts/upload_hf_versioned.py --card-only
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

SRC = Path("data/hf_versioned")
USER = "pieroot"
REPOS = {
    "sft": f"{USER}/timesorter-sft-ko",
    "dpo": f"{USER}/timesorter-dpo-ko",
}

_HEAD = {
    "sft": ("SFT", "할 일 목록을 4축(긴급도·중요도·의존성·시간 제약, 각 1-5)으로 채점하는 "
            "지도학습용 데이터. 각 버전은 그 버전에서 **새로 생성된 증분**만 포함."),
    "dpo": ("DPO", "선호 학습(chosen vs rejected)용 데이터. 각 버전은 그 버전 **증분 선호쌍**만 포함. "
            "v3+는 형식 동일·내용만 오류인 hard negative."),
}
_COLDESC = {
    "sft": "`prompt` / `chosen`(정답 JSON, v1은 자유텍스트) / `persona` / `today`(오늘 날짜, \"\"=미상) / "
           "`source` / `meta`(골격, v3+) / `version`",
    "dpo": "`prompt` / `chosen` / `rejected` / `persona` / `today` / `category`(위반유형) / `source` / `version`",
}


def build_card(kind: str, meta: dict, present: list[str]) -> str:
    label, intro = _HEAD[kind]
    # config 정의 (YAML)
    cfg_lines = ["---", "language: [ko]", "license: apache-2.0",
                 "task_categories: [text-generation]",
                 f"tags: [scheduling, prioritization, korean, {kind}, incremental, timesorter]",
                 "configs:"]
    default_v = "v6" if "v6" in present else present[0]
    for v in present:
        cfg_lines += [f"- config_name: {v}",
                      "  default: true" if v == default_v else None,
                      "  data_files:",
                      "  - split: train",
                      f"    path: data/{v}.parquet"]
    cfg_lines = [x for x in cfg_lines if x is not None]
    cfg_lines.append("---")

    # 버전 표
    rows = ["| 버전 | 행 수 | 제목 | 증분 목적 |", "|------|------|------|----------|"]
    for v in present:
        m = meta[kind][v]
        rows.append(f"| `{v}` | {m['rows']:,} | {m['title']} | {m['purpose']} |")

    # 특징 섹션
    feats = [f"### {v} — {meta[kind][v]['title']} ({meta[kind][v]['rows']:,}행)\n\n"
             f"- **증분 목적**: {meta[kind][v]['purpose']}\n"
             f"- **특징**: {meta[kind][v]['feat']}"
             for v in present]

    body = f"""
# TimeSorter — 한국어 일정 우선순위 {label} 데이터셋 (버전별 증분)

{intro}

> **버저닝 규칙**: `v1`~`v5`는 각 버전에서 새로 생성된 **증분 데이터만** 포함(누적 아님) —
> `vN` 모델 학습 = `v1`..`vN` 증분 누적. **`v6`은 v2~v5 전체를 검수·통합한 자립형(self-contained)
> 데이터셋**으로, 증분 누적 없이 **`v6` 단독으로 학습 가능**(기본 config).
> v6의 v2 구간은 opus 하위 에이전트로 전수 검수·수정했다.
> SFT 데이터: [{REPOS['sft']}](https://huggingface.co/datasets/{REPOS['sft']}) ·
> DPO 데이터: [{REPOS['dpo']}](https://huggingface.co/datasets/{REPOS['dpo']})

## 버전별 증분

{chr(10).join(rows)}

- 컬럼: {_COLDESC[kind]}
- 각 config는 단일 버전 증분. 전체를 보려면 모든 config를 로드해 concat.

## 버전별 특징·증분 목적

{chr(10).join(feats)}

## 생성 방법 — 시나리오 골격 우선 (v3+)

1. 프로그램이 골격 확정: 태스크별 마감 일시·지난 여부·의존 체인·리스크 문구
2. LLM은 텍스트·정답만 생성 (텍스트 gpt-5.4-mini, 정답 gpt-5.4 / Claude Sonnet 4.6)
3. 골격 규칙 자동 검증(verify_chosen) 통과분만 수록 — meta 컬럼에 골격 보존

## 학습 결과 (Qwen3.5-4B, held-out n=30 골격 자동 채점)

| 단계 | 통과율 |
|------|-------|
| 어댑터 없음 | 0.0% (스키마 미준수) |
| + SFT (v1~v4 누적) | 90.0% |
| + DPO (v5) | 90.0% |

학습 코드·모델: https://github.com/jung-geun/TimeSorter
"""
    return "\n".join(cfg_lines) + "\n" + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card-only", action="store_true")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN 환경변수가 필요합니다.")
    api = HfApi(token=token)
    meta = json.loads((SRC / "VERSIONS.json").read_text())

    for kind, repo in REPOS.items():
        files = sorted(f for f in (SRC / kind).glob("v*.parquet") if "_selfcontained" not in f.stem)
        present = [f.stem for f in files]
        api.create_repo(repo, repo_type="dataset", private=args.private, exist_ok=True)
        if not args.card_only:
            for f in files:
                api.upload_file(path_or_fileobj=str(f), path_in_repo=f"data/{f.name}",
                                repo_id=repo, repo_type="dataset")
                print(f"  [업로드] {repo} :: data/{f.name}")
        card = build_card(kind, meta, present)
        api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                        repo_id=repo, repo_type="dataset")
        print(f"  [카드] https://huggingface.co/datasets/{repo}\n")


if __name__ == "__main__":
    main()
