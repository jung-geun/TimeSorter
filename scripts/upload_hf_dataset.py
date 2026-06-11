#!/usr/bin/env python
"""TimeSorter 데이터셋을 HuggingFace Hub에 업로드.

업로드 구성 (repo_type=dataset):
  sft/      scheduler_v3, scheduler_v4_openai, scheduler_v4_claude, scheduler_v3_combined
  dpo/      dpo_pairs_v2, dpo_pairs_v3
  eval/     scheduler_v3_eval
  README.md 데이터 카드 (구성·생성 방법·검증 규칙)

사용:
  uv run python scripts/upload_hf_dataset.py --repo pieroot/timesorter-scheduler-ko
  uv run python scripts/upload_hf_dataset.py --repo ... --private
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

_FILES = {
    "sft/scheduler_v3.parquet": "data/scheduler_v3.parquet",
    "sft/scheduler_v4_openai.parquet": "data/scheduler_v4_openai.parquet",
    "sft/scheduler_v4_claude.parquet": "data/scheduler_v4_claude.parquet",
    "sft/scheduler_v3_combined.parquet": "data/scheduler_v3_combined.parquet",
    "sft/scheduler_v4_extra.parquet": "data/scheduler_v4_extra.parquet",
    "dpo/dpo_pairs_v3.parquet": "data/dpo_pairs_v3.parquet",
    "dpo/dpo_pairs_v4_extra.parquet": "data/dpo_pairs_v4_extra.parquet",
    "eval/scheduler_v3_eval.parquet": "data/scheduler_v3_eval.parquet",
}

_CARD = """\
---
language: [ko]
license: apache-2.0
task_categories: [text-generation]
tags: [scheduling, prioritization, korean, dpo, synthetic]
---

# TimeSorter — 한국어 일정 우선순위 정렬 데이터셋

할 일 목록을 4축(긴급도·중요도·의존성·시간 제약, 각 1-5)으로 채점하고 우선순위를 결정하는
JSON 응답(chosen) 학습 데이터. Qwen3-4B SFT/DPO/GRPO 파인튜닝에 사용.

## 구성

| 경로 | 행 수 | 설명 |
|------|-------|------|
{rows}

## 생성 방법 — 시나리오 골격 우선 (skeleton-first)

1. 프로그램이 시나리오 골격을 먼저 확정: 태스크별 마감 일시·이미 지났는지 여부·의존 체인·리스크 문구
2. LLM은 골격에 맞는 자연어 태스크 텍스트와 chosen JSON만 생성
   - 텍스트: gpt-5.4-mini / chosen: gpt-5.4 (v4_openai), Claude Sonnet 4.6 (v4_claude)
3. 골격 규칙으로 자동 검증 후 통과분만 수록 (지난 일정 후순위, 동일일 시각 오름차순,
   체인 연속 배치, 리스크 importance≥4, 무마감 1위 금지 등)
4. 골격은 `meta` 컬럼(JSON)에 보존 — DPO hard negative 생성·자동 평가·GRPO 보상에 재사용

## 시나리오

- `dated_mixed` / `past_split`: 오늘 날짜 주입 + 이미 지난/유효 일정 분리
- `intraday`: 같은 날 서로 다른 시각 마감 (오전 > 오후)
- `dependency_chain`: 작성→검토→발송 체인 연속 배치
- `risk` / `am_escalation`: 에스컬레이션·위약금 등 리스크 → importance/urgency 상향
- `no_today`: 오늘 날짜 미상 — 절대 날짜 상대 정렬만 수행, '지남' 단정 금지
- `relative`: 내일/모레/다음 주 화요일 등 상대 표현 환산

컬럼: `prompt`(할 일 목록), `chosen`(4축 JSON), `persona`, `today`(ISO 또는 빈 문자열=미상),
`source`(시나리오), `meta`(골격, 일부 파일).

생성 코드: https://github.com/jung-geun/TimeSorter (scripts/gen_schedule_v3.py 등)
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="pieroot/timesorter-scheduler-ko")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN 환경변수가 필요합니다.")
    api = HfApi(token=token)

    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)

    card_rows = []
    for remote, local in _FILES.items():
        p = Path(local)
        if not p.exists():
            print(f"  [건너뜀] {local} 없음")
            continue
        n = len(pd.read_parquet(p))
        card_rows.append(f"| `{remote}` | {n:,} | {p.stem} |")
        api.upload_file(path_or_fileobj=str(p), path_in_repo=remote,
                        repo_id=args.repo, repo_type="dataset")
        print(f"  [업로드] {remote} ({n:,}행)")

    readme = _CARD.replace("{rows}", "\n".join(card_rows))
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset")
    print(f"[완료] https://huggingface.co/datasets/{args.repo}")
