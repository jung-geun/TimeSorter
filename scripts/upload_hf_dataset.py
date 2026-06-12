#!/usr/bin/env python
"""TimeSorter 통합 데이터셋을 HuggingFace Hub에 업로드.

선행: uv run python scripts/build_hf_release.py  (data/hf_release/ 생성)

repo 구조 (용도별 분리 + 뷰어 configs):
  data/sft_train.parquet      SFT 본 학습 (v2~v5+rework JSON 스키마)
  data/sft_v1_text.parquet    v1 자유 텍스트 (별도 포맷, 6.0K)
  data/dpo_train.parquet      DPO 쌍 v1~v5 통합
  data/grpo_train.parquet     GRPO 프롬프트+골격 meta
  data/eval_heldout.parquet   held-out 평가 (150, seed 47)

새 데이터셋 추가 후 갱신 절차:
  1. uv run python scripts/build_hf_release.py   # data/hf_release/ 재빌드
  2. uv run python scripts/upload_hf_dataset.py  # HF 업로드

사용:
  uv run python scripts/upload_hf_dataset.py --repo pieroot/timesorter-scheduler-ko
  uv run python scripts/upload_hf_dataset.py --card-only  # README만 갱신
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
    "data/sft_train.parquet": "data/hf_release/sft_train.parquet",
    "data/sft_v1_text.parquet": "data/hf_release/sft_v1_text.parquet",
    "data/dpo_train.parquet": "data/hf_release/dpo_train.parquet",
    "data/grpo_train.parquet": "data/hf_release/grpo_train.parquet",
    "data/eval_heldout.parquet": "data/hf_release/eval_heldout.parquet",
}

_CARD = """\
---
language: [ko]
license: apache-2.0
task_categories: [text-generation]
tags: [scheduling, prioritization, korean, dpo, grpo, synthetic]
configs:
- config_name: sft
  default: true
  data_files:
  - split: train
    path: data/sft_train.parquet
- config_name: dpo
  data_files:
  - split: train
    path: data/dpo_train.parquet
- config_name: grpo
  data_files:
  - split: train
    path: data/grpo_train.parquet
- config_name: sft_v1_text
  data_files:
  - split: train
    path: data/sft_v1_text.parquet
- config_name: eval
  data_files:
  - split: test
    path: data/eval_heldout.parquet
---

# TimeSorter — 한국어 일정 우선순위 정렬 데이터셋 (v1~v5 통합)

할 일 목록을 4축(긴급도·중요도·의존성·시간 제약, 각 1-5)으로 채점하고 우선순위를 결정하는
모델 학습용 데이터. Qwen3.5-4B SFT → DPO → GRPO 파인튜닝에 사용.

## Configs

| config | split | 행 수 | 용도 |
|--------|-------|------|------|
{rows}

- **sft** (기본): v2~v4 JSON 스키마 통합. 컬럼 prompt/chosen(JSON)/persona/today/source/meta/version.
  `today`는 시스템 프롬프트에 주입되는 오늘 날짜(ISO, 빈 문자열=날짜 미상 시나리오).
- **dpo**: chosen/rejected 쌍. v3+는 형식 동일·내용만 오류인 hard negative
  (date_confusion/granularity_swap/dependency_scatter/risk_ignore/order_score_mismatch/past_hallucination).
  v5에는 학습된 모델이 실제로 위반한 출력을 rejected로 수집한 **on-policy 351쌍** 포함.
- **grpo**: 골격(meta JSON) 보유 프롬프트 — meta의 태스크별 마감·과거 여부·체인·리스크 정보로
  검증 가능한 보상 함수(RLVR)를 구성할 수 있다.
- **eval**: 학습에 쓰지 않은 seed로 생성한 held-out 150 시나리오 (골격 규칙 자동 채점용).

## 데이터셋별 특징

### sft
- 골격 우선 생성 + 자동 검증 통과분만 수록. `today`(오늘 날짜, ""=미상)·`meta`(골격)·`version` 포함.
- **tier**: `curated`(v3-v5 — Claude Opus 4.8 감사 persona_fit 4.9-5.0, 본 학습 권장) /
  `v2_refusal`(거부 학습, 항상 혼합) / `v2_schedule`(persona_fit 3.3 — 소량만) / `v2_offformat`(비권장)
- 실증: curated + 프롬프트 loss 마스킹 학습 → held-out 골격 통과율 56.7%→77.3% (+20.6%p)

### dpo
- hard negative는 chosen과 형식·길이 동일(길이비 0.96-0.99), **내용만 오류**:
  date_confusion / granularity_swap / dependency_scatter / risk_ignore / order_score_mismatch / past_hallucination
- on-policy 쌍: 학습된 모델이 실제로 위반한 출력을 rejected로 수집
- **tier**: `hard`(권장) / `refusal` / `easy_format`(v2 — urgency_only 길이비 0.19 등 길이 편향 주의) / `legacy_text`(v1 — v3+ 학습 금지)

### grpo
- chosen 없이 prompt+meta(골격)만 — verify_chosen 규칙으로 결정적·무비용 보상(RLVR) 구성
- 보상 예시: 전 규칙 통과 +1.0, 위반당 -0.3, 파싱 실패 -1.0
- 무결성: meta 파싱 100%, eval 누수 0, 프롬프트 중복 0

### eval
- 학습 미사용 seed(47)로 생성한 held-out 150 — 골격 규칙 자동 채점용 기준선

## 생성 방법 — 시나리오 골격 우선 (skeleton-first)

1. 프로그램이 골격 확정: 태스크별 마감 일시·이미 지났는지·의존 체인·리스크 문구
2. LLM은 텍스트와 정답 JSON만 생성 — 텍스트 gpt-5.4-mini, 정답 gpt-5.4 / Claude Sonnet 4.6
3. 골격 규칙 자동 검증 통과분만 수록: 지난 일정 후순위, 동일일 시각 오름차순, 체인 연속 배치,
   리스크 importance≥4, 오늘 미상 시 '지남' 단정 금지, 무마감 1위 금지
4. 골격은 meta 컬럼에 보존 — DPO negative 생성·자동 평가·GRPO 보상에 재사용

## 시나리오 (v3~v4)

dated_mixed / past_split (오늘 기준 지난·유효 분리) · intraday (오전>오후) ·
dependency_chain · risk / am_escalation (불이익 조항) · no_today (날짜 미상 상대 정렬) ·
relative (내일/모레 환산)

생성 코드: https://github.com/jung-geun/TimeSorter
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="pieroot/timesorter-scheduler-ko")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--card-only", action="store_true", help="README(데이터 카드)만 갱신")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN 환경변수가 필요합니다.")
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)

    # 이전 레이아웃 정리 (sft/ dpo/ eval/ 디렉토리)
    existing = api.list_repo_files(args.repo, repo_type="dataset")
    stale = [f for f in existing if f.startswith(("sft/", "dpo/", "eval/"))]
    for f in stale:
        api.delete_file(f, repo_id=args.repo, repo_type="dataset")
        print(f"  [삭제] {f}")

    _CONFIG_LABEL = {
        "sft_train": ("sft", "train", "SFT 본 학습 (v2~v5 JSON, curated 3.4K 포함)"),
        "sft_v1_text": ("sft_v1_text", "train", "v1 자유 텍스트"),
        "dpo_train": ("dpo", "train", "DPO 쌍 v1~v5 (on-policy 351쌍 포함)"),
        "grpo_train": ("grpo", "train", "GRPO 프롬프트+골격 (v3~v5)"),
        "eval_heldout": ("eval", "test", "held-out 자동 평가"),
    }
    card_rows = []
    for remote, local in _FILES.items():
        p = Path(local)
        if not p.exists():
            raise SystemExit(f"{local} 없음 — build_hf_release.py 먼저 실행")
        n = len(pd.read_parquet(p))
        cfg, split, desc = _CONFIG_LABEL[p.stem]
        card_rows.append(f"| `{cfg}` | {split} | {n:,} | {desc} |")
        if not args.card_only:
            api.upload_file(path_or_fileobj=str(p), path_in_repo=remote,
                            repo_id=args.repo, repo_type="dataset")
            print(f"  [업로드] {remote} ({n:,}행)")

    readme = _CARD.replace("{rows}", "\n".join(card_rows))
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset")
    print(f"[완료] https://huggingface.co/datasets/{args.repo}")
