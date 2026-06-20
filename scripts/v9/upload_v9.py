#!/usr/bin/env python
"""v9 데이터셋(신 스키마)을 HF 전용 레포에 업로드 — SFT/DPO 2 config.

신 스키마(JSON-in/out)라 구 4축 레포(timesorter-{sft,dpo}-ko)와 분리.
repo: pieroot/timesorter-scheduler-v9-ko  (config: sft, dpo)

사용: uv run python scripts/v9/upload_v9.py [--private] [--card-only]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
REPO = "pieroot/timesorter-scheduler-v9-ko"
SFT = Path("data/hf_versioned/sft/v9.parquet")
DPO = Path("data/hf_versioned/dpo/v9.parquet")


def card(n_sft: int, n_dpo: int) -> str:
    front = """---
language: [ko]
license: apache-2.0
task_categories: [text-generation]
tags: [scheduling, prioritization, korean, json, agent, timesorter, v9]
configs:
- config_name: sft
  default: true
  data_files:
  - split: train
    path: data/sft.parquet
- config_name: dpo
  data_files:
  - split: train
    path: data/dpo.parquet
---
"""
    body = f"""
# TimeSorter v9 — 한국어 일정 스케줄링 (JSON-in / JSON-out)

앱 연동형 **구조화 JSON 입출력** 스케줄링 데이터셋. 사용자 페르소나·현재 시각·할 일 목록(JSON)을
입력받아 태스크별 우선순위·4축 점수·추천 시간블록·근거(JSON)를 산출한다. 구 4축(1-5 정수) 자연어
스키마를 전면 개편한 신 포맷 — 모델 출력을 앱이 그대로 파싱해 캘린더에 렌더링하는 것을 목표로 한다.

**목적**: "오늘 할 일"을 입력하면 페르소나·마감·의존관계를 고려해 (1) 실행 순서, (2) 설명가능한
4축 점수, (3) 충돌 없는 추천 시간블록을 한 번에 제안하는 개인 비서 코어 모델의 학습 데이터.

- **SFT**: {n_sft:,}행 (`sft` config) — prompt(입력 JSON) / chosen(출력 JSON) / persona / today / source / meta
- **DPO**: {n_dpo:,}쌍 (`dpo` config) — 형식·길이 동일·규칙 1곳만 위반한 hard negative 5종
  (schedule_overlap·deadline_miss·rank_score_mismatch·total_score_wrong·chain_order_break).
  chosen 100% 검증 통과, 차이는 위반 1축뿐. **실제 학습은 약점(체인 순서)에 집중한 서브셋
  2,812쌍**(chain_order_break + rank_score_mismatch)을 사용 — schedule/deadline은 SFT가 이미 100%.

## 입력 스키마
```json
{{
  "current_time": "2026-03-21T08:00:00+09:00",
  "user_persona": {{"occupations": ["..."], "detailed_status": "...", "age": 0,
                    "gender": "male|female", "location": {{"country": "...", "city": "..."}},
                    "bio": "...", "availability": "09:00-18:00"}},
  "tasks": [{{"task_id": "task_001", "title": "...", "memo": "...", "source": "...",
              "deadline": "ISO8601|null", "estimated_duration_minutes": 60}}]
}}
```

## 출력 스키마
```json
{{
  "scheduled_tasks": [{{
    "task_id": "task_001", "title": "...", "priority_rank": 1,
    "scoring": {{"deadline_proximity": 0-10, "task_importance": 0-10,
                "task_chaining": 0-10, "urgency": 0-10, "total_score": 0-10}},
    "reasoning": {{"summary": "...", "chaining_detail": "<체인일 때만>"}},
    "recommended_schedule": {{"start_time": "ISO8601", "end_time": "ISO8601"}}
  }}]
}}
```
`total_score = 0.30·urgency + 0.25·deadline_proximity + 0.30·task_importance + 0.15·task_chaining`

## 생성 방법
1. Nvidia Nemotron-Personas-Korea → 12개 직업군 균등 48 페르소나(이름 제거, 활성/구직 분리)
2. 결정적 골격: 직업군별 도메인·체인테마 + micro-topic(110종)·chain-subject(80종) 순환배정으로 다양화.
   **마감 = 스케줄 종료 + 버퍼**로 항상 실현 가능(마감 위반 없음)
3. 하위 에이전트: **Haiku**=태스크 제목/메모, **Sonnet**=reasoning(주입된 사실 기반), **Opus**=검수
4. 결정적 검증기 게이트(rank 순열·블록 중복·소요 일치·마감 실현·total 재계산·체인 선후)

## 시나리오 다양성
dated_mixed(절대 날짜·시각 혼재) · intraday(당일 시각) · dependency_chain_complex(4-5단계 의존
체인) · risk(리스크 문구→importance 상향) · relative(상대 시각) · past_split(지난/유효 혼합) ·
am_escalation(오전 마감 에스컬레이션) 등. micro-topic 110종 · chain-subject 80종 순환 배정.

## 검수 (2단계 게이트)
1. **결정적 검증기**(`verify_chosen_v9`): rank 순열·시간블록 중복·소요 일치·마감 실현가능·total
   재계산·체인 선후 — 수천 행을 무비용 게이트.
2. **sonnet/opus 전수 검수**: 현실성·페르소나 적합성·점수↔순위↔스케줄↔근거 정합. 1차 1,752행
   62% 통과 → 생성기 결함 수정(현장직 사무체인 차단·risk 사무직군 한정·체인 task_id 단계순 연속) →
   보충분 재생성 76% 통과 → 정제본(1,072) + 보충(470) = **1,542행** 채택.

## 품질
- 태스크 제목 고유율 ~98.7% (과적합 방지), 12 직업군 균일도 ~1.0 (105-142행/군)
- `total_score`는 평가 시 4축에서 공식으로 재계산(recompute) — 모델은 4축만 정확하면 됨
- 학습 코드: https://github.com/jung-geun/TimeSorter
"""
    return front + body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--card-only", action="store_true")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN 필요")
    api = HfApi(token=token)
    n_sft = len(pd.read_parquet(SFT))
    n_dpo = len(pd.read_parquet(DPO))
    api.create_repo(REPO, repo_type="dataset", private=args.private, exist_ok=True)
    if not args.card_only:
        api.upload_file(path_or_fileobj=str(SFT), path_in_repo="data/sft.parquet",
                        repo_id=REPO, repo_type="dataset")
        print(f"  [업로드] {REPO} :: data/sft.parquet ({n_sft:,}행)")
        api.upload_file(path_or_fileobj=str(DPO), path_in_repo="data/dpo.parquet",
                        repo_id=REPO, repo_type="dataset")
        print(f"  [업로드] {REPO} :: data/dpo.parquet ({n_dpo:,}쌍)")
    api.upload_file(path_or_fileobj=card(n_sft, n_dpo).encode(), path_in_repo="README.md",
                    repo_id=REPO, repo_type="dataset")
    print(f"  [카드] https://huggingface.co/datasets/{REPO}")


if __name__ == "__main__":
    main()
