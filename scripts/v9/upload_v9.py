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

## 입력 스키마 `ScheduleInput`
```json
{{
  "current_time": "2026-03-21T08:00:00+09:00",
  "user_persona": {{"occupations": ["마케팅 매니저"], "detailed_status": "...", "age": 34,
                    "gender": "male|female", "location": {{"country": "South Korea", "city": "서울 강남구"}},
                    "bio": "...", "availability": "09:00-18:00"}},
  "tasks": [{{"task_id": "task_001", "title": "분기 실적 보고서 작성", "memo": "미제출 시 위약금",
              "source": "email", "deadline": "2026-03-21T15:00:00+09:00",
              "estimated_duration_minutes": 90}}]
}}
```

**필드 의미** — 모델이 스케줄을 짜기 위해 받는 컨텍스트:

| 필드 | 타입 | 의미 |
|------|------|------|
| `current_time` | str (ISO8601+오프셋) | 스케줄링 기준 "지금" 시각. **모든 마감 임박도·시간블록 배치의 기준점**. 타임존 오프셋 포함(KR=+09:00) |
| `user_persona.occupations` | list[str] | 직업(들). **빈 리스트=무직/구직** → 업무 태스크 대신 생활·구직 과제. 태스크 적합성 판단 |
| `user_persona.detailed_status` | str | 현재 상황·목표 요약(직업적 맥락) — 점수·근거의 배경 |
| `user_persona.age` / `gender` | int / "male"·"female" | 연령·성별 |
| `user_persona.location` | obj | `country`·`city` — 지역 맥락 |
| `user_persona.bio` | str | 성향·생활 패턴(취미·습관) — 어떤 일이 자연스러운지 |
| `user_persona.availability` | str `"HH:MM-HH:MM"` | **하루 가용 시간대**. 추천 시간블록은 반드시 이 창 안에만 배치 |
| `tasks[].task_id` | str | 고유 식별자. **출력에서 그대로 참조**(누락·추가 금지) |
| `tasks[].title` | str | 할 일 제목 |
| `tasks[].memo` | str | 부가 맥락(없으면 `""`). 리스크 문구(위약금·법정기한 등) 포함 가능 |
| `tasks[].source` | str | 유입 출처: email·slack_message·calendar·memo_app·phone_call·sms·kakao·app_push |
| `tasks[].deadline` | str(ISO8601) \| null | 마감 시각. **`null`=마감 없음** |
| `tasks[].estimated_duration_minutes` | int | 예상 소요 시간(분). **추천 시간블록의 길이** |

## 출력 스키마 `ScheduleResponseV9`
```json
{{
  "scheduled_tasks": [{{
    "task_id": "task_001", "title": "분기 실적 보고서 작성", "priority_rank": 1,
    "scoring": {{"deadline_proximity": 9.0, "task_importance": 9.0,
                "task_chaining": 2.0, "urgency": 9.5, "total_score": 8.1}},
    "reasoning": {{"summary": "마감 7시간 전·위약금 리스크로 최우선...", "chaining_detail": ""}},
    "recommended_schedule": {{"start_time": "2026-03-21T08:00:00+09:00",
                             "end_time": "2026-03-21T09:30:00+09:00"}}
  }}]
}}
```

**필드 의미** — `scheduled_tasks[]`는 **입력 task 전부**를 우선순위 순으로 담는다(커버리지 보장):

| 필드 | 타입 | 의미 |
|------|------|------|
| `task_id` / `title` | str | 입력에서 그대로(누락·추가·변형 금지) |
| `priority_rank` | int | **실행 순서**. 1..N 순열, 1=가장 먼저 |
| `scoring.deadline_proximity` | 0–10 | **마감 임박도**. current_time 기준 마감이 가까울수록 ↑ (마감 없음=0) |
| `scoring.urgency` | 0–10 | **지금 당장 착수해야 하는 정도**. 여유(slack) 적을수록·불이익(에스컬레이션/위약금/법정기한) 있을수록 ↑ |
| `scoring.task_importance` | 0–10 | **사용자 목표에 미치는 영향**. 리스크 문구 있으면 9–10으로 상향 |
| `scoring.task_chaining` | 0–10 | **후속 작업 블로킹 정도**. 다른 작업의 선행이면 ↑ (비최종 체인=9 · 최종 체인=6 · 독립=2) |
| `scoring.total_score` | 0–10 | 4축 **가중 종합**(아래 공식). priority_rank 정렬 기준 |
| `reasoning.summary` | str | 이 점수·순위·시간블록인 **근거 1–2문장**(facts 수치 인용, 날조 금지) |
| `reasoning.chaining_detail` | str | 체인 소속 시 **선행/후행 의존관계** 명시. **독립이면 `""`**(빈 문자열) |
| `recommended_schedule.start_time` / `end_time` | str(ISO8601) | **추천 시간블록**. 길이=estimated_duration_minutes, 가용시간 내, 블록 간 비중복, 가능하면 마감 전 완료 |

**total_score 공식** = `0.30·urgency + 0.25·deadline_proximity + 0.30·task_importance + 0.15·task_chaining` (소수 2자리). 평가 시 4축에서 결정적 재계산 — 모델은 **4축만 정확하면** total은 코드로 산출.

**priority_rank 규칙**: total_score 내림차순이 기본, 단 (1) **체인 선행이 후행보다 먼저**, (2) 이미 지난 고정 일정은 최하위.

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

## 학습 결과 (이 데이터로 SFT/DPO 시, n=50, greedy)
`verify_chosen_v9`(구조 규칙 무위반) + gold 대비 점수 오차(↓ 낮을수록 좋음).

| 지표 | 4B base | 4B SFT | 4B DPO | 2B base | 2B SFT | 2B DPO |
|------|---------|--------|--------|---------|--------|--------|
| parse_rate | 0.82 | 0.90 | 0.92 | 0.20 | 0.92 | 0.94 |
| verify_pass | 0.10 | 0.47 | 0.46 | 0.00 | 0.09 | 0.09 |
| chain_order | 0.29 | 0.53 | 0.52 | 0.30 | 0.30 | 0.34 |
| sched_feasible | 0.98 | 1.00 | 1.00 | 0.90 | 0.94 | 0.96 |
| deadline_met | 0.93 | 1.00 | 1.00 | 0.50 | 0.91 | 0.94 |
| rank_exact | 0.40 | 0.59 | 0.59 | 0.29 | 0.39 | 0.38 |
| axis_mae↓ | 2.70 | 0.67 | 0.74 | 3.20 | 1.76 | 1.75 |
| total_mae↓ | 1.95 | 0.66 | 0.72 | 2.14 | 1.63 | 1.60 |

- **SFT가 1차 효과**(규칙·점수 급상승), **DPO ≈ SFT**(선호학습으로 체인 능력은 안 옮겨짐 — 4B·2B 공통).
- **2B는 형식부터 학습**(parse 0.20→0.92), 4B base는 이미 형식을 알아 SFT가 규칙·점수를 끌어올림.
- 2B 천장 < 4B(verify_pass 0.09 vs 0.47) — 규칙 동시충족은 모델 용량 의존.

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
