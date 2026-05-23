# TimeSorter — 데이터셋 명세

## 개요

TimeSorter는 **SFT용 지도학습 데이터**와 **DPO용 선호도 쌍 데이터** 두 종류를 사용합니다.
모든 데이터는 한국어이며 실제 인구 통계를 반영한 다양한 페르소나가 적용됩니다.

---

## 1. SFT 데이터셋 (`data/scheduler_ko_combined.parquet`)

### 구조

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `prompt` | str | 페르소나 + 할 일 목록 (`[XXX 씨의 오늘의 할 일 목록]\n- 할일1\n- 할일2`) |
| `chosen` | str | 4축 기준 우선순위 정렬 응답 (`1) 할일 - 이유\n2) ...`) |
| `persona` | str | 페르소나 요약 (`홍길동 (의사, 35세)`) |
| `source` | str | 데이터 출처 태그 |

### 규모 및 출처

| 출처 | 수량 | 생성 방법 |
|------|------|-----------|
| `events-scheduling` + Nemotron 페르소나 (round 1~4) | 2,000개 | GPT로 영문 시드 → 한국어 로컬라이징, Nemotron-Personas-Korea 100만 풀에서 페르소나 샘플링 |
| `events-scheduling` + 8 제네릭 페르소나 | 3,999개 | GPT로 직장인/학생/프리랜서/부모/시니어/창업자/의료진/연구자 8종 직접 생성 |
| **소계** | **5,999개** | 중복 1건 제거 |
| `ko_Ultrafeedback_binarized` (런타임 혼합) | +500개 | 스케줄링 키워드 필터 후 학습 시 자동 혼합 |
| **학습 실효 총계** | **6,499개** | |

### 길이 분포

| 항목 | 최솟값 | 최댓값 | 평균 | 중앙값 |
|------|--------|--------|------|--------|
| prompt (chars) | 34 | 614 | 149 | 140 |
| chosen (chars) | 22 | 864 | 247 | 188 |

### 페르소나 분포

8개 제네릭 페르소나가 각 500개로 균등 분포하며, Nemotron round별로 다양한 실제 인물형 페르소나가 포함됩니다.

```
직장인     500   학생      500   프리랜서  500   부모     500
시니어     500   창업자    500   의료진    500   연구자   499
+ Nemotron 실제 인물형 페르소나 (68세 건물 경비원, 36세 소방관 등) 2,000개
```

### 프롬프트 형식

```
[홍길동 씨의 오늘의 할 일 목록]
- 논문 초안 검토 (내일 오전 마감)
- 연구실 미팅 (오후 2시)
- 실험 데이터 분석
- 학회 등록비 납부
```

### 정답(chosen) 형식

```
1) 논문 초안 검토 - 내일 오전 마감이라 오늘 안에 반드시 완료해야 함 (긴급도↑)
2) 연구실 미팅 - 오후 2시 고정 일정, 준비 필요 (시간 제약↑)
3) 실험 데이터 분석 - 논문 제출 전 선행 필수 작업 (의존성↑)
4) 학회 등록비 납부 - 마감 여유 있음, 다른 작업 후 처리 가능
```

---

## 2. DPO 데이터셋 (`data/dpo_pairs.parquet`)

### 구조

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `prompt` | str | SFT와 동일한 할 일 목록 프롬프트 |
| `chosen` | str | 4축 기준을 따른 고품질 응답 |
| `rejected` | str | 편향/저품질 응답 (긴급도만 보거나 근거 없이 순서 결정) |
| `persona` | str | 페르소나 요약 |
| `pair` | str | 쌍 조합 코드 (`c1_vs_c4`, `c2_vs_c3`, `c1_vs_c3`) |

### 규모

| 쌍 조합 | 수량 | 설명 |
|---------|------|------|
| `c1_vs_c4` | 497 | gpt-5.5 full guide vs gpt-5.4-mini no guide |
| `c1_vs_c3` | 491 | gpt-5.5 full guide vs gpt-5.4-mini urgency-only |
| `c2_vs_c3` | 481 | gpt-5.5 full guide (Claude 역할) vs gpt-5.4-mini urgency-only |
| **합계** | **1,469** | 500 시나리오 × 3 조합 (TIE 제외) |

> **원래 설계 의도**: C1=Gemini-flash(full), C2=Claude-Sonnet(full), C3=Gemini-lite(urgency), C4=Claude-Haiku(none)  
> **실제 적용**: Anthropic/Google API 미가용으로 모두 OpenAI gpt-5.5/gpt-5.4-mini로 폴백.  
> 폴백 후에도 guide 여부(full vs urgency-only vs no-guide)로 품질 차이를 유지.

### 4-후보 생성 및 판정 방법

```
시나리오 하나당:
  C1: gpt-5.5  + 4축 가이드 전문 (고품질)
  C2: gpt-5.5  + 4축 가이드 전문 (고품질, Claude 역할 대리)
  C3: gpt-5.4-mini + 긴급도만 기준 (편향)
  C4: gpt-5.4-mini + 가이드 없음 (저품질)

Judge: gpt-5.5 → A/B/TIE 판정
  → TIE 제외, 승자=chosen / 패자=rejected
```

### 생성 방식

```bash
# 비동기 생성 (asyncio, concurrency=10)
uv run python scripts/gen_preference_pairs.py \
    --in data/scheduler_ko_combined.parquet \
    --limit 500 \
    --concurrency 10

# 10 시나리오/22초 (동기 대비 7x 향상)
```

---

## 3. 보조 데이터셋

| 파일 | 크기 | 용도 |
|------|------|------|
| `data/events-scheduling.parquet` | — | 영문 시드 이벤트 + 정답 우선순위 |
| `data/nemotron_personas_korea.parquet` | 1.9GB | NVIDIA Nemotron-Personas-Korea, 100만 한국인 페르소나 (gitignore, 재다운로드 가능) |
| `data/ko_Ultrafeedback_binarized.parquet` | 103MB | 한국어 선호도 데이터, 스케줄 키워드 필터 후 혼합 (gitignore) |
| `data/sample_emails/` | — | 검증용 한국어 이메일 5건 |

---

## 4. 데이터 파이프라인

```
[영문 시드]
events-scheduling.parquet
        │
        ├─ gen_nemotron_schedule.py (×4 round, random_state 42/43/44/45)
        │       → Nemotron 페르소나 매칭 + GPT 번역·로컬라이징
        │       → scheduler_ko.parquet, scheduler_nemotron_r2~r4.parquet
        │
        └─ gen_korean_schedule.py (8 페르소나 × 500)
                → scheduler_generic.parquet

[병합]
merge_datasets.py
        → scheduler_ko_combined.parquet (5,999개, 중복 제거)

[DPO 쌍 생성]
gen_preference_pairs.py (async, concurrency=10)
        → 4-후보 생성 + GPT judge
        → dpo_pairs.parquet (1,469쌍)
```
