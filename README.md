# TimeSorter — 한국어 할 일 우선순위 정렬 비서

> **Qwen3.5-4B / 9B**를 한국어 일정 관리 태스크에 특화 파인튜닝하는 SFT → DPO(→GRPO) 파이프라인.
> 할 일 목록을 **긴급도·중요도·의존성·시간 제약** 4축으로 채점해 우선순위를 결정합니다.

---

## 프로젝트 목적

스마트폰·PC에서 "오늘 할 일"을 입력하면 AI가 맥락을 이해해 실행 순서를 제안하는 개인 비서 코어 모델. 단순 키워드 정렬이 아니라, **페르소나**(직장인·학생·부모 등)와 **4축**으로 각 태스크를 1–5점 채점하고 근거를 함께 제시합니다.

```
입력: "임원 보고서 마감(내일), 팀 회의(오후 2시), 점심 약속, 메일 답장 3건"

출력:
1) 임원 보고서 마감  [긴급5·중요5·의존4·시간2] — 내일 마감, 핵심 업무
2) 팀 회의(오후 2시) [긴급4·중요4·의존3·시간4] — 고정 시각, 후속 블로킹
3) 메일 답장 3건     [긴급4·중요3·의존2·시간1] — 긴급하나 고정 시각 없음
4) 점심 약속         [긴급2·중요2·의존1·시간3] — 유연 조정 가능
```

> 위 1–5 점수 예시는 **초기(v3) 자연어 포맷**입니다. 현행 앱 연동 포맷은 아래 **v9 JSON 입출력**으로 전면 개편되었습니다.

---

## 데이터셋 입출력 형식 (v9 — 현행 앱 연동 포맷)

v1~v8의 자연어/4축(1–5) 스키마를 **JSON-in / JSON-out**으로 전면 개편한 현행 포맷. 앱이 모델 출력을 그대로 파싱해 캘린더에 렌더링한다.

### 입력 `ScheduleInput`

```json
{
  "current_time": "2026-03-21T08:00:00+09:00",
  "user_persona": {
    "occupations": ["마케팅 매니저"],
    "detailed_status": "신제품 출시를 앞두고 캠페인 성과를 관리하는 중...",
    "age": 34, "gender": "female",
    "location": {"country": "South Korea", "city": "서울 강남구"},
    "bio": "데이터 기반 의사결정을 선호하는 실무형 리더...",
    "availability": "09:00-18:00"
  },
  "tasks": [
    {"task_id": "task_001", "title": "분기 실적 보고서 작성", "memo": "미제출 시 위약금 발생",
     "source": "email", "deadline": "2026-03-21T15:00:00+09:00",
     "estimated_duration_minutes": 90}
  ]
}
```

| 필드 | 설명 |
|------|------|
| `current_time` | 현재 시각 (ISO8601, 타임존 포함) |
| `user_persona` | occupations·detailed_status·age·gender·location(country/city)·bio·availability(가용 시간대) |
| `tasks[]` | task_id · title · memo · source · **deadline**(없으면 `null`) · **estimated_duration_minutes** |

### 출력 `ScheduleResponseV9`

```json
{
  "scheduled_tasks": [
    {
      "task_id": "task_001",
      "title": "분기 실적 보고서 작성",
      "priority_rank": 1,
      "scoring": {"deadline_proximity": 9.0, "task_importance": 9.0,
                  "task_chaining": 2.0, "urgency": 9.5, "total_score": 8.1},
      "reasoning": {"summary": "마감 7시간 전이고 위약금 리스크가 있어 최우선...", "chaining_detail": ""},
      "recommended_schedule": {"start_time": "2026-03-21T08:00:00+09:00",
                               "end_time": "2026-03-21T09:30:00+09:00"}
    }
  ]
}
```

| 필드 | 설명 |
|------|------|
| `priority_rank` | 실행 순서 (1..N 순열) |
| `scoring` | 4축 **0–10 실수** + `total_score`(가중 종합) |
| `reasoning` | `summary`(점수·순위 근거) + `chaining_detail`(체인일 때만, 독립이면 `""`) |
| `recommended_schedule` | 가용 시간대 내 **비중복 시간블록** (start/end, ISO8601) |

**채점 4축 (0–10)**: `deadline_proximity`(마감 임박도) · `task_importance`(목표 영향·리스크 시 가산) · `task_chaining`(후속 작업 블로킹) · `urgency`(즉시 착수 필요도)

**total_score** = `0.30·urgency + 0.25·deadline_proximity + 0.30·task_importance + 0.15·task_chaining` (평가 시 4축에서 결정적 재계산 — 모델은 4축만 정확하면 됨)

**규칙**: `priority_rank`는 total_score 내림차순(단, **체인 선행이 후행보다 먼저**, 지난 고정 일정은 최하위) · 시간블록은 `current_time` 이후·가용시간 내·서로 겹치지 않게·가능하면 각 task의 `deadline` 전에 완료.

> 데이터셋: HF 공개 [pieroot/timesorter-scheduler-v9-ko](https://huggingface.co/datasets/pieroot/timesorter-scheduler-v9-ko) (SFT 1,542 / DPO 7,438). 구성·생성·검수 상세는 [docs/DATASETS.md](docs/DATASETS.md). **다국어 확장**(EN-US 등 Nemotron 국가별 페르소나 기반) 진행 중.

---

## v9 모델 성능 (4B / 2B, n=50)

신 스키마(v9) 자동 채점 — `verify_chosen_v9`(구조 규칙 무위반) + gold 대비 점수 오차(낮을수록↓). 동일 입력·greedy 디코딩, KR 데이터.

| 지표 | 4B base | 4B SFT | 4B DPO | 2B base | 2B SFT | 2B DPO |
|---|---|---|---|---|---|---|
| parse_rate | 0.82 | 0.90 | 0.92 | 0.20 | 0.92 | 0.94 |
| verify_pass | 0.10 | **0.47** | 0.46 | 0.00 | 0.09 | 0.09 |
| chain_order | 0.29 | 0.53 | 0.52 | 0.30 | 0.30 | 0.34 |
| sched_feasible | 0.98 | 1.00 | 1.00 | 0.90 | 0.94 | 0.96 |
| deadline_met | 0.93 | 1.00 | 1.00 | 0.50 | 0.91 | 0.94 |
| rank_exact | 0.40 | 0.59 | 0.59 | 0.29 | 0.39 | 0.38 |
| axis_mae↓ | 2.70 | 0.67 | 0.74 | 3.20 | 1.76 | 1.75 |
| total_mae↓ | 1.95 | 0.66 | 0.72 | 2.14 | 1.63 | 1.60 |

- **SFT가 1차 효과**: base→SFT에서 규칙 통과·점수 정확도가 급상승. 2B는 parse 0.20→0.92로 **형식부터** 학습, 4B는 base가 이미 형식을 알아 규칙·점수에서 향상.
- **DPO ≈ SFT**: 4B·2B 모두 DPO가 SFT 대비 거의 불변(chain_order만 미세 변동). **선호학습(DPO)으로 체인 능력은 안 옮겨진다**는 프로젝트 일관 실증(v7 SFT-only 결정과 동일 결).
- **2B 천장 < 4B**: 2B는 유효 JSON은 잘 뽑지만(parse 0.94) verify_pass 0.09로 4B(0.47)의 1/5 — 모든 규칙 동시충족은 모델 용량에 의존.
- **다국어(EN-US) 파일럿**: 238행 SFT만으로 chain_order 0.61(KR 4B SFT 0.53 상회) — Track B 파이프라인 작동 입증. ([docs/DATASETS.md](docs/DATASETS.md) v9 참고)

---

## 실험 결과 (구 v3 스키마)

held-out 150 시나리오를 골격 규칙(`verify_chosen`)으로 $0·결정론 자동 채점. (지난 일정 순위·당일 시각 순서·체인 연속성·리스크 importance·무마감 1위)

![학습 여정](assets/chart_milestones.png)

### 모델별 통과율 (held-out)

| 모델 | 어댑터 | 전체 | Dated | Intraday | Risk | Relative | Chain | N |
|------|--------|------|-------|----------|------|----------|-------|---|
| Qwen3-4B-Instruct | DPO v3 | 56.7% | 38% | 67% | 95% | 100% | 30% | 150 |
| Qwen3-4B-Instruct | SFT v4 | 77.3% | 77% | 87% | 95% | 100% | 43% | 150 |
| Qwen3.5-4B | no adapter | 0.0% | 0% | 0% | 0% | 0% | 0% | 150 |
| **Qwen3.5-4B** | **SFT+DPO** | **85.3%** | 96% | 90% | 100% | 93% | 47% | 150 |
| **Qwen3.5-9B** | **SFT+DPO** | **88.7%** | 98% | 97% | 95% | 93% | 57% | 150 |

![4B vs 9B](assets/chart_4b_vs_9b.png)

### 핵심 발견

| 단계 | 변화 | 원인 |
|------|------|------|
| DPO v3 → SFT v4 | 56.7% → **77.3%** (+20.6%p) | **데이터 큐레이션 + prompt loss 마스킹** |
| SFT → DPO | ±0 (체인 불변) | DPO는 선호 조정 — 능력 부재(체인)는 못 옮김 |
| 4B → 9B | 85.3% → **88.7%** (+3.4%p, n=150) | 모델 크기는 약점 평준화에만, 체인은 여전히 미해결 |

1. **성능 점프는 모델 구조가 아니라 데이터 품질·loss 마스킹에서.** DPO/GRPO 선호 학습 단독으로는 한계 약점을 못 옮긴다.
2. **파인튜닝의 1차 효과는 출력 스키마 준수.** no-adapter는 추론은 하지만(내용 34%) JSON 규격을 안 지켜 0% — SFT가 앱이 파싱할 규격으로 고정. ([상세](presentation/01_model_comparison/content_analysis.md))
3. **의존성 체인(47~57%)은 4B·9B 모두 미해결** — 모델 크기가 아닌 체인 SFT 데이터 보강이 필요.

> 상세 분석: [docs/VALIDATION.md](docs/VALIDATION.md) · 실험 보고서: [experiments/](experiments/) · 발표자료: [presentation/](presentation/)

---

## 학습 · 사용 방법

### 1. 환경 설정

```bash
make setup-dgx       # Linux CUDA (또는 setup-mac / docker-build)
```

`.env`:
```
OPENAI_API_KEY=sk-...   # 데이터 생성
HF_TOKEN=hf_...         # 모델·데이터 다운로드
HF_HOME=models          # 로컬 캐시
```

### 2. 데이터 준비

```bash
make download          # HF 데이터셋
make download-models   # Qwen3.5 가중치
```

데이터셋은 HF에 버전별로 공개: [SFT](https://huggingface.co/datasets/pieroot/timesorter-sft-ko) · [DPO](https://huggingface.co/datasets/pieroot/timesorter-dpo-ko)

### 3. 학습

```bash
uv run python scripts/train_sft.py    # SFT (Qwen3.5-4B, curated + loss 마스킹)
uv run python scripts/train_dpo.py    # DPO (on-policy + hard tier)
```

### 4. 추론 · 검증

```bash
# 추론
uv run python -m timesorter.infer --adapter outputs/dpo_q35_4b_v5 --schema-version v3 \
  --persona "직장인" --today 2026-06-15 \
  --prompt "보고서 마감(오늘 17시), 팀 회의(14시), 메일 답장 3건"

# 검증 (held-out 자동 채점)
bash scripts/validate_model.sh outputs/dpo_q35_4b_v5
```

학습 어댑터도 HF 공개: [SFT v4](https://huggingface.co/pieroot/timesorter-qwen3.5-4b-sft-v4) · [DPO v5](https://huggingface.co/pieroot/timesorter-qwen3.5-4b-dpo-v5)

---

## 문서

| 문서 | 내용 |
|------|------|
| [docs/DATASETS.md](docs/DATASETS.md) | 데이터셋 v1~v9 구성·차이·생성 방법 (v9 신 스키마 포함) |
| [VERSIONING.md](VERSIONING.md) | 버전별 증분 규칙 · HuggingFace 사용법 |
| [docs/TRAINING.md](docs/TRAINING.md) | 학습 설정·하드웨어·검증·모듈 구조 |
| [docs/VALIDATION.md](docs/VALIDATION.md) | 상세 검증 분석·학습 지표 이력 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 개선 계획·남은 한계 |
| [docs/SETUP.md](docs/SETUP.md) · [docs/SERVING.md](docs/SERVING.md) | 환경 설정 · vLLM 서빙 |
| [experiments/](experiments/) | 실험별 상세 보고서 (RTX 12GB·4090 9B) |
| [presentation/](presentation/) | 발표자료 (PPT) + 모델 비교 분석 |
