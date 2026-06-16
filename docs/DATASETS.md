# 데이터셋 구성 (v1 ~ v7)

> 버전별 증분·HuggingFace 사용법은 [VERSIONING.md](../VERSIONING.md) 참고.
> - SFT 증분: [pieroot/timesorter-sft-ko](https://huggingface.co/datasets/pieroot/timesorter-sft-ko)
> - DPO 증분: [pieroot/timesorter-dpo-ko](https://huggingface.co/datasets/pieroot/timesorter-dpo-ko)

## v1 — 자유 텍스트 우선순위 응답

| 파일 | 행 수 | 설명 |
|------|-------|------|
| `scheduler_ko.parquet` | 1,200 | GPT 생성 한국어 스케줄 기본 셋 |
| `scheduler_generic.parquet` | 3,000 | 다양한 일상 태스크 확장 |
| `scheduler_nemotron_r2~r4.parquet` | 각 ~2,500 | Nemotron 페르소나 기반 다양화 3라운드 |
| **`scheduler_ko_combined.parquet`** | **5,999** | v1 SFT 통합 |

```
1) 보고서 마감 - 외부 고객 신뢰와 직결된 마감이라 가장 우선합니다.
2) 팀 회의 - 협업에 필수적인 정보 공유 자리입니다.
```

## v2 — 4축 점수 JSON 응답

| 파일 | 행 수 | 설명 |
|------|-------|------|
| `scheduler_v2_regen.parquet` | ~6,000 | v1 데이터를 v2 JSON으로 재생성 |
| `scheduler_v2_nemotron_extra.parquet` | ~3,000 | Nemotron 페르소나 v2 추가 |
| **`scheduler_v2_combined.parquet`** | **10,958** | v2 SFT 통합 |
| `dpo_pairs_v2.parquet` | 15,338쌍 | 선호/비선호 응답 쌍 (v2 JSON) |

```json
{"tasks":[{"id":1,"text":"보고서 마감"}],"priority_order":[1],
 "scores":[{"task_id":1,"urgency":5,"importance":5,"dependency":4,"time_constraint":2,"reason":"내일 마감"}]}
```

## v3 — 시나리오 골격 기반 (날짜 추론·마감 세분화·의존성·리스크)

| 파일 | 행 수 | 설명 |
|------|-------|------|
| `scheduler_v3.parquet` | 1,978 | 골격 검증 통과 신규 시나리오 (today + meta 골격) |
| **`scheduler_v3_combined.parquet`** | **5,678** | v3 신규 + v2 replay |
| `dpo_pairs_v3.parquet` | 3,722쌍 | hard negative 5종 + v2 replay 1.5K |
| `scheduler_v3_eval.parquet` | 150 | held-out 평가셋 (seed 47) |

## v4 — 이중 생성 경로 + 신규 시나리오 3종

OpenAI(gpt-5.4-mini 텍스트 + gpt-5.4 chosen)와 Claude(Sonnet 4.6 / Opus 4.8)가 같은 골격을 병렬로 채움 → 둘 다 `verify_chosen()` 통과 필수(97-98%).

| 파일 | 행 수 | 설명 |
|------|-------|------|
| `scheduler_v4_extra.parquet` | 989 | OpenAI + Claude 병렬 생성 |
| `dpo_pairs_v4_extra.parquet` | 1,978쌍 | hard negative 5종 |

| 신규 시나리오 | 행 수 | 핵심 신호 |
|---------|------|----------|
| `past_split` | 249 | 지난/유효 일정 혼합 — 지난 일정 하위 배치 |
| `no_today` | 245 | 오늘 미상 — '지남' 단정 금지 |
| `dated_mixed` | 247 | 절대 날짜·시각 혼재 |
| `am_escalation` | 248 | 오전 마감 + 에스컬레이션 → urgency=5 |

## v5 — on-policy DPO + 의존성 체인 보강

| 파일 | 행 수 | 설명 |
|------|-------|------|
| `scheduler_v5_claude.parquet` | 389 | Claude 생성 — dependency_chain 비중 확대 |
| `dpo_pairs_v5.parquet` | 1,368쌍 | hard negative 6종 + on-policy 228 + refusal 200 |
| `dpo_pairs_v5_onpolicy.parquet` | 351쌍 | on-policy 전용 (SFT v4 모델 실제 오류) |

**v5 DPO 카테고리:** onpolicy 228 · order_score_mismatch 230 · granularity_swap 220 · dependency_scatter 185 · date_confusion 177 · risk_ignore 92 · past_hallucination 36 · refusal 200 = **1,368**

## v6 — 통합·검수 자립형 데이터셋 (권장)

v2~v5 전체를 검수·통합한 단일 데이터셋. **v6 단독으로 학습 가능**(증분 누적 불필요).

| 구분 | 행 수 | 구성 |
|------|------|------|
| SFT v6 | 14,314 | 검수 v2 10,958 + 큐레이션 v3-v5 3,356 |
| DPO v6 | 17,894 | 검수 v2 15,321 + 큐레이션 v3/v5 2,573 |

v2 구간은 opus 하위 에이전트 264개로 전수 검수·수정(SFT priority↔점수 1,188건, DPO 4,732건). drop 0(최대한 보존). 상세: [VERSIONING.md](../VERSIONING.md).

## v7 — 의존성 체인 특화 보강 (페르소나별 복잡 스케줄)

v6의 유일한 미해결 약점인 **의존성 체인(47~57%)** 을 타깃해, 페르소나별 4–5단계 긴 체인과 다중(최대 2개) 체인이 섞인 복잡 스케줄을 신규 시나리오 `dependency_chain_complex`로 대량 생성.

| 파일 | 행 수 | 설명 |
|------|------|------|
| `scheduler_v7_chain.parquet` | **968** | 체인 특화 SFT 증분 (검수 통과분) |
| `scheduler_v7_chain_eval.parquet` | 50 | 체인 전용 held-out 평가셋 (seed 777, 학습 미사용) |
| `hf_versioned/sft/v7_selfcontained.parquet` | 15,282 | v6(14,314) + 체인(968) 자립형 |

**DPO는 v6 그대로** — 체인은 선호 학습(DPO)으로 못 옮긴다는 실증(SFT=DPO 불변)에 따라 v7은 **SFT-only 증분**.

**생성·검수 (2단계 게이트, 1,400 생성 → 968 수록, ~70%)**:
1. **생성**: Claude Sonnet 4.6 + Haiku 4.5 하위 에이전트가 골격을 채움. Haiku는 복잡 골격에서 템플릿 반복·구조 무시가 잦아 대부분 Sonnet으로 재생성(품질 격차 실측).
2. **결정론 검증** (`verify_chosen`): 체인 연속 배치·단계 순서·선행 dependency≥4 등 골격 규칙. 1,400→1,385 통과.
3. **opus 블라인드 검수**: **텍스트만** 보고 체인을 복원하게 한 뒤 골격 정답과 대조. 텍스트만으로 체인·순서가 복원 안 되는 행("독심술 학습")을 제거 — 50행/배치 단위로 신뢰도 확보(94~98%, 100행 배치는 노이즈 커서 폐기).
4. **dedup**: train 내부 중복 + train/eval 누출 제거. 전역 고유 태스크 92.5%.

> 핵심 교훈: `verify_chosen`(라벨↔골격 정합)만으로는 "텍스트에 체인 신호가 없는데 라벨만 맞는" 행을 못 거른다. opus 블라인드 검수가 그 갭을 막는 핵심 게이트. 체인 개선 수치는 학습 후 v6모델 vs v7모델을 이 held-out 50으로 비교해야 확정.

---

## v2 ↔ v3 설계 철학 차이 (실측)

**v2는 "GPT가 만든 그럴듯한 정답"을 그대로 믿었고, v3는 "프로그램이 정한 골격"을 GPT가 채운 뒤 골격으로 재검증했다.**

### SFT 데이터 (v2 9,758행 vs v3 신규 1,978행)

| 지표 | v2 | v3 | 의미 |
|------|----|----|------|
| 절대 날짜 포함 프롬프트 | 1.9% | **85.8%** | v2는 날짜 추론 재료 없음 |
| 시각(HH:MM) 표현 | 30.3% | **95.9%** | 오전/오후 세분화 가능 |
| 리스크 문구 | 0.2% | **15.2%** | importance 상향 트리거 |
| `today` 컬럼 | 없음 | **전 행** | 학습·추론 날짜 일치 |
| dependency 1점 비율 | 39% | 24% | 체인이 의존성 축 사용 |

### DPO 쌍 (v2 15,338쌍 vs v3 3,722쌍)

| 지표 | v2 | v3 | 의미 |
|------|----|----|------|
| rejected가 chosen과 동일 스키마 | 56% | **84%** | v2는 형식 위반 다수 |
| rejected/chosen 길이비 | 0.77 | **1.21** | v2 rejected는 짧아 길이로 구분 가능 |
| 학습 결과 rewards/margins | 17.7 (과분리) | 0.16 | v3는 끝까지 "어려운" 구분 학습 |

> v2 DPO가 accuracies 1.0으로 "완벽 수렴"하고도 검증 실패한 이유 — 형식·길이만 배우면 만점이 나오는 쉬운 문제였다. v3는 같은 형식·길이에서 내용만 틀린 쌍이라 margins가 낮게 유지되는 게 정상이다.

---

## 데이터 구축 방법 (골격 우선, skeleton-first)

1. 프로그램이 시나리오 골격 확정 — 태스크별 마감·과거 여부·의존 체인·리스크 문구
2. LLM은 텍스트(gpt-5.4-mini)와 정답 JSON(gpt-5.4 / Claude Sonnet 4.6)만 생성
3. 골격 규칙 자동 검증(`verify_chosen`) 통과분만 수록 (위반 시 1회 피드백 재생성, 실패 시 폐기)
4. 골격은 `meta` 컬럼에 보존 — DPO negative 생성·자동 평가(`eval_scheduler.py`)·GRPO 보상에 재사용

원본: `nvidia/Nemotron-Personas-Korea`(페르소나) + `anakin87/events-scheduling`(이벤트) → 한국어 할 일로 번역·증강 (상세: [VERSIONING.md](../VERSIONING.md)).

---

## 용도별 데이터셋 특징 (SFT / DPO / GRPO)

> 감사 상세: [DATASET_AUDIT.md](DATASET_AUDIT.md)

### SFT
- **골격 우선 생성** — 골격 규칙 통과분만 수록. `today`·`meta`·`version` 컬럼 포함.
- **tier**: `curated`(v3-v5, persona_fit 4.9-5.0, 본 학습 권장) / `rework`(v1/v2 재가공) / `v2_refusal`(거부, 항상 혼합) / `v2_schedule`(persona_fit 3.3, 소량만) / `v2_offformat`(비권장)
- 실증: curated + 프롬프트 loss 마스킹 → held-out 56.7%→77.3% (+20.6%p)

### DPO
- hard negative는 chosen과 **형식·길이 동일(0.96-0.99), 내용만 오류**: date_confusion·granularity_swap·dependency_scatter·risk_ignore·order_score_mismatch·past_hallucination
- on-policy 쌍: 학습된 모델 실제 오류를 rejected로 수집
- **tier**: `hard`(권장) / `refusal` / `easy_format`(v2 — 길이 편향 주의) / `legacy_text`(v1, v3+ 학습 금지)

### GRPO
- 쌍 없이 모델 생성 k샘플을 골격 규칙(`verify_chosen`)으로 채점 — RLVR(결정적·무비용 보상)
- 구성: prompt + persona + today + meta(골격). 보상: 전 규칙 통과 +1.0 / 위반당 -0.3 / 파싱 실패 -1.0

### eval
- 학습 미사용 seed(47) held-out 150. `eval_scheduler.py`가 골격 규칙으로 $0 자동 채점 — 모든 어댑터 비교 기준.

---

## HuggingFace 갱신 절차

```bash
uv run python scripts/build_versioned_datasets.py   # 버전별 증분 빌드
uv run python scripts/upload_hf_versioned.py        # SFT/DPO 2개 repo 업로드
```
