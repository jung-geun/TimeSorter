# 검증 결과 상세 분석

> 핵심 벤치마크 결과(통과율 표·차트)는 [README](../README.md) 참고. 이 문서는 상세 분석·이력.

## v4에서 나타난 특징과 변화 (2026-06-12)

**v4는 모델 구조가 아니라 "무엇으로, 어떻게 학습하느냐"를 바꾼 사이클이다.**

1. **데이터 선별 (curated tier)**: Opus 4.8 의미 감사로 v2 데이터 결함(persona_fit 3.3, 표본 35% 페르소나-할일 불일치, 비스케줄 오염 3,943행) 정량 확인 → 골격 검증 + persona_fit 4.9-5.0인 v3-v5만으로 재구성 (6,056행 = curated 3,356 + refusal 1,200 + v2_schedule 1,500).
2. **프롬프트 loss 마스킹**: ~600토큰 시스템 프롬프트에 걸리던 loss 제거(`completion_only_loss`) → 학습 밀도 ~2.5배.
3. **생성 경로 이원화**: OpenAI(gpt-5.4) + Claude(Sonnet 4.6/Opus 4.8) 병렬 — 같은 골격 검증 통과로 품질 수렴.
4. **신규 시나리오 3종**: past_split·no_today·am_escalation.

**held-out 150 실측 (dpo_v3 → sft_v4)**

| 지표 | v3 | v4 | 변화 |
|------|----|----|------|
| 전 규칙 통과율 | 56.7% | **77.3%** | **+20.6%p** |
| guard rerank | 62.7% | **78.7%** | +16.0%p |
| past_rank 위반 | 52건 | **15건** | -71% |
| dated_mixed 통과 | 38% | **77%** | 날짜 추론 |
| intraday 통과 | 67% | **87%** | 오전/오후 |

**바뀌지 않은 것 (v6 과제)**: dependency_chain 통과율 — 체인은 소량 DPO·GRPO 모두에서 불변. 선호 학습이 아니라 체인 SFT 데이터 보강이 필요.

---

## DPO 학습 지표 이력

### Qwen3.5-4B DPO v5
train_loss 0.166 · reward_acc 98.9% · margin 3.50 (on-policy hard-negative라 분리 큼)

### Qwen3-4B DPO v3 (116 steps)
train_loss 0.635 · accuracies 0.876 · margins 0.164 — 형식 동일·내용만 틀린 hard negative라 margins 낮게 유지(정상)

### Qwen3-4B DPO v2 (956 steps)
train_loss 0.0043 · accuracies 1.0 · margins 17.7 — 형식·길이 차이만 배운 과분리(검증 FAIL)

### Mac MPS (초기, 300샘플 5epoch)
SFT v1 loss 0.977/acc 76.5% · SFT v2 loss 0.415/acc 90.0%

> epoch별 전체: [TRAINING_LOG.md](TRAINING_LOG.md) · wandb: https://wandb.ai/pieroot-pieroot/drl-qwen3

---

## gpt-5.5 교차 검증 (오늘=2026-05-24 주입)

| 어댑터 | 커버리지 | 우선순위 | 추론 | 4축일관성 | 종합 | 판정 |
|--------|--------|--------|-----|---------|------|------|
| SFT v1 / DPO v1 | 3/5 | 2/5 | 2/5 | — | 2/5 | ❌ FAIL |
| SFT v2 / DPO v2 | 4/5 | 2/5 | 2/5 | 2/5 | 2/5 | ❌ FAIL |
| SFT v3 + rerank | 3/5 | 3/5 | 2/5 | 3/5 | 3/5 | 🔶 PARTIAL |
| **SFT v4 + rerank** | **4/5** | **3/5** | **3/5** | 2/5 | **3/5** | **🔶 PARTIAL** |

> 판사 분산 주의: 실행마다 기준이 흔들림 — n=1 판사 점수보다 held-out 골격 통과율이 신뢰할 회귀 지표.
> 방법: Phase 1 gpt-5.5가 원본 독립 분석해 기준 답 생성 → Phase 2 모델 출력 비교 채점. `--today`로 날짜 주입, `+rerank`=ScoreRanker 후처리.

---

## 위반 유형별 변화 (Qwen3-4B, 150샘플)

| 위반 | DPO v3 | GRPO v4 | SFT v4 | DPO v5 |
|------|--------|---------|--------|--------|
| `past_rank` | 52 | 44 | **15** | 15 |
| `chain` | 39 | 38 | 30 | **29** |
| `intraday_order` | 12 | 12 | 5 | 5 |
| **합계** | **105** | **96** | **51** | **50** |

---

## 포맷(스키마) vs 추론(내용) 분리 채점

no-FT 모델은 `tasks`를 `["문자열"]`로 출력해 스키마 채점이 전부 실패하지만, `priority_order`·`scores`는 정상 출력한다. tasks id를 위치 기반 재구성하면 추론 내용만 따로 채점 가능(`scripts/content_eval.py`).

| 모델 (n=150) | 스키마 | 내용 |
|------|-------|------|
| Qwen3.5-4B (no FT) | 0.0% | **34.0%** |
| Qwen3.5-4B-Base (no FT) | 14.7% | 31.3% |
| + SFT v4 / + DPO v5 | 85.3% | 85.3% |

> "0%"는 거의 전부 포맷 문제 — 추론 능력의 1/3은 이미 있었고, 파인튜닝이 ① 앱 규격으로 고정 + ② 날짜·체인 약점에 실제 추론 보강. 상세: [presentation/01_model_comparison/content_analysis.md](../presentation/01_model_comparison/content_analysis.md)

---

## (기록) v2 검증 3대 실패 모드 — v3에서 해결

1. **날짜 혼동** (가장 심각): 이메일 날짜(5/23)와 추론 시점(5/24) 미구분 → 지난 일정에 urgency=5 부여. 원인: 학습 데이터에 `today ≠ email_date` 케이스 부재. **해결**: 시스템 프롬프트에 오늘 날짜 고정 주입 + 지난 일정 강등 규칙.
2. **오전/오후 마감 미반영**: 같은 날 오전 마감이 17:00 마감보다 임박함을 미인식. **해결**: 시각 기준 차등 채점.
3. **태스크 분리 비일관**: 작성→업로드→발송 체인을 1·7·8위로 분산. **해결**: dependency 4-5 + 연속 배치 규칙 + order_score_mismatch DPO.
