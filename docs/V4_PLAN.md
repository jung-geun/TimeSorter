# v4 개선 사이클 — 학습 구조 재분석과 개선 방법 탐구

> 작성: 2026-06-11. 기준: v3 완료 시점 (gpt-5.5 판사 3/5 PARTIAL, 목표 4/5 PASS)

## 1. 파인튜닝 과정 재분석 — 현재 구조의 빈틈 7가지

### 1.1 SFT가 프롬프트 토큰에도 loss를 건다 (가장 즉효성 큰 수정)

`train_sft.py`는 messages 포맷 데이터셋을 그대로 SFTTrainer에 넘기는데, trl 1.5의
`assistant_only_loss` 기본값이 False라 **시스템 프롬프트(~600토큰) + 유저 프롬프트(~200토큰)
에도 cross-entropy가 걸린다**. 응답(~500토큰)은 전체 시퀀스의 40% 수준 — 학습 신호의 절반
이상이 "매번 동일한 시스템 프롬프트 암기"에 소모된다.

**v4 수정**: `assistant_only_loss: true` (SFTConfig). Qwen3 chat template이
`{% generation %}` 태그를 지원하지 않으면 prompt-completion 포맷으로 변환 후
`completion_only_loss: true` 사용. 같은 epoch 수에서 응답 학습 밀도가 ~2.5배 올라간다.

### 1.2 eval 분리가 전혀 없다

train/eval split 없이 100% 학습, `save_strategy: epoch`로 마지막 체크포인트를 무조건 채택.
과적합 감지·체크포인트 선택 근거가 없다. 판사 검증은 이메일 5건 시나리오 **1개**(n=1)라
분산이 커서(동일 모델이 실행마다 ±1점) 개선 신호로 쓰기 어렵다.

**v4 수정 (구현 완료)**: `scripts/eval_scheduler.py` — 골격(meta)이 있는 held-out 시나리오에
모델을 호출해 verify_chosen() 규칙(지난 일정 순위, 동일일 시각 순서, 체인 연속성·순서,
리스크 importance, 무마감 1위 금지)을 자동 채점. `data/scheduler_v3_eval.parquet`(seed 47,
학습 미사용)로 측정하며, 판사 없이 $0으로 회귀 추적이 가능하다.

#### dpo_v3 held-out 실측 (150 시나리오, 2026-06-11)

| 후처리 | 전 규칙 통과율 | dated_mixed | chain | intraday | relative | risk |
|--------|--------------|------------|-------|----------|----------|------|
| 없음 (모델 원순서) | 56.7% | 38% | 30% | 67% | 100% | 95% |
| 전면 rerank (가중합) | **43.3% ↓** | 30% | 10% | 37% | 100% | 91% |
| **guard rerank (지난 일정만 강등)** | **62.7% ↑** | 55% | 30% | 67% | 100% | 95% |

**핵심 발견 — 전면 rerank는 역효과**: 1-5점 4축은 같은 날 내 시각 순서·체인 단계 순서를
표현하지 못하므로, 가중합 정렬이 모델이 priority_order에 담은 그 정보를 파괴한다
(chain 30%→10%, intraday 67%→37%). 판사 시나리오에서 rerank가 유효했던 것은 해당 출력의
순서가 통째로 역전된 특수 케이스였기 때문. → v4부터 기본 후처리는 `rerank_guard`
(urgency≤1 AND time_constraint≤1 시그니처만 최하위 강등, 모델 순서 보존)로 하고,
전면 rerank는 스키마 v4의 deadline 필드가 생겨 D-day 산술이 가능해진 뒤에만 사용한다.

**남은 위반의 성격** (guard 기준): past_rank 41건 중 다수는 모델이 지난 일정에
u1/t1 시그니처를 주지 못한 채점 오류(가드가 못 잡음), chain 39건은 체인 연속성·순서의
모델 수준 오류 — 각각 A1·C2(on-policy DPO)의 직접 타깃이다.

### 1.3 판사 시나리오가 1개뿐

sample_emails 5건(직장인 도메인)이 유일한 E2E 시나리오. 한 도메인에 과적합된 판단이다.
**v4 수정**: 학사 일정/프리랜서 계약/의료 예약 등 3-5개 시나리오 세트를 만들어 판정 평균.
각 세트는 이메일 Date ≠ 오늘 케이스를 반드시 포함.

### 1.4 추출기가 품질 상한을 결정하는데 학습 대상이 아니다

coverage 3/5의 원인은 모델이 아니라 gpt-5.4-mini 추출기였다 (미팅 "참석"을 "회신"으로 대체,
원문 세부 항목 손실). 추출은 프롬프트 의존이라 버전 관리도 안 된다.
**v4 옵션**: (a) 추출 태스크를 같은 4B 모델에 멀티태스크로 학습 (이메일 원문 → tasks JSON,
~1K 합성 이메일) — 온디바이스 단독 동작 달성, (b) 추출 스키마를 구조화해 마감을 ISO 필드로
분리 (자유 텍스트 괄호 표기보다 손실 적음). 권장: (b) 먼저, (a)는 v5.

### 1.5 스키마에 구조화된 deadline 필드가 없다

모델이 마감을 reason 텍스트에만 남기므로 후처리(rank.py)가 날짜를 알 수 없고, 발표 자료의
D-day 감쇠표(D+ 1.00 / D-0 0.95 / D-1 0.85 / …)를 적용할 수 없다. "지난 일정 처리"도
모델의 텍스트 추론에만 의존한다.

**v4 수정 (스키마 v4)**: `scores[].deadline: "YYYY-MM-DD HH:MM" | null` 추가.
ScoreRanker가 (오늘, deadline)에서 deadline_score를 결정적으로 계산해 가중합에 편입 —
지난 일정 강등이 모델 실수와 무관하게 보장된다. 학습 데이터는 골격에 deadline이 이미
있으므로 재생성 비용이 거의 없다 (chosen JSON에 필드만 추가).

### 1.6 DPO negative가 off-policy다

v3 hard negative는 프로그램이 chosen을 변형해 만든 것이라 **모델이 실제로 저지르는 오류
분포와 다르다** (예: SFT v3가 보인 "전체 역순 출력"은 negative에 없던 패턴).
**v4 수정 — on-policy iterative DPO**: eval_scheduler 실행에서 규칙 위반한 모델 출력
자체를 rejected로, 골격 준수 출력(또는 rerank 교정본)을 chosen으로 수집해 1라운드 더 학습.
모델의 진짜 실수를 직접 벌점한다.

### 1.7 검증 가능한 보상 함수가 있는데 RL을 안 쓰고 있다

verify_chosen()은 **결정적·무비용 보상 함수**다 (RLVR 조건 충족). DPO의 쌍 구성 없이
GRPO(trl `GRPOTrainer`)로 직접 최적화 가능: 프롬프트당 k개 샘플 생성 → 위반 수 기반 보상 →
정책 업데이트. 판사·GPT 비용 $0, 골격 규칙이 곧 보상.
**제약**: 12GB에서 4B 생성 k개 + 학습 동시 수행은 빠듯 — vLLM rollout 서버 연동 또는
DGX에서 수행. v4에서는 소규모 파일럿(k=4, 500 프롬프트)으로 효과 검증 후 v5 확대.

### (기타 기록)

- 추론은 vLLM `guided_json` 제약 디코딩, 학습은 자유 디코딩 — 분포 불일치는 작지만 존재
- DPO `precompute_ref_batch_size=1`이 wall clock의 절반을 차지 (12GB 한계, DGX에서 해소)
- rank.py 가중치(0.40/0.25/0.20/0.05/0.10)는 수동값 — 판사 기준 순위에 그리드 서치 여지

## 2. v4 사이클 설계 (우선순위순)

| 순서 | 작업 | 기대 효과 | 비용 |
|------|------|----------|------|
| A1 | **오전 마감+에스컬레이션 urgency=5 시나리오 500행** (PR류 — 판사 1위 불일치 직접 대응) | priority 2→3 | ~$2 |
| A2 | 지난 일정 reason 자연화 — 텍스트 생성 가이드에 "어제/그제 마감이 지난" 표현 강제 | reasoning 2→3 | $0 (재생성 시 포함) |
| A3 | 추출 스키마 구조화 (마감 ISO 필드, 참석/회신 구분 규칙) + 판사 시나리오 3세트 | coverage 3→4 | ~$1 |
| B1 | `assistant_only_loss` + eval split + eval_scheduler 기반 체크포인트 선택 | 학습 효율 ~2.5× | $0 |
| B2 | 스키마 v4: deadline 필드 + ScoreRanker D-day 감쇠 | 날짜 안전장치 결정화 | $0 |
| C1 | SFT v4 (v3 데이터 + A1·A2, 1.5-2 epochs) | | GPU ~2h |
| C2 | **on-policy DPO**: eval 위반 출력 = rejected 수집 → 1 epoch | 실제 오류 분포 벌점 | GPU ~1.5h |
| C3 | (파일럿) GRPO k=4 × 500 프롬프트, verify_chosen 보상 | RLVR 검증 | GPU ~3h |
| D | held-out 150 골격 통과율 + 판사 3세트 평균으로 판정 | | ~$3 |

**PASS 기준 제안**: held-out 골격 전 규칙 통과율 ≥ 90% (rerank 포함) AND 판사 3세트 평균 ≥ 4/5.

## 3. 역할 분담 다이어그램 (v4 목표 구조)

```
이메일/캘린더 ─→ 추출(구조화 스키마, 마감 ISO) ─→ 4B 모델(4축 점수 + deadline 필드)
                                                      │
                            ScoreRanker(가중합 + D-day 감쇠) ←─ 오늘 날짜
                                                      │
                                  우선순위 출력 ─→ 사용자 수정 ─→ FeedbackRecord
                                                                      │
                eval_scheduler(골격 보상) ←─ on-policy DPO/GRPO ←─────┘
```

모델은 "채점"에 집중하고, "정렬·날짜 산술"은 결정적 코드가 보장하는 구조 —
발표 자료의 Feature Generator/ScoreRanker 설계와 학습 모델의 분별력을 결합한 최종 형태다.
