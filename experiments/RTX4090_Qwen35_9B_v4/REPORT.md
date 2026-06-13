# TimeSorter — RTX 4090 × 1 | Qwen3.5-9B | SFT v4 + DPO v4 실험 보고서

> **실험 환경**: RTX 4090 (24 GB VRAM) × 1 GPU · Qwen3.5-9B · QLoRA 4-bit (NF4)  
> **데이터셋**: `sft_v4_train` (6,056행) → `dpo_pairs_v4_extra` (1,978쌍)  
> **목적**: RTX 12GB Qwen3-4B 대비 상위 모델에서의 동일 파이프라인 성능 검증

---

## 1. 실험 설정

### 1-1. 모델 · 하드웨어

| 항목 | 값 |
|------|----|
| 베이스 모델 | `Qwen/Qwen3.5-9B` |
| 파라미터 수 | 9.4B |
| 어휘 크기 | 248,064 (Qwen3.5 신규 vocab) |
| VRAM | 24 GB (RTX 4090) |
| 정밀도 | QLoRA 4-bit NF4 (bf16 compute) |
| LoRA rank | r=16, α=32, dropout=0.05 |

### 1-2. SFT 하이퍼파라미터 (`configs/sft_4090_1x_9b_v4.yaml`)

| 항목 | 값 |
|------|----|
| 학습률 | 2.0e-5 (cosine, warmup 3%) |
| 배치 크기 | 1 × 32 grad_accum = eff_batch 32 |
| 에폭 | 3 |
| 최대 시퀀스 길이 | 1,536 tokens |
| 옵티마이저 | adamw_8bit |
| 특이사항 | prompt_completion=true (응답 토큰에만 loss) |

### 1-3. DPO 하이퍼파라미터 (`configs/dpo_4090_1x_9b_v4.yaml`)

| 항목 | 값 |
|------|----|
| 베이스 어댑터 | `outputs/sft_4090_1x_9b_v4` |
| 데이터셋 | `dpo_pairs_v4_extra` (1,978쌍) |
| 학습률 | 5.0e-7 (cosine, warmup 5%) |
| 배치 크기 | 1 × 32 grad_accum = eff_batch 32 |
| 에폭 | 1 |
| 최대 시퀀스 길이 | 1,280 tokens (1,536 → OOM으로 축소) |
| β (DPO 온도) | 0.1 |
| 특이사항 | precompute_ref_log_probs=true (ref forward 사전 계산) |

---

## 2. v4 데이터셋 구성

### 2-1. SFT 학습 데이터 (`sft_v4_train`, 6,056행)

| 출처 | 행 수 | 설명 |
|------|-------|------|
| curated v3-v5 | 3,356 | persona_fit 4.9-5.0, 골격 검증 통과 |
| refusal | 1,200 | 비할일 입력 거절 예시 |
| v2_schedule 표본 | 1,500 | 구버전 호환 다양성 확보 |

**v4 핵심 개선**: OpenAI + Claude 이중 생성 경로, prompt_completion loss 마스킹 (학습 밀도 ~2.5× 향상)

### 2-2. DPO 학습 데이터 (`dpo_pairs_v4_extra`, 1,978쌍)

| 유형 | 쌍 수 | 설명 |
|------|-------|------|
| order_score_mismatch | 점수와 순서가 모순된 rejected | hard negative |
| granularity_swap | 마감 단위(시각↔날짜) 혼동 | hard negative |
| date_confusion | 날짜 혼동 (어제/내일 뒤바꿈) | hard negative |
| past_hallucination | 지난 일정 상위 배치 | 172쌍 |
| risk_ignore | 리스크 조건 무시 | hard negative |

---

## 3. 평가 방법론

### 3-1. 골격 규칙 기반 자동 채점 (주 지표)

held-out 150개 시나리오에 대해 5가지 규칙 위반 여부 자동 채점:

| 규칙 | 내용 |
|------|------|
| `past_rank` | 지난 일정이 유효 일정보다 상위 배치 |
| `intraday_order` | 당일 마감이 이른 일정이 늦은 일정보다 후순위 |
| `chain` | 의존성 체인의 단계 순서 위반 |
| `risk_importance` | 리스크 조건 있는 태스크 중요도 미반영 |
| `none_first` | 마감 없는 태스크가 1위에 배치됨 |

### 3-2. 평가 데이터 분포 (held-out 150개)

| 시나리오 | 개수 | 설명 |
|----------|------|------|
| `dated_mixed` | 53 | 절대 날짜 혼재 (과거/오늘/미래) |
| `intraday` | 30 | 당일 시각 순서 |
| `dependency_chain` | 30 | 선후관계 체인 |
| `risk` | 22 | 리스크 조건부 태스크 |
| `relative` | 15 | 상대 날짜 표현 |

---

## 4. 예상 실험 결과 (실행 대기)

> ⚠️ 이 섹션은 실험 실행 전 기준치 예측입니다. 실제 결과로 업데이트 필요.

### 4-1. 기준선 (RTX 12GB · Qwen3-4B-Instruct)

| 모델 | 전체 통과율 | dated | intraday | risk | relative | chain |
|------|------------|-------|----------|------|----------|-------|
| SFT v4 (4B baseline) | **77.3%** | 77% | 87% | 95% | 100% | 43% |
| DPO v5 (4B baseline) | **77.3%** | 77% | 87% | 95% | 100% | 43% |

### 4-2. 9B 모델 기대치

9B 모델은 4B 대비 파라미터 2.3× 증가로 다음 개선이 예상됨:
- **dependency_chain**: 추론 깊이 향상으로 43% → 60-70% 예상
- **dated_mixed**: 날짜 계산 정확도 향상으로 77% → 85-90% 예상
- **전체**: 77.3% → 83-88% 예상

---

## 5. 실험 실행 방법

```bash
# SFT 학습
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_sft.py \
  --config configs/sft_4090_1x_9b_v4.yaml

# DPO 학습 (SFT 완료 후)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_dpo.py \
  --config configs/dpo_4090_1x_9b_v4.yaml

# 검증
bash scripts/validate_model.sh outputs/dpo_4090_1x_9b_v4
```

---

## 6. 주요 관찰 사항 (예정)

실험 완료 후 다음 항목 기록:
- [ ] SFT train_loss 및 eval_loss 수렴 곡선
- [ ] DPO reward_accuracy 및 reward_margin 추이
- [ ] 시나리오별 통과율 세부 분석
- [ ] 4B 대비 chain 시나리오 개선 폭
- [ ] 추론 속도 및 메모리 사용량 비교

---

*생성일: 2026-06-13 | 담당: TimeSorter Pipeline*
