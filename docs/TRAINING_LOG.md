# 학습 이력 (Training Log)

Mac (Apple M-series MPS) 환경에서 Qwen3.5-4B LoRA SFT 3회 실험 결과.

---

## 실험 환경

| 항목 | 값 |
|------|----|
| 모델 | Qwen/Qwen3.5-4B |
| 어댑터 | LoRA r=8, alpha=16, dropout=0.05 |
| 학습 방식 | SFT (SFTTrainer, TRL) |
| 디바이스 | Apple MPS (bfloat16, 4bit 미사용) |
| 샘플 수 | 300 (max_samples) |
| 에폭 | 5 |
| 배치 (eff) | 8 (bs=1 × grad_accum=8) |
| lr | 2.0e-5, cosine decay, warmup_ratio=0.03 |
| max_seq_length | 1024 |

---

## 실험 1 — SFT v1 (자유 텍스트 응답)

- **wandb run**: `sft-mac-4b-5ep` (rjibqpc5)
- **데이터셋**: `data/scheduler_ko_combined.parquet` (5,999개 중 300개 샘플)
- **schema_version**: v1 (자유 텍스트 우선순위 목록)
- **train_loss**: 1.295 | **최종 accuracy**: 76.5%
- **출력 형식**: 번호+이름+이유 자유 텍스트

| epoch | loss | token_accuracy |
|-------|------|----------------|
| 0.13 | 2.420 | 51.3% |
| 0.40 | 2.191 | 54.8% |
| 0.67 | 1.951 | 58.7% |
| 1.00 | 1.751 | 62.9% |
| 1.32 | 1.423 | 70.2% |
| 1.59 | 1.278 | 72.8% |
| 2.00 | 1.155 | 74.3% |
| 2.64 | 1.047 | 75.9% |
| 3.00 | 1.025 | 76.3% |
| 3.56 | 0.995 | 76.5% |
| 4.00 | 0.969 | 76.9% |
| 4.75 | 0.987 | 76.6% |
| 5.00 | 0.977 | 76.5% |

**비고**: epoch 2 이후 loss 개선이 완만해져 plateau 진입. 자유 텍스트 형식 학습 성공.

---

## 실험 2 — SFT v2 데이터 불일치 (실패 케이스)

- **wandb run**: `sft-mac-4b-5ep-v2` (cl5jmusx)
- **데이터셋**: `data/scheduler_ko_combined.parquet` (v1 자유 텍스트 응답) + v2 시스템 프롬프트
- **schema_version**: v2 (설정은 v2이나 응답 데이터는 v1)
- **train_loss**: 0.855 | **최종 accuracy**: 87.4%
- **출력 형식**: JSON 파싱 실패 → 자유 텍스트 폴백

| epoch | loss | token_accuracy |
|-------|------|----------------|
| 0.13 | 2.420 | 51.3% |
| 1.00 | 1.751 | 62.9% |
| 2.00 | 0.637 | 86.7% |
| 3.00 | 0.581 | 87.2% |
| 4.00 | 0.569 | 87.4% |
| 5.00 | 0.538 | 87.4% |

**원인**: 시스템 프롬프트는 v2(JSON 요구)이지만 타겟 응답이 v1 자유 텍스트라 모델이 잘못된 매핑을 학습. loss는 낮지만 실제 JSON 미출력.

---

## 실험 3 — SFT v2 정상 (JSON 4축 응답)

- **wandb run**: `sft-mac-4b-5ep-v2-correct` (numjwc18)
- **데이터셋**: `data/scheduler_v2_combined.parquet` (10,958개 중 300개 샘플)
- **schema_version**: v2 (4축 점수 JSON 응답)
- **train_loss**: 0.641 | **최종 accuracy**: 90.0%
- **출력 형식**: JSON 파싱 성공 → 4축 점수 + 우선순위 정렬

| epoch | loss | token_accuracy |
|-------|------|----------------|
| 0.13 | 1.541 | 66.5% |
| 0.40 | 1.389 | 69.1% |
| 0.67 | 1.179 | 73.3% |
| 1.00 | 0.993 | 77.6% |
| 1.32 | 0.742 | 83.1% |
| 1.59 | 0.585 | 87.0% |
| 2.00 | 0.534 | 88.1% |
| 2.64 | 0.457 | 89.3% |
| 3.00 | 0.427 | 89.7% |
| 3.56 | 0.424 | 89.6% |
| 4.00 | 0.417 | 89.8% |
| 4.75 | 0.398 | 90.2% |
| 5.00 | 0.415 | 89.97% |

**비고**: epoch 1.5~2 구간에서 loss가 급감하며 JSON 구조 패턴 습득. epoch 3 이후 수렴.

---

## 3회 실험 비교

| 실험 | 데이터 | train_loss | acc@ep5 | JSON 출력 |
|------|--------|-----------|---------|-----------|
| v1 SFT | scheduler_ko_combined (v1) | 1.295 | 76.5% | — (자유 텍스트) |
| v2 불일치 | scheduler_ko_combined (v1 응답) | 0.855 | 87.4% | 실패 |
| **v2 정상** | **scheduler_v2_combined (v2 JSON)** | **0.641** | **90.0%** | **성공** |

**핵심 교훈**: schema_version은 시스템 프롬프트뿐 아니라 타겟 응답 형식까지 일치해야 함. 데이터셋 선택이 학습 결과를 결정.

---

## 다음 단계 (v1-v2 당시)

- [x] DPO 학습: v7/v8/v9 DPO 파인튜닝 완료
- [x] 더 많은 샘플: v9 SFT 1993행 전체 활용
- [ ] DGX 환경: 9B 모델 재학습 검토 중

---

## 실험 4 — v9 SFT (JSON-in/out, 신 스키마, KR+EN)

- **wandb run**: `sft_q35_4b_v9combined`
- **디바이스**: RTX 3060 12GB (CUDA), int4 + LoRA
- **LoRA**: r=16, alpha=32, dropout=0.05
- **데이터셋**: `data/scheduler_v9_combined.parquet` (KR 1,358 + EN 635 = 1,993행; max_seq=3584로 295행 truncation 제외 → 실 사용 1,528행)
- **schema_version**: v9 (JSON-in/JSON-out, 4축 0-10 scoring, time-block, chain_pairs, is_overdue)
- **에폭**: 2 | **배치**: 1 × grad_accum 16 | **lr**: 2.0e-5, cosine | **max_seq_length**: 3,584
- **VRAM peak**: ~9.8GB (OOM 방지: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`)
- **체크포인트**: `outputs/sft_q35_4b_v9combined/` (epoch 1: checkpoint-96, epoch 2: checkpoint-192)

### 검증기 규칙 (verify_chosen_v9, 8-규칙)
| # | 규칙 | 설명 |
|---|------|------|
| R1 | rank 순열 | priority_rank가 1..N 순열 |
| R2 | 블록 중복 없음 | 시간블록 간 겹침 0 |
| R3 | 소요시간 일치 | end-start ≈ estimated_duration_minutes (±5분) |
| R4 | 마감 실현 | 마감 전 완료 (가능한 경우) |
| R5 | total 재계산 | 4축 가중합 일치 (±0.01) |
| R6 | 체인 선후 | 선행 task 시작 < 후행 task 시작 |
| R7 | chaining_detail 조건부 | 체인이면 비어있지 않아야 함 |
| R8 | is_overdue 긴급 유지 | 마감 초과 + urgency≥8 → rank 높게 유지 |

---

## 실험 5 — v9 DPO (chain_order_break + rank_score_mismatch 집중)

- **wandb run**: `dpo_q35_4b_v9combined`
- **베이스 어댑터**: `outputs/sft_q35_4b_v9combined`
- **데이터셋**: `data/dpo_pairs_v9_focus.parquet` (2,812쌍: chain_order_break + rank_score_mismatch)
- **DPO beta**: 0.1 | **max_length**: 3,584 | **에폭**: 1
- **출력**: `outputs/dpo_q35_4b_v9combined/`

---

## v9 종합 평가 결과

### 이전 평가셋 (50행, KR 전용, `data/scheduler_v9.parquet`)

| 모델 | verify_pass | chain_order | rank_exact | axis_mae |
|------|------------|-------------|------------|----------|
| Base (Qwen3.5-4B) | 9.8% | 29.3% | 0.404 | 2.703 |
| SFT v9 (1,993행) | 46.7% | 53.3% | 0.591 | 0.671 |
| DPO v9 (focus) | 45.7% | 52.2% | 0.589 | 0.743 |

### 신규 평가셋 (59행, KR 31 + EN 28, `data/scheduler_v9_eval.parquet`)

| 모델 | parse_rate | verify_pass | chain_order | rank_exact | axis_mae | total_mae |
|------|-----------|-------------|-------------|------------|----------|-----------|
| Base (Qwen3.5-4B) | 88.0% | 4.5% | 34.1% | 0.363 | 2.949 | 2.134 |
| **SFT v9combined** | **96.0%** | **66.7%** | **77.1%** | **0.682** | **0.800** | **0.826** |
| DPO v9combined | 98.0% | 65.3% | 75.5% | 0.667 | 0.823 | 0.849 |

> 평가 완료: 2026-06-23 | SFT: verify_pass +20pp vs 구 v9 | DPO ≈ SFT (동일 패턴)

---

## 데이터셋 증강 계획 (v9.1)

| Phase | 목표 | 방법 |
|-------|------|------|
| Phase 1 | EN +650행 (`build_en3` 63배치 채우기) | haiku+sonnet fill → assemble_v9 |
| Phase 2 | is_overdue +300행 (KR/EN 각 150) | overdue_ratio=0.4 scaffold |
| Phase 3 | DPO hard negative +500쌍 | overdue_rank_drop 카테고리 |
| Phase 4 | 고복잡도(10-13태스크) +100행 | n_tasks=10~13 scaffold |
