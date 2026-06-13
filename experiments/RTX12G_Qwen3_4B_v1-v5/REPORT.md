# TimeSorter — RTX 3080 Ti 12GB | Qwen3-4B-Instruct-2507 | v1~v5 전체 실험 보고서

> **실험 환경**: RTX 3080 Ti (12 GB VRAM) × 1 GPU  
> **모델 계보**: Qwen3-4B-Instruct-2507 (v1~v5) → Qwen3.5-4B (v4+v5 재학습)  
> **기간**: 2026-05 ~ 2026-06

---

## 1. 실험 개요

한국어 일정 관리 특화 파인튜닝 파이프라인 (`TimeSorter`)의 전체 실험 이력.  
단일 RTX 3080 Ti (12 GB VRAM) 환경에서 메모리 제약을 극복하며 SFT → DPO → GRPO 사이클을 반복.

### 핵심 성과

| 체크포인트 | 전체 통과율 | 주요 개선 |
|-----------|-----------|---------|
| DPO v3 (베이스라인) | 56.7% | — |
| GRPO v4 파일럿 | 57.3% | 동률 (KL 도즈 부족) |
| **SFT v4 (curated + loss masking)** | **77.3%** | **+20.6%p 돌파** |
| DPO v5 (on-policy) | 77.3% | 동률 (chain은 소량 DPO로 이동 없음) |
| **Qwen3.5-4B SFT+DPO v5** | **90.0%\*** | 모델 업그레이드 효과 |

\* 30개 샘플 검증 (로컬 QLoRA inference)

---

## 2. 실험 이력 상세

### 2-1. 학습 이력 표

| 단계 | 어댑터 | 데이터셋 | 샘플 수 | train_loss | 검증 |
|------|--------|---------|--------|-----------|------|
| SFT v1 | `sft_rtx12g_4b` | scheduler_ko_combined | 5,999 | — | 자유 텍스트 출력 |
| DPO v1 | `dpo_rtx12g_4b` | dpo_pairs_v1 | ~5K쌍 | — | — |
| SFT v2 | `sft_rtx12g_4b_v2` | scheduler_v2_combined | 10,958 | — | 4축 JSON (가중치 미보존) |
| DPO v2 | `dpo_rtx12g_4b_v2` | dpo_pairs_v2 | 15,280쌍 | 0.0043 | — |
| SFT v3 | `sft_rtx12g_4b_v3` | scheduler_v3_combined | 5,678 | 0.328 | today 주입 |
| DPO v3 | `dpo_rtx12g_4b_v3` | dpo_pairs_v3 | 3,722쌍 | 0.635 | **56.7%** |
| GRPO v4 | `grpo_rtx12g_4b_v4` | grpo_prompts_v4 | 256×4생성 | reward 0.38 | **57.3%** |
| **SFT v4** | `sft_rtx12g_4b_v4` | sft_v4_train (curated) | 6,056 | 0.315 | **77.3%** |
| DPO v5 | `dpo_rtx12g_4b_v5` | dpo_pairs_v5 (on-policy) | 1,368쌍 | 0.678 | **77.3%** |
| *(Qwen3.5)* SFT v4 | `sft_q35_4b_v4` | sft_v4_train | 6,056 | — | — |
| *(Qwen3.5)* **DPO v5** | `dpo_q35_4b_v5` | dpo_pairs_v5 | 1,368쌍 | **0.1706** | **90.0%\*** |

---

## 3. 검증 결과 상세 분석

### 3-1. 전체 통과율 추이

```
DPO v3          ████████████████████████████░░░░░░░░░░░░░  56.7% (85/150)
GRPO v4         █████████████████████████████░░░░░░░░░░░░  57.3% (86/150)
SFT v4          ██████████████████████████████████████░░░  77.3% (116/150)
DPO v5          ██████████████████████████████████████░░░  77.3% (116/150)
Qwen3.5 DPO v5  ████████████████████████████████████████░  90.0% (27/30)*
```

### 3-2. 시나리오별 통과율 (held-out 150개, Qwen3-4B 계열)

| 시나리오 | DPO v3 | GRPO v4 | SFT v4 | DPO v5 |
|----------|--------|---------|--------|--------|
| dated_mixed (53개) | 38% (20/53) | 40% (21/53) | **77%** (41/53) | 77% (41/53) |
| intraday (30개) | 67% (20/30) | 67% (20/30) | **87%** (26/30) | 87% (26/30) |
| risk (22개) | 95% (21/22) | 95% (21/22) | **95%** (21/22) | 95% (21/22) |
| relative (15개) | **100%** (15/15) | 100% (15/15) | 100% (15/15) | 100% (15/15) |
| dependency_chain (30개) | 30% (9/30) | 30% (9/30) | **43%** (13/30) | 43% (13/30) |
| **전체** | **56.7%** | **57.3%** | **77.3%** | **77.3%** |

### 3-3. 위반 유형별 건수 분석

| 위반 유형 | DPO v3 | GRPO v4 | SFT v4 | DPO v5 |
|-----------|--------|---------|--------|--------|
| `past_rank` (지난 일정 상위배치) | **52** | 44 | **15** | 15 |
| `chain` (체인 순서 위반) | 39 | 38 | **30** | 29 |
| `intraday_order` (당일 시각 순서) | 12 | 12 | **5** | 5 |
| `none_first` (무마감 1위) | 1 | 1 | 1 | 1 |
| `parse_or_count` (파싱 실패) | 1 | 1 | 0 | 0 |

### 3-4. guard rerank 효과

| 모델 | 무처리 | guard rerank | 차이 |
|------|--------|-------------|------|
| DPO v3 | 56.7% | 62.7% | +6.0%p |
| GRPO v4 | 57.3% | 62.0% | +4.7%p |
| SFT v4 | 77.3% | 78.7% | +1.4%p |
| DPO v5 | 77.3% | 77.3% | ±0 |

> SFT v4 이후 모델 자체 품질이 높아 guard rerank 효과 축소 — 모델이 이미 규칙을 내재화.

### 3-5. Qwen3.5-4B 검증 결과 (30개 샘플)

| 시나리오 | 통과 | 위반 유형 |
|----------|------|---------|
| dated_mixed (9개) | 9/9 (100%) | 없음 |
| dependency_chain (6개) | 4/6 (67%) | chain순서 2건, chain_dep_score 2건 |
| intraday (6개) | 6/6 (100%) | 없음 |
| relative (4개) | 4/4 (100%) | 없음 |
| risk (5개) | 4/5 (80%) | past_rank 1건 |
| **전체** | **27/30 (90.0%)** | chain 4건, past_rank 1건 |

---

## 4. 핵심 인사이트

### 4-1. 성능 점프의 원인 (DPO v3 56.7% → SFT v4 77.3%)

**+20.6%p 돌파는 모델 아키텍처가 아니라 데이터·학습 방법론에서 나왔다.**

#### 원인 1: 데이터 품질 큐레이션

v3 이전 데이터의 구조적 결함이 Opus 4.8 의미 감사로 확인됨:
- persona_fit 평균 3.3 (목표: 4.9-5.0)
- 35% 샘플에서 페르소나-할일 불일치
- 3,943행이 비스케줄 오염(잡무, 일기 등)

SFT v4는 골격 검증 + persona_fit 4.9-5.0 조건을 모두 통과한 6,056행만 사용:
```
curated v3-v5: 3,356행
refusal:        1,200행
v2_schedule:    1,500행 (다양성 확보)
```

#### 원인 2: Prompt Loss 마스킹

기존 SFT는 ~600 토큰 시스템 프롬프트에도 cross-entropy가 걸려 학습 신호의 50%+ 가 "프롬프트 암기"에 소모됨.  
`prompt_completion=true` + `completion_only_loss`로 응답 토큰에만 loss를 걸어 **학습 밀도 ~2.5배 향상**.

#### 원인 3: 이중 생성 경로

OpenAI(gpt-5.4 계열)와 Claude(Sonnet 4.6/Opus 4.8)를 병렬로 운용하여 같은 골격에 대한 다양한 표현을 확보. 두 경로 모두 동일한 골격 검증을 통과해야 하므로 품질 수렴.

### 4-2. DPO/GRPO가 SFT와 동률인 이유

```
dependency_chain: SFT v4 43% → DPO v5 43% (불변)
                  GRPO v4    57.3% (동률)
```

- **체인 시나리오는 소량 DPO로 개선되지 않음**: 선호 학습이 아니라 체인 SFT 데이터 자체가 부족
- **GRPO KL 도즈 부족**: KL ≈ 0.0005 (목표 0.01-0.1) — 256 프롬프트는 너무 적음
- **결론**: 체인 약점 해결은 chain 특화 SFT 데이터 보강이 선행되어야 함

### 4-3. 잔여 약점과 다음 단계

| 약점 | 현황 | 해결 방향 |
|------|------|---------|
| `dependency_chain` 43% | 모든 체크포인트에서 불변 | chain 특화 SFT 데이터 3× 증량 (v5 일부 개선 중) |
| `past_rank` 15건 | SFT v4에서 크게 감소(52→15) | 추가 past_hallucination DPO |
| `none_first` 1건 | 고질적 1건 | 프롬프트 규칙 강화 |

---

## 5. 메모리 최적화 기록 (RTX 12GB 한계 극복)

### 5-1. Qwen3.5 248K vocab OOM 문제

Qwen3.5는 vocab 크기 248,064로 인해 `[2, T, 248K]` logits가 ~1.1 GiB — 12GB GPU에서 표준 DPO가 불가능했음.

**해결책**: `MemEfficientDPOTrainer` 구현

```python
# chosen/rejected를 [1,T] 순차 처리
# 청킹된 log_softmax (chunk_size=128 토큰)
# 수동 gradient 주입 (proxy backward)
# accelerate convert_to_fp32 우회 (keep_fp32_wrapper=False)
```

### 5-2. Qwen3.5 CUDA 버그 우회

`torch_chunk_gated_delta_rule`의 in-place 비연속 뷰 할당 버그 (`cudaErrorIllegalAddress`):

```python
# train_dpo.py main() 시작 시 monkey-patch
import transformers.models.qwen3_5.modeling_qwen3_5 as _qwen35
_qwen35.is_fast_path_available = True  # fla의 chunk_gated_delta_rule 강제 사용
```

---

## 6. 재현 방법

```bash
# 환경 설정
uv sync
uv pip install flash-linear-attention  # Qwen3.5 필수

# SFT v4 (Qwen3.5-4B)
uv run python scripts/train_sft.py --config configs/sft_rtx12g_q35_4b_v4.yaml

# DPO v5 (Qwen3.5-4B)
uv run python scripts/train_dpo.py --config configs/dpo_rtx12g_q35_4b_v5.yaml

# 검증
bash scripts/validate_model.sh outputs/dpo_q35_4b_v5
```

---

*생성일: 2026-06-13 | 실험 환경: RTX 3080 Ti 12GB × 1*
