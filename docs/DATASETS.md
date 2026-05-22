# 데이터셋 분석 및 활용 계획

## 프로젝트 목표 요약

Qwen3-4B / 8B Instruct 모델에 **한국어 preference 데이터로 DPO**를 적용한다.
SFT 없이 DPO 단독으로 진행하며, 모델이 더 나은 한국어 응답을 선택하는 방향으로 학습시킨다.

---

## 1. 메인 데이터셋 — `maywell/ko_Ultrafeedback_binarized`

### 기본 정보

| 항목 | 값 |
|---|---|
| 총 샘플 수 | 61,966 |
| 컬럼 | `prompt`, `chosen`, `rejected` |
| 언어 | 한국어 |
| 출처 | UltraFeedback를 한국어 번역·재구성한 binarized DPO 포맷 |
| HuggingFace Hub | `maywell/ko_Ultrafeedback_binarized` |

### 길이 분포 (문자 수 기준)

| 통계 | prompt | chosen | rejected |
|---|---|---|---|
| min | 3 | 1 | 1 |
| p25 | 52 | 133 | 104 |
| p50 | 174 | 494 | 305 |
| p75 | 401 | 1,087 | 760 |
| p90 | 901 | 1,601 | 1,303 |
| p95 | 1,495 | 1,968 | 1,721 |
| p99 | 4,064 | 3,466 | 2,940 |
| max | 91,329 | 49,681 | 61,101 |
| mean | 406 | 721 | 551 |

한국어 Qwen3 토크나이저 기준 문자당 약 1.5-2 토큰이므로,
p99 prompt (~4,064 chars) ≈ 2,700 토큰 → `max_prompt_length=1024` 필터에서 제거됨.
p90 prompt (~901 chars) ≈ 600 토큰 → 통과.

### 품질 지표

- **빈 응답**: chosen 0건, rejected 1건 (무시 가능)
- **chosen이 rejected보다 긴 비율**: 60.0%
  - 좋은 응답(chosen)이 더 길고 상세한 경향 — DPO 학습에 유리한 신호
  - 40%는 rejected가 더 긴 경우로, 길이보다 품질로 레이블된 pair임
- **짧은 prompt (< 20자)**: 1,372건 (2.2%) — 노이즈 아님, 정상적인 짧은 질문 ("키티 호크는 무엇으로 유명합니까?" 등)

### 포맷 호환성 (DPOTrainer)

```
prompt   → 순수 한국어 질문 텍스트 (plain string)
chosen   → 순수 한국어 어시스턴트 응답 (plain string, 응답만)
rejected → 순수 한국어 어시스턴트 응답 (plain string, 응답만)
```

TRL `DPOTrainer(processing_class=tokenizer)`가 내부적으로 Qwen3 chat template을 적용:
```
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
{chosen 또는 rejected}
<|im_end|>
```

**결론: DPOTrainer와 직접 호환. 어댑터 불필요.**

### 필터링 후 예상 잔존 샘플 수

현재 설정 (`max_prompt_len=1024`, `max_response_len=1024` 토큰) 기준:
- prompt p90 ≈ 600 토큰 → ~90%+ 통과 예상
- chosen/rejected p90 ≈ 1,000-1,100 토큰 → 약간 필터됨
- **예상 잔존: 약 50,000-55,000개** (전체의 80-90%)

실제 수치는 학습 시작 시 `[data] 길이 필터 후: X/Y개 유지` 로그로 확인.

---

## 2. 보조 데이터셋 — `SJ-Donald/orca-dpo-pairs-ko`

### 기본 정보

| 항목 | 값 |
|---|---|
| 총 샘플 수 | 36,009 |
| 컬럼 | `system`, `question`, `chosen`, `rejected` |
| 언어 | 한국어 |
| 출처 | ORCA DPO pairs 한국어 번역 |
| HuggingFace Hub | `SJ-Donald/orca-dpo-pairs-ko` |

### 길이 분포 (문자 수 기준)

| 통계 | question | chosen | rejected |
|---|---|---|---|
| p50 | 163 | 224 | 337 |
| p90 | 1,062 | 857 | 929 |
| p99 | 1,602 | 1,455 | 1,518 |

메인 데이터셋보다 전반적으로 짧고 고른 분포. 필터링 손실 적을 것으로 예상.

### 스키마 차이 — 어댑터 필요

메인 데이터셋과 달리 `prompt` 컬럼이 없고, `system` 필드가 별도로 존재한다.

```
system   → "당신은 AI 비서입니다. 상세하고 긴 답변을 생성해야 합니다."
question → prompt에 해당
chosen   → 선호 응답
rejected → 비선호 응답
```

`loader.py`에 어댑터 함수를 추가하면 병합 사용 가능:

```python
def adapt_orca_ko(row: dict) -> dict:
    # system 프롬프트를 prefix로 붙이거나, question만 사용
    return {
        "prompt": row["question"],
        "chosen": row["chosen"],
        "rejected": row["rejected"],
    }
```

**현재 상태**: 미구현, 학습에 미사용.

### rejected > chosen인 비율

ORCA 데이터셋에서는 rejected가 더 긴 경우가 많다 (예: rejected 337자 vs chosen 224자 p50).
이는 rejected가 과잉 설명이거나 hallucination을 포함한 장황한 응답이기 때문으로,
길이 자체가 품질 지표가 아님을 다시 확인.

---

## 3. 학습 파이프라인 분석

### 전체 흐름

```
YAML config
    ↓
RunConfig.from_yaml()
    ↓
load_model_and_tokenizer()   ← Qwen3 + LoRA 부착
    ↓
load_dpo_dataset()           ← HF Hub 로드 → 길이 필터
    ↓
DPOConfig(**training_kwargs)
    ↓
DPOTrainer(model, ref_model=None, ...)
    ↓
trainer.train()  →  trainer.save_model()
```

### ref_model=None 트릭

메모리를 절감하기 위해 `ref_model=None`을 전달하고 PEFT adapter-disable 방식을 활용한다.
DPOTrainer가 reference logprobs를 계산할 때 내부적으로 LoRA adapter를 disable하여
동일 모델이 base model 역할을 겸한다.

```
[학습 시]  model with LoRA adapter → policy logprobs
[ref 계산] model with adapter disabled → reference logprobs
```

GPU 메모리를 ref_model 없이 절반으로 줄일 수 있어 DGX 8B 학습에 필수적.

### 환경별 학습 설정

| 항목 | mac_train (MPS) | dgx_4b | dgx_8b |
|---|---|---|---|
| 모델 | Qwen3-1.7B | Qwen3-4B-Instruct-2507 | Qwen3-8B |
| lora_r / alpha | 8 / 16 | 16 / 32 | 16 / 32 |
| per_device_bs | 1 | 4 | 2 |
| grad_accum | 4 | 2 | 4 |
| effective_bs | 4 | 8 | 8 |
| max_length | 2048 | 2048 | 2048 |
| max_prompt_length | 1024 | 1024 | 1024 |
| learning_rate | 1.0e-6 | 5.0e-6 | 5.0e-6 |
| beta | 0.1 | 0.1 | 0.1 |
| grad_checkpointing | off | off | on |
| bf16 (Trainer flag) | off (MPS는 load-time bf16) | on | on |

### DPO 하이퍼파라미터 선택 근거

| 파라미터 | 값 | 근거 |
|---|---|---|
| `beta` | 0.1 | DPO 원논문 기본값. KL 페널티 강도. 낮을수록 공격적 업데이트 |
| `learning_rate` | 1e-6 (Mac), 5e-6 (DGX) | DPO는 SFT보다 lr을 10-100x 낮게 설정해야 collapse 방지 |
| `warmup_ratio` | 0.05 (Mac), 0.03 (DGX) | 낮은 lr에서 초반 불안정 방지 |
| `lora_r` | 8 (1.7B), 16 (4B/8B) | 모델 크기 대비 adapter 표현력 균형 |

---

## 4. 목표 부합 여부 점검

| 목표 | 현황 | 평가 |
|---|---|---|
| Qwen3-4B/8B에 DPO 적용 | DGX config 준비 완료 | 완료 |
| 한국어 preference 데이터 사용 | ko_Ultrafeedback_binarized 적용 | 완료 |
| DPO 단독 (SFT 없음) | SFT 파이프라인 없음 | 완료 |
| Mac에서 smoke 테스트 | 1.7B MPS 설정 완료 | 완료 |
| DGX에서 풀 학습 | 4B/8B config 준비 완료 | 준비됨 |
| DPO 붕괴 방지 | lr 수정, beta 명시, 길이 필터 추가 | 수정 완료 |
| 학습 모니터링 (wandb) | DPOTrainer → wandb 연동 | 완료 |
| 보조 데이터셋 활용 | orca-dpo-pairs-ko 어댑터 미구현 | 미완료 |

### 현재 가장 큰 리스크

1. **학습 후 성능 검증 수단 없음** — MT-Bench 또는 단순 prompt 비교 평가 미구현
2. **단일 데이터셋 의존** — ko_Ultrafeedback_binarized만 사용 중, 다양성 부족 가능
3. **DPO base model 품질** — SFT 없이 Instruct 모델에 바로 DPO 적용. Instruct 모델이 이미 RLHF/DPO를 거쳤으므로 distribution mismatch 가능성 있음

---

## 5. 데이터셋 활용 로드맵

### Phase 1 (현재) — 메인 데이터셋 단독

- `maywell/ko_Ultrafeedback_binarized` 전체 사용
- 길이 필터 적용: prompt ≤ 1024 tokens, response ≤ 1024 tokens
- 예상 유효 샘플: ~50,000-55,000개

### Phase 2 (선택적 확장) — 보조 데이터셋 병합

`loader.py`에 어댑터 추가 후 두 데이터셋을 합쳐 ~95,000개로 확장.

```python
# loader.py에 추가 예정
def load_multi_dataset(
    datasets: list[dict],  # [{"name": ..., "adapter": ...}, ...]
    ...
) -> Dataset:
    ...
```

| 데이터셋 | 샘플 수 | 특징 |
|---|---|---|
| ko_Ultrafeedback_binarized | ~50,000 (필터 후) | 다양한 주제, 긴 응답 |
| orca-dpo-pairs-ko | ~35,000 (필터 후) | 지시 수행, 짧고 명확한 응답 |
| 합계 | ~85,000 | 두 분포의 보완 |

### Phase 3 (미래) — 도메인 특화

특정 도메인(의료, 법률, 코딩 등)의 한국어 preference 데이터 추가 수집 또는 합성.
이 단계는 현재 프로젝트 범위 외.
