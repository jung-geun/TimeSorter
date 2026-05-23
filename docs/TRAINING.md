# TimeSorter — 학습 방법론 및 결과

## 개요

Qwen3.5 모델에 2단계 파인튜닝을 적용합니다.
1. **SFT (Supervised Fine-Tuning)**: 6,499개 한국어 스케줄링 예제로 태스크 형식 학습
2. **DPO (Direct Preference Optimization)**: 1,469개 선호도 쌍으로 추론 품질 향상

---

## 1. 베이스 모델

| 항목 | 4B | 9B |
|------|----|----|
| 모델 | `Qwen/Qwen3.5-4B` | `Qwen/Qwen3.5-9B` |
| 컨텍스트 | 32K (학습 시 2K로 제한) | 32K |
| 아키텍처 | GQA, SwiGLU, RoPE | GQA, SwiGLU, RoPE |
| 라이선스 | Apache 2.0 | Apache 2.0 |

---

## 2. 공통 설정 — QLoRA

12GB VRAM 환경에서 전체 파라미터 미세조정 대신 NF4 4-bit 양자화 + LoRA 어댑터를 사용합니다.

```
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",       # Normalized Float 4 — Gaussian 분포 최적화
    bnb_4bit_use_double_quant=True,  # 양자화 상수도 재양자화 (추가 0.4bit/param 절감)
    bnb_4bit_compute_dtype=bfloat16  # 연산 정밀도
)
```

LoRA 어댑터 대상 모듈: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `r` | 16 | 랭크 (학습 파라미터 ≈ 16M) |
| `alpha` | 32 | 스케일 = alpha/r = 2.0 |
| `dropout` | 0.05 | 정규화 |
| 훈련 가능 비율 | ~0.4% | 전체 4B 대비 |

---

## 3. SFT 학습

### 목표

모델이 `[페르소나 씨의 오늘의 할 일 목록]\n- 항목` 형식의 입력을 받아
`1) 항목 - 이유\n2) ...` 형식의 4축 우선순위 출력을 생성하도록 학습합니다.

### 설정 (`configs/sft_rtx12g_4b.yaml`)

```yaml
model_name: Qwen/Qwen3.5-4B
dataset: data/scheduler_ko_combined.parquet
ko_ultrafeedback_n: 500        # 런타임 혼합 (스케줄 키워드 필터)
max_seq_length: 2048

training_args:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32   # effective batch = 32
  num_train_epochs: 5
  learning_rate: 2.0e-5
  optim: adamw_8bit                 # 8-bit Adam (메모리 절반)
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  gradient_checkpointing: true      # 활성화 재계산 (VRAM 절감)
  packing: true                     # 짧은 시퀀스 병합 (패딩 낭비 제거)
```

### 학습 결과

| 지표 | 값 |
|------|-----|
| 총 스텝 | 225 steps |
| 에폭 | 5 epoch |
| 소요 시간 | ~3시간 (RTX 3060 12GB) |
| `train_loss` (최종) | **1.179** |
| `mean_token_accuracy` | **77.55%** |
| 체크포인트 | `outputs/sft_rtx12g_4b/` |

### 관찰된 현상

- **Packing 경고**: flash-attn 미설치 환경에서는 `Using SDPA attention implementation for packing` 경고 발생. 기능에는 영향 없음 (SDPA 폴백으로 정상 학습).
- **Loss 수렴**: 5 epoch에 걸쳐 안정적으로 하강. 초기 1.8→최종 1.1 수준.
- **형식 학습 완료**: 학습 후 모델이 `1) 항목 - 이유` 형식을 일관되게 출력함.

---

## 4. DPO 학습

### 목표

SFT 모델이 단순 형식 모방에서 벗어나 **4축 기준(긴급도·중요도·의존성·시간 제약)에 근거한 추론**을 선호하도록 선호도 신호를 주입합니다.

### ref_model=None PEFT 트릭

표준 DPO는 학습 모델과 별도의 참조 모델이 필요해 메모리를 2배 사용합니다.
PEFT LoRA 어댑터를 비활성화하면 베이스 모델이 자연스럽게 참조 모델 역할을 수행하므로, 모델 1개로 12GB 내에서 DPO 학습이 가능합니다.

```python
# train_dpo.py 에서
ref_model = None  # PEFT disable = 어댑터 OFF 상태가 reference
```

### 설정 (`configs/dpo_rtx12g_4b.yaml`)

```yaml
sft_adapter: outputs/sft_rtx12g_4b
dataset: data/dpo_pairs.parquet
max_prompt_len: 512

training_args:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32   # effective batch = 32
  num_train_epochs: 2
  learning_rate: 5.0e-7             # SFT보다 40× 작은 LR (과도한 이탈 방지)
  beta: 0.1                         # KL 패널티 강도
  max_length: 1024
  optim: adamw_8bit
```

### 학습 결과

| 지표 | 값 |
|------|-----|
| 총 스텝 | 92 steps |
| 에폭 | 2 epoch |
| 소요 시간 | ~39분 |
| `train_loss` (최종) | **0.117** |
| `reward_accuracy` (peak) | **98.8%** |
| 체크포인트 | `outputs/dpo_rtx12g_4b/` |

### 관찰된 현상

- **reward_accuracy 급상승**: 초반 10 스텝 이내에 70% → 90%대로 진입. DPO 데이터 품질이 충분히 명확한 선호 신호를 담고 있다는 의미.
- **loss 급락**: 0.7 → 0.1 수준으로 빠르게 하강. 과적합 징후이기도 하나, 검증 결과 추론 형식과 근거 품질은 SFT 단독 대비 개선.
- **KL 안정성**: beta=0.1로 베이스 모델 분포에서 과도하게 이탈하지 않음.

---

## 5. 학습 파이프라인 실행

### auto (VRAM 자동 감지, 권장)

실행 시점 VRAM·GPU 수·모델 크기를 감지해 배치 크기·4bit·grad\_accum을 자동 산출합니다.
config에 `auto_batch: true`가 설정된 경우 활성화됩니다.

```bash
make pipeline-auto      # SFT → DPO (v1)
make pipeline-auto-v2   # SFT → DPO (v2 JSON 스키마)
```

VRAM별 자동 산출 결과 (4B 기준, eff\_batch=32 유지):

| VRAM | bs/GPU | 4bit | grad\_accum |
|------|--------|------|------------|
| < 14 GB | 1 | True | 32 |
| 14–22 GB | 2 | False | 16 |
| 22–30 GB | 4 | False | 8 |
| 30–50 GB | 8 | False | 4 |
| 50–90 GB | 16 | False | 2 |
| > 90 GB | 32 | False | 1 |

9B 모델은 VRAM 임계값이 `4/9` 비율로 축소 적용됩니다.

### RTX 4090 × 2 (24 GB × 2, DDP)

```bash
make pipeline-4090-2x-4b     # 4B, bf16 LoRA
make pipeline-4090-2x-4b-v2  # 4B v2 JSON 스키마
```

`configs/accelerate_4090_2x.yaml` (2-GPU DDP, bf16)을 사용합니다.
멀티 GPU 직접 실행:

```bash
uv run accelerate launch --config_file configs/accelerate_4090_2x.yaml \
  -m drl.train_sft --config configs/sft_auto.yaml
```

### RTX 12GB — Docker (flash-attn 포함)

```bash
make docker-build       # 이미지 빌드 (최초 1회, ~30분)
make pipeline-docker    # SFT → DPO
make pipeline-docker    # v2: make sft-docker-v2 → make dpo-docker-v2
```

### DGX (120 GB)

```bash
make pipeline-dgx-4b   # Qwen3.5-4B
make pipeline-dgx-8b   # Qwen3.5-9B
```

### RTX 12GB — 호스트 직접

```bash
make pipeline-rtx12g-4b     # v1
make pipeline-rtx12g-4b-v2  # v2
```

---

## 6. 장점 및 한계

### 장점

- **12GB VRAM에서 완전한 2단계 파인튜닝 가능**: NF4 QLoRA + adamw_8bit + gradient_checkpointing 조합
- **형식 일관성 높음**: SFT 후 `1) 항목 - 이유` 출력 형식이 99% 이상 유지됨
- **다양한 페르소나 적용**: 8개 직군 균등 분포 + Nemotron 실제 인물형 2,000개
- **DPO 데이터 품질**: chosen(4축 full-guide) vs rejected(긴급도만/가이드 없음)의 명확한 품질 차이로 judge가 TIE 없이 판정 가능

### 한계

- **마감일 절대화 미흡**: "내일 오전", "5/24 오후 5시" 같은 상대/절대 혼재 표현에서 날짜 계산 오류 발생
- **환각**: 존재하지 않는 미팅 시간(예: "오전 10시")을 생성하거나, 실제 일정(오후 2시)을 누락하는 사례 있음
- **의존 관계 파악 부족**: "참석 가능 여부 확인 → 참석" 같은 선행-후행 관계를 독립 태스크로 분리하지 못함
- **담당자/세부 조항 누락**: 수신자명, 계약서 조항 번호 등 이메일 본문의 구체적 실행 정보가 스케줄에서 탈락
