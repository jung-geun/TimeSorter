# drl — Qwen3 할 일 우선순위 정렬 비서

Qwen3-4B / 8B 모델을 **할 일 우선순위 정렬** 태스크에 맞게 3단계로 파인튜닝하는 파이프라인.

```
Stage 0: 환경 설정
Stage 1: 데이터 생성  (한국어 스케줄 데이터 + preference pair)
Stage 2: SFT          (스케줄 태스크 구조 학습)
Stage 3: DPO          (SFT 체크포인트 위에서 선호도 정렬)
```

---

## Stage 0 — 환경 설정

### 1. 의존성 설치

```bash
# uv가 없으면 먼저 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# Mac (MPS)
make setup-mac

# DGX Spark (ARM64 CUDA)
make setup-dgx

# RTX 12GB (Docker, flash-attn 포함)
make docker-build
```

### 2. API 키 설정

`.env` 파일을 프로젝트 루트에 생성:

```bash
ANTHROPIC_API_KEY=sk-ant-...   # 없으면 claude CLI (OAuth) 자동 사용
OPENAI_API_KEY=sk-...          # 데이터 생성 및 GPT-4o judge용
GOOGLE_API_KEY=...             # 없으면 gemini CLI (OAuth) 자동 사용

HF_TOKEN=hf_...                # HuggingFace 모델/데이터셋 다운로드
WANDB_API_KEY=...              # 학습 모니터링 (선택)
```

> **CLI 자동 폴백**: `ANTHROPIC_API_KEY`가 없으면 `claude -p` (Claude Code OAuth),
> `GOOGLE_API_KEY`가 없으면 `gemini CLI` → `OpenAI`로 자동 전환됩니다.

### 3. 동작 확인

```bash
make test    # 7 passed
make smoke   # 단순 파이프라인 연기 테스트
make lint    # ruff check
```

---

## Stage 1 — 데이터 생성

SFT 학습용 **한국어 스케줄 데이터**와 DPO 학습용 **preference pair**를 순서대로 생성합니다.

### Step 1-A: 한국어 스케줄 데이터 생성

두 가지 방식으로 데이터를 생성하고 `scripts/merge_datasets.py`로 합칩니다.

#### 방식 1 — Nemotron 페르소나 기반 (고품질, 권장)

`nvidia/Nemotron-Personas-Korea` (100만 개 한국 실제 인구통계 페르소나)와
영문 시드 `anakin87/events-scheduling`을 1:1 매칭해 생성합니다.

```bash
# 의존 데이터셋 먼저 다운로드
uv run python scripts/download_nemotron.py        # data/nemotron_personas_korea.parquet
uv run python scripts/download_datasets.py        # data/events-scheduling.parquet 등

# 생성 (랜덤 시드별로 다른 페르소나 조합, 동시 15개 비동기 처리)
uv run python scripts/gen_nemotron_schedule.py --out data/scheduler_ko.parquet --random-state 42
uv run python scripts/gen_nemotron_schedule.py --out data/scheduler_nemotron_r2.parquet --random-state 123
uv run python scripts/gen_nemotron_schedule.py --out data/scheduler_nemotron_r3.parquet --random-state 456
uv run python scripts/gen_nemotron_schedule.py --out data/scheduler_nemotron_r4.parquet --random-state 789
```

각 실행마다 500개 시나리오 × 고유 Nemotron 페르소나(나이·직업·지역·문체 포함) 생성.

#### 방식 2 — Generic 페르소나 기반 (8종)

Claude / OpenAI / Gemini로 8개 페르소나 유형 × 500 시나리오를 번역·확장합니다.

```bash
# 전체 실행 (8 페르소나 × 500 = 4,000개)
uv run python scripts/gen_korean_schedule.py --provider openai --out data/scheduler_generic.parquet

# 특정 provider/model 지정
uv run python scripts/gen_korean_schedule.py --provider gemini --model gemini-3.1-flash-lite
uv run python scripts/gen_korean_schedule.py --provider claude

# dry-run (5개)
make gen-schedule LIMIT=5
```

**지원 페르소나**: 직장인, 학생, 프리랜서, 부모, 시니어, 창업자, 의료진, 연구자

체크포인트(`data/scheduler_generic.ckpt.jsonl`) 기반 재개 가능.

#### 병합

```bash
# 모든 parquet을 data/scheduler_ko_combined.parquet으로 합산 (중복 제거 포함)
uv run python scripts/merge_datasets.py
```

| 소스 | 샘플 수 | 특징 |
|------|---------|------|
| Nemotron r1~r4 | 2,000 | 100만 한국 실제 페르소나, 나이·직업·지역·사투리 반영 |
| Generic 8 페르소나 | 4,000 | 직장인/학생/프리랜서/부모/시니어/창업자/의료진/연구자 |
| ko_Ultrafeedback 필터 | 최대 500 | 학습 시 자동 혼합 (`ko_ultrafeedback_n` 설정) |
| **합계** | **~6,500** | |

### Step 1-B: Preference pair 생성

각 시나리오에 대해 4개 후보를 생성하고 GPT-4o가 judge합니다.

| 후보 | 모델 | 가이드 |
|------|------|--------|
| C1 | `gemini-2.5-flash` | 4축 풀 가이드 (고품질) |
| C2 | `claude-sonnet-4-6` | 4축 풀 가이드 (고품질) |
| C3 | `gemini-1.5-flash` | 긴급도만 강조 (편향) |
| C4 | `claude-haiku-4-5` | 가이드 없음 (저품질) |
| Judge | `gpt-4o` | 4축 기준 A/B/TIE 판정 |

페어 조합: `c1_vs_c4`, `c2_vs_c3`, `c1_vs_c3` — TIE 제외 후 저장

```bash
make gen-pairs          # Step 1-A 완료 후 실행
make gen-pairs LIMIT=5  # dry-run

# 체크포인트(data/dpo_pairs.ckpt.jsonl)에서 이어서 실행 가능
uv run python scripts/gen_preference_pairs.py
```

산출물: `data/dpo_pairs.parquet` (`prompt`, `chosen`, `rejected`, `persona`, `pair` 컬럼)

### 한 번에 실행

```bash
make gen-data           # gen-schedule → gen-pairs 순차 실행
make gen-data LIMIT=5   # dry-run
```

### 데이터셋 분석

```bash
uv run python scripts/analyze_dataset.py     # 콘솔 통계
uv run python scripts/gen_dataset_report.py  # docs/dataset_analysis.md 생성
```

---

## Stage 2 — SFT (스케줄 태스크 학습)

`data/scheduler_ko_combined.parquet`와 `ko_ultrafeedback_n` 혼합 데이터로
Qwen3에 "할 일 목록 → 우선순위 정렬" 구조와 한국어 출력 포맷을 학습시킵니다.

### RTX 12GB — Docker (flash-attn 포함, 권장)

```bash
# 이미지 빌드 (최초 1회, torch 2.5.1+cu124 + flash_attn 2.7.4)
make docker-build

# SFT / DPO / 전체 파이프라인
make sft-docker
make dpo-docker
make pipeline-docker

# 인터랙티브 디버깅
make docker-shell

# 추론
make infer-docker ADAPTER=outputs/sft_rtx12g_4b PROMPT="보고서 작성(내일 마감), 점심 약속, 메일 답장"
```

| 설정 | RTX 12GB |
|------|----------|
| 모델 | Qwen/Qwen3-4B-Instruct-2507 |
| config | `configs/sft_rtx12g_4b.yaml` |
| 배치 | 1 × 32 = 32 (eff) |
| lr | 2e-5 |
| epochs | 5 |
| packing | ✓ (flash_attention_2) |
| optimizer | adamw_8bit |
| 어댑터 | `outputs/sft_rtx12g_4b/` |

### DGX Spark (120GB VRAM)

```bash
make sft-dgx-4b   # 4B 모델
make sft-dgx-8b   # 8B 모델
```

| 설정 | 4B | 8B |
|------|----|----|
| 모델 | Qwen/Qwen3-4B-Instruct-2507 | Qwen/Qwen3-8B |
| config | `configs/sft_dgx_4b.yaml` | `configs/sft_dgx_8b.yaml` |
| 배치 | 16 × 2 = 32 | 8 × 4 = 32 |
| lr | 2e-5 | 2e-5 |
| epochs | 2 | 2 |

### Mac (스모크 테스트)

```bash
make sft-smoke   # Qwen3-1.7B, 2 steps, 64 samples
```

---

## Stage 3 — DPO (선호도 정렬)

Stage 2 SFT 체크포인트 위에서 `data/dpo_pairs.parquet`로 DPO를 적용합니다.

```bash
# RTX 12GB (Docker)
make dpo-docker

# DGX
make dpo-dgx-4b
make dpo-dgx-8b

# generic
make dpo-final
```

| 설정 | 4B | 8B |
|------|----|----|
| config | `configs/dpo_dgx_4b.yaml` | `configs/dpo_dgx_8b.yaml` |
| beta | 0.1 | 0.05 |
| lr | 5e-7 | 5e-7 |
| eff_bs | 32 | 32 |
| epochs | 2 | 2 |

### Stage 2 + 3 한 번에

```bash
make pipeline-docker      # RTX 12GB Docker
make pipeline-dgx-4b      # DGX 4B
make pipeline-dgx-8b      # DGX 8B
```

---

## 추론

```bash
# 호스트 직접
make infer ADAPTER=outputs/dpo_dgx_4b \
  PROMPT="보고서 작성(내일 마감), 점심 약속, 메일 답장 3건, 운동"

# Docker
make infer-docker ADAPTER=outputs/sft_rtx12g_4b \
  PROMPT="보고서 작성(내일 마감), 점심 약속, 메일 답장 3건, 운동"
```

예상 출력:
```
1) 보고서 작성 (긴급도↑, 마감 임박)
2) 메일 답장 3건 (의존성: 보고서 관련 회신 포함 가능)
3) 점심 약속 (시간 고정)
4) 운동 (중요도 있으나 시간 유연)
```

---

## Makefile 타겟 전체

| 타겟 | 설명 |
|------|------|
| `make setup-mac` | Mac(MPS) 환경 의존성 설치 |
| `make setup-dgx` | DGX(ARM64 CUDA) 환경 의존성 설치 |
| `make docker-build` | Docker 이미지 빌드 (timesorter:cu124) |
| `make docker-shell` | 인터랙티브 컨테이너 셸 |
| `make test` | pytest 전체 실행 |
| `make lint` | ruff check |
| `make smoke` | 스모크 테스트 |
| `make gen-schedule [LIMIT=N]` | Generic 페르소나 스케줄 데이터 생성 |
| `make gen-pairs [LIMIT=N]` | DPO preference pair 생성 |
| `make gen-data [LIMIT=N]` | gen-schedule → gen-pairs 순차 실행 |
| `make sft` | SFT (configs/sft_scheduler.yaml) |
| `make sft-smoke` | Mac 스모크 SFT |
| `make sft-dgx-4b` | DGX 4B SFT |
| `make sft-dgx-8b` | DGX 8B SFT |
| `make sft-rtx12g-4b` | RTX 12GB 호스트 SFT |
| `make sft-docker` | RTX 12GB Docker SFT (flash-attn) |
| `make dpo-final` | DPO (configs/dpo_final.yaml) |
| `make dpo-dgx-4b` | DGX 4B DPO |
| `make dpo-dgx-8b` | DGX 8B DPO |
| `make dpo-docker` | RTX 12GB Docker DPO |
| `make pipeline-dgx-4b` | DGX 4B SFT → DPO |
| `make pipeline-dgx-8b` | DGX 8B SFT → DPO |
| `make pipeline-docker` | RTX 12GB Docker SFT → DPO |
| `make infer ADAPTER=... PROMPT=...` | 호스트 추론 |
| `make infer-docker ADAPTER=... PROMPT=...` | Docker 추론 |

---

## 전체 순서 한눈에

```bash
# 0. 환경
make docker-build       # 또는 make setup-dgx

# 1. 데이터 다운로드 (최초 1회)
uv run python scripts/download_datasets.py
uv run python scripts/download_nemotron.py

# 1. 데이터 생성
uv run python scripts/gen_nemotron_schedule.py --out data/scheduler_ko.parquet --random-state 42
uv run python scripts/gen_nemotron_schedule.py --out data/scheduler_nemotron_r2.parquet --random-state 123
uv run python scripts/gen_korean_schedule.py --provider openai --out data/scheduler_generic.parquet
uv run python scripts/merge_datasets.py
make gen-pairs

# 2. SFT
make sft-docker         # 또는 make sft-dgx-4b

# 3. DPO
make dpo-docker         # 또는 make dpo-dgx-4b

# 추론
make infer-docker ADAPTER=outputs/sft_rtx12g_4b PROMPT="..."
```

---

## 모듈 구조

```
src/drl/
├── device.py          — MPS/CUDA/CPU 감지 → DeviceProfile (flash-attn 자동 감지)
├── config.py          — YAML → RunConfig 파싱
├── model.py           — Qwen3 로딩 + LoRA / SFT 어댑터 부착
├── data/
│   ├── loader.py      — HF 데이터셋 / 로컬 parquet → DPO 포맷
│   ├── scheduler.py   — scheduler_ko_combined.parquet + ko_Ultrafeedback 혼합 → ChatML
│   └── augment.py     — Gemini/Claude 생성 + GPT-4o judge
├── train_sft.py       — SFTTrainer 래퍼
├── train_dpo.py       — DPOTrainer 래퍼
└── infer.py           — 어댑터 로드 + 텍스트 생성

scripts/
├── gen_nemotron_schedule.py  — Nemotron 페르소나 기반 스케줄 생성 (비동기, --random-state)
├── gen_korean_schedule.py    — Generic 8 페르소나 번역 + 확장 (체크포인트 재개 가능)
├── merge_datasets.py         — 여러 parquet 병합 → scheduler_ko_combined.parquet
├── gen_preference_pairs.py   — 4후보 생성 + judge → DPO parquet
├── download_datasets.py      — HF 데이터셋 다운로드
├── download_nemotron.py      — Nemotron-Personas-Korea 다운로드 (1M 샘플)
├── analyze_dataset.py        — 데이터셋 콘솔 통계 출력
├── gen_dataset_report.py     — 데이터셋 분석 → docs/dataset_analysis.md
└── translate_locally.py      — 로컬 모델 번역

configs/
├── sft_rtx12g_4b.yaml                        — RTX 12GB Docker SFT (flash-attn)
├── sft_dgx_4b.yaml / sft_dgx_8b.yaml        — DGX SFT
├── sft_mac_smoke.yaml / sft_scheduler.yaml
├── dpo_rtx12g_4b.yaml                        — RTX 12GB DPO
├── dpo_dgx_4b.yaml / dpo_dgx_8b.yaml        — DGX DPO
├── dpo_final.yaml
├── dgx_4b.yaml / dgx_8b.yaml                — generic DPO
└── mac_smoke.yaml / mac_train.yaml

data/                          (*.parquet, *.jsonl → Git LFS)
├── scheduler_ko_combined.parquet   — SFT 학습 데이터 합본 (5,999행)
├── scheduler_ko.parquet            — Nemotron round 1 (500행)
├── scheduler_nemotron_r2.parquet   — Nemotron round 2 (500행, seed=123)
├── scheduler_nemotron_r3.parquet   — Nemotron round 3 (500행, seed=456)
├── scheduler_nemotron_r4.parquet   — Nemotron round 4 (500행, seed=789)
├── scheduler_generic.parquet       — Generic 8 페르소나 (4,000행)
├── dpo_pairs.parquet               — DPO 페어
├── events-scheduling.parquet       — 영문 시드 원본 (500행)
│
│   # 아래 파일은 .gitignore — download_*.py로 재생성
├── ko_Ultrafeedback_binarized.parquet   (61,966행, ~103MB)
├── nemotron_personas_korea.parquet      (1,000,000행, ~1.9GB)
└── orca-dpo-pairs-ko.parquet

docs/
└── dataset_analysis.md
```

---

## 환경 요구사항

| 환경 | HW | Python | 특이사항 |
|------|----|--------|---------|
| RTX 12GB Docker | x86_64 CUDA 12.4 | 3.11 | flash_attn 2.7.4, torch 2.5.1+cu124, QLoRA 4-bit |
| DGX Spark | ARM64 + Blackwell CUDA | 3.11 | bf16 LoRA (ARM64 bnb wheel 미존재) |
| MacBook Pro | Apple Silicon (MPS) | 3.11 | bitsandbytes 미사용, bf16 LoRA |

### Docker 이미지 상세 (`timesorter:cu124`)

- Base: `nvidia/cuda:12.4.1-devel-ubuntu22.04`
- `torch==2.5.1+cu124`
- `flash-attn==2.7.4` (pre-built wheel, cxx11abiFALSE)
- `bitsandbytes>=0.43` (QLoRA NF4)
- `/opt/venv` 격리 환경 (호스트 `.venv`와 충돌 없음)

---

## 참고

- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)
- [PEFT LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [nvidia/Nemotron-Personas-Korea](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea)
- [maywell/ko_Ultrafeedback_binarized](https://huggingface.co/datasets/maywell/ko_Ultrafeedback_binarized)
- [`docs/dataset_analysis.md`](docs/dataset_analysis.md)
