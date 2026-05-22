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
```

### 2. API 키 설정

`.env` 파일을 프로젝트 루트에 생성:

```bash
ANTHROPIC_API_KEY=sk-ant-...   # 없으면 claude CLI (OAuth) 자동 사용
OPENAI_API_KEY=sk-...          # GPT-4o judge용, 필수
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

영문 시드(`anakin87/events-scheduling`)를 번역 모델로 한국어화하고
페르소나(직장인/학생/프리랜서/부모)별로 확장합니다.

```bash
# 전체 실행 (~500 시나리오)
make gen-schedule

# dry-run (5개만)
make gen-schedule LIMIT=5

# 번역 모델 선택 (기본: gemini)
uv run python scripts/gen_korean_schedule.py --provider gemini   # 기본
uv run python scripts/gen_korean_schedule.py --provider claude
uv run python scripts/gen_korean_schedule.py --provider openai
uv run python scripts/gen_korean_schedule.py --provider gemini --model gemini-3.1-flash-lite
```

산출물: `data/scheduler_ko.parquet` (`prompt`, `chosen`, `persona`, `source`, `original_idx` 컬럼)

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
# Step 1-A 완료 후 실행
make gen-pairs

# dry-run
make gen-pairs LIMIT=5

# 체크포인트(data/dpo_pairs.ckpt.jsonl)에서 이어서 실행 가능 — 재실행 안전
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
# 콘솔 출력 (페르소나 통계, 포맷 검증, 랜덤 샘플)
uv run python scripts/analyze_dataset.py

# 마크다운 리포트 생성 → docs/dataset_analysis.md
uv run python scripts/gen_dataset_report.py
```

자세한 분석은 [`docs/dataset_analysis.md`](docs/dataset_analysis.md) 참조.

---

## Stage 2 — SFT (스케줄 태스크 학습)

`data/scheduler_ko.parquet`를 사용해 Qwen3에 "할 일 목록 → 우선순위 정렬" 구조와 한국어 출력 포맷을 학습시킵니다.

> Stage 1 완료(`data/scheduler_ko.parquet` 존재) 후 실행하세요.

### DGX Spark (120GB VRAM, 권장)

```bash
# 4B 모델 (bs=16, eff_bs=32)
make sft-dgx-4b

# 8B 모델 (bs=8, eff_bs=32)
make sft-dgx-8b
```

| 설정 | 4B | 8B |
|------|----|----|
| 모델 | Qwen/Qwen3-4B-Instruct-2507 | Qwen/Qwen3-8B |
| config | `configs/sft_dgx_4b.yaml` | `configs/sft_dgx_8b.yaml` |
| 배치 | 16 × 2 = 32 | 8 × 4 = 32 |
| lr | 2e-5 | 2e-5 |
| epochs | 2 | 2 |
| packing | ✓ | ✓ |
| optimizer | adamw_torch_fused | adamw_torch_fused |

어댑터 저장 위치: `outputs/sft_dgx_4b/` 또는 `outputs/sft_dgx_8b/`

### Mac (스모크 테스트)

```bash
make sft-smoke   # Qwen3-1.7B, 2 steps, 64 samples — 수분 내 완료
```

### 스케줄러 단독 SFT

```bash
make sft   # configs/sft_scheduler.yaml 사용
```

---

## Stage 3 — DPO (선호도 정렬)

Stage 2 SFT 체크포인트를 초기점으로 `data/dpo_pairs.parquet`로 DPO를 적용합니다.

> Stage 2 완료(`outputs/sft_dgx_4b` 또는 `outputs/sft_dgx_8b` 존재) 후 실행하세요.

```bash
# 4B
make dpo-dgx-4b

# 8B
make dpo-dgx-8b

# generic DPO (dpo_final.yaml)
make dpo-final
```

| 설정 | 4B | 8B |
|------|----|----|
| config | `configs/dpo_dgx_4b.yaml` | `configs/dpo_dgx_8b.yaml` |
| beta | 0.1 | 0.05 |
| lr | 5e-7 | 5e-7 |
| eff_bs | 32 | 32 |
| epochs | 2 | 2 |

어댑터 저장 위치: `outputs/dpo_dgx_4b/` 또는 `outputs/dpo_dgx_8b/`

### Stage 2 + 3 한 번에

```bash
make pipeline-dgx-4b   # sft-dgx-4b → dpo-dgx-4b
make pipeline-dgx-8b   # sft-dgx-8b → dpo-dgx-8b
```

---

## 추론

```bash
make infer ADAPTER=outputs/dpo_dgx_4b \
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
| `make test` | pytest 전체 실행 |
| `make lint` | ruff check |
| `make smoke` | 스모크 테스트 (scripts/smoke.sh) |
| `make gen-schedule [LIMIT=N]` | 한국어 스케줄 데이터 생성 |
| `make gen-pairs [LIMIT=N]` | DPO preference pair 생성 |
| `make gen-data [LIMIT=N]` | gen-schedule → gen-pairs 순차 실행 |
| `make sft` | SFT (configs/sft_scheduler.yaml) |
| `make sft-smoke` | Mac 스모크 SFT |
| `make sft-dgx-4b` | DGX 4B SFT |
| `make sft-dgx-8b` | DGX 8B SFT |
| `make dpo-final` | DPO (configs/dpo_final.yaml) |
| `make dpo-dgx-4b` | DGX 4B DPO |
| `make dpo-dgx-8b` | DGX 8B DPO |
| `make pipeline-dgx-4b` | DGX 4B SFT → DPO |
| `make pipeline-dgx-8b` | DGX 8B SFT → DPO |
| `make train-mac` | Mac DPO (configs/mac_train.yaml) |
| `make train-4b` | DGX DPO (configs/dgx_4b.yaml) |
| `make train-8b` | DGX DPO (configs/dgx_8b.yaml) |
| `make infer ADAPTER=... PROMPT=...` | 추론 |

---

## 전체 순서 한눈에

```bash
# 0. 환경
make setup-mac          # 또는 make setup-dgx
make test

# 1. 데이터 생성
make gen-data           # (dry-run: make gen-data LIMIT=5)

# 2. SFT
make sft-dgx-4b         # 또는 sft-dgx-8b

# 3. DPO
make dpo-dgx-4b         # 또는 dpo-dgx-8b

# 추론
make infer ADAPTER=outputs/dpo_dgx_4b PROMPT="..."
```

---

## 모듈 구조

```
src/drl/
├── device.py          — MPS/CUDA/CPU 감지 → DeviceProfile
├── config.py          — YAML → RunConfig 파싱
├── model.py           — Qwen3 로딩 + LoRA / SFT 어댑터 부착
├── data/
│   ├── loader.py      — HF 데이터셋 / 로컬 parquet → DPO 포맷
│   ├── scheduler.py   — scheduler_ko.parquet → ChatML SFT 포맷
│   └── augment.py     — Gemini/Claude 생성 + GPT-4o judge
├── train_sft.py       — SFTTrainer 래퍼
├── train_dpo.py       — DPOTrainer 래퍼
└── infer.py           — 어댑터 로드 + 텍스트 생성

scripts/
├── gen_korean_schedule.py    — 영문 시드 → 한국어 parquet (번역 + 페르소나 확장)
├── gen_nemotron_schedule.py  — Nemotron 페르소나 기반 스케줄 생성
├── gen_preference_pairs.py   — 4후보 생성 + judge → DPO parquet
├── analyze_dataset.py        — 데이터셋 콘솔 통계 출력
├── gen_dataset_report.py     — 데이터셋 분석 → docs/dataset_analysis.md
├── download_datasets.py      — HF 데이터셋 다운로드
├── download_nemotron.py      — Nemotron 페르소나 데이터 다운로드
└── translate_locally.py      — 로컬 모델 번역

configs/
├── sft_dgx_4b.yaml / sft_dgx_8b.yaml    — DGX SFT
├── sft_mac_smoke.yaml / sft_scheduler.yaml
├── dpo_dgx_4b.yaml / dpo_dgx_8b.yaml    — DGX DPO
├── dpo_final.yaml
├── dgx_4b.yaml / dgx_8b.yaml            — generic DPO
└── mac_smoke.yaml / mac_train.yaml       — Mac 전용

data/
├── scheduler_ko.parquet         — SFT 학습 데이터 (500행)
├── dpo_pairs.parquet            — DPO 페어 (현재 13행)
├── dpo_pairs.ckpt.jsonl         — 생성 체크포인트 (재개 가능)
├── events-scheduling.parquet    — 영문 시드 원본
├── ko_Ultrafeedback_binarized.parquet
├── orca-dpo-pairs-ko.parquet
└── nemotron_personas_korea.parquet

docs/
└── dataset_analysis.md          — 데이터셋 분석 리포트 (gen_dataset_report.py 생성)
```

---

## 환경 요구사항

| 환경 | HW | Python | 특이사항 |
|------|----|--------|---------|
| MacBook Pro | Apple Silicon (MPS) | 3.11 | bitsandbytes 미사용, bf16 LoRA |
| DGX Spark | ARM64 + Blackwell (CUDA) | 3.11 | bf16 LoRA (ARM64 bnb wheel 미존재) |

## 참고

- [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)
- [PEFT LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [`docs/dataset_analysis.md`](docs/dataset_analysis.md) — 데이터셋 분석 리포트
