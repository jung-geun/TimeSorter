# drl — Qwen3.5 할 일 우선순위 정렬 비서

Qwen3.5-4B / 9B 모델을 **한국어 할 일 우선순위 정렬** 태스크에 맞게 SFT → DPO 2단계 파인튜닝하는 파이프라인.

---

## 빠른 시작

### 1. 환경 설정

```bash
make setup-mac      # Mac (MPS)
make setup-dgx      # DGX / Linux ARM64 CUDA
make docker-build   # RTX GPU (Docker, flash-attn 포함)
```

`.env` 파일에 API 키 설정:

```
OPENAI_API_KEY=sk-...   # 데이터 생성 · judge 필수
HF_TOKEN=hf_...         # 모델 다운로드
WANDB_API_KEY=...       # 학습 모니터링 (선택)
```

→ 상세: [docs/SETUP.md](docs/SETUP.md)

### 2. 데이터 준비

```bash
make download          # HF 데이터셋 4종 다운로드
make download-models   # Qwen3.5-2B / 4B / 9B 가중치 사전 캐싱
make gen-data          # 스케줄 데이터 + preference pair 생성
```

→ 상세: [docs/DATASET.md](docs/DATASET.md)

### 3. 학습

```bash
make pipeline-auto      # VRAM 자동 감지 (모든 GPU, 권장)
make pipeline-auto-v2   # v2 JSON 스키마 버전
```

하드웨어를 직접 지정하려면:

| 환경 | VRAM | 명령 |
|------|------|------|
| Mac smoke | — | `make sft-smoke` |
| RTX 12GB Docker | 12 GB | `make pipeline-docker` |
| RTX 4090 × 2 | 24 GB×2 | `make pipeline-4090-2x-4b` |
| DGX 4B | 120 GB | `make pipeline-dgx-4b` |
| DGX 9B | 120 GB | `make pipeline-dgx-8b` |

`make pipeline-auto`는 실행 시점 VRAM과 GPU 수·모델 크기를 감지해 배치 크기·4bit·grad\_accum을 자동 산출합니다.

→ 상세: [docs/TRAINING.md](docs/TRAINING.md)

### 4. 추론 / 서빙

```bash
# 단일 추론
make infer ADAPTER=outputs/dpo_auto \
  PROMPT="보고서 작성(내일 마감), 팀 미팅(오후 2시), 메일 답장"

# vLLM OpenAI API 서빙 (포트 8000)
make serve-docker
make email-pipeline   # 이메일 → 스케줄 파이프라인
```

→ 상세: [docs/SERVING.md](docs/SERVING.md)

### 5. 검증

```bash
make validate   # gpt-5.5 판사 교차 검증
```

→ 상세: [docs/VALIDATION.md](docs/VALIDATION.md)

---

## 모듈 구조

```
src/drl/
├── device.py        — VRAM 감지 + auto_batch_config (GPU/모델 크기 자동 조정)
├── config.py        — YAML → RunConfig
├── model.py         — Qwen3.5 로딩 + LoRA / multi-GPU DDP 대응
├── data/
│   ├── loader.py    — HF 데이터셋 / parquet → DPO 포맷
│   ├── scheduler.py — SFT 데이터 → ChatML
│   ├── augment.py   — LLM 생성 + GPT judge
│   └── schema.py    — v2 JSON 스키마 정의
├── train_sft.py     — SFTTrainer 래퍼
├── train_dpo.py     — DPOTrainer 래퍼 (ref_model=None PEFT 트릭)
└── infer.py         — 어댑터 로드 + 텍스트 생성

configs/
├── sft_auto.yaml / dpo_auto.yaml           — 하드웨어 무관 (auto_batch)
├── sft_auto_v2.yaml / dpo_auto_v2.yaml     — v2 JSON 스키마 버전
├── sft_4090_2x_4b.yaml / dpo_4090_2x_4b.yaml  — RTX 4090 × 2
├── sft_rtx12g_4b.yaml / dpo_rtx12g_4b.yaml    — RTX 12GB QLoRA
├── sft_dgx_4b.yaml / sft_dgx_8b.yaml      — DGX 4B / 9B
└── accelerate_4090_2x.yaml                 — 2-GPU DDP 설정
```

---

## 상세 문서

| 문서 | 내용 |
|------|------|
| [docs/SETUP.md](docs/SETUP.md) | 환경 설정, Docker, API 키 |
| [docs/DATASET.md](docs/DATASET.md) | 데이터셋 명세, 생성 파이프라인 |
| [docs/TRAINING.md](docs/TRAINING.md) | 학습 설정, 결과, 하드웨어별 옵션 |
| [docs/SERVING.md](docs/SERVING.md) | vLLM 서빙, 이메일 파이프라인, API |
| [docs/VALIDATION.md](docs/VALIDATION.md) | 교차 검증 방법 및 분석 |
| [docs/BACKLOG.md](docs/BACKLOG.md) | 개선 계획 |
