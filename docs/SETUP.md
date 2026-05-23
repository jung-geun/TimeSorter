# 환경 설정

## 사전 요구사항

| 환경 | OS / HW | Python | 특이사항 |
|------|---------|--------|---------|
| RTX GPU Docker | x86_64 + CUDA 12.4 | 3.11 | flash_attn, QLoRA 4-bit |
| DGX Spark | ARM64 + Blackwell CUDA | 3.11 | bf16 LoRA (bitsandbytes ARM64 미지원) |
| Mac | Apple Silicon (MPS) | 3.11 | bf16 LoRA |

[uv](https://docs.astral.sh/uv/) 패키지 매니저가 필요합니다:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 설치

### Mac (MPS)

```bash
make setup-mac
```

Qwen3.5-2B 기준으로 smoke test 가능. 학습은 제한적 (bf16, MPS 폴백).

### DGX / Linux (ARM64 CUDA)

```bash
make setup-dgx
```

bitsandbytes ARM64 wheel이 없어 4-bit QLoRA 미지원. bf16 LoRA로 동작.

### RTX GPU — Docker (권장)

```bash
# 이미지 빌드 (최초 1회, ~30분)
make docker-build
```

Docker 이미지 `timesorter:cu124` 포함 스택:

| 패키지 | 버전 |
|--------|------|
| Base | `nvidia/cuda:12.4.1-devel-ubuntu22.04` |
| PyTorch | `2.5.1+cu124` |
| flash-attn | `2.7.4` |
| bitsandbytes | `≥0.43` (QLoRA NF4) |

---

## API 키 설정

프로젝트 루트에 `.env` 파일 생성:

```bash
# 데이터 생성·판사 모델 (필수)
OPENAI_API_KEY=sk-...

# HuggingFace 모델/데이터셋 다운로드
HF_TOKEN=hf_...

# 학습 모니터링 (선택)
WANDB_API_KEY=...

# Claude / Gemini (데이터 생성 선택적 활용)
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

> `ANTHROPIC_API_KEY` 없으면 `claude -p` (OAuth) 자동 폴백.
> `GOOGLE_API_KEY` 없으면 `gemini CLI` → `OpenAI` 순으로 폴백.

---

## 동작 확인

```bash
make test    # pytest (7 passed)
make lint    # ruff check
make smoke   # 파이프라인 smoke test
```

Mac에서 SFT smoke test:

```bash
make sft-smoke   # Qwen3.5-2B, 2 steps, 64 samples
```

---

## 모델 / 데이터셋 사전 다운로드

```bash
make download          # HF 데이터셋 4종 (events-scheduling, ko_Ultrafeedback 등)
make download-models   # Qwen3.5-2B / 4B / 9B 가중치 HF 캐시에 저장
```

학습 실행 시 `from_pretrained()`가 자동으로 다운로드하므로, 이 단계는 사전 캐싱용입니다.
