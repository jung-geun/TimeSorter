export HF_HOME ?= $(CURDIR)/models

.PHONY: smoke train-mac train-4b train-8b sft dpo-final gen-data infer setup-mac setup-dgx test lint \
        download download-models \
        sft-rtx12g-4b dpo-rtx12g-4b pipeline-rtx12g-4b \
        sft-rtx12g-4b-v2 dpo-rtx12g-4b-v2 pipeline-rtx12g-4b-v2 \
        sft-4090-2x-4b dpo-4090-2x-4b pipeline-4090-2x-4b \
        sft-4090-2x-4b-v2 dpo-4090-2x-4b-v2 pipeline-4090-2x-4b-v2 \
        sft-auto dpo-auto pipeline-auto \
        sft-auto-v2 dpo-auto-v2 pipeline-auto-v2 \
        docker-build sft-docker dpo-docker pipeline-docker infer-docker docker-shell \
        sft-docker-v2 dpo-docker-v2 \
        serve-build serve-docker serve-stop serve-sft-docker serve-sft-stop \
        email-pipeline email-pipeline-sft email-extract email-pipeline-v2 \
        gen-data-v2 \
        validate validate-sft validate-and-pipeline validate-and-pipeline-sft

download:
	uv run python scripts/download_datasets.py

# HuggingFace 모델 가중치 사전 다운로드 (학습 시 자동으로도 되지만, 미리 캐싱)
download-models:
	uv run huggingface-cli download Qwen/Qwen3.5-2B
	uv run huggingface-cli download Qwen/Qwen3.5-4B
	uv run huggingface-cli download Qwen/Qwen3.5-9B

smoke:
	bash scripts/smoke.sh

# Stage 1: 스케줄 SFT
sft:
	uv run python -m drl.train_sft --config configs/sft_scheduler.yaml

sft-smoke:
	PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false \
	uv run python -m drl.train_sft --config configs/sft_mac_smoke.yaml

# Stage 2: 데이터 생성 (dry-run은 LIMIT=5 make gen-data)
gen-schedule:
	uv run python scripts/gen_korean_schedule.py $(if $(LIMIT),--limit $(LIMIT),)

gen-pairs:
	uv run python scripts/gen_preference_pairs.py $(if $(LIMIT),--limit $(LIMIT),)

gen-data: gen-schedule gen-pairs

# v2 데이터 생성 (Phase B-G 순차 실행)
# 각 phase는 --limit 으로 dry-run 후 확인
gen-data-v2:
	@echo "v2 데이터 생성은 phase별로 수동 실행하세요. docs/PLAN_V2.md 참고"

# Stage 3: DPO (SFT 체크포인트 위에서)
dpo-final:
	uv run python -m drl.train_dpo --config configs/dpo_final.yaml

# DGX Spark (120GB) 전용 파이프라인
sft-dgx-4b:
	uv run python -m drl.train_sft --config configs/sft_dgx_4b.yaml

sft-dgx-8b:
	uv run python -m drl.train_sft --config configs/sft_dgx_8b.yaml

dpo-dgx-4b:
	uv run python -m drl.train_dpo --config configs/dpo_dgx_4b.yaml

dpo-dgx-8b:
	uv run python -m drl.train_dpo --config configs/dpo_dgx_8b.yaml

# 전체 DGX 파이프라인 순차 실행 (4B)
pipeline-dgx-4b: sft-dgx-4b dpo-dgx-4b

# 전체 DGX 파이프라인 순차 실행 (8B)
pipeline-dgx-8b: sft-dgx-8b dpo-dgx-8b

# 12GB VRAM (RTX 3060/4070/4080 등) — QLoRA 4-bit, 4B 모델
sft-rtx12g-4b:
	uv run python -m drl.train_sft --config configs/sft_rtx12g_4b.yaml

dpo-rtx12g-4b:
	uv run python -m drl.train_dpo --config configs/dpo_rtx12g_4b.yaml

pipeline-rtx12g-4b: sft-rtx12g-4b dpo-rtx12g-4b

# v2 — JSON 스키마 + 4축 점수 학습 (데이터 준비 후 실행)
sft-rtx12g-4b-v2:
	uv run python -m drl.train_sft --config configs/sft_rtx12g_4b_v2.yaml

dpo-rtx12g-4b-v2:
	uv run python -m drl.train_dpo --config configs/dpo_rtx12g_4b_v2.yaml

pipeline-rtx12g-4b-v2: sft-rtx12g-4b-v2 dpo-rtx12g-4b-v2

# RTX 4090 × 2 (24GB × 2) — bf16 LoRA, DDP 2-GPU
# 실행 전: pip install accelerate 확인
sft-4090-2x-4b:
	uv run accelerate launch --config_file configs/accelerate_4090_2x.yaml \
	  -m drl.train_sft --config configs/sft_4090_2x_4b.yaml

dpo-4090-2x-4b:
	uv run accelerate launch --config_file configs/accelerate_4090_2x.yaml \
	  -m drl.train_dpo --config configs/dpo_4090_2x_4b.yaml

pipeline-4090-2x-4b: sft-4090-2x-4b dpo-4090-2x-4b

sft-4090-2x-4b-v2:
	uv run accelerate launch --config_file configs/accelerate_4090_2x.yaml \
	  -m drl.train_sft --config configs/sft_4090_2x_4b_v2.yaml

dpo-4090-2x-4b-v2:
	uv run accelerate launch --config_file configs/accelerate_4090_2x.yaml \
	  -m drl.train_dpo --config configs/dpo_4090_2x_4b_v2.yaml

pipeline-4090-2x-4b-v2: sft-4090-2x-4b-v2 dpo-4090-2x-4b-v2

# 하드웨어 무관 — 실행 시점 VRAM으로 bs/grad_accum/4bit 자동 산출
# 단일 GPU: uv run python -m drl.train_sft --config configs/sft_auto.yaml
# 멀티 GPU: accelerate launch --config_file configs/accelerate_4090_2x.yaml -m drl.train_sft ...
sft-auto:
	uv run python -m drl.train_sft --config configs/sft_auto.yaml

dpo-auto:
	uv run python -m drl.train_dpo --config configs/dpo_auto.yaml

pipeline-auto: sft-auto dpo-auto

sft-auto-v2:
	uv run python -m drl.train_sft --config configs/sft_auto_v2.yaml

dpo-auto-v2:
	uv run python -m drl.train_dpo --config configs/dpo_auto_v2.yaml

pipeline-auto-v2: sft-auto-v2 dpo-auto-v2

sft-docker-v2:
	$(_DOCKER_RUN) make sft-rtx12g-4b-v2

dpo-docker-v2:
	$(_DOCKER_RUN) make dpo-rtx12g-4b-v2

# 기존 DPO (generic)
train-mac:
	PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false \
	uv run python -m drl.train_dpo --config configs/mac_train.yaml

train-4b:
	uv run python -m drl.train_dpo --config configs/dgx_4b.yaml

train-8b:
	uv run python -m drl.train_dpo --config configs/dgx_8b.yaml

# 사용: make infer ADAPTER=outputs/dpo_final PROMPT="보고서 작성(내일 마감), 점심 약속, 메일 답장"
infer:
	uv run python -m drl.infer --adapter $(ADAPTER) --prompt "$(PROMPT)"

setup-mac:
	bash scripts/setup_mac.sh

setup-dgx:
	bash scripts/setup_dgx.sh

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/

# ── Docker (CUDA 12.4 + flash-attn) ─────────────────────────────────────────
DOCKER_IMAGE ?= timesorter:cu124

_DOCKER_RUN = docker run --rm --gpus all \
	-v $(PWD):/workspace \
	-v /workspace/.venv \
	-v $(CURDIR)/models:/root/.cache/huggingface \
	--env-file .env \
	$(DOCKER_IMAGE)

docker-build:
	docker build -t $(DOCKER_IMAGE) .

sft-docker:
	$(_DOCKER_RUN) make sft-rtx12g-4b

dpo-docker:
	$(_DOCKER_RUN) make dpo-rtx12g-4b

pipeline-docker:
	$(_DOCKER_RUN) make pipeline-rtx12g-4b

# 사용: make infer-docker ADAPTER=outputs/sft_rtx12g_4b PROMPT="할 일 목록..."
infer-docker:
	$(_DOCKER_RUN) make infer ADAPTER=$(ADAPTER) PROMPT="$(PROMPT)"

docker-shell:
	docker run --rm -it --gpus all \
	-v $(PWD):/workspace \
	-v /workspace/.venv \
	-v $(CURDIR)/models:/root/.cache/huggingface \
	--env-file .env \
	$(DOCKER_IMAGE) bash

# ── vLLM 서빙 ────────────────────────────────────────────────────────────────
SERVE_IMAGE   ?= vllm/vllm-openai:v0.8.5
BASE_MODEL    ?= Qwen/Qwen3-4B-Instruct-2507
ADAPTER       ?= outputs/dpo_rtx12g_4b
LORA_NAME     ?= scheduler
SERVE_PORT    ?= 8000
GPU_MEM_UTIL  ?= 0.85
MAX_MODEL_LEN ?= 2048
EMAIL_DIR     ?= data/sample_emails
PERSONA       ?= 직장인

# 공통 vLLM 인자 빌더 (인자: 컨테이너 내 어댑터 경로, lora 이름)
# vllm/vllm-openai 이미지 ENTRYPOINT = python3 -m vllm.entrypoints.openai.api_server
# → docker run <image> 뒤에는 인자만 넘겨야 함
define _VLLM_ARGS
--model $(BASE_MODEL) \
  --enable-lora \
  --lora-modules $(2)=/workspace/$(1) \
  --dtype bfloat16 \
  --max-model-len $(MAX_MODEL_LEN) \
  --gpu-memory-utilization $(GPU_MEM_UTIL) \
  --max-lora-rank 16 \
  --host 0.0.0.0 --port 8000
endef

# vLLM 서빙 이미지 빌드 (Dockerfile.serve 사용, 선택적)
serve-build:
	docker build -f Dockerfile.serve -t timesorter-serve:latest .

# DPO 서버 기동 (포트 8000)
# 중지: make serve-stop  또는  docker stop timesorter-serve
serve-docker:
	docker run -d --name timesorter-serve --rm --gpus all \
	  -v $(CURDIR)/models:/root/.cache/huggingface \
	  -v $(PWD)/outputs:/workspace/outputs \
	  -p $(SERVE_PORT):8000 \
	  $(SERVE_IMAGE) \
	  $(call _VLLM_ARGS,$(ADAPTER),$(LORA_NAME))
	@echo ""
	@echo "[DPO 서버 기동] 로드까지 약 30~60초 소요됩니다."
	@echo "  헬스체크: curl http://localhost:$(SERVE_PORT)/health"
	@echo "  모델목록: curl http://localhost:$(SERVE_PORT)/v1/models"
	@echo "  중지:     make serve-stop"

serve-stop:
	docker stop timesorter-serve 2>/dev/null || true

# SFT 어댑터 서버 (포트 8001, DPO 서버와 동시 기동 가능)
# 중지: make serve-sft-stop  또는  docker stop timesorter-serve-sft
SFT_ADAPTER   ?= outputs/sft_rtx12g_4b
SFT_LORA_NAME ?= sft
SFT_PORT      ?= 8001

serve-sft-docker:
	docker run -d --name timesorter-serve-sft --rm --gpus all \
	  -v $(CURDIR)/models:/root/.cache/huggingface \
	  -v $(PWD)/outputs:/workspace/outputs \
	  -p $(SFT_PORT):8000 \
	  $(SERVE_IMAGE) \
	  $(call _VLLM_ARGS,$(SFT_ADAPTER),$(SFT_LORA_NAME))
	@echo ""
	@echo "[SFT 서버 기동] 로드까지 약 30~60초 소요됩니다."
	@echo "  헬스체크: curl http://localhost:$(SFT_PORT)/health"
	@echo "  모델목록: curl http://localhost:$(SFT_PORT)/v1/models"
	@echo "  중지:     make serve-sft-stop"
	@echo "  (DPO 서버: make serve-docker  →  port $(SERVE_PORT))"

serve-sft-stop:
	docker stop timesorter-serve-sft 2>/dev/null || true

# 이메일 → 스케줄 파이프라인 (vLLM 서버 필요)
# 사용: make email-pipeline EMAIL_DIR=data/sample_emails PERSONA=직장인
email-pipeline:
	uv run python scripts/email_to_schedule.py \
	  --email-dir $(EMAIL_DIR) \
	  --persona "$(PERSONA)" \
	  --server-url http://localhost:$(SERVE_PORT) \
	  --model $(LORA_NAME) \
	  --out outputs/schedule_result.json

# v2 모델로 이메일 파이프라인 (JSON 스키마 출력)
email-pipeline-v2:
	uv run python scripts/email_to_schedule.py \
	  --email-dir $(EMAIL_DIR) \
	  --persona "$(PERSONA)" \
	  --server-url http://localhost:$(SERVE_PORT) \
	  --model $(LORA_NAME) \
	  --schema-version v2 \
	  --out outputs/schedule_result_v2.json

# SFT 모델로 이메일 파이프라인 (포트 8001)
email-pipeline-sft:
	uv run python scripts/email_to_schedule.py \
	  --email-dir $(EMAIL_DIR) \
	  --persona "$(PERSONA)" \
	  --server-url http://localhost:$(SFT_PORT) \
	  --model $(SFT_LORA_NAME) \
	  --out outputs/schedule_result_sft.json

# 태스크 추출만 (vLLM 서버 불필요, OpenAI만 사용)
email-extract:
	uv run python scripts/email_to_schedule.py \
	  --email-dir $(EMAIL_DIR) \
	  --persona "$(PERSONA)" \
	  --extract-only

# ── 교차 검증 ─────────────────────────────────────────────────────────────────
JUDGE_MODEL   ?= gpt-5.5
RESULT_FILE   ?= outputs/schedule_result.json

# 기존 결과 파일을 판사 모델로 검증
# 사용: make validate [RESULT_FILE=...] [JUDGE_MODEL=gpt-5.5]
validate:
	uv run python scripts/validate_schedule.py \
	  --result $(RESULT_FILE) \
	  --email-dir $(EMAIL_DIR) \
	  --judge $(JUDGE_MODEL) \
	  --out outputs/validation_result.json

# SFT 결과 검증
validate-sft:
	uv run python scripts/validate_schedule.py \
	  --result outputs/schedule_result_sft.json \
	  --email-dir $(EMAIL_DIR) \
	  --judge $(JUDGE_MODEL) \
	  --out outputs/validation_result_sft.json

# 파이프라인 실행 후 즉시 검증 (순차)
# 사용: make validate-and-pipeline EMAIL_DIR=data/sample_emails
validate-and-pipeline: email-pipeline validate

# SFT 파이프라인 실행 후 즉시 검증
validate-and-pipeline-sft: email-pipeline-sft validate-sft
