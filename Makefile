.PHONY: smoke train-mac train-4b train-8b sft dpo-final gen-data infer setup-mac setup-dgx test lint \
        sft-rtx12g-4b dpo-rtx12g-4b pipeline-rtx12g-4b \
        docker-build sft-docker dpo-docker pipeline-docker infer-docker docker-shell \
        serve-build serve-docker serve-stop email-pipeline email-extract

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
	-v $(HOME)/.cache/huggingface:/root/.cache/huggingface \
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
	-v $(HOME)/.cache/huggingface:/root/.cache/huggingface \
	--env-file .env \
	$(DOCKER_IMAGE) bash

# ── vLLM 서빙 ────────────────────────────────────────────────────────────────
SERVE_IMAGE   ?= timesorter-serve:latest
ADAPTER       ?= outputs/dpo_rtx12g_4b
LORA_NAME     ?= scheduler
SERVE_PORT    ?= 8000
GPU_MEM_UTIL  ?= 0.85
EMAIL_DIR     ?= data/sample_emails
PERSONA       ?= 직장인

# vLLM 서빙 이미지 빌드 (Dockerfile.serve 사용)
serve-build:
	docker build -f Dockerfile.serve -t $(SERVE_IMAGE) .

# vLLM 서버 기동 (백그라운드 데몬, GPU 점유)
# 중지: make serve-stop  또는  docker stop timesorter-serve
serve-docker:
	docker run -d --name timesorter-serve --rm --gpus all \
	  -v $(HOME)/.cache/huggingface:/root/.cache/huggingface \
	  -v $(PWD)/outputs:/workspace/outputs \
	  -p $(SERVE_PORT):8000 \
	  -e LORA_PATH=$(ADAPTER) \
	  -e LORA_NAME=$(LORA_NAME) \
	  -e GPU_MEM_UTIL=$(GPU_MEM_UTIL) \
	  $(SERVE_IMAGE)
	@echo ""
	@echo "[서버 기동] 로드까지 약 30~60초 소요됩니다."
	@echo "  헬스체크: curl http://localhost:$(SERVE_PORT)/health"
	@echo "  모델목록: curl http://localhost:$(SERVE_PORT)/v1/models"
	@echo "  중지:     make serve-stop"

serve-stop:
	docker stop timesorter-serve 2>/dev/null || true

# 이메일 → 스케줄 파이프라인 (vLLM 서버 필요)
# 사용: make email-pipeline EMAIL_DIR=data/sample_emails PERSONA=직장인
email-pipeline:
	uv run python scripts/email_to_schedule.py \
	  --email-dir $(EMAIL_DIR) \
	  --persona "$(PERSONA)" \
	  --server-url http://localhost:$(SERVE_PORT) \
	  --model $(LORA_NAME) \
	  --out outputs/schedule_result.json

# 태스크 추출만 (vLLM 서버 불필요, OpenAI만 사용)
email-extract:
	uv run python scripts/email_to_schedule.py \
	  --email-dir $(EMAIL_DIR) \
	  --persona "$(PERSONA)" \
	  --extract-only
