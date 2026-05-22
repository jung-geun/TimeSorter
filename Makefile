.PHONY: smoke train-mac train-4b train-8b sft dpo-final gen-data infer setup-mac setup-dgx test lint \
        sft-rtx12g-4b dpo-rtx12g-4b pipeline-rtx12g-4b \
        docker-build sft-docker dpo-docker pipeline-docker infer-docker docker-shell

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
