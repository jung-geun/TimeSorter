FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_NO_SYNC=1 \
    PATH="/opt/venv/bin:/root/.local/bin:$PATH" \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
        build-essential git curl ca-certificates ninja-build && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace

# uv.lock은 호스트 torch 2.12+cu130 기준 — lock 없이 pyproject만으로 설치
COPY pyproject.toml ./

# 1단계: venv 생성 + torch 2.5.1+cu124 (flash-attn 2.7.4 pre-built wheel과 ABI 매칭)
RUN uv venv /opt/venv --python 3.11 && \
    uv pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu124

# 2단계: 나머지 의존성 (torch 제외, lock 없이)
RUN uv pip install \
    "transformers>=4.46,<4.60" \
    "trl>=0.12,<0.20" \
    "peft>=0.13" \
    "datasets>=3.0" \
    "accelerate>=1.0" \
    pyyaml \
    huggingface-hub \
    "wandb>=0.27.0" \
    "python-dotenv>=1.0" \
    "anthropic>=0.28" \
    "openai>=1.30" \
    "pandas>=2.0" \
    "pyarrow>=14.0" \
    "bitsandbytes>=0.43" \
    packaging

# 3단계: flash-attn pre-built wheel (torch 2.5 + cu12 + cp311 + cxx11abiFALSE — PyPI torch는 old ABI)
RUN uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

# 소스 복사 후 로컬 패키지 설치
COPY . .
RUN uv pip install -e . --no-deps

CMD ["bash"]
