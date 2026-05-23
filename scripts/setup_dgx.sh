#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== DGX Spark (CUDA / ARM64) 환경 셋업 ==="

if ! command -v uv &>/dev/null; then
  echo "[ERROR] uv가 없습니다."
  echo "  설치: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

uv venv -p 3.11 .venv
echo "✓ .venv 생성 (Python 3.11)"

# DGX Spark: ARM64 + Blackwell (CUDA 12.4 호환)
# bitsandbytes ARM64 prebuilt wheel 미존재 → quant extra 미설치
source .venv/bin/activate

# PyTorch CUDA wheel 먼저 설치 (ARM64 CUDA wheel은 nightly 또는 cu124 인덱스 필요)
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
echo "✓ PyTorch CUDA wheel 설치"

# 나머지 의존성 (quant 제외)
uv sync --extra dev
echo "✓ 의존성 설치 완료"

python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA 가용: {torch.cuda.get_device_name(0)}')
else:
    print('[WARN] CUDA 미감지 — 드라이버/환경 확인 필요')
"

echo ""
echo "셋업 완료. 4B 학습 시작:"
echo "  uv run python -m timesorter.train_dpo --config configs/dgx_4b.yaml"
