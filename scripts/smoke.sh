#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# MPS에서 미지원 연산을 CPU로 폴백
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

echo "=== 스모크 테스트 (Qwen3-1.7B, MPS, 2 steps) ==="

uv run python -m drl.train_dpo --config configs/mac_smoke.yaml

echo ""
echo "=== 완료 ==="
ls -lh outputs/mac_smoke/ 2>/dev/null && echo "어댑터 저장 확인" || echo "[WARN] outputs/mac_smoke 없음"
