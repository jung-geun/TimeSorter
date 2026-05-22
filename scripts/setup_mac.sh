#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== MacBook (MPS) 환경 셋업 ==="

if ! command -v uv &>/dev/null; then
  echo "[ERROR] uv가 없습니다."
  echo "  설치: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

uv venv -p 3.11 .venv
echo "✓ .venv 생성 (Python 3.11)"

uv sync --extra dev
echo "✓ 의존성 설치 완료"

echo ""
echo "셋업 완료. 스모크 테스트:"
echo "  bash scripts/smoke.sh"
