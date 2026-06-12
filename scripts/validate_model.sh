#!/usr/bin/env bash
# 학습된 어댑터를 docker(vLLM) 또는 로컬(transformers)에 올려 간단 검증.
#
# 사용:
#   bash scripts/validate_model.sh                                 # 기본 어댑터, docker, 빠른 검증(30)
#   bash scripts/validate_model.sh outputs/dpo_rtx12g_4b_v5        # 어댑터 지정
#   MODE=local bash scripts/validate_model.sh outputs/sft_rtx12g_4b_v4   # GPU 서버 없이 단건 추론
#   FULL=1 bash scripts/validate_model.sh outputs/...              # held-out 150 전체 + guard
#   KEEP=1 bash scripts/validate_model.sh outputs/...              # 검증 후 서버 유지
#
# 출력: 골격 규칙 자동 채점 결과(통과율·시나리오별·위반 유형) + 샘플 추론 1건
set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER="${1:-outputs/sft_rtx12g_4b_v4}"
MODE="${MODE:-docker}"
FULL="${FULL:-0}"
KEEP="${KEEP:-0}"
PORT="${PORT:-8000}"
EVAL_SET="${EVAL_SET:-data/scheduler_v3_eval.parquet}"

[ -f "$ADAPTER/adapter_model.safetensors" ] || { echo "[에러] $ADAPTER 에 어댑터 가중치 없음"; exit 1; }

if [ "$MODE" = "local" ]; then
  echo "=== 로컬 추론 검증 (서버 불필요, 단건) ==="
  uv run python -m timesorter.infer --adapter "$ADAPTER" --schema-version v3 \
    --today "$(date +%F)" \
    --prompt "- 보고서 제출 (오늘 17:00까지)
- PR 리뷰 (오늘 오전까지, 미처리 시 팀장 에스컬레이션)
- 어제 14:00 회의 자료 정리 ($(date -d yesterday +%F) 14:00)
- 책상 정리"
  exit 0
fi

# 어댑터가 기록한 실제 베이스 모델 사용 (Qwen3 구세대/Qwen3.5 신세대 어댑터 모두 호환)
BASE_MODEL=$(uv run python -c "import json;print(json.load(open('$ADAPTER/adapter_config.json'))['base_model_name_or_path'])")
SERVE_IMAGE="${SERVE_IMAGE:-vllm/vllm-openai:latest}"   # Qwen3.5는 v0.8.5 미지원
echo "=== docker(vLLM) 서빙 검증: $ADAPTER (base: $BASE_MODEL) ==="
docker stop timesorter-serve 2>/dev/null && sleep 3 || true

docker run -d --name timesorter-serve --rm --gpus all \
  -v "$PWD/models:/root/.cache/huggingface" \
  -v "$PWD/outputs:/workspace/outputs" \
  -p "$PORT:8000" "$SERVE_IMAGE" \
  --model "$BASE_MODEL" --enable-lora \
  --lora-modules "scheduler=/workspace/$ADAPTER" \
  --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --max-lora-rank 16 --host 0.0.0.0 --port 8000 >/dev/null

echo -n "[대기] vLLM 기동 중"
for i in $(seq 1 60); do
  curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q scheduler && break
  echo -n "."; sleep 10
done
echo " 준비 완료"

LIMIT_ARGS="--limit 30"; [ "$FULL" = "1" ] && LIMIT_ARGS=""
echo; echo "--- 골격 규칙 자동 채점 (무처리) ---"
uv run python scripts/eval_scheduler.py --input "$EVAL_SET" $LIMIT_ARGS \
  --server-url "http://localhost:$PORT"
if [ "$FULL" = "1" ]; then
  echo; echo "--- guard rerank ---"
  uv run python scripts/eval_scheduler.py --input "$EVAL_SET" --rerank --rerank-mode guard \
    --server-url "http://localhost:$PORT"
fi

echo; echo "--- 샘플 추론 ---"
uv run python - <<PYEOF
import sys; sys.path.insert(0, 'src')
import datetime
from openai import OpenAI
from timesorter.data.schema import (SCHEDULER_SYSTEM_PROMPT_V3, ScheduleResponse,
                                    parse_or_repair, render_system_prompt, response_to_text)
today = datetime.date.today().isoformat()
yest = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
client = OpenAI(api_key="EMPTY", base_url="http://localhost:$PORT/v1")
r = client.chat.completions.create(model="scheduler", max_tokens=2048, temperature=0.0,
    extra_body={"guided_json": ScheduleResponse.model_json_schema()},
    messages=[{"role": "system", "content": render_system_prompt(SCHEDULER_SYSTEM_PROMPT_V3, "직장인", today=today)},
              {"role": "user", "content": f"[직장인의 오늘의 할 일 목록]\n- 보고서 제출 ({today} 17:00까지)\n- PR 리뷰 ({today} 오전까지, 미처리 시 팀장 에스컬레이션)\n- 회의 자료 정리 ({yest} 14:00 회의)\n- 책상 정리"}])
print(response_to_text(parse_or_repair(r.choices[0].message.content)))
PYEOF

if [ "$KEEP" != "1" ]; then
  echo; echo "[정리] 서버 중지 (유지하려면 KEEP=1)"
  docker stop timesorter-serve >/dev/null
fi
