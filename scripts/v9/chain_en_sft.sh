#!/usr/bin/env bash
# v9 EN-US 파일럿 SFT 체인 — 238행으로 end-to-end 학습 증명.
# probe → full SFT → eval(EN base) → eval(EN SFT). schema=v9_en(영어 프롬프트).
# 설계: 전역 set -e 금지, 단계별 START/END(exit) 로깅.
cd /mnt/hdd/WD_8TB/code/TimeSorter || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=outputs/v9/chain_en_sft.log
mkdir -p outputs/v9

stamp(){ date '+%F %T'; }
S(){ echo "[$(stamp)] >>> START : $1" | tee -a "$LOG"; }
E(){ echo "[$(stamp)] <<< END   : $1 (exit=$2)" | tee -a "$LOG"; }
nonempty(){ [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }

CFG=configs/sft_rtx12g_q35_4b_v9_en.yaml
echo "[$(stamp)] ===== EN SFT 체인 시작 =====" | tee -a "$LOG"

# 1) 프로브
S "probe-EN-SFT"
PROBE=1 uv run python scripts/v9/train_sft_v9.py --config "$CFG" >>"$LOG" 2>&1
PR=$?; E "probe-EN-SFT" $PR; sleep 10

if [ $PR -eq 0 ]; then
  # 2) 전체 SFT
  S "train-EN-SFT"
  uv run python scripts/v9/train_sft_v9.py --config "$CFG" >>"$LOG" 2>&1
  E "train-EN-SFT" $?; sleep 10

  if nonempty outputs/sft_q35_4b_v9_en; then
    # 3) eval: EN base
    S "eval-EN-base"
    uv run python scripts/v9/eval_v9.py --model Qwen/Qwen3.5-4B --adapter base \
      --schema v9_en --data data/scheduler_v9_en.parquet --n 40 \
      --out outputs/v9/eval_en_base.json >>"$LOG" 2>&1
    E "eval-EN-base" $?; sleep 10
    # 4) eval: EN SFT
    S "eval-EN-SFT"
    uv run python scripts/v9/eval_v9.py --model Qwen/Qwen3.5-4B --adapter outputs/sft_q35_4b_v9_en \
      --schema v9_en --data data/scheduler_v9_en.parquet --n 40 \
      --out outputs/v9/eval_en_sft.json >>"$LOG" 2>&1
    E "eval-EN-SFT" $?
  else
    echo "[$(stamp)] EN SFT 어댑터 없음 — eval 중단" | tee -a "$LOG"
  fi
else
  echo "[$(stamp)] EN SFT 프로브 실패 — 전체 중단" | tee -a "$LOG"
fi

echo "[$(stamp)] ===== EN SFT 체인 종료 =====" | tee -a "$LOG"
