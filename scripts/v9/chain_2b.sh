#!/usr/bin/env bash
# v9 소형 모델(Qwen3.5-2B) 학습 검증 체인 — 단일 GPU라 4B DPO 종료 후 자동 순차 실행.
# 설계(advisor): 전역 set -e 금지(앞 단계 산출물 보호) · 단계별 start/end/exit 로깅 ·
#   SFT를 DPO보다 먼저 평가(핵심 질문 "소형도 학습되나"를 SFT만으로 먼저 확정) ·
#   각 학습 전 PROBE 스모크 · pgrep 패턴으로 4B 종료 대기 · GPU 단계 간 sleep.
cd /mnt/hdd/WD_8TB/code/TimeSorter || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=outputs/v9/chain_2b.log
mkdir -p outputs/v9

stamp(){ date '+%F %T'; }
S(){ echo "[$(stamp)] >>> START : $1" | tee -a "$LOG"; }
E(){ echo "[$(stamp)] <<< END   : $1 (exit=$2)" | tee -a "$LOG"; }
nonempty(){ [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }

echo "[$(stamp)] ===== 2B 체인 시작 =====" | tee -a "$LOG"

# 0) 4B DPO 종료 대기 (pgrep 패턴 — 이미 끝났으면 즉시 통과)
S "wait-4B-DPO"
while pgrep -f "train_dpo_v9.py --config configs/dpo_rtx12g_q35_4b_v9.yaml" >/dev/null 2>&1; do
  sleep 60
done
E "wait-4B-DPO" 0
sleep 20

# 1) 4B DPO 평가 (비치명적 — 실패/부재해도 2B 진행)
S "eval-4B-DPO"
if nonempty outputs/dpo_q35_4b_v9; then
  uv run python scripts/v9/eval_v9.py --model Qwen/Qwen3.5-4B --adapter outputs/dpo_q35_4b_v9 \
    --n 50 --out outputs/v9/eval_4b_dpo.json >>"$LOG" 2>&1
  E "eval-4B-DPO" $?
else
  echo "[$(stamp)] 4B DPO 출력 없음 — eval 건너뜀" | tee -a "$LOG"; E "eval-4B-DPO" skip
fi
sleep 10

# 2) 2B SFT: 프로브 → 전체
S "probe-2B-SFT"
PROBE=1 uv run python scripts/v9/train_sft_v9.py --config configs/sft_rtx12g_q35_2b_v9.yaml >>"$LOG" 2>&1
PR=$?; E "probe-2B-SFT" $PR
sleep 10
if [ $PR -eq 0 ]; then
  S "train-2B-SFT"
  uv run python scripts/v9/train_sft_v9.py --config configs/sft_rtx12g_q35_2b_v9.yaml >>"$LOG" 2>&1
  E "train-2B-SFT" $?
  sleep 10
else
  echo "[$(stamp)] 2B SFT 프로브 실패 — 전체 SFT 중단" | tee -a "$LOG"
fi

# 3) 2B base + 2B SFT 평가 (SFT 증거 먼저 확보)
if nonempty outputs/sft_q35_2b_v9; then
  S "eval-2B-base"
  uv run python scripts/v9/eval_v9.py --model Qwen/Qwen3.5-2B --adapter base \
    --n 50 --out outputs/v9/eval_2b_base.json >>"$LOG" 2>&1
  E "eval-2B-base" $?; sleep 10
  S "eval-2B-SFT"
  uv run python scripts/v9/eval_v9.py --model Qwen/Qwen3.5-2B --adapter outputs/sft_q35_2b_v9 \
    --n 50 --out outputs/v9/eval_2b_sft.json >>"$LOG" 2>&1
  E "eval-2B-SFT" $?; sleep 10

  # 4) 2B DPO: 프로브 → 전체
  S "probe-2B-DPO"
  PROBE=1 uv run python scripts/v9/train_dpo_v9.py --config configs/dpo_rtx12g_q35_2b_v9.yaml >>"$LOG" 2>&1
  PR=$?; E "probe-2B-DPO" $PR; sleep 10
  if [ $PR -eq 0 ]; then
    S "train-2B-DPO"
    uv run python scripts/v9/train_dpo_v9.py --config configs/dpo_rtx12g_q35_2b_v9.yaml >>"$LOG" 2>&1
    E "train-2B-DPO" $?; sleep 10
    # 5) 2B DPO 평가
    if nonempty outputs/dpo_q35_2b_v9; then
      S "eval-2B-DPO"
      uv run python scripts/v9/eval_v9.py --model Qwen/Qwen3.5-2B --adapter outputs/dpo_q35_2b_v9 \
        --n 50 --out outputs/v9/eval_2b_dpo.json >>"$LOG" 2>&1
      E "eval-2B-DPO" $?
    else
      echo "[$(stamp)] 2B DPO 출력 없음 — eval 건너뜀" | tee -a "$LOG"
    fi
  else
    echo "[$(stamp)] 2B DPO 프로브 실패 — 전체 DPO 중단" | tee -a "$LOG"
  fi
else
  echo "[$(stamp)] 2B SFT 어댑터 없음 — DPO/eval 중단" | tee -a "$LOG"
fi

echo "[$(stamp)] ===== 2B 체인 종료 =====" | tee -a "$LOG"
