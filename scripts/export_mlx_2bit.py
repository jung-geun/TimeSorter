#!/usr/bin/env python
"""Merged full model → MLX 2-bit 변환 + 자동 평가 (Apple Silicon 전용).

파이프라인:
  1. mlx_lm.convert  : HF merged model → MLX safetensors (2-bit 양자화)
  2. eval (선택)     : held-out parquet에 MLX 추론 → verify_chosen 채점

사전 작업 (Linux):
  uv run python scripts/merge_adapter.py \
      --adapter outputs/sft_q35_4b_v7 \
      --out outputs/sft_q35_4b_v7_merged

Mac에서 실행:
  # 변환만
  python scripts/export_mlx_2bit.py \
      --merged outputs/sft_q35_4b_v7_merged \
      --out outputs/sft_q35_4b_v7_mlx_2bit

  # 변환 + held-out 전체 평가 (scheduler_v3_eval.parquet 150행)
  python scripts/export_mlx_2bit.py \
      --merged outputs/sft_q35_4b_v7_merged \
      --out outputs/sft_q35_4b_v7_mlx_2bit \
      --eval data/scheduler_v3_eval.parquet

  # 변환 + 체인 전용 평가 (scheduler_v7_chain_eval.parquet 50행)
  python scripts/export_mlx_2bit.py \
      --merged outputs/sft_q35_4b_v7_merged \
      --out outputs/sft_q35_4b_v7_mlx_2bit \
      --eval data/scheduler_v7_chain_eval.parquet \
      --eval-out outputs/eval_mlx_2bit_chain.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))


# ── 공통 헬퍼 ─────────────────────────────────────────────────────────────────

def _require_mlx() -> None:
    try:
        import mlx  # noqa: F401
    except ImportError:
        print("[ERROR] mlx-lm이 설치되지 않았습니다.")
        print("  pip install mlx-lm")
        sys.exit(1)

    import platform
    if platform.system() != "Darwin":
        print("[WARN] MLX는 Apple Silicon(macOS)에서만 가속됩니다.")
        print("  변환은 가능하지만 추론 속도가 매우 느릴 수 있습니다.")


# ── 1. 변환 ──────────────────────────────────────────────────────────────────

def convert_2bit(merged_path: Path, out_path: Path, q_group_size: int = 64) -> None:
    """mlx_lm.convert로 HF full model → MLX 2-bit safetensors."""
    _require_mlx()
    from mlx_lm import convert

    if out_path.exists() and any(out_path.iterdir()):
        print(f"[변환] 이미 존재: {out_path} — 재변환 건너뜀 (--force로 강제)")
        return

    print(f"[변환] {merged_path} → {out_path}  (q_bits=2, group={q_group_size})")
    convert(
        hf_path=str(merged_path),
        mlx_path=str(out_path),
        quantize=True,
        q_bits=2,
        q_group_size=q_group_size,
    )
    size_mb = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / 1e6
    print(f"[완료] {out_path}  ({size_mb:.0f} MB)")


# ── 2. 평가 ──────────────────────────────────────────────────────────────────

def _infer_mlx(model, tokenizer, prompt: str, persona: str, today: str,
               max_tokens: int = 1024, thinking: bool = False) -> str:
    from mlx_lm import generate
    from timesorter.data.schema import SCHEDULER_SYSTEM_PROMPT_V3, render_system_prompt

    today_line = f"\n오늘 날짜: {today}" if today else ""
    system_content = render_system_prompt(SCHEDULER_SYSTEM_PROMPT_V3, persona) + today_line

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    chat_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    return generate(model, tokenizer, prompt=chat_input, max_tokens=max_tokens, verbose=False)


def _violation_kind(msg: str) -> str:
    if "지난 일정" in msg:
        return "past_rank"
    if "마감(태스크" in msg or "후순위" in msg:
        return "intraday_order"
    if "체인" in msg:
        return "chain"
    if "리스크" in msg:
        return "risk_importance"
    if "마감 없는" in msg:
        return "none_first"
    if "urgency/time_constraint" in msg:
        return "past_score"
    return "parse_or_count"


def evaluate(mlx_path: Path, eval_parquet: Path, eval_out: Path | None,
             max_tokens: int, thinking: bool) -> None:
    _require_mlx()
    from mlx_lm import load
    import pandas as pd
    from timesorter.data.schema import parse_or_repair
    from gen_schedule_v3 import Skeleton, TaskSpec, verify_chosen  # noqa: F401

    print(f"[eval] MLX 모델 로드: {mlx_path}")
    model, tokenizer = load(str(mlx_path))

    print(f"[eval] 데이터 로드: {eval_parquet}")
    df = pd.read_parquet(eval_parquet)
    print(f"[eval] {len(df)}행 평가 시작")

    per_scenario: dict[str, Counter] = {}
    details = []

    for i, row in df.iterrows():
        meta_raw = row.get("meta")
        if not meta_raw:
            continue
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw

        skel = Skeleton(
            scenario=meta["scenario"],
            today=meta["today"],
            tasks=[TaskSpec(**t) for t in meta["tasks"]],
        )

        raw = _infer_mlx(
            model, tokenizer,
            prompt=row["prompt"],
            persona=meta.get("persona", "직장인"),
            today=meta["today"],
            max_tokens=max_tokens,
            thinking=thinking,
        )
        out_json = parse_or_repair(raw)
        errors = verify_chosen(skel, out_json)

        sc = per_scenario.setdefault(skel.scenario, Counter())
        if errors:
            sc["fail"] += 1
            for e in errors:
                sc[_violation_kind(e)] += 1
        else:
            sc["pass"] += 1

        details.append({"idx": int(i), "scenario": skel.scenario, "errors": errors})

        done = sum(c["pass"] + c["fail"] for c in per_scenario.values())
        if done % 10 == 0:
            print(f"  진행 {done}/{len(df)}", flush=True)

    # ── 결과 출력 ──────────────────────────────────────────────────────────────
    print("\n═══════════════════════════════════════════════════════")
    print("시나리오별 통과율 (MLX 2-bit)")
    print("═══════════════════════════════════════════════════════")
    total_pass = total_fail = 0
    for name, c in sorted(per_scenario.items()):
        p, f = c["pass"], c["fail"]
        total_pass += p; total_fail += f
        rate = p / (p + f) * 100 if (p + f) else 0
        print(f"  {name:30s}  {p:3d}/{p+f:3d}  {rate:5.1f}%")
    overall = total_pass / (total_pass + total_fail) * 100 if (total_pass + total_fail) else 0
    print(f"  {'전체':30s}  {total_pass:3d}/{total_pass+total_fail:3d}  {overall:5.1f}%")
    print("═══════════════════════════════════════════════════════")

    result = {
        "model": str(mlx_path),
        "eval_file": str(eval_parquet),
        "n": len(df),
        "overall_pass_rate": round(overall / 100, 4),
        "per_scenario": {k: dict(v) for k, v in per_scenario.items()},
        "details": details,
    }

    if eval_out:
        eval_out.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n[저장] {eval_out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="merged model → MLX 2-bit 변환 + 평가")
    parser.add_argument("--merged", required=True, help="merge_adapter.py 출력 디렉토리")
    parser.add_argument("--out", required=True, help="MLX 2-bit 출력 디렉토리")
    parser.add_argument("--force", action="store_true", help="이미 변환된 경우도 재실행")
    parser.add_argument("--q-group-size", type=int, default=64, help="양자화 그룹 크기 (기본 64)")
    parser.add_argument("--eval", default=None,
                        help="평가용 parquet (예: data/scheduler_v3_eval.parquet)")
    parser.add_argument("--eval-out", default=None, help="평가 결과 JSON 저장 경로")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--thinking", action="store_true", help="Qwen3 thinking mode")
    args = parser.parse_args()

    merged = Path(args.merged)
    out = Path(args.out)

    if args.force and out.exists():
        import shutil
        shutil.rmtree(out)

    convert_2bit(merged, out, q_group_size=args.q_group_size)

    if args.eval:
        eval_out = Path(args.eval_out) if args.eval_out else None
        evaluate(out, Path(args.eval), eval_out, args.max_tokens, args.thinking)


if __name__ == "__main__":
    main()
