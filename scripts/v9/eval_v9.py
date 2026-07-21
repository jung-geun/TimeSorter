#!/usr/bin/env python
"""v9 정량 성능 평가 — 배치 생성 + total_score 후처리 + gold 대비 실측 지표.

--adapter <path> 또는 'base'(SFT 없는 기본 Qwen3.5). 같은 입력 집합으로 비교.
total_score는 모델 4축으로 후처리 재계산(recompute_total_scores) 후 평가.

지표(행별→평균):
  parse_rate          유효 JSON 파싱율
  coverage_rate       입력 task_id 전체를 정확히 출력
  verify_pass_rate    verify_chosen_v9 무위반(total 후처리 적용)
  rank_exact          gold와 priority_rank 정확 일치 비율
  axis_mae            4축 점수 평균절대오차(vs gold)
  total_mae           후처리 total vs gold total 평균절대오차
  sched_feasible_rate 시간블록 중복없음 + start≥now
  chain_order_rate    체인 선후 순서 준수
  deadline_met_rate   마감 있는 태스크가 블록 내 마감 전 완료

사용:
  uv run python scripts/v9/eval_v9.py --adapter outputs/sft_q35_4b_v9 --n 50 --batch 4 --out outputs/v9/eval_sft.json
  uv run python scripts/v9/eval_v9.py --adapter base --n 50 --batch 4 --out outputs/v9/eval_base.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import torch

sys.path.insert(0, "src")
from timesorter.data import schema_v9 as S  # noqa: E402
from timesorter.data.schema import system_prompt_for  # noqa: E402
from timesorter.device import detect  # noqa: E402
from timesorter.model import load_model_and_tokenizer  # noqa: E402


def _prompt_str(tok, input_json: str, schema: str = "v9") -> str:
    msgs = [{"role": "system", "content": system_prompt_for(schema)},
            {"role": "user", "content": input_json}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)


def _metrics(inp: S.ScheduleInput, out, gold, chain_pairs):
    """단일 행 지표 dict (out은 파싱된 ScheduleResponseV9 또는 None)."""
    m = {"parsed": out is not None}
    if out is None:
        return m
    in_ids = {t.task_id for t in inp.tasks}
    out_ids = [s.task_id for s in out.scheduled_tasks]
    m["coverage"] = sorted(out_ids) == sorted(in_ids)
    errs = S.verify_chosen_v9(inp, out, chain_pairs)
    m["verify_pass"] = not errs
    m["chain_order"] = not any("체인 순서" in e for e in errs)
    m["sched_feasible"] = not any(("중복" in e) or ("current_time 이전" in e) for e in errs)
    m["deadline_met"] = not any("실현불가" in e for e in errs)

    # gold 대비 (공통 task_id에서)
    gmap = {s.task_id: s for s in gold.scheduled_tasks} if gold else {}
    omap = {s.task_id: s for s in out.scheduled_tasks}
    common = [t for t in omap if t in gmap]
    if common:
        m["rank_exact"] = sum(omap[t].priority_rank == gmap[t].priority_rank
                              for t in common) / len(common)
        axes = ("deadline_proximity", "task_importance", "task_chaining", "urgency")
        diffs = [abs(getattr(omap[t].scoring, a) - getattr(gmap[t].scoring, a))
                 for t in common for a in axes]
        m["axis_mae"] = sum(diffs) / len(diffs)
        m["total_mae"] = sum(abs(omap[t].scoring.total_score - gmap[t].scoring.total_score)
                             for t in common) / len(common)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="어댑터 경로 또는 'base'")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B",
                    help="베이스 모델 (어댑터와 동일 계열이어야 함, 예: Qwen/Qwen3.5-2B)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--data", default="data/scheduler_v9.parquet")
    ap.add_argument("--schema", default="v9", help="시스템 프롬프트 버전 (EN-US: v9_en)")
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--out", default="outputs/v9/eval_result.json")
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(args.data).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    profile = detect()
    adapter = None if args.adapter == "base" else args.adapter
    model, tok = load_model_and_tokenizer(
        model_name=args.model, profile=profile,
        lora_r=16, lora_alpha=32, lora_dropout=0.0, use_4bit=True,
        gradient_checkpointing=False, sft_adapter_path=adapter)
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # 프롬프트가 너무 길지 않은 행 우선(생성 여유) — n개 수집
    rows = []
    for _, r in df.iterrows():
        ps = _prompt_str(tok, str(r["prompt"]), schema=args.schema)
        if len(tok(ps, add_special_tokens=False)["input_ids"]) <= 1900:
            rows.append((r, ps))
        if len(rows) >= args.n:
            break
    print(f"[eval] {args.adapter}: {len(rows)}행 평가 (batch {args.batch})")

    all_m = []
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        prompts = [ps for _, ps in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j, (r, _) in enumerate(chunk):
            out_ids = gen[j][enc["input_ids"].shape[1]:]
            text = tok.decode(out_ids, skip_special_tokens=True)
            parsed = S.parse_lenient_v9(text)
            if parsed is not None:
                S.recompute_total_scores(parsed)
            inp = S.ScheduleInput.model_validate_json(str(r["prompt"]))
            gold = S.parse_lenient_v9(str(r["chosen"]))
            if gold is not None:
                S.recompute_total_scores(gold)
            cp = [tuple(p) for p in json.loads(r["meta"]).get("chain_pairs", [])]
            all_m.append(_metrics(inp, parsed, gold, cp))
        print(f"  {min(i+args.batch, len(rows))}/{len(rows)} 완료")

    # 집계
    n = len(all_m)
    def rate(k):
        v = [m[k] for m in all_m if k in m]
        return round(sum(v) / len(v), 3) if v else None
    summary = {
        "adapter": args.adapter, "n": n,
        "parse_rate": round(sum(m["parsed"] for m in all_m) / n, 3),
        "coverage_rate": rate("coverage"),
        "verify_pass_rate": rate("verify_pass"),
        "chain_order_rate": rate("chain_order"),
        "sched_feasible_rate": rate("sched_feasible"),
        "deadline_met_rate": rate("deadline_met"),
        "rank_exact": rate("rank_exact"),
        "axis_mae": rate("axis_mae"),
        "total_mae": rate("total_mae"),
    }
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n=== 결과 ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if torch.cuda.is_available():
        print(f"[VRAM] peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
