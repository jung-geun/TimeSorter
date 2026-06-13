#!/usr/bin/env python3
"""Qwen3.5 벤치마크 — base / instruct / fine-tuned 어댑터 통과율 측정.

평가 대상:
  - Qwen3.5-4B (instruct, no adapter)
  - Qwen3.5-4B-Base (no RLHF, no adapter)
  - outputs/sft_q35_4b_v4   (Qwen3.5-4B + SFT v4)
  - outputs/dpo_q35_4b_v5   (Qwen3.5-4B + SFT + DPO v5)
  - outputs/sft_q35_4b_base_v4  (Base + SFT v4, 학습 완료 후)
  - outputs/dpo_q35_4b_base_v5  (Base + SFT + DPO v5, 학습 완료 후)

사용:
  uv run python scripts/benchmark_qwen35.py                   # 전체 (사용 가능한 것만)
  uv run python scripts/benchmark_qwen35.py --target instruct_base --limit 30
  uv run python scripts/benchmark_qwen35.py --target sft_q35  --limit 150
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore")

# ── 평가 대상 정의 ──────────────────────────────────────────────────────────
TARGETS = {
    "instruct_base": {
        "label": "Qwen3.5-4B\n(no adapter)",
        "base_model": "Qwen/Qwen3.5-4B",
        "adapter": None,
        "is_base_model": False,
    },
    "base_no_rlhf": {
        "label": "Qwen3.5-4B-Base\n(no adapter)",
        "base_model": "Qwen/Qwen3.5-4B-Base",
        "adapter": None,
        "is_base_model": True,
    },
    "sft_q35": {
        "label": "SFT v4\n(Qwen3.5-4B)",
        "base_model": "Qwen/Qwen3.5-4B",
        "adapter": "outputs/sft_q35_4b_v4",
        "is_base_model": False,
    },
    "dpo_q35": {
        "label": "DPO v5\n(Qwen3.5-4B)",
        "base_model": "Qwen/Qwen3.5-4B",
        "adapter": "outputs/dpo_q35_4b_v5",
        "is_base_model": False,
    },
    "sft_q35_base": {
        "label": "SFT v4\n(Qwen3.5-4B-Base)",
        "base_model": "Qwen/Qwen3.5-4B-Base",
        "adapter": "outputs/sft_q35_4b_base_v4",
        "is_base_model": True,
    },
    "dpo_q35_base": {
        "label": "DPO v5\n(Qwen3.5-4B-Base)",
        "base_model": "Qwen/Qwen3.5-4B-Base",
        "adapter": "outputs/dpo_q35_4b_base_v5",
        "is_base_model": True,
    },
}


def _patch_qwen35() -> None:
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as _q
        if not _q.is_fast_path_available:
            _q.is_fast_path_available = True
            print("[patch] Qwen3.5 fast path forced True")
    except Exception:
        pass


def _violation_kind(msg: str) -> str:
    if "지난 일정" in msg or "urgency" in msg:
        return "past_rank"
    if "마감(태스크" in msg or "후순위" in msg:
        return "intraday_order"
    if "체인" in msg:
        return "chain"
    if "리스크" in msg:
        return "risk_importance"
    if "마감 없는" in msg:
        return "none_first"
    return "parse_or_count"


def run_eval(cfg: dict, eval_file: str, limit: int, out_path: str) -> dict:
    from collections import Counter

    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from gen_schedule_v3 import Skeleton, TaskSpec, verify_chosen
    from timesorter.data.schema import SCHEDULER_SYSTEM_PROMPT_V3, render_system_prompt

    _patch_qwen35()

    base_model = cfg["base_model"]
    adapter_path = cfg.get("adapter")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(f"[load] {base_model}" + (f" + {adapter_path}" if adapter_path else ""))
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_cfg,
        device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        model = base
    model.eval()
    print("[loaded]")

    df = pd.read_parquet(eval_file).head(limit)
    per_kind: Counter = Counter()
    per_scenario: dict[str, dict] = {}
    details = []
    n_clean = 0

    for _, row in df.iterrows():
        meta = json.loads(row["meta"])
        skel = Skeleton(
            scenario=meta["scenario"], today=meta["today"],
            specs=[TaskSpec(**s) for s in meta["specs"]],
        )
        system = render_system_prompt(
            SCHEDULER_SYSTEM_PROMPT_V3, str(row["persona"]), today=str(row["today"]))
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(row["prompt"])},
        ]

        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except Exception:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=800, do_sample=False,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
                temperature=None, top_p=None,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(gen, skip_special_tokens=True).strip()

        errors = verify_chosen(skel, raw)
        sc = meta["scenario"]
        if sc not in per_scenario:
            per_scenario[sc] = {"n": 0, "violated": 0}
        per_scenario[sc]["n"] += 1
        if not errors:
            n_clean += 1
        else:
            per_scenario[sc]["violated"] = per_scenario[sc].get("violated", 0) + 1
            for e in errors:
                per_kind[_violation_kind(e)] += 1
        details.append({"idx": len(details), "scenario": sc, "errors": errors, "raw": raw[:500]})

        if len(details) % 5 == 0:
            print(f"  [{len(details)}/{limit}] clean={n_clean} ({n_clean/len(details)*100:.0f}%)")

    result = {
        "label": cfg["label"],
        "base_model": base_model,
        "adapter": adapter_path,
        "total": len(df),
        "clean": n_clean,
        "pass_rate": n_clean / len(df),
        "per_scenario": per_scenario,
        "per_kind": dict(per_kind),
        "details": details,
    }
    Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}  {n_clean}/{len(df)} = {n_clean/len(df)*100:.1f}%")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", default="all",
        help="all, or comma-separated keys: instruct_base,base_no_rlhf,sft_q35,dpo_q35,sft_q35_base,dpo_q35_base")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--eval-file", default="data/scheduler_v3_eval.parquet")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--force", action="store_true", help="기존 결과 덮어쓰기")
    args = parser.parse_args()

    if args.target == "all":
        keys = list(TARGETS.keys())
    else:
        keys = [k.strip() for k in args.target.split(",")]

    for key in keys:
        cfg = TARGETS[key]

        # 어댑터 필요한데 없으면 skip
        if cfg["adapter"] and not Path(cfg["adapter"]).exists():
            print(f"[skip] {key}: adapter {cfg['adapter']} not found")
            continue

        out_path = f"{args.out_dir}/eval_qwen35_{key}_n{args.limit}.json"
        if Path(out_path).exists() and not args.force:
            print(f"[skip] {out_path} already exists (--force to overwrite)")
            continue

        print(f"\n{'='*50}")
        print(f"Target: {cfg['label'].replace(chr(10), ' ')}")
        print(f"{'='*50}")
        run_eval(cfg, args.eval_file, args.limit, out_path)


if __name__ == "__main__":
    main()
