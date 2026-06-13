#!/usr/bin/env python3
"""Base model 벤치마크 — 어댑터 없는 베이스 모델의 통과율 측정.

사용:
  uv run python scripts/benchmark_base_models.py --models all --limit 30
  uv run python scripts/benchmark_base_models.py --models qwen35_4b --limit 30
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

BASE_MODELS = {
    "qwen35_4b_instruct": {
        "model_id": "Qwen/Qwen3.5-4B",
        "label": "Qwen3.5-4B-Instruct (base)",
        "is_qwen35": True,
        "enable_thinking": False,
    },
    "qwen35_4b_base": {
        "model_id": "Qwen/Qwen3.5-4B-Base",
        "label": "Qwen3.5-4B-Base (no RLHF)",
        "is_qwen35": True,
        "enable_thinking": False,
    },
}


def _patch_qwen35():
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as _q
        if not _q.is_fast_path_available:
            _q.is_fast_path_available = True
    except Exception:
        pass


def eval_model(model_id: str, is_qwen35: bool, enable_thinking: bool,
               eval_file: str, limit: int, out_path: str) -> dict:
    from collections import Counter

    import pandas as pd
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from gen_schedule_v3 import Skeleton, TaskSpec, verify_chosen
    from timesorter.data.schema import SCHEDULER_SYSTEM_PROMPT_V3, render_system_prompt

    if is_qwen35:
        _patch_qwen35()

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(f"[load] {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_cfg,
        device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    print(f"[loaded] {model_id}")

    df = pd.read_parquet(eval_file).head(limit)
    per_kind = Counter()
    per_scenario: dict[str, dict] = {}
    details = []
    n_clean = 0

    for i, row in df.iterrows():
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
            chat_kwargs = {}
            if enable_thinking:
                chat_kwargs["enable_thinking"] = True
            else:
                try:
                    chat_kwargs["enable_thinking"] = False
                except Exception:
                    pass
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **chat_kwargs)
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
                from timesorter.data.schema import parse_or_repair
                if "지난 일정" in e or "urgency" in e:
                    per_kind["past_rank"] += 1
                elif "마감(태스크" in e or "후순위" in e:
                    per_kind["intraday_order"] += 1
                elif "체인" in e:
                    per_kind["chain"] += 1
                elif "리스크" in e:
                    per_kind["risk_importance"] += 1
                elif "마감 없는" in e:
                    per_kind["none_first"] += 1
                else:
                    per_kind["parse_or_count"] += 1
        details.append({"idx": i, "scenario": sc, "errors": errors})

        if (len(details)) % 5 == 0:
            print(f"  [{len(details)}/{limit}] clean so far: {n_clean}")

    result = {
        "model": model_id,
        "total": len(df),
        "clean": n_clean,
        "pass_rate": n_clean / len(df),
        "per_scenario": per_scenario,
        "per_kind": dict(per_kind),
        "details": details,
    }
    Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}  pass_rate={n_clean}/{len(df)} = {n_clean/len(df)*100:.1f}%")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all",
                        help="all, or comma-separated: qwen3_4b_instruct,qwen35_4b_instruct,qwen35_4b_base")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--eval-file", default="data/scheduler_v3_eval.parquet")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    if args.models == "all":
        keys = list(BASE_MODELS.keys())
    else:
        keys = [k.strip() for k in args.models.split(",")]

    for key in keys:
        cfg = BASE_MODELS[key]
        out_path = f"{args.out_dir}/eval_base_{key}_n{args.limit}.json"
        if Path(out_path).exists():
            print(f"[skip] {out_path} already exists")
            continue
        print(f"\n=== {cfg['label']} ===")
        eval_model(
            model_id=cfg["model_id"],
            is_qwen35=cfg["is_qwen35"],
            enable_thinking=cfg["enable_thinking"],
            eval_file=args.eval_file,
            limit=args.limit,
            out_path=out_path,
        )


if __name__ == "__main__":
    main()
