#!/usr/bin/env python3
"""Qwen3.5-9B 4-way 벤치마크 — base / instruct / SFT / DPO 동일 평가.

⚠️ 실행 환경: RTX 4090(24GB) 등 9B 적재 가능 GPU 필요.
   9B sft/dpo 어댑터(outputs/sft_4090_1x_9b_v4, dpo_4090_1x_9b_v4)는
   학습한 4090 서버에 존재해야 한다. (RTX 3080 Ti 12GB에서는 9B 적재 불가)

4B의 benchmark_qwen35.py와 동일한 골격 규칙 자동 채점(verify_chosen).
schema-strict 통과율 + (no-FT는) content-level(스키마 무시) 통과율 동시 산출.

사용 (4090 서버):
  uv run python scripts/benchmark_9b.py --target all --limit 150
  uv run python scripts/benchmark_9b.py --target sft_9b,dpo_9b --limit 150 --force
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

from content_eval import repair_to_response, content_violations  # noqa: E402

TARGETS = {
    "base_no_rlhf": {"label": "Qwen3.5-9B-Base (no FT)",
                     "base_model": "Qwen/Qwen3.5-9B-Base", "adapter": None},
    "instruct_base": {"label": "Qwen3.5-9B (no adapter)",
                      "base_model": "Qwen/Qwen3.5-9B", "adapter": None},
    "sft_9b": {"label": "SFT v4 (Qwen3.5-9B)",
               "base_model": "Qwen/Qwen3.5-9B", "adapter": "outputs/sft_4090_1x_9b_v4"},
    "dpo_9b": {"label": "DPO v4 (Qwen3.5-9B)",
               "base_model": "Qwen/Qwen3.5-9B", "adapter": "outputs/dpo_4090_1x_9b_v4"},
}


def _patch():
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as _q
        if not _q.is_fast_path_available:
            _q.is_fast_path_available = True
    except Exception:
        pass


def _kind(msg: str) -> str:
    if "지난" in msg or "urgency" in msg: return "past_rank"
    if "마감(태스크" in msg or "후순위" in msg: return "intraday_order"
    if "체인" in msg: return "chain"
    if "리스크" in msg: return "risk_importance"
    if "마감 없는" in msg: return "none_first"
    return "parse_or_count"


def run(cfg, eval_file, limit, out_path):
    from collections import Counter
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from gen_schedule_v3 import Skeleton, TaskSpec, verify_chosen
    from timesorter.data.schema import SCHEDULER_SYSTEM_PROMPT_V3, render_system_prompt

    _patch()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(cfg["adapter"] or cfg["base_model"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=bnb,
                                                 device_map="auto", torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    if cfg["adapter"]:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg["adapter"])
    model.eval()

    df = pd.read_parquet(eval_file).head(limit)
    per_sc, per_kind = {}, Counter()
    schema_ok = content_ok = 0
    for _, row in df.iterrows():
        meta = json.loads(row["meta"])
        skel = Skeleton(scenario=meta["scenario"], today=meta["today"],
                        specs=[TaskSpec(**s) for s in meta["specs"]])
        system = render_system_prompt(SCHEDULER_SYSTEM_PROMPT_V3, str(row["persona"]), today=str(row["today"]))
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": str(row["prompt"])}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=1536, do_sample=False,
                                 pad_token_id=tok.eos_token_id or tok.pad_token_id,
                                 temperature=None, top_p=None)
        raw = tok.decode(gen[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        sv = verify_chosen(skel, raw)
        resp = repair_to_response(raw)
        cv = content_violations(skel, resp) if resp else ["JSON 블록 없음"]
        sc = meta["scenario"]
        per_sc.setdefault(sc, {"n": 0, "schema": 0, "content": 0})
        per_sc[sc]["n"] += 1
        if not sv: schema_ok += 1; per_sc[sc]["schema"] += 1
        else:
            for e in sv: per_kind[_kind(e)] += 1
        if not cv: content_ok += 1; per_sc[sc]["content"] += 1

    res = {"label": cfg["label"], "base_model": cfg["base_model"], "adapter": cfg.get("adapter"),
           "total": len(df), "schema_pass": schema_ok, "content_pass": content_ok,
           "per_scenario": per_sc, "per_kind": dict(per_kind)}
    Path(out_path).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}  schema {schema_ok}/{len(df)} ({schema_ok/len(df)*100:.1f}%) | "
          f"content {content_ok}/{len(df)} ({content_ok/len(df)*100:.1f}%)")
    del model; torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="all")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--eval-file", default="data/scheduler_v3_eval.parquet")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    keys = list(TARGETS) if args.target == "all" else [k.strip() for k in args.target.split(",")]
    for k in keys:
        cfg = TARGETS[k]
        if cfg["adapter"] and not Path(cfg["adapter"]).exists():
            print(f"[skip] {k}: adapter {cfg['adapter']} 없음 (4090 서버에서 실행 필요)")
            continue
        out = f"{args.out_dir}/eval_9b_{k}_n{args.limit}.json"
        if Path(out).exists() and not args.force:
            print(f"[skip] {out} 이미 존재 (--force)"); continue
        print(f"\n=== {cfg['label']} ===")
        run(cfg, args.eval_file, args.limit, out)


if __name__ == "__main__":
    main()
