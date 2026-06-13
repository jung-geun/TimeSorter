#!/usr/bin/env python3
"""n=30 content-level 벤치마크 — base/instruct를 schema·content 두 기준으로 채점.

각 출력에 대해:
  - schema_pass: verify_chosen(raw)  — 현행 (포맷 준수 필요)
  - content_pass: tasks id 재구성 후 verify_chosen — 추론 내용만

출력: presentation/01_model_comparison/content_n30.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from content_eval import repair_to_response, content_violations  # noqa: E402

MODELS = [
    {"key": "base", "label": "Qwen3.5-4B-Base (RLHF 미적용)", "base_model": "Qwen/Qwen3.5-4B-Base"},
    {"key": "instruct", "label": "Qwen3.5-4B (instruct, 어댑터 없음)", "base_model": "Qwen/Qwen3.5-4B"},
]
LIMIT = 30
OUT = Path("presentation/01_model_comparison/content_n30.json")


def _patch():
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as _q
        if not _q.is_fast_path_available:
            _q.is_fast_path_available = True
    except Exception:
        pass


def main():
    import pandas as pd
    from collections import Counter
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from gen_schedule_v3 import Skeleton, TaskSpec, verify_chosen
    from timesorter.data.schema import SCHEDULER_SYSTEM_PROMPT_V3, render_system_prompt

    _patch()
    df = pd.read_parquet("data/scheduler_v3_eval.parquet").head(LIMIT)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)

    out = {"limit": LIMIT, "models": {}}
    for m in MODELS:
        print(f"\n=== {m['label']} ===")
        tok = AutoTokenizer.from_pretrained(m["base_model"], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            m["base_model"], quantization_config=bnb, device_map="auto",
            dtype=torch.bfloat16, trust_remote_code=True).eval()

        rows = []
        schema_ok = content_ok = 0
        sc_schema, sc_content = Counter(), Counter()  # 시나리오별
        sc_total = Counter()
        for _, row in df.iterrows():
            meta = json.loads(row["meta"])
            skel = Skeleton(scenario=meta["scenario"], today=meta["today"],
                            specs=[TaskSpec(**s) for s in meta["specs"]])
            system = render_system_prompt(SCHEDULER_SYSTEM_PROMPT_V3,
                                          str(row["persona"]), today=str(row["today"]))
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": str(row["prompt"])}]
            try:
                text = tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
            except Exception:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=900, do_sample=False,
                                     pad_token_id=tok.eos_token_id or tok.pad_token_id,
                                     temperature=None, top_p=None)
            raw = tok.decode(gen[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            schema_viol = verify_chosen(skel, raw)
            resp = repair_to_response(raw)
            content_viol = content_violations(skel, resp) if resp else ["JSON 블록 없음"]
            s_pass = len(schema_viol) == 0
            c_pass = len(content_viol) == 0
            schema_ok += s_pass; content_ok += c_pass
            sc = meta["scenario"]; sc_total[sc] += 1
            sc_schema[sc] += s_pass; sc_content[sc] += c_pass
            rows.append({"scenario": sc, "schema_pass": s_pass, "content_pass": c_pass,
                         "content_viol": content_viol})

        out["models"][m["key"]] = {
            "label": m["label"], "total": LIMIT,
            "schema_pass": schema_ok, "content_pass": content_ok,
            "per_scenario": {sc: {"n": sc_total[sc], "schema": sc_schema[sc],
                                  "content": sc_content[sc]} for sc in sc_total},
            "rows": rows,
        }
        print(f"  스키마 통과: {schema_ok}/{LIMIT} ({schema_ok/LIMIT*100:.1f}%)")
        print(f"  내용 통과:   {content_ok}/{LIMIT} ({content_ok/LIMIT*100:.1f}%)")
        del model; torch.cuda.empty_cache()

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
