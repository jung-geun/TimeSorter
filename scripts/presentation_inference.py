#!/usr/bin/env python3
"""발표자료 Part 1 — 4개 모델 동일 쿼리 추론 + verify_chosen 정답판정.

대상 모델:
  base     = Qwen/Qwen3.5-4B-Base   (RLHF 미적용 사전학습 베이스)
  instruct = Qwen/Qwen3.5-4B        (instruct/thinking, 어댑터 없음)
  sft      = Qwen/Qwen3.5-4B + outputs/sft_q35_4b_v4
  dpo      = Qwen/Qwen3.5-4B + outputs/dpo_q35_4b_v5

eval set(scheduler_v3_eval.parquet)에서 시나리오별 대표 쿼리를 뽑아
각 모델의 전체 raw 출력을 캡처하고, verify_chosen()으로 결정론적 정답/위반을 기록.

출력: presentation/01_model_comparison/raw_outputs.json
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

# 시나리오별 대표 인덱스 (dependency_chain은 약점이라 2개)
QUERY_INDICES = [0, 1, 3, 8, 9, 11]

MODELS = [
    {"key": "base",     "label": "Qwen3.5-4B-Base (RLHF 미적용)",
     "base_model": "Qwen/Qwen3.5-4B-Base", "adapter": None},
    {"key": "instruct", "label": "Qwen3.5-4B (instruct, 어댑터 없음)",
     "base_model": "Qwen/Qwen3.5-4B", "adapter": None},
    {"key": "sft",      "label": "Qwen3.5-4B + SFT v4",
     "base_model": "Qwen/Qwen3.5-4B", "adapter": "outputs/sft_q35_4b_v4"},
    {"key": "dpo",      "label": "Qwen3.5-4B + SFT v4 + DPO v5",
     "base_model": "Qwen/Qwen3.5-4B", "adapter": "outputs/dpo_q35_4b_v5"},
]

OUT = Path("presentation/01_model_comparison")
OUT.mkdir(parents=True, exist_ok=True)
RAW_PATH = OUT / "raw_outputs.json"


def _patch_qwen35():
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as _q
        if not _q.is_fast_path_available:
            _q.is_fast_path_available = True
            print("[patch] Qwen3.5 fast path forced True")
    except Exception:
        pass


def describe_expected(skel) -> str:
    """골격에서 사람이 읽을 수 있는 기대 정답 설명 생성 (하위 에이전트 채점 근거용)."""
    lines = [f"시나리오: {skel.scenario}", f"오늘: {skel.today or '(미상)'}"]
    lines.append("태스크 골격 (입력 id 기준):")
    for s in sorted(skel.specs, key=lambda x: x.idx):
        parts = [f"  - id={s.idx} kind={s.kind}"]
        if s.deadline:
            parts.append(f"마감={s.deadline}")
        if s.is_past:
            parts.append("[지난 일정→최하위·urgency/tc≤2]")
        if s.chain_group:
            parts.append(f"[체인{s.chain_group}-{s.chain_pos}단계→순서연속·dependency≥4]")
        if s.risk:
            parts.append(f"[리스크:{s.risk_clause}→importance≥4]")
        if s.rel_expr:
            parts.append(f"[상대표현:{s.rel_expr}]")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def main():
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from gen_schedule_v3 import Skeleton, TaskSpec, verify_chosen
    from timesorter.data.schema import SCHEDULER_SYSTEM_PROMPT_V3, render_system_prompt

    _patch_qwen35()

    df = pd.read_parquet("data/scheduler_v3_eval.parquet")
    queries = []
    for idx in QUERY_INDICES:
        row = df.iloc[idx]
        meta = json.loads(row["meta"])
        skel = Skeleton(scenario=meta["scenario"], today=meta["today"],
                        specs=[TaskSpec(**s) for s in meta["specs"]])
        queries.append({
            "idx": int(idx),
            "scenario": meta["scenario"],
            "persona": str(row["persona"]),
            "today": str(row["today"]),
            "prompt": str(row["prompt"]),
            "expected": describe_expected(skel),
            "_skel": skel,
        })
    print(f"[queries] {len(queries)}개 선택: {[q['scenario'] for q in queries]}")

    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)

    results = {"queries": [{k: v for k, v in q.items() if k != "_skel"} for q in queries],
               "models": {}}

    for m in MODELS:
        print(f"\n{'='*55}\n[model] {m['label']}\n{'='*55}")
        tok = AutoTokenizer.from_pretrained(m["adapter"] or m["base_model"],
                                            trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            m["base_model"], quantization_config=bnb_cfg, device_map="auto",
            dtype=torch.bfloat16, trust_remote_code=True)
        if m["adapter"]:
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, m["adapter"])
        else:
            model = base
        model.eval()

        model_results = []
        for q in queries:
            system = render_system_prompt(SCHEDULER_SYSTEM_PROMPT_V3,
                                          q["persona"], today=q["today"])
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": q["prompt"]}]
            try:
                text = tok.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=False)
            except Exception:
                text = tok.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=900, do_sample=False,
                                     pad_token_id=tok.eos_token_id or tok.pad_token_id,
                                     temperature=None, top_p=None)
            gen = out[0][inputs["input_ids"].shape[1]:]
            raw = tok.decode(gen, skip_special_tokens=True).strip()  # 전체 캡처
            errors = verify_chosen(q["_skel"], raw)
            model_results.append({
                "idx": q["idx"], "scenario": q["scenario"],
                "raw_output": raw,
                "passed": len(errors) == 0,
                "violations": errors,
            })
            verdict = "✅PASS" if not errors else f"❌{len(errors)}건"
            print(f"  [{q['scenario']:18}] {verdict}  (출력 {len(raw)}자)")

        results["models"][m["key"]] = {"label": m["label"], "results": model_results}

        del model, base
        torch.cuda.empty_cache()

    RAW_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n[saved] {RAW_PATH}")

    # 요약
    print("\n=== 모델별 통과율 (이번 6쿼리, 정량지표는 n=30 벤치마크 참조) ===")
    for key, mv in results["models"].items():
        n_pass = sum(1 for r in mv["results"] if r["passed"])
        print(f"  {mv['label']:40} {n_pass}/{len(mv['results'])}")


if __name__ == "__main__":
    main()
