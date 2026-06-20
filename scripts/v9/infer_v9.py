#!/usr/bin/env python
"""v9 SFT 어댑터 추론 검증 — 실제 v9 JSON 생성 + 파싱/검증 확인.

사용: uv run python scripts/v9/infer_v9.py --adapter outputs/sft_q35_4b_v9 --n 3
"""
from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, "src")
from timesorter.data import schema_v9 as S  # noqa: E402
from timesorter.data.scheduler import _to_chatml  # noqa: E402
from timesorter.device import detect  # noqa: E402
from timesorter.model import load_model_and_tokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="outputs/sft_q35_4b_v9")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--data", default="data/scheduler_v9.parquet")
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(args.data)
    profile = detect()
    model, tok = load_model_and_tokenizer(
        model_name="Qwen/Qwen3.5-4B", profile=profile,
        lora_r=16, lora_alpha=32, lora_dropout=0.0, use_4bit=True,
        gradient_checkpointing=False, sft_adapter_path=args.adapter,
    )
    model.eval()

    # 다양한 행(체인/비체인) 추출
    import json as _json
    idxs = []
    for i in range(len(df)):
        meta = _json.loads(df.iloc[i]["meta"])
        if len(meta.get("chain_pairs", [])) > 0 and len([x for x in idxs if x[1]]) < 2:
            idxs.append((i, True))
        elif len([x for x in idxs if not x[1]]) < 1:
            idxs.append((i, False))
        if len(idxs) >= args.n:
            break

    ok_parse = ok_verify = 0
    for i, is_chain in idxs:
        r = df.iloc[i]
        from timesorter.data.schema import system_prompt_for
        msgs = [{"role": "system", "content": system_prompt_for("v9")},
                {"role": "user", "content": str(r["prompt"])}]
        pstr = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                       enable_thinking=False)
        inputs = tok(pstr, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=2200, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        out_text = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        inp = S.ScheduleInput.model_validate_json(str(r["prompt"]))
        parsed = S.parse_lenient_v9(out_text)
        chain_pairs = [tuple(p) for p in _json.loads(r["meta"]).get("chain_pairs", [])]
        print(f"\n{'='*60}\n[row {i}] chain={is_chain} · 입력 태스크 {len(inp.tasks)}개")
        if parsed is None:
            print("  ❌ 파싱 실패. 생성 앞 300자:", repr(out_text[:300]))
            continue
        ok_parse += 1
        errs = S.verify_chosen_v9(inp, parsed, chain_pairs)
        if not errs:
            ok_verify += 1
        n_out = len(parsed.scheduled_tasks)
        print(f"  ✅ 파싱 OK · scheduled_tasks {n_out}개 (입력 {len(inp.tasks)}) · 검증 {'통과' if not errs else f'위반 {len(errs)}'}")
        if errs:
            for e in errs[:3]:
                print("     -", e)
        if parsed.scheduled_tasks:
            t = parsed.scheduled_tasks[0]
            print(f"     예) rank{t.priority_rank} '{t.title}' total={t.scoring.total_score} "
                  f"sched={t.recommended_schedule.start_time[-14:-9]}~{t.recommended_schedule.end_time[-14:-9]}")
            print(f"        reasoning: {t.reasoning.summary[:80]}")

    print(f"\n=== 결과: 파싱 {ok_parse}/{len(idxs)} · 검증통과 {ok_verify}/{len(idxs)} ===")


if __name__ == "__main__":
    main()
