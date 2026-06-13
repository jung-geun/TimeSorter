#!/usr/bin/env python
"""DPO 어댑터 Docker 검증 스크립트.

사용:
  docker run --rm --gpus '"device=0"' \
    -v $PWD/models:/root/.cache/huggingface \
    -v $PWD/outputs:/workspace/outputs \
    -v $PWD/data:/workspace/data \
    -v $PWD/src:/workspace/src \
    -v $PWD/scripts:/workspace/scripts \
    -e ADAPTER=outputs/dpo_4090_1x_9b_v4 \
    -e EVAL_SET=data/scheduler_v3_eval.parquet \
    timesorter:cu124 \
    python /workspace/scripts/docker_validate_dpo.py
"""
from __future__ import annotations

import json
import os
import sys
import datetime
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/workspace/src")
sys.path.insert(0, "/workspace/scripts")

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from timesorter.data.schema import (
    SCHEDULER_SYSTEM_PROMPT_V3,
    ScheduleResponse,
    parse_or_repair,
    render_system_prompt,
    format_today,
)
from gen_schedule_v3 import Skeleton, TaskSpec, verify_chosen

ADAPTER = os.environ.get("ADAPTER", "outputs/dpo_4090_1x_9b_v4")
EVAL_SET = os.environ.get("EVAL_SET", "data/scheduler_v3_eval.parquet")
OUT_JSON = os.environ.get("OUT_JSON", "outputs/validation_dpo_v4.json")
LIMIT = int(os.environ.get("LIMIT", "0")) or None

ADAPTER_PATH = f"/workspace/{ADAPTER}"
EVAL_PATH = f"/workspace/{EVAL_SET}"
OUT_PATH = f"/workspace/{OUT_JSON}"


def load_model():
    import json as _json
    cfg = _json.load(open(f"{ADAPTER_PATH}/adapter_config.json"))
    base = cfg["base_model_name_or_path"]
    print(f"[모델] base={base}, adapter={ADAPTER_PATH}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    print(f"[모델] 로드 완료 — {torch.cuda.memory_allocated()/1e9:.1f} GB 사용")
    return model, tokenizer


def infer(model, tokenizer, system: str, user: str, max_new_tokens: int = 1536) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    # enable_thinking=False: <think>\n\n</think> 즉시 닫아 JSON 직출력 (thinking 토큰 낭비 방지)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # thinking 잔여 텍스트 제거: </think> 이후만 취함
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[-1].lstrip("\n").lstrip()
    return raw


def violation_kind(msg: str) -> str:
    if "지난 일정" in msg or "urgency/time_constraint" in msg:
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


def main():
    print("=" * 60)
    print(f"TimeSorter DPO v4 검증 — {datetime.date.today()}")
    print(f"어댑터: {ADAPTER}")
    print(f"평가셋: {EVAL_SET}")
    print("=" * 60)

    model, tokenizer = load_model()

    df = pd.read_parquet(EVAL_PATH)
    if LIMIT:
        df = df.head(LIMIT)
    print(f"\n[평가] {len(df)}개 시나리오 로드\n")

    results = []
    per_scenario: dict[str, Counter] = {}
    per_kind = Counter()
    n_clean = 0
    errors_list = []

    for i, row in df.iterrows():
        today_str = str(row.get("today", "") or "")
        if not today_str:
            today_str = datetime.date.today().isoformat()

        persona = str(row.get("persona", "직장인"))
        system = render_system_prompt(SCHEDULER_SYSTEM_PROMPT_V3, persona, today=today_str)
        user = str(row["prompt"])

        try:
            raw = infer(model, tokenizer, system, user)
            parsed = parse_or_repair(raw)
            out_json = parsed.model_dump_json() if parsed else ""
        except Exception as e:
            raw = ""
            out_json = ""
            print(f"  [{i+1:3d}] 추론 오류: {e}")

        skel_raw = row.get("meta", None)
        violations: list[str] = []
        if skel_raw and out_json:
            try:
                meta = json.loads(skel_raw) if isinstance(skel_raw, str) else skel_raw
                skel = Skeleton(
                    scenario=meta["scenario"],
                    today=meta["today"],
                    specs=[TaskSpec(**s) for s in meta["specs"]],
                )
                violations = verify_chosen(skel, out_json)
            except Exception as e:
                violations = [f"검증 오류: {e}"]

        scenario = str(row.get("source", "unknown"))
        if scenario not in per_scenario:
            per_scenario[scenario] = Counter({"total": 0, "clean": 0})
        per_scenario[scenario]["total"] += 1

        if violations:
            for v in violations:
                per_kind[violation_kind(v)] += 1
            errors_list.append({"idx": i, "scenario": scenario, "violations": violations})
            status = "FAIL"
        else:
            n_clean += 1
            per_scenario[scenario]["clean"] += 1
            status = "OK"

        results.append({
            "idx": int(i),
            "scenario": scenario,
            "today": today_str,
            "status": status,
            "violations": violations,
            "output_snippet": out_json[:200] if out_json else "",
        })

        if (i + 1) % 10 == 0:
            pct = n_clean / (i + 1) * 100
            print(f"  진행: {i+1:3d}/{len(df)} | 통과율 {pct:.1f}%")

    total = len(df)
    pass_rate = n_clean / total * 100

    print("\n" + "=" * 60)
    print(f"  전체 통과율: {n_clean}/{total} = {pass_rate:.1f}%")
    print("=" * 60)
    print("\n[시나리오별 통과율]")
    for sc, cnt in sorted(per_scenario.items()):
        pct = cnt["clean"] / cnt["total"] * 100 if cnt["total"] else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {sc:30s} {cnt['clean']:2d}/{cnt['total']:2d} [{bar}] {pct:.0f}%")

    print("\n[위반 유형별 집계]")
    for kind, cnt in per_kind.most_common():
        print(f"  {kind:25s} {cnt:4d}건")

    # 샘플 추론 1건 (정성 출력)
    print("\n" + "=" * 60)
    print("[샘플 추론 — 정성 검토]")
    sample_today = datetime.date.today().isoformat()
    sample_yest = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    sample_user = (
        f"[직장인의 오늘의 할 일 목록]\n"
        f"- 보고서 제출 ({sample_today} 17:00까지)\n"
        f"- PR 리뷰 ({sample_today} 오전까지, 미처리 시 팀장 에스컬레이션)\n"
        f"- 회의 자료 정리 ({sample_yest} 14:00 회의, 아직 미완료)\n"
        f"- 책상 정리\n"
        f"- 팀 점심 예약 ({sample_today} 12:00 이전)"
    )
    sample_sys = render_system_prompt(SCHEDULER_SYSTEM_PROMPT_V3, "직장인", today=sample_today)
    sample_out = infer(model, tokenizer, sample_sys, sample_user)
    parsed_sample = parse_or_repair(sample_out)
    if parsed_sample:
        print(f"입력: {sample_user[:100]}...")
        print(f"\n출력 (priority_order): {parsed_sample.priority_order}")
        for t in parsed_sample.tasks:
            sc_list = [s for s in parsed_sample.scores if s.task_id == t.id]
            sc_str = ""
            if sc_list:
                sc = sc_list[0]
                sc_str = f" [U={sc.urgency} I={sc.importance} D={sc.dependency} T={sc.time_constraint}]"
            print(f"  {t.id}. {t.text}{sc_str}")
    else:
        print("(파싱 실패)")
        print(sample_out[:500])

    # 결과 저장
    summary = {
        "adapter": ADAPTER,
        "eval_set": EVAL_SET,
        "date": str(datetime.date.today()),
        "total": total,
        "n_clean": n_clean,
        "pass_rate_pct": round(pass_rate, 2),
        "per_scenario": {k: dict(v) for k, v in per_scenario.items()},
        "violation_kinds": dict(per_kind),
        "errors": errors_list[:20],
        "details": results,
    }
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {OUT_PATH}")
    print("\n검증 완료.")


if __name__ == "__main__":
    main()
