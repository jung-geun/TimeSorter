#!/usr/bin/env python
"""on-policy DPO 쌍 수집 — 현재 모델이 실제로 틀리는 출력을 rejected로.

서빙 중인 모델에 골격(meta) 보유 프롬프트를 temperature>0으로 N회 생성시키고,
verify_chosen 위반 출력을 rejected로, 데이터셋의 검증된 chosen을 chosen으로 묶는다.
프로그램 변형 negative(off-policy)와 달리 모델의 진짜 오류 분포를 벌점한다.

사용 (vLLM 서빙 필요):
  uv run python scripts/gen_onpolicy_pairs.py --total 800 \
      --out data/dpo_pairs_v5_onpolicy.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from timesorter.data.schema import (
    SCHEDULER_SYSTEM_PROMPT_V3,
    ScheduleResponse,
    parse_lenient,
    render_system_prompt,
)

from gen_schedule_v3 import Skeleton, TaskSpec, verify_chosen

_WEIGHTS = {"dependency_chain": 4.0, "past_split": 2.0, "dated_mixed": 1.5}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/hf_release/grpo_train.parquet")
    parser.add_argument("--sft", default="data/hf_release/sft_train.parquet")
    parser.add_argument("--total", type=int, default=800)
    parser.add_argument("--n-samples", type=int, default=2, help="프롬프트당 생성 수")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--model", default="scheduler")
    parser.add_argument("--out", default="data/dpo_pairs_v5_onpolicy.parquet")
    parser.add_argument("--seed", type=int, default=46)
    args = parser.parse_args()

    g = pd.read_parquet(args.input)
    sft = pd.read_parquet(args.sft)
    chosen_map = dict(zip(sft["prompt"], sft["chosen"]))

    scen = g["meta"].apply(lambda m: json.loads(m)["scenario"])
    w = scen.map(_WEIGHTS).fillna(1.0)
    sample = g.sample(min(args.total, len(g)), weights=w, random_state=args.seed)
    print(f"[수집] {len(sample)}개 프롬프트 × {args.n_samples}생성 (temp {args.temperature})")
    print(scen[sample.index].value_counts().to_string())

    client = OpenAI(api_key="EMPTY", base_url=f"{args.server_url}/v1")
    pairs, stats = [], Counter()

    for i, (_, row) in enumerate(sample.iterrows()):
        md = json.loads(row["meta"])
        skel = Skeleton(scenario=md["scenario"], today=md["today"],
                        specs=[TaskSpec(**s) for s in md["specs"]])
        chosen = chosen_map.get(row["prompt"])
        if not chosen:
            stats["no_chosen"] += 1
            continue
        system = render_system_prompt(
            SCHEDULER_SYSTEM_PROMPT_V3, str(row["persona"]), today=str(row["today"]))
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": str(row["prompt"])}],
                max_tokens=2048, temperature=args.temperature, n=args.n_samples,
                extra_body={"guided_json": ScheduleResponse.model_json_schema()},
            )
        except Exception as e:
            stats["api_error"] += 1
            if stats["api_error"] <= 2:
                print("  [API 오류]", str(e)[:100])
            continue
        for ch in resp.choices:
            raw = (ch.message.content or "").strip()
            parsed = parse_lenient(raw)
            if parsed is None or not parsed.tasks:
                stats["unparseable"] += 1
                continue
            rejected = parsed.model_dump_json()
            if rejected == chosen:
                stats["identical"] += 1
                continue
            errors = verify_chosen(skel, rejected)
            if not errors:
                stats["clean_gen"] += 1
                continue
            kind = errors[0][:10]
            pairs.append({
                "prompt": str(row["prompt"]),
                "chosen": str(chosen),
                "rejected": rejected,
                "persona": str(row["persona"]),
                "today": str(row["today"]),
                "category": "onpolicy",
                "source": f"onpolicy_{skel.scenario}",
            })
            stats[f"viol:{kind}"] += 1
            break  # 프롬프트당 1쌍
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(sample)} — 쌍 {len(pairs)}")

    print(f"\n[완료] on-policy 쌍 {len(pairs)}개")
    for k, v in stats.most_common(12):
        print(f"  {k}: {v}")
    if pairs:
        df = pd.DataFrame(pairs).drop_duplicates(subset=["prompt"]).reset_index(drop=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)
        print(f"[저장] {args.out} ({len(df)}쌍)")
        print(df["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
