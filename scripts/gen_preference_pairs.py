#!/usr/bin/env python
"""한국어 스케줄 시나리오에서 4종 후보 생성 + GPT judge로 DPO pair 구축.

산출물: data/dpo_pairs.parquet  (prompt, chosen, rejected, persona 컬럼)

사용:
  uv run python scripts/gen_preference_pairs.py                     # 전체
  uv run python scripts/gen_preference_pairs.py --limit 5           # dry-run
  uv run python scripts/gen_preference_pairs.py --in data/scheduler_ko.parquet
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_PAIR_COMBOS = [
    ("c1", "c4"),  # 고품질 vs 저품질
    ("c2", "c3"),  # 고품질 vs 편향
    ("c1", "c3"),  # 고품질 vs 편향
]


def _load_checkpoint(ckpt_path: Path) -> tuple[list[dict], set[int]]:
    pairs: list[dict] = []
    done: set[int] = set()
    if not ckpt_path.exists():
        return pairs, done
    with ckpt_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done.add(rec["_seed_idx"])
            pairs.append({k: v for k, v in rec.items() if not k.startswith("_")})
    print(f"[체크포인트] {len(done)}개 시나리오 이미 완료 — 이어서 진행합니다.")
    return pairs, done


async def process_scenario(
    idx: int,
    prompt: str,
    persona: str,
    ckpt_f,
    ckpt_lock: asyncio.Lock,
) -> list[dict]:
    from drl.data.augment import async_generate_four_candidates, async_judge_pair

    try:
        c1, c2, c3, c4 = await async_generate_four_candidates(prompt, persona)
    except Exception as e:
        print(f"  [스킵] 시나리오 {idx+1} 후보 생성 실패: {e}")
        return []

    candidates = {"c1": c1, "c2": c2, "c3": c3, "c4": c4}
    judge_tasks = [
        async_judge_pair(prompt, persona, candidates[h], candidates[l])
        for h, l in _PAIR_COMBOS
    ]
    verdicts = await asyncio.gather(*judge_tasks, return_exceptions=True)

    row_pairs: list[dict] = []
    for (high_key, low_key), verdict in zip(_PAIR_COMBOS, verdicts):
        if isinstance(verdict, Exception) or verdict == "TIE":
            continue
        a, b = candidates[high_key], candidates[low_key]
        chosen = a if verdict == "A" else b
        rejected = b if verdict == "A" else a
        pair = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "persona": persona,
            "pair": f"{high_key}_vs_{low_key}",
        }
        row_pairs.append(pair)

    if row_pairs:
        async with ckpt_lock:
            for pair in row_pairs:
                ckpt_f.write(
                    json.dumps({**pair, "_seed_idx": idx}, ensure_ascii=False) + "\n"
                )
            ckpt_f.flush()
        print(f"  [완료] 시나리오 {idx+1}: {len(row_pairs)}개 pair 저장")

    return row_pairs


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", default="data/scheduler_ko.parquet")
    parser.add_argument("--out", default="data/dpo_pairs.parquet")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    import pandas as pd

    df = pd.read_parquet(args.input)
    if args.limit:
        df = df.head(args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path.with_suffix(".ckpt.jsonl")

    all_pairs, done = _load_checkpoint(ckpt_path)

    pending = [(i, row) for i, row in enumerate(df.itertuples()) if i not in done]
    print(f"[시작] 처리 대상: {len(pending)}개 시나리오 (동시 처리: {args.concurrency})")

    sem = asyncio.Semaphore(args.concurrency)

    async def worker(idx: int, row) -> list[dict]:
        async with sem:
            prompt = str(row.prompt)
            persona = str(getattr(row, "persona", "직장인"))
            return await process_scenario(idx, prompt, persona, ckpt_f, ckpt_lock)

    ckpt_lock = asyncio.Lock()
    with ckpt_path.open("a") as ckpt_f:
        tasks = [worker(i, row) for i, row in pending]
        results = await asyncio.gather(*tasks)

    for result in results:
        all_pairs.extend(result)

    import pandas as pd  # noqa: F811
    result_df = pd.DataFrame(all_pairs)
    result_df.to_parquet(str(out_path), index=False)
    print(f"\n[완료] 총 {len(result_df)}개 pair 저장 → {out_path}")
    print(f"  TIE 제외 유효 비율: {len(result_df)}/{len(df) * len(_PAIR_COMBOS)}")


if __name__ == "__main__":
    asyncio.run(main_async())
