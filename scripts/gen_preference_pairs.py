#!/usr/bin/env python
"""한국어 스케줄 시나리오에서 4종 후보 생성 + GPT-4o judge로 DPO pair 구축.

산출물: data/dpo_pairs.parquet  (prompt, chosen, rejected, persona 컬럼)

사용:
  uv run python scripts/gen_preference_pairs.py                     # 전체
  uv run python scripts/gen_preference_pairs.py --limit 5           # dry-run
  uv run python scripts/gen_preference_pairs.py --in data/scheduler_ko.parquet
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_PAIR_COMBOS = [
    ("c1", "c4"),  # Gemini full vs Claude no-guide
    ("c2", "c3"),  # Claude full vs Gemini urgency-only
    ("c1", "c3"),  # Gemini full vs Gemini urgency-only
]


def _load_checkpoint(ckpt_path: Path) -> tuple[list[dict], set[int]]:
    """체크포인트 JSONL 로드. (pairs, done_idx_set) 반환."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", default="data/scheduler_ko.parquet")
    parser.add_argument("--out", default="data/dpo_pairs.parquet")
    parser.add_argument("--limit", type=int, default=None, help="처리할 시나리오 수 제한")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    import pandas as pd
    from drl.data.augment import generate_four_candidates, judge_pair

    df = pd.read_parquet(args.input)
    if args.limit:
        df = df.head(args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path.with_suffix(".ckpt.jsonl")

    pairs, done = _load_checkpoint(ckpt_path)

    with ckpt_path.open("a") as ckpt_f:
        for i, row in enumerate(df.itertuples()):
            if i in done:
                continue
            prompt = row.prompt
            persona = getattr(row, "persona", "직장인")
            print(f"[{i+1}/{len(df)}] 후보 생성: {prompt[:40]}...")

            try:
                c1, c2, c3, c4 = generate_four_candidates(prompt, persona)
            except Exception as e:
                print(f"  [스킵] 후보 생성 실패: {e}")
                continue

            candidates = {"c1": c1, "c2": c2, "c3": c3, "c4": c4}
            row_new_pairs: list[dict] = []

            for high_key, low_key in _PAIR_COMBOS:
                a = candidates[high_key]
                b = candidates[low_key]
                verdict = judge_pair(prompt, persona, a, b)
                if verdict == "TIE":
                    continue
                chosen = a if verdict == "A" else b
                rejected = b if verdict == "A" else a
                pair = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "persona": persona,
                    "pair": f"{high_key}_vs_{low_key}",
                }
                pairs.append(pair)
                row_new_pairs.append(pair)

            for pair in row_new_pairs:
                ckpt_f.write(
                    json.dumps({**pair, "_seed_idx": i}, ensure_ascii=False) + "\n"
                )
            ckpt_f.flush()

            time.sleep(0.5)

    result = pd.DataFrame(pairs)
    result.to_parquet(args.out, index=False)
    print(f"[done] {len(result)}개 pair 저장 → {args.out}")
    print(f"  TIE 제외 후 유효 페어 비율: {len(result)}/{len(df)*len(_PAIR_COMBOS):.0f}")


if __name__ == "__main__":
    main()
