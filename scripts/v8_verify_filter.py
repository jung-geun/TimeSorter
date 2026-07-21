#!/usr/bin/env python
"""생성 JSONL에 verify_chosen 적용 — 통과 행만 필터링.

사용:
  uv run python scripts/v8_verify_filter.py \
      --skel outputs/v8_chain/skeletons/batch_00.jsonl \
      --gen  outputs/v8_chain/gen/batch_00.jsonl \
      --out  outputs/v8_chain/verified/batch_00.jsonl

  # 전체 배치 일괄 처리
  for i in $(seq -f "%02g" 0 17); do
    uv run python scripts/v8_verify_filter.py \
        --skel outputs/v8_chain/skeletons/batch_${i}.jsonl \
        --gen  outputs/v8_chain/gen/batch_${i}.jsonl \
        --out  outputs/v8_chain/verified/batch_${i}.jsonl
  done
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from gen_schedule_v3 import Skeleton, TaskSpec, verify_chosen


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def skel_from_row(row: dict) -> Skeleton:
    meta = json.loads(row["meta"]) if isinstance(row["meta"], str) else row["meta"]
    specs = [TaskSpec(**s) for s in meta["specs"]]
    return Skeleton(scenario=meta["scenario"], today=meta["today"], specs=specs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skel", required=True, help="골격 배치 JSONL")
    ap.add_argument("--gen", required=True, help="생성 JSONL (id, tasks, chosen)")
    ap.add_argument("--out", required=True, help="통과 행 출력 JSONL")
    args = ap.parse_args()

    skels = {r["id"]: r for r in load_jsonl(args.skel)}
    gen_rows = load_jsonl(args.gen)

    passed, failed, missing = [], 0, 0
    for row in gen_rows:
        sid = row.get("id")
        if sid not in skels:
            missing += 1
            continue
        skel = skel_from_row(skels[sid])
        chosen_str = json.dumps(row.get("chosen", {}), ensure_ascii=False)
        errors = verify_chosen(skel, chosen_str)
        if not errors:
            passed.append(row)
        else:
            failed += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in passed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(gen_rows)
    rate = len(passed) / total * 100 if total else 0
    print(f"[{Path(args.gen).name}] {len(passed)}/{total} 통과 ({rate:.0f}%) | 실패={failed} 누락={missing}")


if __name__ == "__main__":
    main()
