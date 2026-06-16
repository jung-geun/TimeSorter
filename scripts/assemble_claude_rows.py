#!/usr/bin/env python
"""Claude 에이전트 생성 결과 검증·병합.

입력: 스켈레톤 JSONL(export_skeletons.py) + 에이전트 출력 JSONL
  에이전트 출력 행: {"id": "...", "tasks": ["...", ...], "chosen": {...}}
검증: 태스크 수 일치, 리스크 문구 포함, verify_chosen 골격 규칙 전체.
출력: parquet (prompt/chosen/persona/today/source/meta) — source=v4_claude_<scenario>

사용:
  uv run python scripts/assemble_claude_rows.py \
      --skeletons "outputs/skeletons_v4_part*.jsonl" \
      --agent-out "outputs/claude_gen_v4_part*.jsonl" \
      --out data/scheduler_v4_claude.parquet
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from timesorter.data.schema import parse_lenient

from gen_schedule_v3 import Skeleton, TaskSpec, verify_chosen


def load_jsonl(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  [경고] {path}:{ln} JSON 파싱 실패: {e}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skeletons", required=True)
    parser.add_argument("--agent-out", required=True)
    parser.add_argument("--out", default="data/scheduler_v4_claude.parquet")
    parser.add_argument("--source-prefix", default="v4_claude", help="source 태그 접두")
    parser.add_argument("--keep-ids", default="", help="이 파일(줄당 id)에 있는 id만 수록 — 블라인드 감사 통과분 필터")
    args = parser.parse_args()

    keep: set[str] | None = None
    if args.keep_ids:
        keep = {ln.strip() for ln in Path(args.keep_ids).read_text(encoding="utf-8").splitlines() if ln.strip()}
        print(f"keep-ids 필터: {len(keep)}개 id만 수록")

    skels = {r["id"]: r for r in load_jsonl(args.skeletons)}
    outputs = load_jsonl(args.agent_out)
    print(f"스켈레톤 {len(skels)}개, 에이전트 출력 {len(outputs)}개")

    rows, rejected = [], Counter()
    seen = set()
    for o in outputs:
        sid = o.get("id")
        if sid not in skels or sid in seen:
            rejected["unknown_or_dup_id"] += 1
            continue
        if keep is not None and sid not in keep:
            rejected["not_in_keep_ids"] += 1
            continue
        seen.add(sid)
        sk = skels[sid]
        meta = json.loads(sk["meta"])
        skel = Skeleton(scenario=meta["scenario"], today=meta["today"],
                        specs=[TaskSpec(**s) for s in meta["specs"]])

        tasks = o.get("tasks", [])
        if len(tasks) != len(skel.specs):
            rejected["task_count"] += 1
            continue
        # 리스크 문구 포함 확인
        ok = True
        for s in skel.specs:
            if s.risk and s.risk_clause:
                core = s.risk_clause.split("—")[0].strip()[:6]
                if core and not any(core in t for t in tasks):
                    ok = False
        if not ok:
            rejected["risk_clause_missing"] += 1
            continue

        chosen_str = json.dumps(o.get("chosen", {}), ensure_ascii=False)
        if parse_lenient(chosen_str) is None:
            rejected["chosen_schema"] += 1
            continue
        errors = verify_chosen(skel, chosen_str)
        if errors:
            rejected["rule:" + errors[0][:24]] += 1
            continue

        task_lines = "\n".join(f"- {t}" for t in tasks)
        rows.append({
            "prompt": f"[{sk['persona_name']} 씨의 오늘의 할 일 목록]\n{task_lines}",
            "chosen": chosen_str,
            "persona": sk["persona_label"],
            "today": sk["today"],
            "source": f"{args.source_prefix}_{skel.scenario}",
            "meta": sk["meta"],
        })

    print(f"통과 {len(rows)} / 거부 {sum(rejected.values())}")
    for k, v in rejected.most_common(10):
        print(f"  {k}: {v}")
    if rows:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(args.out, index=False)
        print(f"[저장] {args.out} ({len(rows)}행)")
        print(pd.DataFrame(rows)["source"].value_counts().to_string())
