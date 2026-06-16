#!/usr/bin/env python
"""체인 텍스트-추론성 블라인드 감사 입력 생성 + 채점.

핵심 아이디어(advisor): 라벨은 골격에서 나오지만, 추론 시 모델은 '텍스트만' 본다.
체인 시나리오는 텍스트만 읽고도 어떤 태스크가 의존 체인을 이루고 그 순서가 무엇인지
식별 가능해야 한다. 그렇지 않으면 '독심술 학습'이 된다.

- build: 에이전트 생성물에서 (id, tasks) 만 추출 → Opus 가 골격 없이 체인 추론.
- score: Opus 의 블라인드 추론 vs 골격 정답(meta) 비교 → 일치분만 통과.

사용:
  uv run python scripts/build_blind_audit.py build \
      --skeletons "outputs/skeletons_v7chain_part1.jsonl" \
      --agent-out "outputs/claude_gen_v7chain_part1.jsonl" \
      --out outputs/blind_audit_in_p1.jsonl
  # (Opus 에이전트가 각 행에 chains=[[idx..],..] 추론 → outputs/blind_audit_out_p1.jsonl)
  uv run python scripts/build_blind_audit.py score \
      --skeletons "outputs/skeletons_v7chain_part1.jsonl" \
      --audit-out "outputs/blind_audit_out_p1.jsonl" \
      --pass-ids outputs/blind_pass_p1.txt
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def load_jsonl(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def true_chains(meta: dict) -> list[list[int]]:
    """골격 meta → 체인 그룹별 [단계순 idx] 리스트."""
    groups: dict[int, list[tuple[int, int]]] = {}
    for s in meta["specs"]:
        if s.get("chain_group"):
            groups.setdefault(s["chain_group"], []).append((s["chain_pos"], s["idx"]))
    out = []
    for g in sorted(groups):
        out.append([idx for _, idx in sorted(groups[g])])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--skeletons", required=True)
    b.add_argument("--agent-out", required=True)
    b.add_argument("--out", required=True)

    s = sub.add_parser("score")
    s.add_argument("--skeletons", required=True)
    s.add_argument("--audit-out", required=True)
    s.add_argument("--pass-ids", required=True)
    args = ap.parse_args()

    if args.cmd == "build":
        skels = {r["id"]: r for r in load_jsonl(args.skeletons)}
        outs = load_jsonl(args.agent_out)
        rows = []
        for o in outs:
            sid = o.get("id")
            if sid not in skels:
                continue
            tasks = o.get("tasks", [])
            rows.append({
                "id": sid,
                # 1-기반 번호로 태스크 텍스트만 제시 (골격/정답 미포함)
                "tasks": {str(i): t for i, t in enumerate(tasks, 1)},
            })
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[build] {args.out}: {len(rows)}행 (텍스트만)")
        return

    # score
    skels = {r["id"]: r for r in load_jsonl(args.skeletons)}
    audit = {r["id"]: r for r in load_jsonl(args.audit_out)}
    passed, stats = [], Counter()
    for sid, sk in skels.items():
        if sid not in audit:
            stats["missing_audit"] += 1
            continue
        truth = sorted([sorted_chain for sorted_chain in true_chains(json.loads(sk["meta"]))])
        pred_raw = audit[sid].get("chains", [])
        # 예측 정규화: 그룹 내부는 추론 순서 유지(순서 비교용), 그룹 집합은 정렬
        pred = sorted([[int(x) for x in grp] for grp in pred_raw])
        # 순서까지 정확히 일치해야 통과 (체인 식별 + 단계 순서)
        truth_ordered = sorted(true_chains(json.loads(sk["meta"])))
        if pred == truth_ordered:
            passed.append(sid)
            stats["pass"] += 1
        else:
            stats["fail"] += 1
    Path(args.pass_ids).parent.mkdir(parents=True, exist_ok=True)
    Path(args.pass_ids).write_text("\n".join(passed), encoding="utf-8")
    total = stats["pass"] + stats["fail"]
    rate = stats["pass"] / total * 100 if total else 0
    print(f"[score] 통과 {stats['pass']}/{total} ({rate:.0f}%) · {dict(stats)}")
    print(f"[score] 통과 id → {args.pass_ids}")


if __name__ == "__main__":
    main()
