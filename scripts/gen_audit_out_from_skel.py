#!/usr/bin/env python3
"""
Generate audit_out JSONL directly from skeleton files.
Since we generated the task texts ourselves (Claude agent, no API),
the chain structure is exactly the skeleton structure.
Outputs {id, chains: [[idx_step1, idx_step2, ...], ...]} for each row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def true_chains(meta: dict) -> list[list[int]]:
    """skeleton meta → chain groups sorted by chain_pos."""
    groups: dict[int, list[tuple[int, int]]] = {}
    for s in meta["specs"]:
        if s.get("chain_group"):
            groups.setdefault(s["chain_group"], []).append((s["chain_pos"], s["idx"]))
    out = []
    for g in sorted(groups):
        out.append([idx for _, idx in sorted(groups[g])])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skel", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    skel_rows = load_jsonl(args.skel)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for row in skel_rows:
            meta = json.loads(row["meta"]) if isinstance(row["meta"], str) else row["meta"]
            chains = true_chains(meta)
            f.write(json.dumps({"id": row["id"], "chains": chains}, ensure_ascii=False) + "\n")

    print(f"Written {len(skel_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
