#!/usr/bin/env python
"""held-out eval용 hard 행 선별 — 정상 스캐폴드 풀에서 난도 상위만 골라 재배치.

빌드스크립트(gen_row 로직) 수정 없이 "심화" 달성: 풀을 과생성한 뒤 복잡도 점수
(체인 수·체인 길이·태스크 수·risk·tight 마감)로 상위 K를 직업군별 stratify 선별.
각 행에 difficulty 특성을 meta로 태깅 → 타입별 통과율 보고용. v9는 마감=스케줄+버퍼라
난도를 올려도 gold는 항상 실현가능(검증기 통과 유지).

사용:
  uv run python scripts/v9/select_hard_eval.py --pool outputs/v9/build_en_evalpool \
    --out-dir outputs/v9/build_en_eval --k 50
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

def difficulty(row: dict) -> dict:
    slots = row["slots"]
    n_tasks = len(slots)
    cg = defaultdict(int)
    for s in slots:
        if s["chain_group"]:
            cg[s["chain_group"]] += 1
    n_chains = len(cg)
    max_chain = max(cg.values()) if cg else 0
    has_risk = any(s["risk"] for s in slots)
    # slots는 tier 대신 kind 보유: "today"=오늘 마감(임박), "future"/"none"/"chain"
    n_tight = sum(1 for s in slots if s.get("kind") == "today")
    score = n_chains * 3 + max_chain * 1.5 + n_tasks * 0.5 + (2 if has_risk else 0) + n_tight * 0.8
    return {"score": score, "n_tasks": n_tasks, "n_chains": n_chains,
            "max_chain": max_chain, "has_risk": has_risk, "n_tight": n_tight}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="스캐폴드 풀 디렉토리(.../build_*/)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(f"{args.pool}/scaffold/batch_*.json")):
        if not re.match(r"batch_\d{3}\.json$", Path(f).name):
            continue
        rows.extend(json.loads(Path(f).read_text()))
    for r in rows:
        r["_diff"] = difficulty(r)

    # 직업군별 stratify: 카테고리별 난도순 정렬 후 라운드로빈으로 K개 채움
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["occ_category"]].append(r)
    for c in by_cat:
        by_cat[c].sort(key=lambda r: r["_diff"]["score"], reverse=True)
    picked, ci = [], {c: 0 for c in by_cat}
    cats = sorted(by_cat)
    while len(picked) < args.k and any(ci[c] < len(by_cat[c]) for c in cats):
        for c in cats:
            if ci[c] < len(by_cat[c]):
                picked.append(by_cat[c][ci[c]]); ci[c] += 1
                if len(picked) >= args.k:
                    break

    out = Path(args.out_dir)
    (out / "scaffold").mkdir(parents=True, exist_ok=True)
    bs = args.batch_size
    nb = (len(picked) + bs - 1) // bs
    for b in range(nb):
        chunk = picked[b * bs:(b + 1) * bs]
        for j, r in enumerate(chunk):
            r["row_id"] = b * bs + j
        (out / "scaffold" / f"batch_{b:03d}.json").write_text(json.dumps(chunk, ensure_ascii=False))
    (out / "manifest.json").write_text(json.dumps({"n": len(picked), "n_batch": nb, "batch_size": bs}, ensure_ascii=False))

    import statistics
    print(f"[saved] {out}/scaffold — {len(picked)}행 / {nb}배치 (풀 {len(rows)})")
    diffs = [r["_diff"] for r in picked]
    print(f"  난도 평균: tasks {statistics.mean(d['n_tasks'] for d in diffs):.1f} · "
          f"chains {statistics.mean(d['n_chains'] for d in diffs):.1f} · "
          f"max_chain {statistics.mean(d['max_chain'] for d in diffs):.1f} · "
          f"risk {sum(d['has_risk'] for d in diffs)}/{len(diffs)} · "
          f"tight {statistics.mean(d['n_tight'] for d in diffs):.1f}")
    from collections import Counter
    print("  직업군:", dict(Counter(r["occ_category"] for r in picked)))


if __name__ == "__main__":
    main()
