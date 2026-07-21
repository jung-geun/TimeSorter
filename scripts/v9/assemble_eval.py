#!/usr/bin/env python
"""v9 held-out hard eval 조립 — KR+EN 통합, lang 컬럼 + 난도 meta, 강화 verifier + opus 엄격 게이트.

입력: v9_eval_fill 워크플로 결과 JSON([{dir,lang,b,haiku,sonnet,opus}]).
게이트: verify_chosen_v9(강화: 스케줄순서 포함) 무위반 AND opus pass AND realism>=--min-realism.
출력: data/scheduler_v9_eval.parquet + data/hf_versioned/eval/v9.parquet
  컬럼: prompt·chosen·persona·today·source·meta·lang·version  (meta에 난도 특성 태깅 → 타입별 통과율 보고용)

사용:
  uv run python scripts/v9/assemble_eval.py --llm outputs/v9/eval/llm_result.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/v9")
from timesorter.data import schema_v9 as S  # noqa: E402
from assemble_v9 import assemble, load_scaffold, _generic_pattern, has_generic_title, _norm_title  # noqa: E402
from select_hard_eval import difficulty  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True)
    ap.add_argument("--min-realism", type=int, default=4)  # eval은 엄격
    ap.add_argument("--out", default="data/scheduler_v9_eval.parquet")
    ap.add_argument("--out-hf", default="data/hf_versioned/eval/v9.parquet")
    args = ap.parse_args()

    raw = json.loads(Path(args.llm).read_text())
    if isinstance(raw, str):
        raw = json.loads(raw)

    # dir/lang별로 분리 → 각 스캐폴드에 대해 조립
    groups: dict[tuple[str, str], list] = {}
    for r in raw:
        groups.setdefault((r["dir"], r["lang"]), []).append(r)

    rows, stats = [], Counter()
    seen: set = set()
    for (sdir, lang), items in groups.items():
        scaffold = load_scaffold(f"{sdir}/scaffold")
        titles, reasons, opus = {}, {}, {}
        for it in items:
            for h in it.get("haiku") or []:
                titles[h["row_id"]] = {t["task_id"]: t for t in h["tasks"]}
            for s in it.get("sonnet") or []:
                reasons[s["row_id"]] = {x["task_id"]: x for x in s["reasoning"]}
            for o in it.get("opus") or []:
                opus[o["row_id"]] = o
        pat = _generic_pattern(lang)
        for rid, scaf in scaffold.items():
            if rid not in titles or rid not in reasons:
                stats["no_llm"] += 1; continue
            if has_generic_title(titles[rid], pat):
                stats["generic"] += 1; continue
            try:
                inp, out, errors = assemble(scaf, titles[rid], reasons[rid])
            except Exception:
                stats["pydantic"] += 1; continue
            if errors:
                stats["verify_fail"] += 1; continue
            rev = opus.get(rid, {})
            if not rev.get("pass", True):
                stats["opus_fail"] += 1; continue
            if rev.get("realism", 5) < args.min_realism:
                stats["low_realism"] += 1; continue
            tset = frozenset(_norm_title(t.title) for t in inp.tasks)
            if tset in seen:
                stats["dup"] += 1; continue
            seen.add(tset)
            diff = difficulty(scaf)
            rows.append({
                "prompt": S.format_input_v9(inp),
                "chosen": S.format_for_sft_v9(out),
                "persona": json.dumps(inp.user_persona.model_dump(), ensure_ascii=False),
                "today": scaf["current_time"][:10],
                "source": f"v9eval_{lang}_{scaf['occ_category']}",
                "meta": json.dumps({"occ_category": scaf["occ_category"],
                                    "chain_pairs": scaf["chain_pairs"],
                                    "n_tasks": diff["n_tasks"], "n_chains": diff["n_chains"],
                                    "max_chain": diff["max_chain"], "has_risk": diff["has_risk"],
                                    "n_tight": diff["n_tight"]}, ensure_ascii=False),
                "lang": lang,
                "version": "v9_eval",
            })

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_hf).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    df.to_parquet(args.out_hf, index=False)
    print(f"[saved] {args.out} · {args.out_hf} — 수록 {len(df)}행")
    print("탈락:", dict(stats))
    if len(df):
        print("lang:", dict(Counter(r["lang"] for r in rows)))
        print("직업군:", dict(Counter(json.loads(r["meta"])["occ_category"] for r in rows)))
        print(f"난도 평균: tasks {df['meta'].map(lambda m: json.loads(m)['n_tasks']).mean():.1f} · "
              f"chains {df['meta'].map(lambda m: json.loads(m)['n_chains']).mean():.1f} · "
              f"risk행 {sum(json.loads(m)['has_risk'] for m in df['meta'])}")


if __name__ == "__main__":
    main()
