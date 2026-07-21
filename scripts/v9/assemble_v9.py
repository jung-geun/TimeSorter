#!/usr/bin/env python
"""v9 조립·검증·게이트 → SFT parquet.

입력:
  outputs/v9/build/scaffold/batch_*.json   (결정적: scoring/rank/schedule/slots/persona/facts)
  <llm-result>.json                        (workflow 결과: [{batch, haiku, sonnet, opus}])
게이트: verify_chosen_v9 무위반 AND opus pass AND realism >= --min-realism.
출력:
  data/scheduler_v9.parquet                (증분 SFT)
  data/hf_versioned/sft/v9.parquet         (업로드용)

사용:
  uv run python scripts/v9/assemble_v9.py --llm outputs/v9/build/llm_result.json --min-realism 3
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from timesorter.data import schema_v9 as S  # noqa: E402

import re

# 막연·일반(placeholder) 제목 — 이런 제목이 있으면 행 탈락(다양성/품질)
_GENERIC_KO = re.compile(r"^(작업\s*진행|일반\s*업무|할\s*일\s*처리|업무\s*관련|기타\s*업무|업무\s*처리)")


def _generic_pattern(lang: str):
    if lang == "en":
        from content_en import GENERIC_TITLE
        return re.compile(GENERIC_TITLE, re.IGNORECASE)
    return _GENERIC_KO


def _norm_title(t: str) -> str:
    return re.sub(r"\s+", "", t).lower()


def has_generic_title(titles: dict, pat) -> bool:
    return any(pat.match(t.get("title", "").strip()) for t in titles.values())


# placeholder/템플릿 memo — LLM 미보강(게으른 haiku) 흔적. 이런 행은 탈락(품질).
_PLACEHOLDER_MEMO = re.compile(
    r"^\s*(complete (this|the) task\b|complete step\s*\d|work on .*-\s*step\s*\d|"
    r"begin work on\b|next phase of\b|continue working on\b|"
    r"작업\s*진행|할\s*일\s*처리|이\s*작업\s*(수행|완료)|단계\s*\d+\s*진행)",
    re.IGNORECASE)


def _strip_nonword(s: str) -> str:
    return re.sub(r"[^\w]", "", s.lower())


def has_placeholder_content(titles: dict) -> bool:
    """memo가 placeholder 템플릿이거나 제목을 그대로 반복하면 True (LLM 미보강 결함)."""
    for t in titles.values():
        memo = (t.get("memo") or "").strip()
        title = (t.get("title") or "").strip()
        if not memo:
            continue
        if _PLACEHOLDER_MEMO.match(memo):
            return True
        # "Complete <title>" / "<title> 반복" 류 제목 그대로 베끼기
        mw, tw = _strip_nonword(memo), _strip_nonword(title)
        if tw and (mw == tw or mw == "complete" + tw):
            return True
    return False


def load_scaffold(scaffold_dir: str = "outputs/v9/build/scaffold") -> dict[int, dict]:
    out = {}
    for f in sorted(glob.glob(f"{scaffold_dir}/batch_*.json")):
        # 정규 batch_NNN.json 만(에이전트 부산물 제외)
        if not re.match(r"batch_\d{3}\.json$", Path(f).name):
            continue
        for r in json.loads(Path(f).read_text()):
            out[r["row_id"]] = r
    return out


def load_llm(path: str, lang: str | None = None) -> tuple[dict, dict, dict]:
    """Load LLM fill results. Pass lang='ko'/'en' when file mixes both languages
    (KR and EN scaffolds share row_id 0-N, so unfiltered loads cause collisions)."""
    raw = Path(path).read_text()
    arr = json.loads(raw)
    if isinstance(arr, str):
        arr = json.loads(arr)
    haiku, sonnet, opus = {}, {}, {}
    for batch in arr:
        if lang is not None and batch.get("lang") != lang:
            continue
        for h in batch.get("haiku") or []:
            haiku[h["row_id"]] = {t["task_id"]: t for t in h["tasks"]}
        for s in batch.get("sonnet") or []:
            sonnet[s["row_id"]] = {r["task_id"]: r for r in s["reasoning"]}
        for o in batch.get("opus") or []:
            opus[o["row_id"]] = o
    return haiku, sonnet, opus


def assemble(scaf: dict, titles: dict, reasons: dict):
    persona = {k: v for k, v in scaf["persona"].items() if k != "_meta"}
    in_tasks = []
    for sl in scaf["slots"]:
        tid = sl["task_id"]
        ht = titles.get(tid, {})
        in_tasks.append(S.InputTask(
            task_id=tid, title=ht.get("title") or tid, memo=ht.get("memo", ""),
            source=ht.get("source", ""), deadline=sl["deadline"],
            estimated_duration_minutes=sl["estimated_duration_minutes"]))
    inp = S.ScheduleInput(current_time=scaf["current_time"],
                          user_persona=S.UserPersona(**persona), tasks=in_tasks)
    sched_tasks = []
    for tid in sorted(scaf["priority_rank"], key=lambda t: scaf["priority_rank"][t]):
        rs = reasons.get(tid, {})
        sched_tasks.append(S.ScheduledTask(
            task_id=tid, title=titles.get(tid, {}).get("title") or tid,
            priority_rank=scaf["priority_rank"][tid], scoring=S.Scoring(**scaf["scoring"][tid]),
            reasoning=S.Reasoning(summary=rs.get("summary", ""),
                                  chaining_detail=rs.get("chaining_detail", "")),
            recommended_schedule=S.RecommendedSchedule(**scaf["schedule"][tid])))
    out = S.ScheduleResponseV9(scheduled_tasks=sched_tasks)
    errors = S.verify_chosen_v9(inp, out, [tuple(p) for p in scaf["chain_pairs"]])
    return inp, out, errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True)
    ap.add_argument("--min-realism", type=int, default=3)
    ap.add_argument("--lang", default="ko", choices=["ko", "en"])
    ap.add_argument("--scaffold-dir", default="outputs/v9/build/scaffold")
    ap.add_argument("--out-sft", default="data/scheduler_v9.parquet")
    ap.add_argument("--out-hf", default="data/hf_versioned/sft/v9.parquet")
    args = ap.parse_args()

    pat = _generic_pattern(args.lang)
    version = "v9_en" if args.lang == "en" else "v9"
    src_tag = "v9en" if args.lang == "en" else "v9"
    scaffold = load_scaffold(args.scaffold_dir)
    titles, reasons, opus = load_llm(args.llm, lang=args.lang)

    rows, stats = [], {"verify_fail": 0, "opus_fail": 0, "low_realism": 0,
                       "no_llm": 0, "pydantic": 0, "generic": 0, "dup_row": 0}
    seen_titlesets: set[frozenset] = set()
    all_titles: list[str] = []
    for rid, scaf in scaffold.items():
        if rid not in titles or rid not in reasons:
            stats["no_llm"] += 1
            continue
        if has_generic_title(titles[rid], pat):
            stats["generic"] += 1
            continue
        if has_placeholder_content(titles[rid]):
            stats.setdefault("placeholder", 0)
            stats["placeholder"] += 1
            continue
        try:
            inp, out, errors = assemble(scaf, titles[rid], reasons[rid])
        except Exception as e:  # pydantic 검증 실패 등
            stats["pydantic"] += 1
            continue
        if errors:
            stats["verify_fail"] += 1
            continue
        rev = opus.get(rid, {})
        if not rev.get("pass", True):
            stats["opus_fail"] += 1
            continue
        if rev.get("realism", 5) < args.min_realism:
            stats["low_realism"] += 1
            continue
        # 행 단위 중복 제거: 동일 제목 집합이면 drop
        tset = frozenset(_norm_title(t.title) for t in inp.tasks)
        if tset in seen_titlesets:
            stats["dup_row"] += 1
            continue
        seen_titlesets.add(tset)
        all_titles.extend(_norm_title(t.title) for t in inp.tasks)
        rows.append({
            "prompt": S.format_input_v9(inp),
            "chosen": S.format_for_sft_v9(out),
            "persona": json.dumps(inp.user_persona.model_dump(), ensure_ascii=False),
            "today": scaf["current_time"][:10],
            "source": f"{src_tag}_{scaf['occ_category']}",
            "meta": json.dumps({"occ_category": scaf["occ_category"],
                                "chain_pairs": scaf["chain_pairs"],
                                "n_tasks": len(scaf["slots"])}, ensure_ascii=False),
            "version": version,
        })

    df = pd.DataFrame(rows)
    Path(args.out_sft).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_hf).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_sft, index=False)
    df.to_parquet(args.out_hf, index=False)
    print(f"[saved] {args.out_sft} · {args.out_hf} — 수록 {len(df)}행 / 후보 {len(scaffold)}")
    print("탈락:", stats)
    if len(df):
        import collections
        print("source 분포:", dict(collections.Counter(r["source"] for r in rows)))
        uniq = len(set(all_titles))
        print(f"태스크 제목: 총 {len(all_titles)} · 고유 {uniq} "
              f"({100*uniq/len(all_titles):.1f}%)")
        dup = collections.Counter(all_titles)
        top = [(t, n) for t, n in dup.most_common(8) if n > 1]
        if top:
            print("최다 중복 제목:", [(t[:18], n) for t, n in top])


if __name__ == "__main__":
    main()
