#!/usr/bin/env python
"""v9 샘플 조립 — 결정적 스캐폴드 + LLM 산출(haiku/sonnet) → ScheduleInput/Response + 검증.

입력:
  outputs/v9/sample_scaffold.jsonl   (scoring/rank/schedule/slots/persona/chain_pairs)
  outputs/v9/sample_llm.json         (workflow 결과: [{index, tasks, reasoning, review}])
출력:
  data/v9/sample.jsonl               (SFT 행: prompt/chosen/persona/current_time/source/meta/version)
  outputs/v9/sample_preview.md       (사람이 읽을 미리보기)

사용:
  uv run python scripts/v9/assemble_sample.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from timesorter.data import schema_v9 as S  # noqa: E402

SCAF = Path("outputs/v9/sample_scaffold.jsonl")
LLM = Path("outputs/v9/sample_llm.json")
OUT_JSONL = Path("data/v9/sample.jsonl")
OUT_MD = Path("outputs/v9/sample_preview.md")


def assemble_row(scaf: dict, llm: dict) -> tuple[S.ScheduleInput, S.ScheduleResponseV9, list[str]]:
    persona = {k: v for k, v in scaf["persona"].items() if k != "_meta"}
    title_map = {t["task_id"]: t for t in llm["tasks"]}
    reason_map = {r["task_id"]: r for r in llm["reasoning"]}

    in_tasks = []
    for sl in scaf["slots"]:
        tid = sl["task_id"]
        ht = title_map.get(tid, {})
        in_tasks.append(S.InputTask(
            task_id=tid, title=ht.get("title", tid), memo=ht.get("memo", ""),
            source=ht.get("source", ""), deadline=sl["deadline"],
            estimated_duration_minutes=sl["estimated_duration_minutes"]))
    inp = S.ScheduleInput(current_time=scaf["current_time"],
                          user_persona=S.UserPersona(**persona), tasks=in_tasks)

    sched_tasks = []
    for tid in sorted(scaf["priority_rank"], key=lambda t: scaf["priority_rank"][t]):
        sc = scaf["scoring"][tid]
        rs = reason_map.get(tid, {})
        sched_tasks.append(S.ScheduledTask(
            task_id=tid, title=title_map.get(tid, {}).get("title", tid),
            priority_rank=scaf["priority_rank"][tid],
            scoring=S.Scoring(**sc),
            reasoning=S.Reasoning(summary=rs.get("summary", ""),
                                  chaining_detail=rs.get("chaining_detail", "")),
            recommended_schedule=S.RecommendedSchedule(**scaf["schedule"][tid])))
    out = S.ScheduleResponseV9(scheduled_tasks=sched_tasks)
    chain_pairs = [tuple(p) for p in scaf["chain_pairs"]]
    errors = S.verify_chosen_v9(inp, out, chain_pairs)
    return inp, out, errors


def main() -> None:
    scaffold = [json.loads(l) for l in SCAF.open()]
    llm_rows = json.loads(LLM.read_text())
    if isinstance(llm_rows, str):
        llm_rows = json.loads(llm_rows)
    llm_by_idx = {r["index"]: r for r in llm_rows}

    rows, previews = [], []
    n_pass_verify = n_pass_opus = 0
    for i, scaf in enumerate(scaffold):
        if i not in llm_by_idx:
            print(f"  [skip] row {i}: LLM 결과 없음")
            continue
        llm = llm_by_idx[i]
        inp, out, errors = assemble_row(scaf, llm)
        review = llm.get("review", {})
        ok_v = not errors
        ok_o = bool(review.get("pass"))
        n_pass_verify += ok_v
        n_pass_opus += ok_o
        rows.append({
            "prompt": S.format_input_v9(inp),
            "chosen": S.format_for_sft_v9(out),
            "persona": json.dumps(inp.user_persona.model_dump(), ensure_ascii=False),
            "current_time": inp.current_time,
            "source": f"v9_{scaf['scenario']}",
            "meta": json.dumps({"scenario": scaf["scenario"],
                                "chain_pairs": scaf["chain_pairs"]}, ensure_ascii=False),
            "version": "v9",
            "_verify_errors": errors,
            "_opus_review": review,
        })
        flag = "✅" if ok_v else "❌"
        print(f"  row {i} [{scaf['scenario']}] {flag} verify({len(errors)} err) "
              f"opus(pass={ok_o} realism={review.get('realism')})")
        if errors:
            for e in errors[:4]:
                print(f"       - {e}")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items()
                                if not k.startswith("_")}, ensure_ascii=False) + "\n")

    # 미리보기 (처음 2행 풀 본문)
    md = [f"# v9 샘플 미리보기 ({len(rows)}행, verify통과 {n_pass_verify}, opus통과 {n_pass_opus})\n"]
    for r in rows[:2]:
        md.append(f"\n## source={r['source']}\n")
        md.append("### 입력 (prompt)\n```json\n" + r["prompt"] + "\n```\n")
        md.append("### 출력 (chosen)\n```json\n"
                  + json.dumps(json.loads(r["chosen"]), ensure_ascii=False, indent=2) + "\n```\n")
    OUT_MD.write_text("\n".join(md))
    print(f"\n[saved] {OUT_JSONL} ({len(rows)}행) · {OUT_MD}")
    print(f"verify 통과 {n_pass_verify}/{len(rows)} · opus 통과 {n_pass_opus}/{len(rows)}")


if __name__ == "__main__":
    main()
