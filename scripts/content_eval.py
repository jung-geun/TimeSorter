#!/usr/bin/env python3
"""Content-level 평가 — 스키마(포맷)를 무시하고 추론 내용만 채점.

no-FT 모델(instruct/base)은 tasks를 [{id,text}] 대신 ["문자열"]로 출력해
verify_chosen이 'JSON 파싱 실패'로 전부 0% 처리한다.
하지만 priority_order·scores는 정상 출력하므로, tasks id를 위치 기반(1..N)으로
재구성하면 동일한 5종 골격 규칙으로 '추론 내용'만 따로 채점할 수 있다.

→ 포맷 미준수(schema)와 추론 능력(content)을 분리.

사용:
  uv run python scripts/content_eval.py                 # 6쿼리 raw_outputs.json
  uv run python scripts/content_eval.py --n30           # n=30 재추론 후 content 채점
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from timesorter.data.schema import ScheduleResponse, ScoreItem, TaskItem


def repair_to_response(raw: str) -> ScheduleResponse | None:
    """raw 출력에서 priority_order·scores를 추출하고 tasks id를 위치기반 재구성.

    스키마 위반(tasks=["문자열"])이어도 내용 채점이 가능하도록 복구.
    실패(JSON 블록 자체 없음) 시 None.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    start = cleaned.find("{")
    if start == -1:
        return None
    depth, end = 0, -1
    for i, ch in enumerate(cleaned[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None
    try:
        obj = json.loads(cleaned[start:end])
    except Exception:
        return None

    raw_tasks = obj.get("tasks", [])
    # tasks가 문자열 배열이면 위치기반 id 부여, 객체 배열이면 그대로
    tasks = []
    for i, t in enumerate(raw_tasks, 1):
        if isinstance(t, str):
            tasks.append(TaskItem(id=i, text=t))
        elif isinstance(t, dict):
            tasks.append(TaskItem(id=t.get("id", i), text=str(t.get("text", ""))))
    if not tasks:
        return None

    # scores 복구 (task_id 없으면 위치기반)
    scores = []
    for i, s in enumerate(obj.get("scores", []), 1):
        if not isinstance(s, dict):
            continue
        try:
            scores.append(ScoreItem(
                task_id=s.get("task_id", i),
                urgency=int(s.get("urgency", 3)),
                importance=int(s.get("importance", 3)),
                dependency=int(s.get("dependency", 3)),
                time_constraint=int(s.get("time_constraint", 3)),
                reason=str(s.get("reason", "")),
            ))
        except Exception:
            continue

    po = [int(x) for x in obj.get("priority_order", []) if isinstance(x, (int, float))]
    try:
        return ScheduleResponse(tasks=tasks, priority_order=po, scores=scores)
    except Exception:
        # priority_order에 범위 밖 id가 있으면 검증 통과 못 함 → 그대로 평가 위해 우회
        r = ScheduleResponse(tasks=tasks)
        r.priority_order = po
        r.scores = scores
        return r


def content_violations(skel, resp: ScheduleResponse) -> list[str]:
    """verify_chosen의 규칙 검증을 복구된 ScheduleResponse(객체)에 직접 적용."""
    from gen_schedule_v3 import verify_chosen
    # verify_chosen은 JSON 문자열을 받으므로 compact JSON으로 직렬화해 전달
    return verify_chosen(skel, resp.model_dump_json())


def eval_stored(raw_path: str):
    from gen_schedule_v3 import Skeleton, TaskSpec
    import pandas as pd

    data = json.loads(Path(raw_path).read_text())
    df = pd.read_parquet("data/scheduler_v3_eval.parquet")

    # 각 쿼리의 skeleton 복원
    skels = {}
    for q in data["queries"]:
        meta = json.loads(df.iloc[q["idx"]]["meta"])
        skels[q["idx"]] = Skeleton(scenario=meta["scenario"], today=meta["today"],
                                   specs=[TaskSpec(**s) for s in meta["specs"]])

    print(f"{'='*72}")
    print("Content-level 평가 (스키마 무시, 추론 내용만) — 6쿼리")
    print(f"{'='*72}\n")

    result = {}
    for key in ["base", "instruct", "sft", "dpo"]:
        mv = data["models"][key]
        rows = []
        for q, r in zip(data["queries"], mv["results"]):
            skel = skels[q["idx"]]
            resp = repair_to_response(r["raw_output"])
            if resp is None:
                content_pass = False
                content_viol = ["JSON 블록 없음"]
            else:
                content_viol = content_violations(skel, resp)
                content_pass = len(content_viol) == 0
            rows.append({
                "idx": q["idx"], "scenario": q["scenario"],
                "schema_pass": r["passed"],
                "schema_viol": r["violations"],
                "content_pass": content_pass,
                "content_viol": content_viol,
            })
        result[key] = rows
        sp = sum(1 for x in rows if x["schema_pass"])
        cp = sum(1 for x in rows if x["content_pass"])
        print(f"[{mv['label']}]")
        print(f"  스키마 준수 통과(현행): {sp}/6   |   내용만 통과(신규): {cp}/6")
        for x in rows:
            mark_s = "✅" if x["schema_pass"] else "❌"
            mark_c = "✅" if x["content_pass"] else "❌"
            extra = "" if x["content_pass"] else f" — {x['content_viol']}"
            print(f"    {x['scenario']:18} schema {mark_s} | content {mark_c}{extra}")
        print()

    Path("presentation/01_model_comparison/content_eval.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2))
    print("[saved] presentation/01_model_comparison/content_eval.json")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="presentation/01_model_comparison/raw_outputs.json")
    args = ap.parse_args()
    eval_stored(args.raw)


if __name__ == "__main__":
    main()
