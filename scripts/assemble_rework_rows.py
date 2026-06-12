#!/usr/bin/env python
"""rework_legacy_prep.py + 에이전트 출력을 검증·병합.

에이전트 출력 형식:
  {"id": "rework_XX_NNNN", "tasks": ["개선된 태스크1",...], "chosen": {...v3 JSON...}}

검증: JSON 스키마, 4축 점수 범위, 최소 reason 길이, 지난 일정 후순위
출력: data/scheduler_rework_v1v2.parquet (prompt/chosen/persona/today/source/meta)

사용:
  uv run python scripts/assemble_rework_rows.py \\
      --skeletons "outputs/rework_v1v2_part*.jsonl" \\
      --agent-out "outputs/rework_agent_out_part*.jsonl" \\
      --out data/scheduler_rework_v1v2.parquet
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from timesorter.data.schema import parse_lenient, render_system_prompt, system_prompt_for


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
                except json.JSONDecodeError:
                    pass
    return rows


def validate_chosen(chosen_json: str, tasks: list[str], meta: dict) -> list[str]:
    """기본 스키마 검증 (골격 없이 할 수 있는 범위)."""
    errors = []
    try:
        d = json.loads(chosen_json)
    except Exception:
        return ["json_parse_error"]

    task_list = d.get("tasks", [])
    n_tasks = len(task_list)
    task_ids = set(range(1, n_tasks + 1))
    po = d.get("priority_order", [])
    scores = d.get("scores", [])

    if not task_ids:
        errors.append("no_tasks")
    if set(po) != task_ids:
        errors.append("priority_order_mismatch")
    if len(scores) != len(task_ids):
        errors.append("scores_count_mismatch")

    for s in scores:
        for ax in ["urgency", "importance", "dependency", "time_constraint"]:
            v = s.get(ax, 0)
            if not (1 <= v <= 5):
                errors.append(f"invalid_{ax}={v}")
                break
        reason = s.get("reason", "")
        if len(str(reason)) < 5:
            errors.append("reason_too_short")

    # 지난 일정(is_past=True) 태스크가 상위 50%에 배치되면 경고
    specs = meta.get("specs", [])
    if po and specs:
        past_task_ids = set()
        for sp in specs:
            if sp.get("is_past"):
                # 해당 인덱스의 task id 추정 (순서 기반)
                idx = sp.get("idx", -1)
                if 0 <= idx < len(tasks):
                    past_task_ids.add(idx + 1)  # task id는 1-based
        if past_task_ids:
            cutoff = len(po) // 2
            for past_id in past_task_ids:
                if past_id in po[:cutoff]:
                    errors.append("past_task_ranked_high")
                    break

    return errors


def build_prompt(skel_row: dict) -> str:
    """원본 태스크를 v3 시스템 프롬프트와 함께 prompt로 구성."""
    tasks_text = "\n".join(f"- {t}" for t in skel_row.get("original_tasks", []))
    persona = skel_row.get("persona", "")
    # 이름 추출 (예: "홍길동 씨의" → "홍길동")
    name_match = re.match(r"(.+?)\s*씨", persona)
    name = name_match.group(1) if name_match else "사용자"
    header = f"[{name} 씨의 할 일 목록]"
    tmpl = system_prompt_for("v3")
    system = render_system_prompt(tmpl=tmpl, persona=persona, today=skel_row.get("today", ""))
    return f"{system}\n\n{header}\n{tasks_text}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skeletons", required=True)
    parser.add_argument("--agent-out", required=True)
    parser.add_argument("--out", default="data/scheduler_rework_v1v2.parquet")
    args = parser.parse_args()

    skels = {r["id"]: r for r in load_jsonl(args.skeletons)}
    outputs = load_jsonl(args.agent_out)
    print(f"스켈레톤 {len(skels)}개, 에이전트 출력 {len(outputs)}개")

    rows, rejected = [], Counter()
    seen: set[str] = set()

    for o in outputs:
        sid = o.get("id")
        if sid not in skels or sid in seen:
            rejected["unknown_or_dup"] += 1
            continue
        seen.add(sid)
        sk = skels[sid]
        meta = json.loads(sk.get("meta", "{}"))

        # tasks 수 확인
        tasks = o.get("tasks", [])
        if len(tasks) != sk.get("n_tasks", 0):
            rejected["task_count_mismatch"] += 1
            continue

        chosen = o.get("chosen")
        if not chosen:
            rejected["no_chosen"] += 1
            continue

        chosen_str = json.dumps(chosen, ensure_ascii=False) if isinstance(chosen, dict) else str(chosen)
        errs = validate_chosen(chosen_str, tasks, meta)
        if errs:
            fatal = [e for e in errs if e in (
                "json_parse_error", "no_tasks", "priority_order_mismatch",
                "past_task_ranked_high",
            )]
            if fatal:
                rejected["validation_" + fatal[0]] += 1
                continue

        source_type = sk.get("source_type", "v1")
        scenario = sk.get("scenario", "dated_mixed")
        prompt = build_prompt({**sk, "original_tasks": tasks})

        rows.append({
            "prompt": prompt,
            "chosen": chosen_str,
            "persona": sk.get("persona", ""),
            "today": sk.get("today", ""),
            "source": f"rework_{source_type}_{scenario}",
            "meta": sk.get("meta", ""),
        })

    print(f"\n통과: {len(rows)}행")
    print(f"거부: {dict(rejected)}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(args.out, index=False)
        source_dist = df["source"].value_counts().to_dict()
        print(f"\n저장: {args.out}")
        print("source 분포:", source_dist)
    else:
        print("저장할 행 없음")
