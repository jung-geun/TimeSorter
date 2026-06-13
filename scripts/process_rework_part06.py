#!/usr/bin/env python3
"""
Process rework_v1v2_part06.jsonl and generate rework_agent_out_part06.jsonl
Each row: read instruction/meta -> improve tasks -> generate v3 schema JSON
"""

import json
import re
from datetime import datetime, timedelta
from typing import Optional

INPUT_PATH = "/mnt/hdd/WD_8TB/code/TimeSorter/outputs/rework_v1v2_part06.jsonl"
OUTPUT_PATH = "/mnt/hdd/WD_8TB/code/TimeSorter/outputs/rework_agent_out_part06.jsonl"


def parse_deadline(deadline_str: Optional[str]):
    """Parse deadline string to datetime or None"""
    if not deadline_str:
        return None
    try:
        return datetime.strptime(deadline_str, "%Y-%m-%d %H:%M")
    except Exception:
        return None


def fmt_deadline(deadline_str: Optional[str]) -> str:
    """Format deadline for task text inclusion"""
    if not deadline_str:
        return ""
    dt = parse_deadline(deadline_str)
    if not dt:
        return ""
    return f"{dt.month}/{dt.day} {dt.hour:02d}:{dt.minute:02d}까지"


def fmt_deadline_short(deadline_str: Optional[str]) -> str:
    """Short format for parenthetical"""
    if not deadline_str:
        return ""
    dt = parse_deadline(deadline_str)
    if not dt:
        return ""
    return f"{dt.month}/{dt.day} {dt.hour:02d}:{dt.minute:02d}"


def is_am(deadline_str: Optional[str]) -> bool:
    dt = parse_deadline(deadline_str)
    if not dt:
        return False
    return dt.hour < 12


def is_today_deadline(deadline_str: Optional[str], today_str: str) -> bool:
    if not deadline_str or not today_str:
        return False
    dt = parse_deadline(deadline_str)
    today = parse_deadline(today_str + " 00:00")
    if not dt or not today:
        return False
    return dt.date() == today.date()


def is_tomorrow_deadline(deadline_str: Optional[str], today_str: str) -> bool:
    if not deadline_str or not today_str:
        return False
    dt = parse_deadline(deadline_str)
    today = parse_deadline(today_str + " 00:00")
    if not dt or not today:
        return False
    return dt.date() == (today + timedelta(days=1)).date()


def improve_task_text(original: str, spec: dict, today_str: str) -> str:
    """
    Improve task text by incorporating deadline, risk clause, and chain context.
    """
    deadline = spec.get("deadline")
    risk = spec.get("risk", False)
    risk_clause = spec.get("risk_clause", "")
    is_past = spec.get("is_past", False)

    text = original.strip()

    # Remove existing deadline parenthetical if present
    text = re.sub(r'\s*\([^)]*까지\)\s*$', '', text)
    text = re.sub(r'\s*\(\d+/\d+\s+\d+:\d+\)\s*$', '', text)

    if deadline:
        dl_fmt = fmt_deadline_short(deadline)
        # Add deadline to text
        if "까지" not in text and dl_fmt:
            text = f"{text} ({dl_fmt}까지)"

    if risk and risk_clause:
        # Append risk clause if not already present
        if risk_clause not in text:
            text = f"{text} — {risk_clause}"

    return text


def compute_scores_and_order(specs: list, today_str: str) -> tuple:
    """
    Compute urgency, importance, dependency, time_constraint scores and priority order.
    Returns (scores_list, priority_order_list)
    """
    scores = []
    n = len(specs)

    # Identify chain groups
    chain_groups = {}
    for spec in specs:
        cg = spec.get("chain_group", 0)
        if cg != 0:
            if cg not in chain_groups:
                chain_groups[cg] = []
            chain_groups[cg].append(spec)

    # Sort chain members by chain_pos
    for cg in chain_groups:
        chain_groups[cg].sort(key=lambda s: s.get("chain_pos", 0))

    # Check if chain last step is today or tomorrow
    chain_urgent = set()
    for cg, members in chain_groups.items():
        last = members[-1]
        last_dl = last.get("deadline")
        if last_dl and today_str:
            if is_today_deadline(last_dl, today_str) or is_tomorrow_deadline(last_dl, today_str):
                for m in members:
                    chain_urgent.add(m["idx"])

    # Compute scores per task
    for spec in specs:
        idx = spec["idx"]
        is_past = spec.get("is_past", False)
        risk = spec.get("risk", False)
        risk_clause = spec.get("risk_clause", "")
        deadline = spec.get("deadline")
        kind = spec.get("kind", "none")
        chain_group = spec.get("chain_group", 0)
        chain_pos = spec.get("chain_pos", 0)

        reason_parts = []

        if is_past:
            urgency = 1
            importance = 1
            time_constraint = 2
            # dependency: chain membership?
            dependency = 2 if chain_group != 0 else 1
            reason_parts.append(f"마감 {deadline}은 오늘({today_str}) 기준 이미 지난 일정")
            reason_parts.append("긴급도·중요도 최하위 배치")
        else:
            # Base urgency by kind
            if kind == "today":
                urgency = 4
                time_constraint = 4
                reason_parts.append(f"오늘({today_str}) 마감")
            elif kind == "future":
                urgency = 3
                time_constraint = 3
                reason_parts.append(f"미래 마감 {deadline}")
            elif kind == "dated":
                urgency = 3
                time_constraint = 3
                reason_parts.append(f"특정 날짜 마감 {deadline}")
            elif kind == "chain":
                urgency = 3
                time_constraint = 3
                reason_parts.append(f"체인 태스크 (그룹={chain_group}, 순서={chain_pos})")
            elif kind == "risk":
                urgency = 5
                time_constraint = 5
                reason_parts.append(f"리스크 마감 {deadline}")
            elif kind == "intraday":
                urgency = 4
                time_constraint = 4
                reason_parts.append(f"당일 내 마감 {deadline}")
            else:  # none
                urgency = 2
                time_constraint = 2
                reason_parts.append("마감 미지정")

            # AM escalation
            if deadline and is_am(deadline):
                urgency = min(5, urgency + 1)
                time_constraint = min(5, time_constraint + 1)
                reason_parts.append("오전 마감으로 긴급도 상향")

            # Risk override
            if risk:
                urgency = max(urgency, 4)
                importance = 5
                reason_parts.append(f"리스크 문구: '{risk_clause}' → 중요도·긴급도 최상위")
            else:
                importance = min(4, urgency) if urgency >= 3 else 2

            # Chain urgency boost
            if idx in chain_urgent:
                urgency = min(5, urgency + 1)
                importance = min(5, importance + 1)
                reason_parts.append("체인 최하단이 오늘/내일 마감 → 체인 전체 긴급 상향")

            # Dependency
            if chain_group != 0:
                other_members = chain_groups.get(chain_group, [])
                dependency = min(5, len(other_members) + 1)
                reason_parts.append(f"체인 {chain_group}의 {chain_pos}번째 단계 (총 {len(other_members)}개)")
            else:
                dependency = 1

        scores.append({
            "task_id": idx,
            "urgency": urgency,
            "importance": importance if not is_past else 1,
            "dependency": dependency,
            "time_constraint": time_constraint,
            "reason": "; ".join(reason_parts)
        })

    # Determine priority order
    # Sort key:
    # 1. is_past -> bottom
    # 2. chain urgent members -> top, in chain_pos order (grouped together)
    # 3. risk -> very top
    # 4. today AM > today PM > future > no deadline
    # 5. within same level: AM before PM

    def sort_key(spec):
        idx = spec["idx"]
        is_past = spec.get("is_past", False)
        risk = spec.get("risk", False)
        deadline = spec.get("deadline")
        kind = spec.get("kind", "none")
        chain_group = spec.get("chain_group", 0)
        chain_pos = spec.get("chain_pos", 0)

        if is_past:
            return (10, 0, 0, 0, idx)

        # Chain urgent: group together at top, ordered by chain_pos
        if idx in chain_urgent:
            # Find earliest deadline in the chain
            cg = chain_group
            members = chain_groups.get(cg, [])
            last_dl = members[-1].get("deadline") if members else None
            last_dt = parse_deadline(last_dl) if last_dl else datetime.max
            return (0, last_dt.timestamp() if last_dt != datetime.max else 1e18, chain_pos, 0, idx)

        # Risk
        if risk:
            dl_dt = parse_deadline(deadline)
            ts = dl_dt.timestamp() if dl_dt else 1e18
            am_flag = 0 if (dl_dt and dl_dt.hour < 12) else 1
            return (1, ts, am_flag, 0, idx)

        # Today AM
        if deadline and today_str and is_today_deadline(deadline, today_str) and is_am(deadline):
            dl_dt = parse_deadline(deadline)
            return (2, dl_dt.timestamp(), 0, 0, idx)

        # Today PM
        if deadline and today_str and is_today_deadline(deadline, today_str):
            dl_dt = parse_deadline(deadline)
            return (3, dl_dt.timestamp(), 0, 0, idx)

        # Future/dated with deadline
        if deadline:
            dl_dt = parse_deadline(deadline)
            ts = dl_dt.timestamp() if dl_dt else 1e18
            am_flag = 0 if (dl_dt and dl_dt.hour < 12) else 1
            return (4, ts, am_flag, 0, idx)

        # No deadline
        return (5, 0, 0, 0, idx)

    sorted_specs = sorted(specs, key=sort_key)
    priority_order = [s["idx"] for s in sorted_specs]

    return scores, priority_order


def process_row(row: dict) -> dict:
    """Process one row and return output dict"""
    rid = row["id"]
    original_tasks = row["original_tasks"]
    today_str = row.get("today", "")
    meta = json.loads(row["meta"])
    specs = meta["specs"]

    # Improve task texts
    improved_tasks = []
    for spec in specs:
        idx = spec["idx"] - 1  # 0-indexed
        original = original_tasks[idx] if idx < len(original_tasks) else f"태스크 {idx+1}"
        improved = improve_task_text(original, spec, today_str)
        improved_tasks.append(improved)

    # Compute scores and priority order
    scores, priority_order = compute_scores_and_order(specs, today_str)

    # Build v3 JSON
    chosen = {
        "tasks": improved_tasks,
        "priority_order": priority_order,
        "scores": scores
    }

    return {
        "id": rid,
        "tasks": improved_tasks,
        "chosen": chosen
    }


def main():
    processed = 0
    failed = 0
    results = []

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            out = process_row(row)
            results.append(out)
            processed += 1
        except Exception as e:
            print(f"ERROR on line {i+1}: {e}")
            failed += 1

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done. Processed: {processed}, Failed: {failed}")
    print(f"Output: {OUTPUT_PATH}")
    return processed, failed


if __name__ == "__main__":
    main()
