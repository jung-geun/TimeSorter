#!/usr/bin/env python3
"""
Manually generate v8 chain gen JSONL files from skeleton files.
No API calls. Uses deterministic priority_order computation + handcrafted Korean text.

Rules enforced:
1. Chain tasks: contiguous in priority_order, in chain_pos order
2. Non-final chain steps: dependency = 4 or 5
3. priority_order is permutation of all task ids 1..N
4. "none" tasks must NOT be ranked #1
5. same-day deadlines: earlier time = higher rank
6. past tasks ranked below all live tasks

Usage:
  python scripts/gen_v8_chain_manual.py --batch 00
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SKEL_DIR = Path("outputs/v8_chain/skeletons")
GEN_DIR = Path("outputs/v8_chain/gen")


def load_skeleton(batch: str) -> list[dict]:
    rows = []
    with open(SKEL_DIR / f"batch_{batch}.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_priority_order(specs: list[dict], today: str) -> list[int]:
    """
    Deterministically compute correct priority_order from specs.

    Ranking priority:
    1. Past tasks go last (below all live tasks)
    2. Among live tasks:
       a. today-kind with earlier time goes first (same-day tie-break)
       b. Chain blocks are placed as a unit; within block, in chain_pos order
       c. future tasks after today tasks
       d. none tasks at the very end

    The chain block is placed according to the final step's deadline relative to
    non-chain tasks. If chain has a deadline on final step, it competes with other
    dated tasks; otherwise it goes before none tasks.
    """
    # Group by category
    past = [s for s in specs if s.get("is_past")]
    live = [s for s in specs if not s.get("is_past")]

    none_tasks = [s for s in live if s["kind"] == "none"]

    # Build chain groups
    chain_groups: dict[int, list[dict]] = {}
    for s in live:
        cg = s.get("chain_group", 0)
        if cg:
            chain_groups.setdefault(cg, []).append(s)
    for cg in chain_groups:
        chain_groups[cg].sort(key=lambda s: s["chain_pos"])

    # Non-chain live tasks (today, future, risk but not in a chain)
    non_chain_live = [s for s in live if not s.get("chain_group")]

    # Sort non-chain live tasks by kind priority then deadline
    def sort_key_non_chain(s):
        kind = s["kind"]
        dl = s.get("deadline") or ""
        if kind == "today" or kind == "risk":
            # today tasks: primary sort by time
            return (0, dl)
        elif kind == "future":
            return (1, dl)
        elif kind == "none":
            return (2, dl)
        return (3, dl)

    non_chain_live_sorted = sorted(non_chain_live, key=sort_key_non_chain)

    # Separate none from non-chain dated
    non_chain_dated = [s for s in non_chain_live_sorted if s["kind"] != "none"]

    # Now place chain blocks. A chain block as a unit has a "representative deadline"
    # which is the final step's deadline (if any), otherwise treat as future.
    def chain_representative_deadline(members: list[dict]) -> str:
        final = members[-1]  # last in chain_pos order
        dl = final.get("deadline") or ""
        return dl

    # We need to interleave chain blocks with non-chain dated tasks
    # For simplicity: place chain block right before none tasks,
    # but after today tasks and before future tasks unless final step has a deadline
    # Actually: use the chain's final deadline to determine placement

    # Build ordered result
    # Strategy:
    # 1. Collect today tasks (sorted by time)
    # 2. For each chain group: determine insertion position based on final deadline
    # 3. Future non-chain tasks
    # 4. None tasks
    # 5. Past tasks

    today_tasks = [s for s in non_chain_dated if s["kind"] in ("today", "risk") and (s.get("deadline") or "").startswith(today)]
    future_tasks = [s for s in non_chain_dated if s not in today_tasks]

    # Each chain block as a unit - determine placement score
    # Score = (0=today, 1=after_today_before_future, 2=future, 3=none)
    # If chain's final deadline is today -> place in today group
    # If chain's final deadline is future -> place with future
    # If no deadline -> place between future and none

    chain_units = []
    for cg, members in chain_groups.items():
        final_dl = members[-1].get("deadline") or ""
        if final_dl.startswith(today):
            placement = 0  # today
        elif final_dl:
            placement = 2  # future
        else:
            placement = 1  # between today and future (no deadline)
        chain_units.append((placement, final_dl, cg, members))

    chain_units.sort(key=lambda x: (x[0], x[1]))

    result = []

    # Add today tasks
    result.extend([s["idx"] for s in today_tasks])

    # Add chain blocks that have no deadline (placement=1)
    for placement, final_dl, cg, members in chain_units:
        if placement == 1:
            result.extend([m["idx"] for m in members])

    # Add future tasks
    result.extend([s["idx"] for s in future_tasks])

    # Add chain blocks with future deadlines (placement=2)
    for placement, final_dl, cg, members in chain_units:
        if placement == 2:
            result.extend([m["idx"] for m in members])

    # Add chain blocks with today deadlines (placement=0) -- shouldn't happen often
    # Actually these should go with today tasks - let's fix ordering
    # Re-do: merge today tasks with today-deadline chain blocks by time

    # Simpler approach: flatten everything with a sort key
    result = []

    items = []  # (sort_key_tuple, task_ids_list)

    # today tasks
    for s in today_tasks:
        dl = s.get("deadline") or ""
        items.append(((0, dl), [s["idx"]]))

    # chain units
    for placement, final_dl, cg, members in chain_units:
        if placement == 0:  # today deadline
            items.append(((0, final_dl), [m["idx"] for m in members]))
        elif placement == 1:  # no deadline - after today before future
            items.append(((1, ""), [m["idx"] for m in members]))
        else:  # future deadline
            items.append(((2, final_dl), [m["idx"] for m in members]))

    # future tasks
    for s in future_tasks:
        dl = s.get("deadline") or ""
        items.append(((2, dl), [s["idx"]]))

    # none tasks
    for s in none_tasks:
        items.append(((3, ""), [s["idx"]]))

    # past tasks
    for s in past:
        items.append(((4, ""), [s["idx"]]))

    items.sort(key=lambda x: x[0])

    for _, ids in items:
        result.extend(ids)

    return result


def make_task_text(spec: dict, persona_name: str, today: str, chain_context: dict[int, list[dict]], gen_line: str) -> str:
    """
    Generate Korean task text based on spec.
    Chain tasks share deliverable, each step consumes previous.
    Final chain step names deliverable + completing action with deadline.
    """
    kind = spec["kind"]
    chain_group = spec.get("chain_group", 0)
    chain_pos = spec.get("chain_pos", 0)
    deadline = spec.get("deadline")

    if chain_group:
        members = chain_context[chain_group]
        max_pos = max(m["chain_pos"] for m in members)
        is_final = (chain_pos == max_pos)

        # Generate a deliverable name based on the chain group and some variety
        # Use gen_line hint to understand context
        deliverables = [
            "보고서", "기획서", "제안서", "계획서", "안내문", "신청서",
            "정리 자료", "검토 결과물", "발표 자료", "요약본"
        ]
        # Pick a stable deliverable based on chain_group hash
        deliverable = deliverables[hash(f"{chain_group}_{persona_name}") % len(deliverables)]

        if chain_pos == 1:
            # First step: draft/prepare
            templates = [
                f"{deliverable} 초안 작성",
                f"{deliverable} 기초 자료 수집",
                f"{deliverable} 항목 정리",
                f"{deliverable} 1차 초안 준비",
            ]
            return templates[hash(f"{chain_group}_{persona_name}_1") % len(templates)]
        elif chain_pos == 2:
            prev_templates = [
                f"작성한 {deliverable} 검토",
                f"정리한 {deliverable} 내용 확인",
                f"준비한 {deliverable} 수정",
                f"초안 {deliverable} 피드백 반영",
            ]
            return prev_templates[hash(f"{chain_group}_{persona_name}_2") % len(prev_templates)]
        elif chain_pos == 3:
            templates = [
                f"검토한 {deliverable} 최종 수정",
                f"확인된 {deliverable} 내용 보완",
                f"수정된 {deliverable} 다듬기",
                f"피드백 반영한 {deliverable} 정리",
            ]
            return templates[hash(f"{chain_group}_{persona_name}_3") % len(templates)]
        elif chain_pos == 4:
            if is_final and deadline:
                date_str = deadline[:10]
                time_str = deadline[11:16]
                templates = [
                    f"확정된 {deliverable} 제출 ({date_str} {time_str}까지)",
                    f"완성된 {deliverable} 발송 ({date_str} {time_str}까지)",
                    f"최종 {deliverable} 제출 ({date_str} {time_str}까지)",
                ]
                return templates[hash(f"{chain_group}_{persona_name}_4") % len(templates)]
            else:
                templates = [
                    f"완성된 {deliverable} 최종 검토",
                    f"확정된 {deliverable} 전달 준비",
                ]
                return templates[hash(f"{chain_group}_{persona_name}_4") % len(templates)]
        elif chain_pos == 5:
            if is_final and deadline:
                date_str = deadline[:10]
                time_str = deadline[11:16]
                templates = [
                    f"완성된 {deliverable} 최종 제출 ({date_str} {time_str}까지)",
                    f"확정된 {deliverable} 전달 완료 ({date_str} {time_str}까지)",
                ]
                return templates[hash(f"{chain_group}_{persona_name}_5") % len(templates)]
            else:
                templates = [
                    f"완성된 {deliverable} 배포",
                    f"확정된 {deliverable} 공유",
                ]
                return templates[hash(f"{chain_group}_{persona_name}_5") % len(templates)]
        else:
            return f"{deliverable} {chain_pos}단계 처리"

    elif kind == "today":
        if deadline:
            date_str = deadline[:10]
            time_str = deadline[11:16]
            tasks = [
                f"청구서 제출 ({date_str} {time_str}까지)",
                f"서류 제출 ({date_str} {time_str}까지)",
                f"신청서 접수 ({date_str} {time_str}까지)",
                f"결재 요청 ({date_str} {time_str}까지)",
                f"예약 확인 전화 ({date_str} {time_str}까지)",
                f"납부 처리 ({date_str} {time_str}까지)",
                f"회신 메일 전송 ({date_str} {time_str}까지)",
                f"물품 수령 ({date_str} {time_str}까지)",
            ]
            return tasks[hash(f"today_{spec['idx']}_{persona_name}") % len(tasks)]
        return "오늘 중 처리 필요한 업무"

    elif kind == "future":
        if deadline:
            date_str = deadline[:10]
            time_str = deadline[11:16]
            tasks = [
                f"정기 점검 예약 ({date_str} {time_str}까지)",
                f"자료 제출 ({date_str} {time_str}까지)",
                f"행사 참석 준비 ({date_str} {time_str}까지)",
                f"설문 작성 ({date_str} {time_str}까지)",
                f"도서 반납 ({date_str} {time_str}까지)",
                f"약품 수령 ({date_str} {time_str}까지)",
                f"업무 보고서 제출 ({date_str} {time_str}까지)",
            ]
            return tasks[hash(f"future_{spec['idx']}_{persona_name}") % len(tasks)]
        return "향후 처리 예정 업무"

    elif kind == "none":
        tasks = [
            "집 청소 및 정리",
            "장보기 목록 작성",
            "운동 루틴 계획",
            "독서 계속하기",
            "취미 활동 정리",
            "연락처 업데이트",
            "개인 메모 정리",
            "가계부 작성",
        ]
        return tasks[hash(f"none_{spec['idx']}_{persona_name}") % len(tasks)]

    return "기타 업무"


def make_scores(specs: list[dict], today: str) -> list[dict]:
    """Generate scores for each task."""
    scores = []
    for s in specs:
        kind = s["kind"]
        chain_group = s.get("chain_group", 0)
        chain_pos = s.get("chain_pos", 0)
        is_past = s.get("is_past", False)
        deadline = s.get("deadline")

        # Find max chain_pos for this group
        if chain_group:
            max_pos = max(
                sp["chain_pos"] for sp in specs
                if sp.get("chain_group") == chain_group
            )
            is_final = (chain_pos == max_pos)
        else:
            is_final = False

        if is_past:
            urgency = 1
            importance = 2
            dependency = 1
            time_constraint = 1
            reason = "이미 지난 일정으로 긴급성 낮음"
        elif kind == "today":
            urgency = 5
            importance = 4
            dependency = 1
            time_constraint = 5
            reason = f"오늘 {deadline[11:16] if deadline else ''}까지 처리해야 함"
        elif kind == "risk":
            urgency = 5
            importance = 5
            dependency = 1
            time_constraint = 5
            reason = "리스크 상황으로 즉시 처리 필요"
        elif chain_group:
            if is_final:
                # Final chain step: has deadline usually
                if deadline:
                    urgency = 4
                    importance = 4
                    dependency = 1  # final step doesn't need high dependency
                    time_constraint = 4
                    reason = f"체인 최종 단계, {deadline[:10]} 제출 마감"
                else:
                    urgency = 3
                    importance = 4
                    dependency = 1
                    time_constraint = 3
                    reason = "체인 최종 단계"
            else:
                # Non-final chain step: MUST have dependency >= 4
                urgency = 3
                importance = 4
                dependency = 4  # CRITICAL: must be 4 or 5
                time_constraint = 3
                reason = f"체인 {chain_pos}단계, 다음 단계 진행을 위해 먼저 완료 필요"
        elif kind == "future":
            urgency = 2
            importance = 3
            dependency = 1
            time_constraint = 3
            reason = f"미래 마감 ({deadline[:10] if deadline else ''})"
        elif kind == "none":
            urgency = 1
            importance = 2
            dependency = 1
            time_constraint = 1
            reason = "마감 없는 선택적 업무"
        else:
            urgency = 2
            importance = 2
            dependency = 1
            time_constraint = 2
            reason = "일반 업무"

        scores.append({
            "task_id": s["idx"],
            "urgency": urgency,
            "importance": importance,
            "dependency": dependency,
            "time_constraint": time_constraint,
            "reason": reason,
        })

    return scores


def build_prompt(tasks: list[str], persona_label: str, today_display: str) -> str:
    task_lines = "\n".join(f"- {t}" for t in tasks)
    return f"[{persona_label}의 오늘의 할 일 목록]\n{task_lines}"


def process_row(row: dict) -> dict:
    skel_id = row["id"]
    today = row["today"]
    today_display = row["today_display"]
    persona_name = row["persona_name"]
    persona_label = row["persona_label"]
    gen_lines = row["gen_lines"]
    meta_str = row["meta"]

    meta = json.loads(meta_str)
    specs = meta["specs"]

    # Build chain context
    chain_context: dict[int, list[dict]] = {}
    for s in specs:
        cg = s.get("chain_group", 0)
        if cg:
            chain_context.setdefault(cg, []).append(s)

    # Generate task texts
    tasks_text = []
    for s in specs:
        text = make_task_text(s, persona_name, today, chain_context, gen_lines[s["idx"] - 1] if s["idx"] - 1 < len(gen_lines) else "")
        tasks_text.append(text)

    # Compute priority order
    priority_order = compute_priority_order(specs, today)

    # Generate scores
    scores = make_scores(specs, today)

    # Build chosen
    chosen = {
        "tasks": [{"id": s["idx"], "text": tasks_text[s["idx"] - 1]} for s in specs],
        "priority_order": priority_order,
        "scores": scores,
    }

    # Build prompt
    prompt = build_prompt(tasks_text, persona_label, today_display)

    return {
        "id": skel_id,
        "tasks": tasks_text,
        "chosen": chosen,
        "prompt": prompt,
        "persona": persona_label,
        "today": today,
        "source": "v8_dependency_chain_complex",
        "meta": meta_str,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, help="Batch number e.g. 00")
    args = parser.parse_args()

    batch = args.batch
    skel_rows = load_skeleton(batch)
    print(f"Loaded {len(skel_rows)} skeleton rows for batch_{batch}")

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GEN_DIR / f"batch_{batch}.jsonl"

    generated = []
    for i, row in enumerate(skel_rows):
        try:
            result = process_row(row)
            generated.append(result)
        except Exception as e:
            print(f"ERROR on row {i} ({row.get('id')}): {e}")
            raise

    with open(out_path, "w") as f:
        for row in generated:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Written {len(generated)} rows to {out_path}")


if __name__ == "__main__":
    main()
