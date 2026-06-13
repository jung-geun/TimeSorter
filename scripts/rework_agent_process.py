#!/usr/bin/env python3
"""
rework_agent_process.py
한국어 일정 데이터 재가공: rework_v1v2_part00.jsonl → rework_agent_out_part00.jsonl
각 행의 meta spec을 기반으로 태스크 텍스트 개선 및 v3 스키마 JSON 생성
"""

import json
import re
from datetime import datetime, timedelta
from typing import Optional

INPUT_FILE = "/mnt/hdd/WD_8TB/code/TimeSorter/outputs/rework_v1v2_part00.jsonl"
OUTPUT_FILE = "/mnt/hdd/WD_8TB/code/TimeSorter/outputs/rework_agent_out_part00.jsonl"


def format_deadline_kr(deadline_str: str) -> str:
    """
    '2026-03-03 10:00' → '3/3 10:00까지'
    """
    try:
        dt = datetime.strptime(deadline_str, "%Y-%m-%d %H:%M")
        return f"{dt.month}/{dt.day} {dt.hour:02d}:{dt.minute:02d}까지"
    except:
        return deadline_str


def is_today_or_tomorrow(today_str: str, deadline_str: Optional[str]) -> bool:
    """체인 최하단 마감이 오늘/내일인지 확인"""
    if not today_str or not deadline_str:
        return False
    try:
        today = datetime.strptime(today_str, "%Y-%m-%d").date()
        dl = datetime.strptime(deadline_str, "%Y-%m-%d %H:%M").date()
        return dl <= today + timedelta(days=1)
    except:
        return False


def is_morning(deadline_str: Optional[str]) -> bool:
    """오전 마감 여부 (12:00 이전)"""
    if not deadline_str:
        return False
    try:
        dt = datetime.strptime(deadline_str, "%Y-%m-%d %H:%M")
        return dt.hour < 12
    except:
        return False


def improve_task_text(original: str, spec: dict, chain_specs: list, chain_peers: list) -> str:
    """
    골격 spec에 따라 태스크 텍스트를 개선한다.
    - 마감 일시 포함
    - 리스크 문구 포함
    - 체인 단계 표현
    - 지난 일정: 날짜만 표기, '지났다' 표현 금지
    """
    # 원본에서 시각 표기 제거 (예: " (08:33~09:03)" 형태)
    clean = re.sub(r'\s*\(\d{1,2}:\d{2}~\d{1,2}:\d{2}\)', '', original).strip()

    kind = spec.get('kind', 'none')
    deadline = spec.get('deadline')
    is_past = spec.get('is_past', False)
    risk = spec.get('risk', False)
    risk_clause = spec.get('risk_clause', '')
    chain_group = spec.get('chain_group', 0)
    chain_pos = spec.get('chain_pos', 0)

    result = clean

    if kind == 'chain' and chain_group > 0:
        # 체인 단계 처리
        total_steps = len(chain_specs)
        pos = chain_pos
        if deadline:
            dl_str = format_deadline_kr(deadline)
            result = f"{clean} ({pos}단계/{total_steps}단계, {dl_str})"
        else:
            result = f"{clean} ({pos}단계/{total_steps}단계)"
    elif deadline and not is_past:
        # 마감이 있는 일반 태스크
        dl_str = format_deadline_kr(deadline)
        result = f"{clean} ({dl_str})"
    elif is_past and deadline:
        # 지난 일정: 날짜만 표기 ('지났다' 표현 금지)
        try:
            dt = datetime.strptime(deadline, "%Y-%m-%d %H:%M")
            date_str = f"{dt.month}/{dt.day} {dt.hour:02d}:{dt.minute:02d}"
            result = f"{clean} ({date_str})"
        except:
            result = clean
    # kind == 'none': 마감 없음, 날짜·시각 표기 금지
    elif kind == 'none':
        result = clean

    # 리스크 문구 추가
    if risk and risk_clause:
        result = f"{result} — {risk_clause}"

    return result


def score_task(spec: dict, today_str: str, chain_last_deadline: Optional[str]) -> dict:
    """
    v3 스키마 점수 계산
    Returns: {task_id, urgency, importance, dependency, time_constraint, reason}
    """
    idx = spec['idx']
    kind = spec.get('kind', 'none')
    deadline = spec.get('deadline')
    is_past = spec.get('is_past', False)
    risk = spec.get('risk', False)
    chain_group = spec.get('chain_group', 0)
    chain_pos = spec.get('chain_pos', 0)
    risk_clause = spec.get('risk_clause', '')

    reasons = []

    # 기본값
    urgency = 2
    importance = 2
    dependency = 1
    time_constraint = 2

    if is_past:
        # 지난 일정
        urgency = 1
        importance = 1
        time_constraint = 2
        if deadline:
            try:
                dt = datetime.strptime(deadline, "%Y-%m-%d %H:%M")
                reasons.append(f"마감 {dt.month}/{dt.day} {dt.hour:02d}:{dt.minute:02d} — 이미 지난 일정")
            except:
                reasons.append("이미 지난 일정")
        else:
            reasons.append("이미 지난 일정")
    elif risk:
        # 리스크 태스크
        urgency = max(4, urgency)
        importance = max(4, importance)
        if deadline:
            try:
                dt = datetime.strptime(deadline, "%Y-%m-%d %H:%M")
                morning = dt.hour < 12
                if morning:
                    time_constraint = 5
                    urgency = 5
                else:
                    time_constraint = 4
                    urgency = 4
                reasons.append(f"리스크: {risk_clause} (마감 {dt.month}/{dt.day} {dt.hour:02d}:{dt.minute:02d})")
            except:
                time_constraint = 4
                reasons.append(f"리스크: {risk_clause}")
        else:
            time_constraint = 4
            reasons.append(f"리스크: {risk_clause}")
    elif kind == 'chain' and chain_group > 0:
        # 체인 태스크
        dependency = chain_pos  # 앞 단계에 의존
        if chain_last_deadline:
            try:
                dt = datetime.strptime(chain_last_deadline, "%Y-%m-%d %H:%M")
                morning = dt.hour < 12
                chain_today_tmrw = is_today_or_tomorrow(today_str, chain_last_deadline)
                if chain_today_tmrw:
                    urgency = 5
                    importance = 4
                    time_constraint = 5 if morning else 4
                    reasons.append(
                        f"체인 {chain_pos}단계 (최종 마감 {dt.month}/{dt.day} {dt.hour:02d}:{dt.minute:02d} "
                        f"{'오전' if morning else '오후'}, 오늘/내일 마감)"
                    )
                else:
                    urgency = 3
                    importance = 3
                    time_constraint = 3
                    reasons.append(
                        f"체인 {chain_pos}단계 (최종 마감 {dt.month}/{dt.day} {dt.hour:02d}:{dt.minute:02d})"
                    )
            except:
                urgency = 3
                importance = 3
                dependency = chain_pos
                time_constraint = 3
                reasons.append(f"체인 {chain_pos}단계")
        else:
            urgency = 3
            importance = 3
            dependency = chain_pos
            time_constraint = 3
            reasons.append(f"체인 {chain_pos}단계 (마감 미정)")
    elif kind == 'today' and deadline:
        try:
            dt = datetime.strptime(deadline, "%Y-%m-%d %H:%M")
            morning = dt.hour < 12
            time_constraint = 5 if morning else 4
            urgency = 5 if morning else 4
            importance = 3
            reasons.append(
                f"오늘 {'오전' if morning else '오후'} 마감 {dt.month}/{dt.day} {dt.hour:02d}:{dt.minute:02d}"
            )
        except:
            urgency = 4
            importance = 3
            time_constraint = 4
            reasons.append(f"오늘 마감")
    elif kind in ('week', 'month', 'date', 'dated', 'future') and deadline and today_str:
        try:
            dt = datetime.strptime(deadline, "%Y-%m-%d %H:%M")
            today = datetime.strptime(today_str, "%Y-%m-%d").date()
            days_left = (dt.date() - today).days
            morning = dt.hour < 12
            if days_left <= 1:
                urgency = 5 if morning else 4
                importance = 4
                time_constraint = 5 if morning else 4
            elif days_left <= 3:
                urgency = 4
                importance = 3
                time_constraint = 4
            elif days_left <= 7:
                urgency = 3
                importance = 3
                time_constraint = 3
            else:
                urgency = 2
                importance = 3
                time_constraint = 2
            reasons.append(
                f"마감 {dt.month}/{dt.day} {dt.hour:02d}:{dt.minute:02d} "
                f"({'오전' if morning else '오후'}, {days_left}일 후)"
            )
        except:
            urgency = 3
            importance = 3
            time_constraint = 3
            reasons.append(f"마감 있음")
    elif kind in ('dated', 'future') and deadline and not today_str:
        # today 미상: 날짜 기반 판단 금지, 마감 정보만 기록
        try:
            dt = datetime.strptime(deadline, "%Y-%m-%d %H:%M")
            morning = dt.hour < 12
            urgency = 3
            importance = 3
            time_constraint = 3 if morning else 2
            reasons.append(
                f"마감 {dt.month}/{dt.day} {dt.hour:02d}:{dt.minute:02d} "
                f"({'오전' if morning else '오후'}, 오늘 날짜 미상으로 상대적 판단 생략)"
            )
        except:
            urgency = 2
            importance = 2
            time_constraint = 2
            reasons.append("마감 정보 있음 (오늘 날짜 미상)")
    elif kind == 'none' or (not deadline):
        # 마감 없음
        urgency = 2
        importance = 2
        time_constraint = 1
        reasons.append("마감 없음")
    else:
        # 기타 (deadline 있지만 kind 알수없음)
        urgency = 2
        importance = 2
        time_constraint = 2
        reasons.append("마감 정보 있음")

    reason = "; ".join(reasons) if reasons else "일반 태스크"

    return {
        "task_id": idx,
        "urgency": urgency,
        "importance": importance,
        "dependency": dependency,
        "time_constraint": time_constraint,
        "reason": reason
    }


def compute_priority_score(score: dict) -> float:
    """우선순위 정렬을 위한 종합 점수 (높을수록 앞)"""
    # 시간 제약과 긴급도를 우선
    return (
        score['urgency'] * 3.0 +
        score['importance'] * 2.0 +
        score['time_constraint'] * 2.5 +
        (5 - score['dependency']) * 1.0  # dependency 낮을수록 먼저
    )


def process_row(row: dict) -> dict:
    """한 행을 처리하여 출력 형식으로 변환"""
    row_id = row['id']
    today_str = row.get('today', '')
    original_tasks = row['original_tasks']
    meta = json.loads(row['meta'])
    specs = meta['specs']  # 1-indexed by 'idx'

    # spec을 idx 기준으로 매핑
    spec_by_idx = {s['idx']: s for s in specs}

    # 체인 그룹 정보 수집
    chain_groups = {}
    for s in specs:
        cg = s.get('chain_group', 0)
        if cg > 0:
            if cg not in chain_groups:
                chain_groups[cg] = []
            chain_groups[cg].append(s)

    # 체인 그룹별 마지막 마감 (chain_pos 최대인 것의 deadline)
    chain_last_deadlines = {}
    for cg, members in chain_groups.items():
        last = max(members, key=lambda s: s.get('chain_pos', 0))
        chain_last_deadlines[cg] = last.get('deadline')

    # 태스크 텍스트 개선
    improved_tasks = []
    for i, orig in enumerate(original_tasks):
        idx = i + 1
        spec = spec_by_idx.get(idx, {'idx': idx, 'kind': 'none', 'deadline': None,
                                      'is_past': False, 'chain_group': 0, 'chain_pos': 0,
                                      'risk': False, 'risk_clause': '', 'rel_expr': ''})
        cg = spec.get('chain_group', 0)
        chain_specs_for_group = chain_groups.get(cg, [])
        chain_peers = [original_tasks[s['idx'] - 1] for s in chain_specs_for_group
                       if 1 <= s['idx'] <= len(original_tasks)]
        improved = improve_task_text(orig, spec, chain_specs_for_group, chain_peers)
        improved_tasks.append(improved)

    # 점수 계산
    scores = []
    for i in range(len(original_tasks)):
        idx = i + 1
        spec = spec_by_idx.get(idx, {'idx': idx, 'kind': 'none', 'deadline': None,
                                      'is_past': False, 'chain_group': 0, 'chain_pos': 0,
                                      'risk': False, 'risk_clause': '', 'rel_expr': ''})
        cg = spec.get('chain_group', 0)
        cg_last = chain_last_deadlines.get(cg)
        s = score_task(spec, today_str, cg_last)
        scores.append(s)

    # 우선순위 결정
    # 체인은 연속 배치: chain_group별로 묶어서 정렬
    # 1. 체인 그룹의 대표 점수 = 체인 내 최고 urgency/importance로 (체인 첫 멤버 기준)
    # 2. 체인 내부는 chain_pos 순서

    # task idx별 점수 맵
    score_map = {s['task_id']: s for s in scores}

    # 비체인 태스크: 직접 정렬
    non_chain_idxs = [i + 1 for i, _ in enumerate(original_tasks)
                      if spec_by_idx.get(i + 1, {}).get('chain_group', 0) == 0]

    # 체인 그룹: 그룹별 대표 점수로 정렬
    # 대표 점수 = 그룹 내 최대 urgency + time_constraint
    chain_group_rep_score = {}
    for cg, members in chain_groups.items():
        rep = max(members, key=lambda s: compute_priority_score(score_map[s['idx']]))
        chain_group_rep_score[cg] = compute_priority_score(score_map[rep['idx']])

    # 정렬 단위 구성: (sort_key, is_chain, chain_group, task_idx_list)
    sort_units = []

    # 비체인
    for idx in non_chain_idxs:
        sk = compute_priority_score(score_map[idx])
        sort_units.append((sk, False, 0, [idx]))

    # 체인 그룹
    seen_chains = set()
    for cg, members in chain_groups.items():
        if cg in seen_chains:
            continue
        seen_chains.add(cg)
        sorted_members = sorted(members, key=lambda s: s.get('chain_pos', 0))
        idx_list = [s['idx'] for s in sorted_members]
        sk = chain_group_rep_score[cg]
        sort_units.append((sk, True, cg, idx_list))

    # 내림차순 정렬
    sort_units.sort(key=lambda x: x[0], reverse=True)

    # priority_order 생성
    priority_order = []
    for _, _, _, idx_list in sort_units:
        priority_order.extend(idx_list)

    chosen = {
        "tasks": improved_tasks,
        "priority_order": priority_order,
        "scores": scores
    }

    return {
        "id": row_id,
        "tasks": improved_tasks,
        "chosen": chosen
    }


def main():
    processed = 0
    failed = 0
    results = []

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            result = process_row(row)
            results.append(result)
            processed += 1
        except Exception as e:
            print(f"[ERROR] Line {i}: {e}")
            failed += 1

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"처리 완료: {processed}행 성공, {failed}행 실패")
    print(f"출력 파일: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
