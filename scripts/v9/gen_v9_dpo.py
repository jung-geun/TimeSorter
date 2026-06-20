#!/usr/bin/env python
"""v9 DPO 하드네거티브 생성 — chosen(정답)에서 규칙 1곳만 위반한 rejected 파생.

카테고리(형식 동일·내용 한 곳만 오류):
  schedule_overlap   : 인접 두 블록을 겹치게
  deadline_miss      : 마감 있는 태스크 블록을 마감 뒤로 이동
  rank_score_mismatch: total_score 더 높은 태스크를 더 낮은 rank로(순위 역전)
  chain_order_break  : 체인 선행/후행 rank 뒤바꿈
  total_score_wrong  : total_score를 축 가중합과 불일치하게 변조

입력: data/scheduler_v9.parquet (prompt=입력JSON, chosen=출력JSON)
출력: data/dpo_pairs_v9.parquet + data/hf_versioned/dpo/v9.parquet

사용:
  uv run python scripts/v9/gen_v9_dpo.py
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from timesorter.data import schema_v9 as S  # noqa: E402


def _dump(resp: S.ScheduleResponseV9) -> str:
    return S.format_for_sft_v9(resp)


def neg_schedule_overlap(resp, inp, rng):
    r = copy.deepcopy(resp)
    if len(r.scheduled_tasks) < 2:
        return None
    by_rank = sorted(r.scheduled_tasks, key=lambda s: s.priority_rank)
    a, b = by_rank[0], by_rank[1]
    # b의 시작을 a의 블록 중간으로 당겨 겹치게
    sa = S.parse_dt(a.recommended_schedule.start_time)
    ea = S.parse_dt(a.recommended_schedule.end_time)
    mid = sa + (ea - sa) / 2
    dur = S.parse_dt(b.recommended_schedule.end_time) - S.parse_dt(b.recommended_schedule.start_time)
    b.recommended_schedule.start_time = S.iso(mid)
    b.recommended_schedule.end_time = S.iso(mid + dur)
    return r


def neg_deadline_miss(resp, inp, rng):
    r = copy.deepcopy(resp)
    dl_map = {t.task_id: t.deadline for t in inp.tasks}
    cands = [s for s in r.scheduled_tasks if dl_map.get(s.task_id)]
    if not cands:
        return None
    s = rng.choice(cands)
    dl = S.parse_dt(dl_map[s.task_id])
    dur = S.parse_dt(s.recommended_schedule.end_time) - S.parse_dt(s.recommended_schedule.start_time)
    new_start = dl + dt.timedelta(minutes=30)  # 마감 30분 후 시작 → 마감 초과
    s.recommended_schedule.start_time = S.iso(new_start)
    s.recommended_schedule.end_time = S.iso(new_start + dur)
    return r


def neg_rank_score_mismatch(resp, inp, rng):
    r = copy.deepcopy(resp)
    if len(r.scheduled_tasks) < 2:
        return None
    # total_score 최고 태스크와 최저 태스크의 rank를 교환
    hi = max(r.scheduled_tasks, key=lambda s: s.scoring.total_score)
    lo = min(r.scheduled_tasks, key=lambda s: s.scoring.total_score)
    if hi.task_id == lo.task_id or abs(hi.scoring.total_score - lo.scoring.total_score) < 1.0:
        return None
    hi.priority_rank, lo.priority_rank = lo.priority_rank, hi.priority_rank
    return r


def neg_chain_order_break(resp, inp, rng, chain_pairs):
    if not chain_pairs:
        return None
    r = copy.deepcopy(resp)
    a, b = rng.choice(chain_pairs)  # 선행 a, 후행 b
    sm = {s.task_id: s for s in r.scheduled_tasks}
    if a not in sm or b not in sm:
        return None
    sm[a].priority_rank, sm[b].priority_rank = sm[b].priority_rank, sm[a].priority_rank
    return r


def neg_total_score_wrong(resp, inp, rng):
    r = copy.deepcopy(resp)
    s = rng.choice(r.scheduled_tasks)
    delta = rng.choice([-3.0, -2.5, 2.5, 3.0])
    s.scoring.total_score = max(0.0, min(10.0, round(s.scoring.total_score + delta, 2)))
    # 변조가 실제로 불일치하는지 확인(원래와 같으면 skip)
    exp = S.compute_total_score(s.scoring.deadline_proximity, s.scoring.task_importance,
                                s.scoring.task_chaining, s.scoring.urgency)
    if abs(s.scoring.total_score - exp) <= S.TOTAL_TOL:
        return None
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="data/scheduler_v9.parquet")
    ap.add_argument("--out", default="data/dpo_pairs_v9.parquet")
    ap.add_argument("--out-hf", default="data/hf_versioned/dpo/v9.parquet")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_parquet(args.sft)
    rng = random.Random(args.seed)
    pairs = []
    for _, row in df.iterrows():
        inp = S.ScheduleInput.model_validate_json(_to_json(row["prompt"]))
        resp = S.parse_lenient_v9(row["chosen"])
        if resp is None:
            continue
        meta = json.loads(row["meta"])
        chain_pairs = [tuple(p) for p in meta.get("chain_pairs", [])]
        chosen_str = row["chosen"]
        makers = [
            ("schedule_overlap", lambda: neg_schedule_overlap(resp, inp, rng)),
            ("deadline_miss", lambda: neg_deadline_miss(resp, inp, rng)),
            ("rank_score_mismatch", lambda: neg_rank_score_mismatch(resp, inp, rng)),
            ("total_score_wrong", lambda: neg_total_score_wrong(resp, inp, rng)),
        ]
        if chain_pairs:
            makers.append(("chain_order_break",
                           lambda: neg_chain_order_break(resp, inp, rng, chain_pairs)))
        for cat, fn in makers:
            rej = fn()
            if rej is None:
                continue
            rej_str = _dump(rej)
            if rej_str == chosen_str:
                continue
            pairs.append({
                "prompt": row["prompt"], "chosen": chosen_str, "rejected": rej_str,
                "category": cat, "persona": row["persona"], "today": row["today"],
                "source": f"v9_dpo_{cat}", "version": "v9",
            })

    out = pd.DataFrame(pairs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_hf).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    out.to_parquet(args.out_hf, index=False)
    import collections
    print(f"[saved] {args.out} · {args.out_hf} — {len(out)}쌍")
    print("category 분포:", dict(collections.Counter(p["category"] for p in pairs)))


def _to_json(prompt: str) -> str:
    # prompt는 pretty JSON 문자열 — 그대로 파싱 가능
    return prompt


if __name__ == "__main__":
    main()
