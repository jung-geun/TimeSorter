#!/usr/bin/env python
"""v9 결정적 스캐폴딩 — skeleton → 입력 슬롯(facts) + 점수/순위/시간블록.

LLM 비의존(결정적) 부분만 생성한다. 이후 Workflow가:
  haiku → 슬롯 facts에 맞는 title/memo/source 작성
  sonnet → 점수·스케줄 근거로 reasoning(summary, chaining_detail) 작성
  opus → 전체 검수

출력: outputs/v9/sample_scaffold.jsonl (행별 scaffold 레코드)

사용:
  uv run python scripts/v9/build_sample.py --n 12 --seed 7
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from gen_schedule_v3 import build_skeleton  # noqa: E402
from timesorter.data import schema_v9 as S  # noqa: E402

# 시나리오 믹스 (지난 일정 past 는 v9에서 입력 제외 — 미래 배치 일정만)
_SCENARIOS = ["dependency_chain_complex", "dated_mixed", "intraday", "risk",
              "relative", "dependency_chain"]

_DUR_BY_KIND = {"chain": [45, 60, 90], "today": [30, 60, 90], "future": [60, 90, 120],
                "none": [30, 45, 60], "risk": [60, 90]}


def _to_iso(skel_deadline: str | None, rng: random.Random) -> str | None:
    """'YYYY-MM-DD HH:MM' → ISO8601 +09:00. None이면 None."""
    if not skel_deadline:
        return None
    d = dt.datetime.strptime(skel_deadline, "%Y-%m-%d %H:%M").replace(tzinfo=S.KST)
    return d.isoformat()


def _duration(kind: str, rng: random.Random) -> int:
    return rng.choice(_DUR_BY_KIND.get(kind, [60]))


def build_row(persona: dict, scenario: str, today: dt.date, rng: random.Random) -> dict:
    skel = build_skeleton(scenario, today, rng)
    # 지난 일정 제외 → 재인덱싱
    specs = [s for s in skel.specs if not s.is_past]
    if len(specs) < 2:
        specs = skel.specs
    for i, s in enumerate(specs, 1):
        s.idx = i

    current_time = dt.datetime.combine(today, dt.time(8, 0), tzinfo=S.KST)
    now = current_time
    avail = persona["persona"]["availability"]

    # 체인 쌍 (선행 task_id, 후행 task_id) + 멤버 순서
    chains: dict[int, list] = {}
    for s in specs:
        if s.chain_group:
            chains.setdefault(s.chain_group, []).append(s)
    chain_pairs: list[tuple[str, str]] = []
    chain_final_idx: set[int] = set()
    for members in chains.values():
        members.sort(key=lambda s: s.chain_pos)
        for a, b in zip(members, members[1:]):
            chain_pairs.append((f"task_{a.idx:03d}", f"task_{b.idx:03d}"))
        chain_final_idx.add(members[-1].idx)

    # 1차: 마감(ISO)·소요시간 확정
    iso_dl_map, dur_map = {}, {}
    for s in specs:
        tid = f"task_{s.idx:03d}"
        iso_dl_map[tid] = _to_iso(s.deadline, rng)
        dur_map[tid] = _duration(s.kind, rng)

    # 체인별 effective deadline(최종 단계 마감) + 단계별 잔여작업량 누계
    eff_deadline = {}  # tid -> iso or None
    remaining_work = {}  # tid -> minutes (이 단계+이후 단계 소요 합 — 블로커 긴급도)
    for members in chains.values():
        ids = [f"task_{m.idx:03d}" for m in members]
        fin_dl = iso_dl_map[ids[-1]]
        suffix = 0
        for tid in reversed(ids):
            suffix += dur_map[tid]
            remaining_work[tid] = suffix
            eff_deadline[tid] = fin_dl

    # 2차: 슬롯 facts + 결정적 점수
    slots, scoring = [], {}
    durations = dur_map
    for s in specs:
        tid = f"task_{s.idx:03d}"
        in_chain = bool(s.chain_group)
        is_final = s.idx in chain_final_idx
        # 체인 멤버는 최종 마감을 공유, 긴급도는 잔여작업량 기준(블로커=일찍 시작해야 함)
        eff_iso = eff_deadline.get(tid, iso_dl_map[tid])
        eff_dl = S.parse_dt(eff_iso) if eff_iso else None
        slack_dur = remaining_work.get(tid, dur_map[tid])
        dp = S.deadline_proximity_score(now, eff_dl)
        ur = S.urgency_score(now, eff_dl, slack_dur, s.risk)
        imp = S.importance_score(s.kind, s.risk, in_chain)
        ch = S.chaining_score(in_chain, is_final)
        total = S.compute_total_score(dp, imp, ch, ur)
        scoring[tid] = {"deadline_proximity": dp, "task_importance": imp,
                        "task_chaining": ch, "urgency": ur, "total_score": total}
        slots.append({
            "task_id": tid, "kind": s.kind, "deadline": iso_dl_map[tid],
            "estimated_duration_minutes": dur_map[tid],
            "in_chain": in_chain, "chain_group": s.chain_group, "chain_pos": s.chain_pos,
            "is_final_chain": is_final, "risk": s.risk, "risk_clause": s.risk_clause,
            "rel_expr": s.rel_expr,
        })

    # ── priority_rank: 체인=블록 단위, 블록점수=멤버 최대 total, 내림차순 ──
    blocks: list[tuple[float, list[str]]] = []
    used = set()
    for members in chains.values():
        ids = [f"task_{m.idx:03d}" for m in members]
        bscore = max(scoring[i]["total_score"] for i in ids)
        blocks.append((bscore, ids))
        used.update(ids)
    for s in specs:
        tid = f"task_{s.idx:03d}"
        if tid not in used:
            blocks.append((scoring[tid]["total_score"], [tid]))
    blocks.sort(key=lambda b: -b[0])
    ranked_ids = [tid for _, ids in blocks for tid in ids]
    rank = {tid: i for i, tid in enumerate(ranked_ids, 1)}

    # ── 시간블록 ──
    sched = S.build_schedule(now, avail, [(tid, durations[tid]) for tid in ranked_ids])

    return {
        "persona_id": persona["persona_id"],
        "persona": persona["persona"],
        "scenario": scenario,
        "current_time": current_time.isoformat(),
        "slots": slots,
        "scoring": scoring,
        "priority_rank": rank,
        "schedule": {tid: {"start_time": S.iso(st), "end_time": S.iso(en)}
                     for tid, (st, en) in sched.items()},
        "chain_pairs": chain_pairs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--personas", default="data/v9/personas.json")
    ap.add_argument("--out", default="outputs/v9/sample_scaffold.jsonl")
    args = ap.parse_args()

    personas = json.loads(Path(args.personas).read_text())
    rng = random.Random(args.seed)
    rows = []
    for i in range(args.n):
        persona = personas[i % len(personas)]
        scenario = _SCENARIOS[i % len(_SCENARIOS)]
        today = dt.date(2026, 1, 1) + dt.timedelta(days=rng.randint(20, 340))
        rows.append(build_row(persona, scenario, today, rng))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[saved] {out} — {len(rows)} scaffold rows")
    print("시나리오:", [r["scenario"] for r in rows])
    print("태스크 수:", [len(r["slots"]) for r in rows])


if __name__ == "__main__":
    main()
