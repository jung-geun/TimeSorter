#!/usr/bin/env python
"""v9 EN-US 결정적 스캐폴드 생성 — KR build_dataset.py 로직 동일, 콘텐츠/직업/타임존만 미국판.

차이만: content_en(도메인/시나리오 풀)·occupations_en(직업군)·TZ(미국 동부) 사용.
스케줄/마감/점수/순위/체인 로직은 KR과 100% 동일(schema_v9 공유).

출력: outputs/v9/build_en/scaffold/batch_{NN}.json + manifest.json

사용:
  uv run python scripts/v9/build_dataset_en.py --n 300 --batch-size 16 --seed 20260620
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/v9")
from timesorter.data import schema_v9 as S  # noqa: E402
from occupations_en import categorize, CAT_CONFIG, OFFICE_CATS  # noqa: E402
from content_en import (  # noqa: E402
    DOMAINS, WORK_CHAINS, PERSONAL_CHAINS, RISK_CLAUSES, MICRO_TOPICS, CHAIN_SUBJECTS,
    WORK_DOMAIN, MISC_DOMAIN, RISK_DOMAINS, TZ,
)

# tier → (deadline tier, 버퍼 분 범위) — KR과 동일
TIER_TIGHT = "tight_today"
TIER_TODAY = "today"
TIER_SOON = "soon"
TIER_LATER = "later"
TIER_NONE = "none"

_TIER_BUFFER = {
    TIER_TIGHT: (10, 40), TIER_TODAY: (90, 240),
    TIER_SOON: (1440, 2880), TIER_LATER: (4320, 10080),
}
_TIER_RANKKEY = {TIER_TIGHT: 0, TIER_TODAY: 1, TIER_SOON: 2, TIER_LATER: 3, TIER_NONE: 4}
_DUR_CHOICES = [15, 30, 30, 45, 60, 60, 90, 120]


def _avail_start(avail: str) -> dt.time:
    return S._parse_window(avail)[0]


def gen_row(persona: dict, date: dt.date, rng: random.Random,
            dctr: list[int], mctr: dict[str, int]) -> dict:
    occs = persona["persona"]["occupations"]
    cat = persona.get("_meta", {}).get("category") or categorize(" ".join(occs))
    cfg = CAT_CONFIG.get(cat, CAT_CONFIG["general"])
    core_domains, work_ratio = cfg["domains"], cfg["work_ratio"]
    avail = persona["persona"]["availability"]
    start_t = _avail_start(avail)
    current_time = dt.datetime.combine(date, start_t, tzinfo=TZ)

    n_chains = rng.choices([0, 1, 1, 2], weights=[2, 4, 4, 2])[0]
    specs: list[dict] = []
    gid = 0
    used_chain_themes = []
    for _ in range(n_chains):
        gid += 1
        is_work = rng.random() < work_ratio
        pool = WORK_CHAINS if is_work else PERSONAL_CHAINS
        theme = rng.choice([t for t in pool if t not in used_chain_themes] or pool)
        used_chain_themes.append(theme)
        clen = rng.choice([3, 3, 4, 4, 5]) if n_chains == 1 else rng.choice([3, 3, 4])
        fin_tier = rng.choices([TIER_TIGHT, TIER_TODAY, TIER_SOON], weights=[2, 4, 3])[0]
        subject = rng.choice(CHAIN_SUBJECTS.get(theme, [theme]))
        for pos in range(1, clen + 1):
            specs.append({
                "kind": "chain", "chain_group": gid, "chain_pos": pos,
                "is_final_chain": pos == clen, "risk": False, "risk_clause": "",
                "tier": fin_tier if pos == clen else TIER_NONE,
                "chain_theme": theme, "chain_subject": subject,
                "domain_hint": WORK_DOMAIN if is_work else MISC_DOMAIN, "micro_topic": "",
            })
    n_indep = rng.randint(2, 5) if n_chains else rng.randint(4, 7)
    for _ in range(n_indep):
        if rng.random() < 0.55:
            dom = core_domains[dctr[0] % len(core_domains)]
        else:
            dom = DOMAINS[dctr[0] % len(DOMAINS)]
        dctr[0] += 1
        pool = MICRO_TOPICS.get(dom, MICRO_TOPICS[MISC_DOMAIN])
        mi = mctr.get(dom, rng.randrange(len(pool)))
        micro = pool[mi % len(pool)]
        mctr[dom] = mi + 1
        tier = rng.choices([TIER_TIGHT, TIER_TODAY, TIER_SOON, TIER_LATER, TIER_NONE],
                           weights=[2, 4, 3, 2, 3])[0]
        risk = (rng.random() < 0.18 and tier in (TIER_TIGHT, TIER_TODAY, TIER_SOON)
                and cat in OFFICE_CATS and dom in RISK_DOMAINS)
        specs.append({
            "kind": "today" if tier in (TIER_TIGHT, TIER_TODAY) else (
                "future" if tier in (TIER_SOON, TIER_LATER) else "none"),
            "chain_group": 0, "chain_pos": 0, "is_final_chain": False,
            "risk": risk, "risk_clause": rng.choice(RISK_CLAUSES) if risk else "",
            "tier": tier, "chain_theme": "", "chain_subject": "",
            "domain_hint": dom, "micro_topic": micro,
        })

    cb: dict[int, list] = {}
    indep = []
    for s in specs:
        if s["chain_group"]:
            cb.setdefault(s["chain_group"], []).append(s)
        else:
            indep.append(s)
    blocks_list = [sorted(v, key=lambda x: x["chain_pos"]) for v in cb.values()] + [[s] for s in indep]
    rng.shuffle(blocks_list)
    specs = [s for blk in blocks_list for s in blk]
    for i, s in enumerate(specs, 1):
        s["idx"] = i
        s["task_id"] = f"task_{i:03d}"
        s["estimated_duration_minutes"] = rng.choice(_DUR_CHOICES)

    chains: dict[int, list] = {}
    for s in specs:
        if s["chain_group"]:
            chains.setdefault(s["chain_group"], []).append(s)
    for members in chains.values():
        members.sort(key=lambda s: s["chain_pos"])
    chain_pairs = []
    for members in chains.values():
        for a, b in zip(members, members[1:]):
            chain_pairs.append((a["task_id"], b["task_id"]))

    def block_key(spec_or_members):
        if isinstance(spec_or_members, list):
            tier = min(_TIER_RANKKEY[m["tier"]] for m in spec_or_members)
            risk = any(m["risk"] for m in spec_or_members)
        else:
            tier = _TIER_RANKKEY[spec_or_members["tier"]]
            risk = spec_or_members["risk"]
        return (tier, 0 if risk else 1)

    blocks = []
    used = set()
    for members in chains.values():
        blocks.append((block_key(members), [m["task_id"] for m in members]))
        used.update(m["task_id"] for m in members)
    for s in specs:
        if s["task_id"] not in used:
            blocks.append((block_key(s), [s["task_id"]]))
    blocks.sort(key=lambda b: b[0])
    ranked_ids = [tid for _, ids in blocks for tid in ids]
    rank = {tid: i for i, tid in enumerate(ranked_ids, 1)}

    dur_map = {s["task_id"]: s["estimated_duration_minutes"] for s in specs}
    sched = S.build_schedule(current_time, avail,
                             [(tid, dur_map[tid]) for tid in ranked_ids])

    iso_dl = {}
    for s in specs:
        tid = s["task_id"]
        tier = s["tier"]
        if tier == TIER_NONE:
            iso_dl[tid] = None
            continue
        _, end = sched[tid]
        lo, hi = _TIER_BUFFER[tier]
        dl = end + dt.timedelta(minutes=rng.randint(lo, hi))
        iso_dl[tid] = dl.isoformat()

    eff_dl, remaining = {}, {}
    for members in chains.values():
        ids = [m["task_id"] for m in members]
        fin = iso_dl[ids[-1]]
        suf = 0
        for tid in reversed(ids):
            suf += dur_map[tid]
            remaining[tid] = suf
            eff_dl[tid] = fin

    scoring, facts = {}, {}
    for s in specs:
        tid = s["task_id"]
        in_chain = bool(s["chain_group"])
        eff = eff_dl.get(tid, iso_dl[tid])
        eff_d = S.parse_dt(eff) if eff else None
        slack_dur = remaining.get(tid, dur_map[tid])
        dp = S.deadline_proximity_score(current_time, eff_d)
        ur = S.urgency_score(current_time, eff_d, slack_dur, s["risk"])
        imp = S.importance_score(s["kind"], s["risk"], in_chain)
        ch = S.chaining_score(in_chain, s["is_final_chain"])
        total = S.compute_total_score(dp, imp, ch, ur)
        scoring[tid] = {"deadline_proximity": dp, "task_importance": imp,
                        "task_chaining": ch, "urgency": ur, "total_score": total}
        st, en = sched[tid]
        own_dl = iso_dl[tid]
        h2d = round((S.parse_dt(own_dl) - current_time).total_seconds() / 3600, 1) if own_dl else None
        facts[tid] = {
            "rank": rank[tid], "deadline": own_dl,
            "hours_to_deadline": h2d,
            "scheduled": f"{st.strftime('%H:%M')}~{en.strftime('%H:%M')}",
            "deadline_met": True,
            "in_chain": in_chain, "chain_theme": s["chain_theme"],
            "domain": s["domain_hint"], "risk_clause": s["risk_clause"],
        }

    slots = [{
        "task_id": s["task_id"], "kind": s["kind"], "deadline": iso_dl[s["task_id"]],
        "estimated_duration_minutes": dur_map[s["task_id"]],
        "in_chain": bool(s["chain_group"]), "chain_group": s["chain_group"],
        "chain_pos": s["chain_pos"], "is_final_chain": s["is_final_chain"],
        "risk": s["risk"], "risk_clause": s["risk_clause"],
        "chain_theme": s["chain_theme"], "chain_subject": s.get("chain_subject", ""),
        "domain_hint": s["domain_hint"], "micro_topic": s.get("micro_topic", ""),
    } for s in specs]

    return {
        "persona_id": persona["persona_id"], "persona": {k: v for k, v in persona["persona"].items()},
        "occ_category": cat, "current_time": current_time.isoformat(),
        "slots": slots, "scoring": scoring, "priority_rank": rank,
        "schedule": {tid: {"start_time": S.iso(st), "end_time": S.iso(en)}
                     for tid, (st, en) in sched.items()},
        "chain_pairs": chain_pairs, "facts": facts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260620)
    ap.add_argument("--personas", default="data/v9/personas_en.json")
    ap.add_argument("--out-dir", default="outputs/v9/build_en")
    args = ap.parse_args()

    personas = json.loads(Path(args.personas).read_text())
    rng = random.Random(args.seed)
    dctr = [rng.randint(0, 11)]
    mctr: dict[str, int] = {}
    rows = []
    for i in range(args.n):
        persona = personas[rng.randrange(len(personas))]
        date = dt.date(2026, 1, 1) + dt.timedelta(days=rng.randint(0, 364))
        rows.append(gen_row(persona, date, rng, dctr, mctr))

    out = Path(args.out_dir)
    (out / "scaffold").mkdir(parents=True, exist_ok=True)
    bs = args.batch_size
    n_batch = (len(rows) + bs - 1) // bs
    for b in range(n_batch):
        chunk = rows[b * bs:(b + 1) * bs]
        for j, r in enumerate(chunk):
            r["row_id"] = b * bs + j
        (out / "scaffold" / f"batch_{b:03d}.json").write_text(
            json.dumps(chunk, ensure_ascii=False))
    (out / "manifest.json").write_text(json.dumps(
        {"n": len(rows), "n_batch": n_batch, "batch_size": bs, "seed": args.seed},
        ensure_ascii=False, indent=2))

    from collections import Counter
    cats = Counter(r["occ_category"] for r in rows)
    nt = Counter(len(r["slots"]) for r in rows)
    nch = Counter(len(set(s["chain_group"] for s in r["slots"] if s["chain_group"])) for r in rows)
    themes = Counter(s["chain_theme"] for r in rows for s in r["slots"] if s["chain_theme"])
    print(f"[saved] {out}/scaffold/ — {len(rows)}행 / {n_batch}배치")
    print("occ_category:", dict(cats))
    print("태스크수 분포:", dict(sorted(nt.items())))
    print("체인수 분포:", dict(sorted(nch.items())))
    print("체인테마 종수:", len(themes), "| 상위:", themes.most_common(5))


if __name__ == "__main__":
    main()
