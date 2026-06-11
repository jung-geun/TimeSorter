#!/usr/bin/env python
"""v3 DPO 쌍 생성 — 검증 실패 모드를 겨냥한 프로그래매틱 hard negative.

v2 DPO의 rejected(invalid_json·bad_scores 등)는 chosen과 너무 쉽게 구분되어
rewards/margins=17.7로 과분리됐다. v3는 chosen과 형식·길이가 같고 내용만 틀린
rejected를 만들어 실제 실패 모드를 직접 벌점한다.

카테고리 (시나리오 골격 meta 기반, API 호출 없음 · $0):
  date_confusion       지난 일정을 1위 + urgency/time_constraint=5 ("오늘 일정" 착각)
  granularity_swap     같은 날 오전/이른 마감과 늦은 마감의 순위를 뒤바꿈
  dependency_scatter   체인 단계를 역순·분산 배치 + dependency=1
  risk_ignore          리스크 태스크의 importance=2로 강등 + 후순위
  order_score_mismatch 점수는 그대로, priority_order만 점수와 모순되게 뒤집음

입력: data/scheduler_v3.parquet (meta 컬럼 필요)
출력: data/dpo_pairs_v3.parquet (+ --replay-v2 N: v2 쌍 N개 혼합, today 랜덤 부여)

사용:
  uv run python scripts/gen_preference_pairs_v3.py --verify
  uv run python scripts/gen_preference_pairs_v3.py --replay-v2 3000
"""
from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from timesorter.data.schema import ScheduleResponse, parse_lenient

_V3_PATH = "data/scheduler_v3.parquet"
_V2_PAIRS_PATH = "data/dpo_pairs_v2.parquet"
_OUT_PATH = "data/dpo_pairs_v3.parquet"

_CONFUSED_REASONS = [
    "오늘 일정이므로 가장 먼저 처리해야 합니다.",
    "마감이 오늘이라 즉시 착수해야 합니다.",
    "시간이 정해진 오늘 일정이라 최우선입니다.",
]


def _move_to_front(order: list[int], tid: int) -> list[int]:
    out = [t for t in order if t != tid]
    return [tid] + out


def _move_to_back(order: list[int], tid: int) -> list[int]:
    out = [t for t in order if t != tid]
    return out + [tid]


def _swap(order: list[int], a: int, b: int) -> list[int]:
    out = list(order)
    ia, ib = out.index(a), out.index(b)
    out[ia], out[ib] = out[ib], out[ia]
    return out


def make_rejected(
    resp: ScheduleResponse,
    meta: dict,
    rng: random.Random,
) -> list[tuple[str, ScheduleResponse]]:
    """적용 가능한 카테고리별 (category, rejected) 목록 생성."""
    specs = meta["specs"]
    out: list[tuple[str, ScheduleResponse]] = []
    score_by_id = {s.task_id: s for s in resp.scores}

    past_ids = [s["idx"] for s in specs if s.get("is_past")]
    if past_ids:
        tid = rng.choice(past_ids)
        rej = resp.model_copy(deep=True)
        rej.priority_order = _move_to_front(resp.priority_order, tid)
        for s in rej.scores:
            if s.task_id == tid:
                s.urgency, s.time_constraint = 5, 5
                s.reason = rng.choice(_CONFUSED_REASONS)
        out.append(("date_confusion", rej))

    today = meta["today"]
    today_specs = sorted(
        (s for s in specs if s.get("deadline") and s["deadline"].startswith(today)
         and not s.get("is_past")),
        key=lambda s: s["deadline"],
    )
    if len(today_specs) >= 2:
        early, late = today_specs[0]["idx"], today_specs[-1]["idx"]
        pos = {t: i for i, t in enumerate(resp.priority_order)}
        if pos[early] < pos[late]:  # chosen이 올바른 경우에만 뒤집기가 유효한 negative
            rej = resp.model_copy(deep=True)
            rej.priority_order = _swap(resp.priority_order, early, late)
            s_early, s_late = score_by_id.get(early), score_by_id.get(late)
            if s_early and s_late:
                for s in rej.scores:
                    if s.task_id in (early, late):
                        s.urgency = min(s_early.urgency, s_late.urgency)
            out.append(("granularity_swap", rej))

    chain = sorted((s for s in specs if s.get("chain_group")), key=lambda s: s["chain_pos"])
    if len(chain) >= 2:
        rej = resp.model_copy(deep=True)
        order = list(resp.priority_order)
        # 마지막 단계를 맨 앞, 첫 단계를 맨 뒤로 — 체인 역전+분산
        order = _move_to_front(order, chain[-1]["idx"])
        order = _move_to_back(order, chain[0]["idx"])
        rej.priority_order = order
        chain_ids = {s["idx"] for s in chain}
        for s in rej.scores:
            if s.task_id in chain_ids:
                s.dependency = 1
        out.append(("dependency_scatter", rej))

    risk_ids = [s["idx"] for s in specs if s.get("risk")]
    if risk_ids:
        tid = rng.choice(risk_ids)
        rej = resp.model_copy(deep=True)
        rej.priority_order = _move_to_back(resp.priority_order, tid)
        for s in rej.scores:
            if s.task_id == tid:
                s.importance = 2
                s.reason = "불이익 조항은 형식적인 문구라 중요도가 낮습니다."
        out.append(("risk_ignore", rej))

    if len(resp.priority_order) >= 3:
        rej = resp.model_copy(deep=True)
        rej.priority_order = list(reversed(resp.priority_order))
        out.append(("order_score_mismatch", rej))

    # no_today: 오늘 미상인데 임의 태스크를 '이미 지남'으로 단정하는 날짜 환각 negative
    if meta.get("today") == "":
        dated_ids = [s["idx"] for s in specs if s.get("deadline")]
        if dated_ids:
            tid = rng.choice(dated_ids)
            rej = resp.model_copy(deep=True)
            rej.priority_order = _move_to_back(resp.priority_order, tid)
            for s in rej.scores:
                if s.task_id == tid:
                    s.urgency, s.time_constraint = 1, 1
                    s.reason = "이미 지난 일정이므로 우선순위 최하위에 배치합니다."
            out.append(("past_hallucination", rej))

    return out


def build_pairs(df: pd.DataFrame, per_row: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pairs: list[dict] = []
    skipped = 0
    for _, row in df.iterrows():
        resp = parse_lenient(str(row["chosen"]))
        if resp is None or not resp.tasks:
            skipped += 1
            continue
        meta = json.loads(row["meta"])
        candidates = make_rejected(resp, meta, rng)
        rng.shuffle(candidates)
        for category, rej in candidates[:per_row]:
            pairs.append({
                "prompt": str(row["prompt"]),
                "chosen": str(row["chosen"]),
                "rejected": rej.model_dump_json(),
                "persona": str(row["persona"]),
                "today": str(row["today"]),
                "category": category,
                "source": f"v3_{category}",
            })
    if skipped:
        print(f"[경고] 파싱 불가로 건너뜀: {skipped}행")
    return pairs


def load_v2_replay(n: int, seed: int) -> list[dict]:
    """v2 쌍 replay — 거부·형식 선호 유지용. today는 랜덤 부여."""
    p = Path(_V2_PAIRS_PATH)
    if not p.exists() or n <= 0:
        return []
    df = pd.read_parquet(p)
    # 거부 케이스 우선 절반, 나머지 랜덤
    refusal = df[df["source"].astype(str).str.startswith("refusal", na=False)]
    rest = df.drop(refusal.index)
    n_ref = min(len(refusal), n // 2)
    sampled = pd.concat([
        refusal.sample(n_ref, random_state=seed),
        rest.sample(min(len(rest), n - n_ref), random_state=seed),
    ])
    rng = random.Random(seed)
    rows = []
    for _, r in sampled.iterrows():
        today = datetime.date(2026, 1, 1) + datetime.timedelta(days=rng.randint(0, 360))
        rows.append({
            "prompt": str(r["prompt"]),
            "chosen": str(r["chosen"]),
            "rejected": str(r["rejected"]),
            "persona": str(r.get("persona", "직장인")),
            "today": today.isoformat(),
            "category": str(r.get("category", "v2_replay")),
            "source": "v2_replay",
        })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v3 DPO hard negative 쌍 생성")
    parser.add_argument("--input", default=_V3_PATH)
    parser.add_argument("--out", default=_OUT_PATH)
    parser.add_argument("--per-row", type=int, default=2, help="행당 최대 negative 수")
    parser.add_argument("--replay-v2", type=int, default=3000, help="v2 쌍 혼합 수 (0=미사용)")
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"[입력] {args.input}: {len(df)}행")

    pairs = build_pairs(df, args.per_row, args.seed)
    print(f"[hard negative] {len(pairs)}쌍 생성")

    replay = load_v2_replay(args.replay_v2, args.seed)
    if replay:
        print(f"[replay] v2 쌍 {len(replay)}개 혼합")
    all_pairs = pairs + replay

    if args.verify:
        bad = sum(
            1 for p in all_pairs
            if p["source"] != "v2_replay" and (
                parse_lenient(p["rejected"]) is None
                or p["chosen"] == p["rejected"]
            )
        )
        print(f"[검증] rejected 파싱·차별성: {len(pairs) - bad}/{len(pairs)} 통과")
        if bad:
            sys.exit(1)

    out_df = pd.DataFrame(all_pairs).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"[저장] {args.out} ({len(out_df)}쌍)")
    print(out_df["category"].value_counts().to_string())
