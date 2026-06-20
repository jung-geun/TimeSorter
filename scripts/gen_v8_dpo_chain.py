#!/usr/bin/env python
"""v8 체인 특화 DPO 쌍 프로그래매틱 생성.

체인 SFT 데이터(meta 컬럼 보유)를 입력받아 체인 위반 hard negative를 생성한다.
API 호출 없음($0). gen_preference_pairs_v3.py의 dependency_scatter를 포함하고
체인 전용 위반 2종을 추가한다:

  dependency_scatter     체인 최후 단계 맨앞, 최초 단계 맨뒤, 전원 dep=1 (기존)
  chain_internal_swap    체인 내 인접 두 단계의 순서만 뒤바꿈 (위치 연속성은 유지)
  chain_break_mid        중간 체인 단계 하나를 연속 블록 밖으로 이탈시킴

사용:
  uv run python scripts/gen_v8_dpo_chain.py \
      --sft data/scheduler_v8_chain.parquet \
      --out data/dpo_pairs_v8_chain.parquet

  # 기존 v7 체인 SFT로 먼저 테스트
  uv run python scripts/gen_v8_dpo_chain.py \
      --sft data/scheduler_v7_chain.parquet \
      --out data/dpo_pairs_v7_chain.parquet
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from timesorter.data.schema import ScheduleResponse, format_for_sft, parse_lenient


def _move_to_front(order: list[int], tid: int) -> list[int]:
    return [tid] + [t for t in order if t != tid]


def _move_to_back(order: list[int], tid: int) -> list[int]:
    return [t for t in order if t != tid] + [tid]


def _swap_positions(order: list[int], a: int, b: int) -> list[int]:
    out = list(order)
    ia, ib = out.index(a), out.index(b)
    out[ia], out[ib] = out[ib], out[ia]
    return out


def _make_chain_negatives(
    resp: ScheduleResponse,
    specs: list[dict],
    rng: random.Random,
) -> list[tuple[str, ScheduleResponse]]:
    """체인 위반 hard negative 목록 생성. chosen이 올바를 때만 유효."""
    out: list[tuple[str, ScheduleResponse]] = []

    # 체인 그룹별로 단계 순서대로 정렬
    groups: dict[int, list[dict]] = {}
    for s in specs:
        g = s.get("chain_group", 0)
        if g:
            groups.setdefault(g, []).append(s)
    for g in groups:
        groups[g].sort(key=lambda s: s["chain_pos"])

    if not groups:
        return out

    chain_ids_all = {s["idx"] for g in groups.values() for s in g}
    score_by_id = {s.task_id: s for s in resp.scores}

    # ── 1. dependency_scatter (기존 로직) ─────────────────────────────────────
    for g, members in groups.items():
        if len(members) < 2:
            continue
        rej = resp.model_copy(deep=True)
        # 마지막 단계 맨앞, 첫 단계 맨뒤
        rej.priority_order = _move_to_front(rej.priority_order, members[-1]["idx"])
        rej.priority_order = _move_to_back(rej.priority_order, members[0]["idx"])
        for s in rej.scores:
            if s.task_id in chain_ids_all:
                s.dependency = 1
        out.append(("dependency_scatter", rej))
        break  # 첫 번째 체인에만 적용 (중복 억제)

    # ── 2. chain_internal_swap — 인접 두 단계 위치만 교환 ─────────────────────
    # 선택 기준: 가장 긴 체인의 중간 인접 쌍 (step k, step k+1)
    target_chain = max(groups.values(), key=len)
    if len(target_chain) >= 3:
        # 첫 쌍을 제외하고 가운데 인접 쌍 선택 (첫 쌍 swap은 scatter와 겹침)
        k = rng.randint(1, len(target_chain) - 2)  # 1 ≤ k < len-1
        a_id = target_chain[k]["idx"]
        b_id = target_chain[k + 1]["idx"]
        pos = {tid: i for i, tid in enumerate(resp.priority_order)}
        # chosen에서 이미 올바른 순서(a 앞, b 뒤)여야 swap이 유효한 negative
        if pos.get(a_id, 99) < pos.get(b_id, 99):
            rej2 = resp.model_copy(deep=True)
            rej2.priority_order = _swap_positions(rej2.priority_order, a_id, b_id)
            # dep 점수는 유지 (순서만 틀린 케이스)
            out.append(("chain_internal_swap", rej2))

    # ── 3. chain_break_mid — 중간 단계 하나를 맨뒤로 이탈 ────────────────────
    if len(target_chain) >= 4:
        # 체인 중간(first/last 제외) 단계 하나를 통째로 맨뒤로
        mid_candidates = target_chain[1:-1]  # 중간 단계들
        mid = rng.choice(mid_candidates)
        rej3 = resp.model_copy(deep=True)
        rej3.priority_order = _move_to_back(rej3.priority_order, mid["idx"])
        for s in rej3.scores:
            if s.task_id == mid["idx"]:
                s.dependency = 1
        out.append(("chain_break_mid", rej3))

    return out


def process_row(row: pd.Series, rng: random.Random) -> list[dict]:
    meta_raw = row.get("meta")
    if not meta_raw:
        return []
    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
    specs = meta.get("specs", [])

    # 체인 없으면 건너뜀
    if not any(s.get("chain_group", 0) for s in specs):
        return []

    chosen_str = row.get("chosen", "")
    resp = parse_lenient(chosen_str)
    if resp is None or not resp.tasks:
        return []

    pairs = []
    for category, rej in _make_chain_negatives(resp, specs, rng):
        pairs.append({
            "prompt": row["prompt"],
            "chosen": chosen_str,
            "rejected": format_for_sft(rej),
            "reject_category": category,
            "persona": row.get("persona", ""),
            "today": row.get("today", ""),
            "source": f"v8_dpo_chain_{category}",
        })
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True, help="체인 SFT parquet (meta 컬럼 필요)")
    ap.add_argument("--out", required=True, help="DPO 쌍 출력 parquet")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    df = pd.read_parquet(args.sft)
    print(f"[input] {len(df)}행")

    all_pairs: list[dict] = []
    skipped = 0
    for _, row in df.iterrows():
        pairs = process_row(row, rng)
        if pairs:
            all_pairs.extend(pairs)
        else:
            skipped += 1

    if not all_pairs:
        print("[경고] 생성된 쌍이 없습니다. meta 컬럼이 있는 체인 데이터를 입력하세요.")
        return

    out_df = pd.DataFrame(all_pairs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)

    cat_counts = out_df["reject_category"].value_counts().to_dict()
    print(f"[완료] {len(out_df)}쌍 → {args.out}")
    print(f"  카테고리: {cat_counts}")
    print(f"  건너뜀(meta 없음/체인 없음): {skipped}행")


if __name__ == "__main__":
    main()
