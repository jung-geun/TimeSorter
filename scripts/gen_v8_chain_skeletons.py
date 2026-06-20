#!/usr/bin/env python
"""v8 체인 강화용 골격 생성 + 배치 JSONL 분할.

의존성 체인 통과율(47%) 개선을 위한 추가 데이터 생성의 첫 단계.
dependency_chain_complex 골격만 생성해 배치 파일로 분할한다.

복잡도 기준으로 정렬 후 모델 배정:
  - single chain 4-step (복잡도 4)  → haiku 배치
  - 그 외 (5-step, dual chain)       → sonnet 배치

사용:
  uv run python scripts/gen_v8_chain_skeletons.py --total 900 --seed 2026
  # → outputs/v8_chain/skeletons/batch_{i:02d}.jsonl (50행씩)
  # → outputs/v8_chain/manifest.json
"""
from __future__ import annotations

import datetime
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from gen_schedule_v3 import (
    Skeleton,
    TaskSpec,
    _WEEKDAYS_KO,
    _spec_to_fact,
    _spec_to_gen_line,
    build_skeleton,
)
from gen_schedule_v2 import _NEMOTRON_PATH

_BATCH_SIZE = 50
_OUT_ROOT = Path("outputs/v8_chain")
_SCENARIO = "dependency_chain_complex"


def _skeleton_complexity(skel: Skeleton) -> int:
    """복잡도 = 총 체인 단계 수 × 체인 개수 (높을수록 복잡)."""
    n_chains = len({s.chain_group for s in skel.specs if s.chain_group > 0})
    n_steps = sum(1 for s in skel.specs if s.chain_group > 0)
    return n_chains * 10 + n_steps  # dual chain이면 +10으로 명확히 분리


def _assign_model(complexity: int) -> str:
    """단일 4단계(복잡도 14=1×10+4)까지는 haiku, 그 외 sonnet."""
    return "haiku" if complexity <= 14 else "sonnet"


def main(total: int = 900, seed: int = 2026, batch_size: int = _BATCH_SIZE) -> None:
    rng = random.Random(seed)

    # 페르소나 로드
    nem_df = pd.read_parquet(_NEMOTRON_PATH, columns=["persona", "age", "occupation"])
    nem_df = nem_df.sample(min(total * 3, len(nem_df)), random_state=seed).reset_index(drop=True)

    # 골격 생성
    rows: list[dict] = []
    today_base = datetime.date(2026, 7, 1)
    i = 0
    attempts = 0
    while len(rows) < total and attempts < total * 4:
        attempts += 1
        p = nem_df.iloc[i % len(nem_df)]
        persona_raw = str(p["persona"])
        persona_name = persona_raw.split(" 씨는")[0].strip() if " 씨는" in persona_raw else f"사용자{i}"
        persona_label = f"{persona_name} ({p.get('occupation', '직장인')}, {int(p.get('age', 30))}세)"

        today = today_base + datetime.timedelta(days=rng.randint(0, 180))
        skel = build_skeleton(_SCENARIO, today, rng)

        complexity = _skeleton_complexity(skel)
        model = _assign_model(complexity)

        today_h = f"{skel.today} ({_WEEKDAYS_KO[today.weekday()]})"
        gen_lines = [_spec_to_gen_line(s, skel.today) for s in skel.specs]
        facts = "\n".join(_spec_to_fact(s, skel.today) for s in skel.specs)

        row_id = f"v8chain_{seed}_{len(rows):04d}"
        rows.append({
            "id": row_id,
            "today": skel.today,
            "today_display": today_h,
            "persona_name": persona_name,
            "persona_label": persona_label,
            "gen_lines": gen_lines,
            "facts": facts,
            "meta": json.dumps(asdict(skel), ensure_ascii=False),
            "_complexity": complexity,
            "_model": model,
        })
        i += 1

    # 복잡도 오름차순 정렬 (haiku 배치가 앞으로)
    rows.sort(key=lambda r: r["_complexity"])

    # 배치 파일 저장
    skel_dir = _OUT_ROOT / "skeletons"
    skel_dir.mkdir(parents=True, exist_ok=True)

    batches_meta: list[dict] = []
    for b_idx, start in enumerate(range(0, len(rows), batch_size)):
        batch = rows[start: start + batch_size]
        model = batch[0]["_model"]  # 배치 내 첫 행의 모델 (정렬 후 일관적)
        path = skel_dir / f"batch_{b_idx:02d}.jsonl"

        with open(path, "w", encoding="utf-8") as f:
            for r in batch:
                out = {k: v for k, v in r.items() if not k.startswith("_")}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

        batches_meta.append({
            "batch_id": b_idx,
            "path": str(path),
            "n_rows": len(batch),
            "model": model,
            "complexity_range": [batch[0]["_complexity"], batch[-1]["_complexity"]],
        })
        print(f"  batch_{b_idx:02d}: {len(batch)}행, model={model}, "
              f"complexity={batch[0]['_complexity']}-{batch[-1]['_complexity']}")

    manifest = {
        "total": len(rows),
        "batch_size": batch_size,
        "n_batches": len(batches_meta),
        "seed": seed,
        "scenario": _SCENARIO,
        "batches": batches_meta,
    }
    manifest_path = _OUT_ROOT / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    haiku_n = sum(1 for b in batches_meta if b["model"] == "haiku")
    sonnet_n = len(batches_meta) - haiku_n
    print(f"\n[완료] {len(rows)}행 → {len(batches_meta)}배치 "
          f"(haiku={haiku_n}, sonnet={sonnet_n})")
    print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=900)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--batch-size", type=int, default=50)
    args = ap.parse_args()
    main(args.total, args.seed, args.batch_size)
