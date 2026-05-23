#!/usr/bin/env python
"""Phase F: v2 DPO 쌍 생성 — rejected 4 카테고리.

chosen: Phase C+D의 v2 JSON 응답 재활용
rejected 4 카테고리 (각 25%):
  invalid_json    : "JSON 금지, 1) 2) 형식으로 답" 지시 → 형식 위반 (GPT)
  bad_scores      : 점수 전부 3으로 치환 + priority_order 역순 (프로그래매틱)
  urgency_only    : urgency만 기준으로 한 자유 텍스트 응답 (GPT)
  shallow_reason  : reason을 "중요해서"/"급해서" 등으로 교체 (프로그래매틱)

입력: data/scheduler_v2_regen.parquet + data/scheduler_v2_nemotron_extra.parquet
출력: data/dpo_pairs_v2.parquet

사용:
  uv run python scripts/gen_preference_pairs_v2.py --limit 20 --verify
  uv run python scripts/gen_preference_pairs_v2.py
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drl.data.schema import (
    format_for_sft,
    parse_lenient,
    render_system_prompt,
)

load_dotenv()

# ── 상수 ──────────────────────────────────────────────────────────────────────

_CATEGORIES = ["invalid_json", "bad_scores", "urgency_only", "shallow_reason"]

_SHALLOW_REASONS = ["중요해서", "급해서", "먼저 해야 해서", "우선순위가 높아서", "중요한 일이라서"]

# invalid_json rejected 생성 지시
_INVALID_JSON_SYSTEM = """\
당신은 할 일 우선순위를 정렬하는 비서입니다.
반드시 JSON을 사용하지 말고, 아래 형식으로만 답하세요:
1) [할일] - [이유]
2) [할일] - [이유]
...
JSON, 코드 블록, 중괄호를 절대 사용하지 마세요."""

# urgency_only rejected 생성 지시
_URGENCY_ONLY_SYSTEM = """\
당신은 긴급도만을 기준으로 할 일을 정렬하는 비서입니다.
오직 마감일/시작 시각의 임박 정도만 고려해 순서를 정하고,
아래 형식으로 답하세요:
1) [할일] - 긴급도: [상/중/하]
2) [할일] - 긴급도: [상/중/하]
...
중요도, 의존성, 시간 제약 등 다른 요소는 무시하세요."""


# ── 캐시 / 체크포인트 ─────────────────────────────────────────────────────────

class _APICache:
    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._data: dict[str, str] = {}
        if self._path.exists():
            for line in self._path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._data[rec["key"]] = rec["value"]
            print(f"[캐시] {len(self._data)}개 로드")

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

    @staticmethod
    def make_key(messages: list[dict]) -> str:
        content = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()


def _ckpt_load(path: str) -> tuple[list[dict], set[str]]:
    rows: list[dict] = []
    done: set[str] = set()
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done.add(rec["key"])
                rows.append(rec["row"])
    return rows, done


async def _ckpt_append(path: str, key: str, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: open(path, "a").write(
            json.dumps({"key": key, "row": row}, ensure_ascii=False) + "\n"
        ),
    )


# ── API 호출 ──────────────────────────────────────────────────────────────────

async def _call(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    cache: _APICache,
    sem: asyncio.Semaphore,
) -> str:
    key = _APICache.make_key(messages)
    cached = cache.get(key)
    if cached is not None:
        return cached
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.8,
            max_tokens=512,
        )
        result = resp.choices[0].message.content or ""
        cache.set(key, result)
        return result


# ── rejected 생성 함수 ────────────────────────────────────────────────────────

async def _gen_invalid_json(
    client: AsyncOpenAI, model: str, prompt: str, persona: str,
    cache: _APICache, sem: asyncio.Semaphore,
) -> str:
    messages = [
        {"role": "system", "content": render_system_prompt(_INVALID_JSON_SYSTEM, persona)},
        {"role": "user", "content": prompt},
    ]
    return await _call(client, model, messages, cache, sem)


async def _gen_urgency_only(
    client: AsyncOpenAI, model: str, prompt: str, persona: str,
    cache: _APICache, sem: asyncio.Semaphore,
) -> str:
    messages = [
        {"role": "system", "content": render_system_prompt(_URGENCY_ONLY_SYSTEM, persona)},
        {"role": "user", "content": prompt},
    ]
    return await _call(client, model, messages, cache, sem)


def _gen_bad_scores(chosen_json: str) -> str | None:
    """점수를 전부 3으로 교체 + priority_order를 역순으로."""
    parsed = parse_lenient(chosen_json)
    if parsed is None or not parsed.tasks:
        return None
    # 점수 전부 3으로 오염
    for s in parsed.scores:
        s.urgency = 3
        s.importance = 3
        s.dependency = 3
        s.time_constraint = 3
    # priority_order 역순
    parsed.priority_order = list(reversed(parsed.priority_order))
    return format_for_sft(parsed)


def _gen_shallow_reason(chosen_json: str) -> str | None:
    """reason을 얕은 이유로 교체."""
    parsed = parse_lenient(chosen_json)
    if parsed is None or not parsed.scores:
        return None
    for s in parsed.scores:
        s.reason = random.choice(_SHALLOW_REASONS)
    return format_for_sft(parsed)


# ── 소스 데이터 로드 ──────────────────────────────────────────────────────────

_SFT_SOURCES = [
    "data/scheduler_v2_regen.parquet",
    "data/scheduler_v2_nemotron_extra.parquet",
]


def _load_source_rows(limit: int | None) -> list[dict]:
    frames: list[pd.DataFrame] = []
    for path in _SFT_SOURCES:
        p = Path(path)
        if not p.exists():
            print(f"[스킵] {path} — 없음")
            continue
        df = pd.read_parquet(path)
        if "prompt" not in df.columns or "chosen" not in df.columns:
            print(f"[스킵] {path} — prompt/chosen 컬럼 없음")
            continue
        frames.append(df)
        print(f"[로드] {path}: {len(df)}행")
    if not frames:
        raise FileNotFoundError("소스 parquet 없음. Phase C+D 먼저 실행하세요.")
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["prompt"])
    print(f"[소스] 중복 제거 후 총 {len(combined)}개 프롬프트")
    if limit:
        combined = combined.head(limit)
    return combined.to_dict("records")


# ── 메인 ─────────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace) -> None:
    ckpt_path = "outputs/.ckpt_dpo_v2.jsonl"
    cache_path = "outputs/.api_cache_dpo_v2.jsonl"

    client = AsyncOpenAI()
    cache = _APICache(cache_path)
    sem = asyncio.Semaphore(args.concurrency)

    source_rows = _load_source_rows(args.limit)
    existing_rows, done_keys = _ckpt_load(ckpt_path)
    print(f"[DPO v2] {len(existing_rows)}개 이미 완료 — 신규 {len(source_rows) - len(done_keys)}개 처리 예정")

    results: list[dict] = list(existing_rows)
    failed = 0

    async def _process(src: dict) -> None:
        nonlocal failed
        prompt = str(src["prompt"])
        chosen = str(src["chosen"])
        persona = str(src.get("persona", "직장인"))
        row_key = hashlib.md5(prompt.encode()).hexdigest()

        if row_key in done_keys:
            return

        # 4 카테고리 중 2개 랜덤 선택 (per prompt, API 비용 절감)
        cats = random.sample(_CATEGORIES, 2)
        for cat in cats:
            if cat == "bad_scores":
                rejected = _gen_bad_scores(chosen)
            elif cat == "shallow_reason":
                rejected = _gen_shallow_reason(chosen)
            elif cat == "invalid_json":
                rejected = await _gen_invalid_json(client, args.model, prompt, persona, cache, sem)
            else:  # urgency_only
                rejected = await _gen_urgency_only(client, args.model, prompt, persona, cache, sem)

            if rejected is None or not rejected.strip():
                failed += 1
                continue

            row = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "persona": persona,
                "category": cat,
            }
            await _ckpt_append(ckpt_path, f"{row_key}_{cat}", row)
            results.append(row)

    tasks = [_process(src) for src in source_rows]
    done_count = len(existing_rows)
    for coro in asyncio.as_completed(tasks):
        await coro
        done_count += 1
        if done_count % 200 == 0:
            print(f"  진행: {done_count}개 완료 (실패 {failed})")

    print(f"\n[완료] 총 {len(results)}개 DPO 쌍 (실패 {failed})")

    # Phase B 거부 데이터 머지
    refusals_path = Path("data/refusals_dpo_v2.parquet")
    if refusals_path.exists():
        ref_df = pd.read_parquet(str(refusals_path))
        print(f"[머지] Phase B 거부 데이터: {len(ref_df)}쌍 추가")
        results_df = pd.DataFrame(results)
        combined = pd.concat([results_df, ref_df], ignore_index=True)
    else:
        combined = pd.DataFrame(results)

    if args.verify and len(results) > 0:
        chosen_pass = sum(1 for r in results if parse_lenient(r["chosen"]) is not None)
        print(f"\n[검증] chosen parse_lenient: {chosen_pass}/{len(results)} "
              f"({100*chosen_pass/len(results):.1f}%)")
        cat_counts = {}
        for r in results:
            cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1
        print("  카테고리 분포:", cat_counts)

    out_path = "data/dpo_pairs_v2.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"\n[저장] {out_path}  ({len(combined)}행)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="소스 프롬프트 수 상한")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="--limit 20 --verify 단축")
    args = parser.parse_args()

    if args.dry_run:
        args.limit = args.limit or 20
        args.verify = True

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
