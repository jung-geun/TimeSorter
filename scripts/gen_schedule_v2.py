#!/usr/bin/env python
"""v2 스케줄 데이터 생성 — Phase C + D.

두 가지 모드:
  nemotron-extra   : Nemotron 페르소나로 신규 시나리오 +2K 생성 (Phase C, ~$1)
  regenerate-existing : 기존 scheduler_ko_combined.parquet의 prompt에서
                        v2 JSON chosen을 재생성 (Phase D, ~$3)

공통 기능:
  - asyncio + Semaphore(concurrency) 비동기 생성
  - jsonl 체크포인트 (Ctrl+C 후 재실행 시 이어서 진행)
  - parse_lenient 자가 검증, 실패 시 최대 2회 재시도
  - --limit dry-run 게이트

사용:
  # Phase C dry-run
  uv run python scripts/gen_schedule_v2.py --mode nemotron-extra --limit 20 --verify

  # Phase C 전체
  uv run python scripts/gen_schedule_v2.py --mode nemotron-extra --limit 2000

  # Phase D dry-run
  uv run python scripts/gen_schedule_v2.py --mode regenerate-existing --limit 10 --verify

  # Phase D 전체 (게이트: 10 → 200 → None)
  uv run python scripts/gen_schedule_v2.py --mode regenerate-existing --limit 200
  uv run python scripts/gen_schedule_v2.py --mode regenerate-existing
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drl.data.schema import (
    SCHEDULER_SYSTEM_PROMPT_V2,
    format_for_sft,
    parse_lenient,
    render_system_prompt,
)

load_dotenv()

# ── 상수 ──────────────────────────────────────────────────────────────────────

_DEFAULT_MODEL = "gpt-5.4-mini"
_DEFAULT_CONCURRENCY = 15
_DEFAULT_MAX_RETRIES = 3

_NEMOTRON_PATH = "data/nemotron_personas_korea.parquet"
_EXISTING_PATH = "data/scheduler_ko_combined.parquet"

# Phase C: 직장인 60%, 나머지 7 페르소나 40%
_GENERIC_PERSONAS = ["직장인", "학생", "프리랜서", "부모", "시니어", "창업자", "의료진", "연구자"]

# ── GPT 프롬프트 ──────────────────────────────────────────────────────────────

# Step 1: 태스크 목록만 생성 (단순 JSON — 실패율 낮음)
_NEMOTRON_TASK_GEN_PROMPT = """\
아래 페르소나를 위한 현실적인 오늘의 할 일 목록 4-6개를 만드세요.
반드시 아래 형식의 JSON만 응답하세요.

[페르소나]
이름/나이/직업: {persona_name} ({age}세, {occupation})
지역: {district}
배경: {professional_persona_short}

출력 형식:
{{"tasks": ["할일1", "할일2", "할일3", "할일4"]}}"""

# Step 2: SCHEDULER_SYSTEM_PROMPT_V2 를 system으로 써서 JSON 우선순위 생성
_REGEN_SYSTEM = SCHEDULER_SYSTEM_PROMPT_V2


# ── 체크포인트 ────────────────────────────────────────────────────────────────

def _load_checkpoint(ckpt_path: Path) -> tuple[list[dict], set[str]]:
    rows: list[dict] = []
    done_keys: set[str] = set()
    if not ckpt_path.exists():
        return rows, done_keys
    with ckpt_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = rec.get("_key", "")
            if key:
                done_keys.add(key)
            rows.append({k: v for k, v in rec.items() if not k.startswith("_")})
    print(f"[체크포인트] {len(done_keys)}개 이미 완료 — 이어서 진행합니다.")
    return rows, done_keys


def _ensure_ckpt_dir(ckpt_path: Path) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)


async def _append_checkpoint_async(
    ckpt_path: Path,
    row: dict,
    key: str,
    lock: asyncio.Lock,
) -> None:
    async with lock:
        with ckpt_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**row, "_key": key}, ensure_ascii=False) + "\n")


# ── API 캐시 ──────────────────────────────────────────────────────────────────

class _APICache:
    def __init__(self, path: Path | None):
        self._path = path
        self._data: dict[str, str] = {}
        if path and path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self._data[rec["k"]] = rec["v"]
            print(f"[캐시] {len(self._data)}개 로드")

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    async def set(self, key: str, value: str, lock: asyncio.Lock) -> None:
        self._data[key] = value
        if self._path:
            async with lock:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": key, "v": value}, ensure_ascii=False) + "\n")


def _prompt_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# ── Phase C: Nemotron 페르소나 새 시나리오 생성 ───────────────────────────────

async def _call_api(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    cache: _APICache,
    cache_lock: asyncio.Lock,
    max_tokens: int = 600,
    temperature: float = 0.7,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    label: str = "",
) -> str | None:
    cache_key = _prompt_key(json.dumps(messages, ensure_ascii=False))
    cached = cache.get(cache_key)
    if cached:
        return cached
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            raw = resp.choices[0].message.content.strip()
            await cache.set(cache_key, raw, cache_lock)
            return raw
        except Exception as e:
            if attempt == max_retries - 1:
                if label:
                    print(f"  [실패] {label}: {e}")
                return None
            await asyncio.sleep(1)
    return None


async def _gen_nemotron_row(
    idx: int,
    persona_row: dict,
    client: AsyncOpenAI,
    model: str,
    cache: _APICache,
    cache_lock: asyncio.Lock,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> dict | None:
    persona_name_raw = str(persona_row.get("persona", ""))
    persona_name = persona_name_raw.split(" 씨는")[0].strip() if " 씨는" in persona_name_raw else persona_name_raw[:10]
    age = persona_row.get("age", 30)
    occupation = persona_row.get("occupation", "직장인")
    district = persona_row.get("district", "서울")
    # {} 포함 시 .format() 오류 방지
    professional_persona = str(persona_row.get("professional_persona", ""))[:200].replace("{", "（").replace("}", "）")

    # Step 1: 태스크 목록 생성
    task_prompt = _NEMOTRON_TASK_GEN_PROMPT.format(
        persona_name=persona_name,
        age=age,
        occupation=occupation,
        district=district,
        professional_persona_short=professional_persona,
    )
    raw1 = await _call_api(
        client, model,
        [{"role": "user", "content": task_prompt}],
        cache, cache_lock,
        max_tokens=300, temperature=0.9,
        label=f"nemotron-step1 {idx+1}",
    )
    if not raw1:
        return None

    try:
        tasks_json = json.loads(raw1)
        tasks_list = tasks_json.get("tasks", [])
        if not tasks_list or len(tasks_list) < 2:
            return None
    except Exception:
        return None

    # Step 2: SCHEDULER_SYSTEM_PROMPT_V2로 JSON 우선순위 생성
    persona_label = f"{persona_name} ({occupation}, {age}세)"
    task_lines = "\n".join(f"- {t}" for t in tasks_list)
    user_content = f"[{persona_name} 씨의 오늘의 할 일 목록]\n{task_lines}"

    raw2 = await _call_api(
        client, model,
        [
            {"role": "system", "content": render_system_prompt(_REGEN_SYSTEM, persona_label)},
            {"role": "user", "content": user_content},
        ],
        cache, cache_lock,
        max_tokens=700, temperature=0.3,
        label=f"nemotron-step2 {idx+1}",
    )
    if not raw2:
        return None

    parsed = parse_lenient(raw2)
    if parsed is None or not parsed.tasks:
        return None

    return {
        "prompt": user_content,
        "chosen": format_for_sft(parsed),
        "persona": persona_label,
        "source": "nemotron_v2",
    }


async def run_nemotron_extra(
    limit: int | None,
    model: str,
    concurrency: int,
    ckpt_path: Path,
    out_path: Path,
    cache: _APICache,
) -> list[dict]:
    nem_df = pd.read_parquet(_NEMOTRON_PATH)
    print(f"[Nemotron] {len(nem_df):,}개 페르소나 로드")

    # 직장인 비율 60% 확보: occupation 필터링
    wo_mask = nem_df["occupation"].str.contains("사원|직원|관리|기사|기술|사무|영업|서비스|운전|경비|환경|배달|청소", na=False)
    worker_df = nem_df[wo_mask]
    other_df = nem_df[~wo_mask]

    n_total = limit or 2000
    n_worker = int(n_total * 0.6)
    n_other = n_total - n_worker

    worker_sample = worker_df.sample(min(n_worker, len(worker_df)), random_state=46)
    other_sample = other_df.sample(min(n_other, len(other_df)), random_state=46)
    sample_df = pd.concat([worker_sample, other_sample]).reset_index(drop=True)
    sample_df = sample_df.sample(frac=1, random_state=46).reset_index(drop=True)

    print(f"[Nemotron] 샘플링: {len(sample_df)}개 (직장인계열 {len(worker_sample)}, 기타 {len(other_sample)})")

    existing_rows, done_keys = _load_checkpoint(ckpt_path)

    semaphore = asyncio.Semaphore(concurrency)
    cache_lock = asyncio.Lock()
    ckpt_lock = asyncio.Lock()

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    new_rows: list[dict] = []
    counter = {"ok": 0, "fail": 0}

    async def _process(idx: int, row: dict) -> None:
        key = str(row.get("uuid", idx))
        if key in done_keys:
            return
        async with semaphore:
            result = await _gen_nemotron_row(idx, row, client, model, cache, cache_lock)
        if result:
            result_with_key = {**result}
            await _append_checkpoint_async(ckpt_path, result_with_key, key, ckpt_lock)
            new_rows.append(result_with_key)
            counter["ok"] += 1
            if counter["ok"] % 50 == 0:
                print(f"  진행: {counter['ok']}개 완료 (실패 {counter['fail']})")
        else:
            counter["fail"] += 1

    tasks = [
        _process(idx, row.to_dict())
        for idx, row in sample_df.iterrows()
    ]
    await asyncio.gather(*tasks)

    all_rows = existing_rows + new_rows
    print(f"\n[완료] 총 {len(all_rows)}개 (신규 {len(new_rows)}, 실패 {counter['fail']})")
    return all_rows


# ── Phase D: 기존 prompt → v2 JSON 재생성 ────────────────────────────────────

async def _regen_row(
    idx: int,
    prompt: str,
    persona: str,
    client: AsyncOpenAI,
    model: str,
    cache: _APICache,
    cache_lock: asyncio.Lock,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> dict | None:
    raw = await _call_api(
        client, model,
        [
            {"role": "system", "content": render_system_prompt(_REGEN_SYSTEM, persona)},
            {"role": "user", "content": prompt},
        ],
        cache, cache_lock,
        max_tokens=800, temperature=0.3,
        label=f"regen {idx+1}",
    )
    if not raw:
        return None

    parsed = parse_lenient(raw)
    if parsed is None or not parsed.tasks:
        return None

    return {
        "prompt": prompt,
        "chosen": format_for_sft(parsed),
        "persona": persona,
        "source": "regen_v2",
    }


async def run_regenerate_existing(
    limit: int | None,
    model: str,
    concurrency: int,
    ckpt_path: Path,
    out_path: Path,
    cache: _APICache,
) -> list[dict]:
    df = pd.read_parquet(_EXISTING_PATH)
    if limit:
        df = df.head(limit)
    print(f"[재생성] {len(df)}개 프롬프트 처리 예정")

    existing_rows, done_keys = _load_checkpoint(ckpt_path)
    semaphore = asyncio.Semaphore(concurrency)
    cache_lock = asyncio.Lock()
    ckpt_lock = asyncio.Lock()

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    new_rows: list[dict] = []
    counter = {"ok": 0, "fail": 0}

    async def _process(idx: int, prompt: str, persona: str) -> None:
        key = _prompt_key(prompt)
        if key in done_keys:
            return
        async with semaphore:
            result = await _regen_row(idx, prompt, persona, client, model, cache, cache_lock)
        if result:
            await _append_checkpoint_async(ckpt_path, result, key, ckpt_lock)
            new_rows.append(result)
            counter["ok"] += 1
            if counter["ok"] % 100 == 0:
                print(f"  진행: {counter['ok']}개 완료 (실패 {counter['fail']})")
        else:
            counter["fail"] += 1

    tasks = [
        _process(idx, str(row["prompt"]), str(row.get("persona", "직장인")))
        for idx, row in df.iterrows()
    ]
    await asyncio.gather(*tasks)

    all_rows = existing_rows + new_rows
    print(f"\n[완료] 총 {len(all_rows)}개 (신규 {len(new_rows)}, 실패 {counter['fail']})")
    return all_rows


# ── 검증 ─────────────────────────────────────────────────────────────────────

def verify_rows(rows: list[dict]) -> float:
    passed = sum(1 for r in rows if parse_lenient(r["chosen"]) is not None)
    rate = passed / len(rows) if rows else 0.0
    print(f"  parse_lenient 통과: {passed}/{len(rows)} ({rate*100:.1f}%)")
    return rate


# ── 진입점 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v2 스케줄 데이터 생성 (Phase C/D)")
    parser.add_argument(
        "--mode", required=True, choices=["nemotron-extra", "regenerate-existing"],
        help="nemotron-extra(Phase C) | regenerate-existing(Phase D)"
    )
    parser.add_argument("--limit", type=int, default=None, help="처리 행 수 상한")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help=f"GPT 모델 (기본: {_DEFAULT_MODEL})")
    parser.add_argument("--concurrency", type=int, default=_DEFAULT_CONCURRENCY,
                        help=f"비동기 동시 요청 수 (기본: {_DEFAULT_CONCURRENCY})")
    parser.add_argument(
        "--out", default=None,
        help="출력 parquet 경로 (기본: data/scheduler_v2_nemotron_extra.parquet 또는 data/scheduler_v2_regen.parquet)"
    )
    parser.add_argument(
        "--ckpt", default=None,
        help="jsonl 체크포인트 경로 (기본: outputs/.ckpt_<mode>.jsonl)"
    )
    parser.add_argument(
        "--cache", default="outputs/.api_cache.jsonl",
        help="API 응답 캐시 파일 경로"
    )
    parser.add_argument("--verify", action="store_true", help="생성 후 parse_lenient 검증")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("[에러] OPENAI_API_KEY 환경변수가 필요합니다.", file=sys.stderr)
        sys.exit(1)

    mode = args.mode
    is_nemotron = mode == "nemotron-extra"

    default_out = (
        "data/scheduler_v2_nemotron_extra.parquet" if is_nemotron
        else "data/scheduler_v2_regen.parquet"
    )
    out_path = Path(args.out or default_out)
    ckpt_path = Path(args.ckpt or f"outputs/.ckpt_{mode.replace('-','_')}.jsonl")
    cache_path = Path(args.cache) if args.cache else None

    print(f"[gen_schedule_v2] mode={mode}  limit={args.limit}  model={args.model}")
    print(f"  출력: {out_path}")
    print(f"  체크포인트: {ckpt_path}")

    cache = _APICache(cache_path)

    if is_nemotron:
        rows = asyncio.run(run_nemotron_extra(
            limit=args.limit,
            model=args.model,
            concurrency=args.concurrency,
            ckpt_path=ckpt_path,
            out_path=out_path,
            cache=cache,
        ))
    else:
        rows = asyncio.run(run_regenerate_existing(
            limit=args.limit,
            model=args.model,
            concurrency=args.concurrency,
            ckpt_path=ckpt_path,
            out_path=out_path,
            cache=cache,
        ))

    if not rows:
        print("[경고] 생성된 행이 없습니다.")
        sys.exit(1)

    if args.verify:
        print("\n[검증]")
        rate = verify_rows(rows)
        if rate < 0.90:
            print(f"[경고] 통과율 {rate*100:.1f}% < 90% — 출력을 저장하지 않습니다.")
            sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(str(out_path), index=False)
    print(f"\n[저장] {out_path}  ({len(rows)}행)")
