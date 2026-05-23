#!/usr/bin/env python
"""Phase E: HuggingFace 보조 데이터셋 임포트.

두 가지 소스:
  orca-ko  : data/ko_Ultrafeedback_binarized.parquet에서 스케줄링 키워드 필터링 후
             v2 JSON chosen 재생성 → data/orca_ko_filtered_dpo.parquet
  xlam     : Salesforce/xlam-function-calling-60k → 한국어 스케줄 시나리오로 번역 (2단계)
             → data/xlam_ko_scheduled.parquet

공통: 비동기, jsonl 체크포인트, parse_lenient 자가검증, --limit dry-run 게이트.

사용:
  # dry-run
  uv run python scripts/import_hf_datasets.py --source orca-ko --limit 20 --verify
  uv run python scripts/import_hf_datasets.py --source xlam --limit 20 --verify

  # 전체 실행
  uv run python scripts/import_hf_datasets.py --source orca-ko --limit 2000
  uv run python scripts/import_hf_datasets.py --source xlam --limit 5000
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
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

_KO_SCHEDULING_KEYWORDS = [
    "일정", "우선순위", "할 일", "할일", "계획", "스케줄", "업무", "마감",
    "기한", "태스크", "task", "deadline", "schedule", "priority", "회의",
    "프로젝트", "과제", "제출", "납기", "미팅",
]

# xlam에서 스케줄/태스크 관련 함수만 필터
_XLAM_TASK_KEYWORDS = [
    "task", "schedule", "calendar", "remind", "todo", "plan", "meeting",
    "deadline", "appointment", "event", "priority", "organize", "manage",
]

# xlam 인스트럭션 → 한국어 할 일 목록 변환 프롬프트
_XLAM_TRANSLATE_SYSTEM = """\
당신은 영어 작업 지시문을 한국어 '할 일 목록' 시나리오로 변환하는 전문가입니다.
입력: 영어 도구 호출 시나리오 또는 작업 설명
출력: 한국어로 3-7개의 구체적인 할 일 항목(마감/시각 포함)을 나열한 텍스트
형식: 반드시 JSON {"tasks": ["할일1", "할일2", ...]} 으로만 응답하세요.
각 항목은 "~하기", "~제출", "~참석" 형식의 동사형으로 끝내세요.
마감이나 시각이 있으면 괄호 안에 포함하세요. (예: "계약서 검토하기 (내일 오전까지)")
"""

_PERSONAS = ["직장인", "학생", "프리랜서", "부모", "스타트업 창업자", "연구원", "마케터", "디자이너"]


# ── API 캐시 ──────────────────────────────────────────────────────────────────

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


# ── 체크포인트 ────────────────────────────────────────────────────────────────

def _load_checkpoint(path: str) -> set[str]:
    p = Path(path)
    done: set[str] = set()
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done.add(rec["key"])
    return done


async def _append_ckpt(path: str, key: str, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: p.open("a").write(
            json.dumps({"key": key, "row": row}, ensure_ascii=False) + "\n"
        ),
    )


def _load_checkpoint_rows(path: str) -> list[dict]:
    p = Path(path)
    rows: list[dict] = []
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                rows.append(rec["row"])
    return rows


# ── 공통 API 호출 ─────────────────────────────────────────────────────────────

async def _call_api(
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
            temperature=0.9,
            max_tokens=1024,
        )
        result = resp.choices[0].message.content or ""
        cache.set(key, result)
        return result


# ── orca-ko 소스 ──────────────────────────────────────────────────────────────

def _load_orca_ko_prompts(limit: int | None) -> list[tuple[str, str]]:
    """ko_Ultrafeedback에서 스케줄링 키워드 포함 프롬프트 추출. (prompt, persona) 쌍 반환."""
    local = Path("data/ko_Ultrafeedback_binarized.parquet")
    if local.exists():
        df = pd.read_parquet(str(local))
    else:
        from datasets import load_dataset as _ld
        ds = _ld("maywell/ko_Ultrafeedback_binarized", split="train")
        df = ds.to_pandas()

    kw = _KO_SCHEDULING_KEYWORDS

    def _has_kw(text: str) -> bool:
        tl = text.lower()
        return any(k in tl for k in kw)

    mask = df["prompt"].apply(_has_kw)
    filtered = df[mask].reset_index(drop=True)
    print(f"[orca-ko] 스케줄링 필터: {len(df)}개 → {len(filtered)}개")

    if limit is not None:
        filtered = filtered.head(limit)

    import random
    pairs: list[tuple[str, str]] = []
    for _, row in filtered.iterrows():
        persona = random.choice(_PERSONAS)
        pairs.append((str(row["prompt"]), persona))
    return pairs


async def _gen_orca_ko_row(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    persona: str,
    cache: _APICache,
    sem: asyncio.Semaphore,
) -> dict | None:
    messages = [
        {"role": "system", "content": render_system_prompt(SCHEDULER_SYSTEM_PROMPT_V2, persona)},
        {"role": "user", "content": prompt},
    ]
    raw = await _call_api(client, model, messages, cache, sem)
    parsed = parse_lenient(raw)
    if parsed is None:
        # 1회 재시도
        raw = await _call_api(client, model, messages, cache, sem)
        parsed = parse_lenient(raw)
    if parsed is None:
        return None
    return {
        "prompt": prompt,
        "chosen": format_for_sft(parsed),
        "persona": persona,
        "source": "orca_ko_filtered",
    }


async def run_orca_ko(
    limit: int | None,
    model: str,
    concurrency: int,
    verify: bool,
) -> None:
    out_path = "data/orca_ko_filtered_dpo.parquet"
    ckpt_path = "outputs/.ckpt_orca_ko.jsonl"
    cache_path = "outputs/.api_cache_orca_ko.jsonl"

    client = AsyncOpenAI()
    cache = _APICache(cache_path)
    sem = asyncio.Semaphore(concurrency)

    pairs = _load_orca_ko_prompts(limit)
    done_keys = _load_checkpoint(ckpt_path)
    existing_rows = _load_checkpoint_rows(ckpt_path)

    prompts_to_do = [
        (p, persona) for (p, persona) in pairs
        if hashlib.md5(p.encode()).hexdigest() not in done_keys
    ]
    print(f"[orca-ko] {len(existing_rows)}개 이미 완료 — 신규 {len(prompts_to_do)}개 처리 예정")

    results: list[dict] = list(existing_rows)
    failed = 0

    async def _process(prompt: str, persona: str) -> None:
        nonlocal failed
        row = await _gen_orca_ko_row(client, model, prompt, persona, cache, sem)
        if row is None:
            failed += 1
            return
        key = hashlib.md5(prompt.encode()).hexdigest()
        await _append_ckpt(ckpt_path, key, row)
        results.append(row)

    tasks = [_process(p, persona) for p, persona in prompts_to_do]
    done_count = len(existing_rows)

    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        await coro
        done_count += 1
        if done_count % 100 == 0:
            print(f"  진행: {done_count}개 완료 (실패 {failed})")

    print(f"\n[완료] 총 {len(results)}개 (신규 {len(results) - len(existing_rows)}, 실패 {failed})")

    if verify:
        passed = sum(1 for r in results if parse_lenient(r["chosen"]) is not None)
        print(f"\n[검증]\n  parse_lenient 통과: {passed}/{len(results)} ({100*passed/max(len(results),1):.1f}%)")

    if results:
        df = pd.DataFrame(results)
        df.to_parquet(out_path, index=False)
        print(f"\n[저장] {out_path}  ({len(df)}행)")


# ── xlam 소스 ─────────────────────────────────────────────────────────────────

def _load_xlam_instructions(limit: int | None) -> list[str]:
    """xlam 데이터셋에서 스케줄/태스크 관련 인스트럭션 필터링."""
    try:
        from datasets import load_dataset as _ld
        ds = _ld("Salesforce/xlam-function-calling-60k", split="train")
    except Exception as e:
        print(f"[xlam] HF 로드 실패: {e}")
        print("[xlam] data/events-scheduling.parquet 폴백 사용")
        df = pd.read_parquet("data/events-scheduling.parquet")
        instructions = df["prompt"].tolist()
        if limit:
            instructions = instructions[:limit]
        return instructions

    kw = _XLAM_TASK_KEYWORDS

    def _relevant(row: dict) -> bool:
        text = (row.get("query") or row.get("instruction") or "").lower()
        return any(k in text for k in kw)

    filtered = ds.filter(_relevant, desc="xlam 태스크 필터")
    print(f"[xlam] 필터: {len(ds)}개 → {len(filtered)}개")

    if limit is not None:
        filtered = filtered.select(range(min(limit, len(filtered))))

    key = "query" if "query" in filtered.column_names else "instruction"
    return [row[key] for row in filtered]


async def _xlam_step1_translate(
    client: AsyncOpenAI,
    model: str,
    instruction: str,
    cache: _APICache,
    sem: asyncio.Semaphore,
) -> str | None:
    """xlam 인스트럭션 → 한국어 할 일 목록 JSON."""
    messages = [
        {"role": "system", "content": _XLAM_TRANSLATE_SYSTEM},
        {"role": "user", "content": f"다음을 한국어 할 일 목록으로 변환하세요:\n\n{instruction[:600]}"},
    ]
    raw = await _call_api(client, model, messages, cache, sem)
    # {"tasks": [...]} 파싱
    try:
        import re
        m = re.search(r'\{.*"tasks".*\}', raw, re.DOTALL)
        if not m:
            return None
        obj = json.loads(m.group(0))
        tasks = obj.get("tasks", [])
        if not tasks or not isinstance(tasks, list):
            return None
        return "\n".join(f"- {t}" for t in tasks)
    except Exception:
        return None


async def _gen_xlam_row(
    client: AsyncOpenAI,
    model: str,
    instruction: str,
    cache: _APICache,
    sem: asyncio.Semaphore,
) -> dict | None:
    import random
    persona = random.choice(_PERSONAS)

    # step 1: 한국어 할 일 목록 생성
    task_list = await _xlam_step1_translate(client, model, instruction, cache, sem)
    if task_list is None:
        return None

    # step 2: v2 JSON 우선순위 생성
    messages = [
        {"role": "system", "content": render_system_prompt(SCHEDULER_SYSTEM_PROMPT_V2, persona)},
        {"role": "user", "content": task_list},
    ]
    raw = await _call_api(client, model, messages, cache, sem)
    parsed = parse_lenient(raw)
    if parsed is None:
        return None

    return {
        "prompt": task_list,
        "chosen": format_for_sft(parsed),
        "persona": persona,
        "source": "xlam_ko_translated",
    }


async def run_xlam(
    limit: int | None,
    model: str,
    concurrency: int,
    verify: bool,
) -> None:
    out_path = "data/xlam_ko_scheduled.parquet"
    ckpt_path = "outputs/.ckpt_xlam_ko.jsonl"
    cache_path = "outputs/.api_cache_xlam_ko.jsonl"

    client = AsyncOpenAI()
    cache = _APICache(cache_path)
    sem = asyncio.Semaphore(concurrency)

    instructions = _load_xlam_instructions(limit)
    done_keys = _load_checkpoint(ckpt_path)
    existing_rows = _load_checkpoint_rows(ckpt_path)

    to_do = [
        inst for inst in instructions
        if hashlib.md5(inst[:200].encode()).hexdigest() not in done_keys
    ]
    print(f"[xlam] {len(existing_rows)}개 이미 완료 — 신규 {len(to_do)}개 처리 예정")

    results: list[dict] = list(existing_rows)
    failed = 0

    async def _process(inst: str) -> None:
        nonlocal failed
        row = await _gen_xlam_row(client, model, inst, cache, sem)
        if row is None:
            failed += 1
            return
        key = hashlib.md5(inst[:200].encode()).hexdigest()
        await _append_ckpt(ckpt_path, key, row)
        results.append(row)

    tasks_coros = [_process(inst) for inst in to_do]
    done_count = len(existing_rows)

    for i, coro in enumerate(asyncio.as_completed(tasks_coros), 1):
        await coro
        done_count += 1
        if done_count % 100 == 0:
            print(f"  진행: {done_count}개 완료 (실패 {failed})")

    print(f"\n[완료] 총 {len(results)}개 (신규 {len(results) - len(existing_rows)}, 실패 {failed})")

    if verify:
        passed = sum(1 for r in results if parse_lenient(r["chosen"]) is not None)
        print(f"\n[검증]\n  parse_lenient 통과: {passed}/{len(results)} ({100*passed/max(len(results),1):.1f}%)")

    if results:
        df = pd.DataFrame(results)
        df.to_parquet(out_path, index=False)
        print(f"\n[저장] {out_path}  ({len(df)}행)")


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["orca-ko", "xlam"])
    parser.add_argument("--limit", type=int, default=None, help="처리 행 수 상한 (dry-run 게이트)")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI 모델")
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--verify", action="store_true", help="완료 후 parse_lenient 검증")
    parser.add_argument("--dry-run", action="store_true", help="--limit 20 --verify 단축")
    args = parser.parse_args()

    if args.dry_run:
        args.limit = args.limit or 20
        args.verify = True

    print(f"[import_hf] source={args.source}  limit={args.limit}  model={args.model}")

    if args.source == "orca-ko":
        asyncio.run(run_orca_ko(args.limit, args.model, args.concurrency, args.verify))
    else:
        asyncio.run(run_xlam(args.limit, args.model, args.concurrency, args.verify))


if __name__ == "__main__":
    main()
