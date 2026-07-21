#!/usr/bin/env python
"""v8 체인 배치 재생성 — OpenAI gpt-5.4 (sonnet 동급) 사용.

사용:
  uv run python scripts/gen_v8_chain_batches.py --batches 00 01 04 13 17
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

_MODEL = "gpt-5.4"          # sonnet 동급 — stronger model for better dependency compliance
_MAX_CONCURRENCY = 8        # 배치 내 동시 행 처리 수
_OUT_ROOT = Path("outputs/v8_chain")

SYSTEM_PROMPT = """\
You are a Korean to-do scheduling training data generator. You produce structured JSON objects for each input skeleton row.

CRITICAL RULES (machine-verified — violations cause automatic rejection):
1. priority_order: ALL chain tasks of the SAME group must be CONTIGUOUS AND in step order (step1 first, final step last). No non-chain task may appear between chain steps.
2. dependency scores: EVERY NON-FINAL chain step MUST have dependency = 4 or 5. Setting dependency ≤ 3 for any non-final chain step = AUTOMATIC REJECTION. The final chain step (the one with the deadline) may have any dependency value.
3. priority_order must be a permutation of ALL task ids (1..N).
4. "마감 없음" tasks must NOT be ranked #1.
5. Same-day deadlines: earlier time = higher rank.

Output ONLY valid JSON. No markdown fences, no explanation."""

ROW_PROMPT_TEMPLATE = """\
Generate one output object for this skeleton row.

Skeleton:
- id: {id}
- today: {today} ({today_display})
- persona: {persona_label}
- gen_lines (task specs, in skeleton order — task 1 = first gen_line, task 2 = second, etc.):
{gen_lines_str}
- facts (grading reference — tells you EXACTLY which task ids are in which chain and at which step position):
{facts}

STEP 1 — Read the facts carefully. For each task id, note:
  - Is it a chain task? If so, which chain_group and which chain_pos (step 1, 2, 3, ...)?
  - Is the final step? (the one with a deadline in the chain)
  - Is it a standalone today/future/none deadline task?

STEP 2 — Write tasks[] (plain Korean strings, same count and order as gen_lines, task 1 = first string):
  - Chain steps: all steps share the SAME concrete deliverable/subject.
  - Each step explicitly consumes the previous step's output ("작성한 ~", "검토한 ~", "확정된 ~").
  - THE FINAL CHAIN STEP (with deadline) MUST name the deliverable + completing action (e.g. "완성한 보고서 임원 발송 (8/1 17:00까지)").
  - FORBIDDEN finals: "후속 조치 최종 확인", "최종 확인 완료", "마무리 처리".
  - NEVER write "체인", "1단계", "2단계" literally.

STEP 3 — Build priority_order:
  - Read the chain_pos from facts to determine execution order within each chain.
  - Within each chain group, list tasks IN CHAIN_POS ORDER (chain_pos=1 first, last chain_pos last).
  - All tasks of the SAME chain group must be CONTIGUOUS — no non-chain task between them.
  - "마감 없음" tasks must NOT be #1.
  - Same-day deadlines: earlier time = higher rank.

STEP 4 — Build scores for each task:
  - NON-FINAL chain step (chain_pos < max_pos for that group): dependency = 4 or 5 (MANDATORY — 3 or below = REJECTED).
  - Final chain step: any dependency value.
  - Standalone tasks with deadlines: urgency/time_constraint based on proximity to today.

Produce this EXACT JSON structure:
{{
  "id": "{id}",
  "tasks": ["task1 Korean text", "task2 Korean text", ...],
  "chosen": {{
    "tasks": [{{"id": 1, "text": "task1 text"}}, {{"id": 2, "text": "task2 text"}}, ...],
    "priority_order": [N, N, ...],
    "scores": [
      {{"task_id": 1, "urgency": 1-5, "importance": 1-5, "dependency": 1-5, "time_constraint": 1-5, "reason": "한국어 이유"}}
    ]
  }},
  "prompt": "[{persona_name} 씨의 오늘의 할 일 목록]\\n- task1\\n- task2\\n...",
  "persona": "{persona_label}",
  "today": "{today}",
  "source": "v8_dependency_chain_complex",
  "meta": {meta_json}
}}

Output ONLY the JSON object, nothing else."""


async def generate_row(client: AsyncOpenAI, row: dict, semaphore: asyncio.Semaphore) -> dict | None:
    async with semaphore:
        meta_str = row["meta"] if isinstance(row["meta"], str) else json.dumps(row["meta"], ensure_ascii=False)
        gen_lines_str = "\n".join(f"  {i+1}. {line}" for i, line in enumerate(row["gen_lines"]))

        prompt = ROW_PROMPT_TEMPLATE.format(
            id=row["id"],
            today=row["today"],
            today_display=row.get("today_display", row["today"]),
            persona_label=row["persona_label"],
            persona_name=row["persona_name"],
            gen_lines_str=gen_lines_str,
            facts=row["facts"],
            meta_json=meta_str,
        )

        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=_MODEL,
                    max_completion_tokens=2048,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = resp.choices[0].message.content.strip()

                # Strip markdown fences if present
                if text.startswith("```"):
                    text = re.sub(r"^```[a-z]*\n?", "", text)
                    text = re.sub(r"\n?```$", "", text)
                    text = text.strip()

                obj = json.loads(text)

                # Ensure chosen is a dict not a string
                if isinstance(obj.get("chosen"), str):
                    obj["chosen"] = json.loads(obj["chosen"])

                # Ensure meta is the original meta string (verbatim copy)
                obj["meta"] = meta_str

                return obj

            except json.JSONDecodeError as e:
                print(f"  [JSON error] {row['id']} attempt {attempt+1}: {e}", flush=True)
                if attempt == 2:
                    return None
            except Exception as e:
                print(f"  [API error] {row['id']} attempt {attempt+1}: {e}", flush=True)
                if attempt == 2:
                    return None
                await asyncio.sleep(2 ** attempt)

        return None


async def process_batch(client: AsyncOpenAI, batch_id: str) -> int:
    skel_path = _OUT_ROOT / "skeletons" / f"batch_{batch_id}.jsonl"
    out_path = _OUT_ROOT / "gen" / f"batch_{batch_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(skel_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    print(f"[batch_{batch_id}] {len(rows)} rows → {out_path}", flush=True)

    semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
    tasks = [generate_row(client, row, semaphore) for row in rows]
    results = await asyncio.gather(*tasks)

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (row, result) in enumerate(zip(rows, results)):
            if result is not None:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                written += 1
            else:
                print(f"  [SKIP] {row['id']} — generation failed", flush=True)

    print(f"[batch_{batch_id}] wrote {written}/{len(rows)} rows", flush=True)

    # Spot-check: print 2 sample task arrays
    with open(out_path, encoding="utf-8") as f:
        sample_rows = [json.loads(l) for l in f if l.strip()]
    for sr in sample_rows[:2]:
        print(f"  sample {sr['id']}: {sr['tasks']}", flush=True)

    return written


async def main(batch_ids: list[str]) -> None:
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    for batch_id in batch_ids:
        await process_batch(client, batch_id)
        print(flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="+", required=True, help="배치 ID 목록 (e.g. 00 01 04 13 17)")
    args = ap.parse_args()
    asyncio.run(main(args.batches))
