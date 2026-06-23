#!/usr/bin/env python
"""v9 학습셋 opus/gemini 품질 감사 (agy 기반).

각 행의 reasoning 품질을 LLM으로 평가하고 저품질 행을 pruning.

사용:
  uv run python scripts/v9/audit_v9_agy.py \
    --data data/scheduler_v9_en4.parquet \
    --out outputs/v9/audit_en4.json \
    --model "Gemini 3.5 Flash (Low)" \
    --workers 8

  uv run python scripts/v9/audit_v9_agy.py \
    --data data/scheduler_v9_combined.parquet \
    --out outputs/v9/audit_combined.json \
    --model "Gemini 3.5 Flash (Low)" \
    --workers 8

  # prune 적용
  uv run python scripts/v9/audit_v9_agy.py \
    --prune outputs/v9/audit_combined.json \
    --threshold 3.0 \
    --data data/scheduler_v9_combined.parquet \
    --out-pruned data/scheduler_v9_combined_pruned.parquet
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

AGY_BIN = "/home/pieroot/.local/bin/agy"

AUDIT_PROMPT_TMPL = """\
You are a dataset quality auditor for a scheduling AI system.

Evaluate this training example on 4 dimensions (score 1-5 each):
1. **summary_quality**: Does each task's reasoning.summary clearly explain WHY this task got its priority score? Is it specific and coherent?
2. **chain_detail_quality**: For tasks in dependency chains, does chaining_detail explain the dependency clearly? (Score 3 if no chains present)
3. **persona_fit**: Do task titles/memos fit the given user persona and occupation?
4. **schedule_coherence**: Is the overall schedule logically consistent (time blocks, priorities)?

Return ONLY valid JSON:
{{"summary_quality": <1-5>, "chain_detail_quality": <1-5>, "persona_fit": <1-5>, "schedule_coherence": <1-5>, "flag": "<ok|warn|reject>", "reason": "<one sentence>"}}

flag rules: ok=all>=3, warn=any==2, reject=any==1 or avg<2.5

--- EXAMPLE ---
Persona: {persona_snippet}
Tasks (first 3): {tasks_snippet}
"""


def build_audit_input(row: pd.Series) -> str:
    try:
        persona = row.get("persona", "")
        if isinstance(persona, str) and len(persona) > 200:
            persona = persona[:200] + "..."

        chosen = json.loads(row["chosen"])
        tasks = chosen.get("scheduled_tasks", [])[:3]
        tasks_snippet = []
        for t in tasks:
            tasks_snippet.append({
                "title": t.get("title", ""),
                "priority_rank": t.get("priority_rank"),
                "scoring": {k: v for k, v in (t.get("scoring") or {}).items()},
                "summary": (t.get("reasoning") or {}).get("summary", "")[:120],
                "chaining_detail": (t.get("reasoning") or {}).get("chaining_detail", "")[:80],
            })
        return AUDIT_PROMPT_TMPL.format(
            persona_snippet=str(persona)[:300],
            tasks_snippet=json.dumps(tasks_snippet, ensure_ascii=False, indent=None),
        )
    except Exception as e:
        return f"Error building prompt: {e}"


def call_agy(prompt: str, model: str, timeout: int = 60) -> dict:
    try:
        result = subprocess.run(
            [AGY_BIN, "--model", model, "--print", prompt,
             "--dangerously-skip-permissions"],
            capture_output=True, text=True, timeout=timeout,
        )
        raw = result.stdout.strip()
        if not raw:
            return {"error": "empty_response"}
        # strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(l for l in lines if not l.startswith("```"))
        # extract JSON block
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
        return {"error": f"no JSON in response: {raw[:200]}"}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except json.JSONDecodeError as e:
        return {"error": f"json_parse: {e}"}
    except Exception as e:
        return {"error": str(e)}


def audit_row(idx: int, row: pd.Series, model: str) -> dict:
    prompt = build_audit_input(row)
    result = call_agy(prompt, model)
    result["row_idx"] = idx
    result["version"] = row.get("version", "")
    return result


def run_audit(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.data).reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {args.data}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # resume support
    results: list[dict] = []
    done_idxs: set[int] = set()
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        results = existing
        done_idxs = {r["row_idx"] for r in results if "row_idx" in r}
        print(f"Resuming: {len(done_idxs)} already done")

    todo = [(i, df.iloc[i]) for i in range(len(df)) if i not in done_idxs]
    print(f"To audit: {len(todo)} rows | model: {args.model} | workers: {args.workers}")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(audit_row, i, row, args.model): i for i, row in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            if n % 20 == 0 or n == len(todo):
                elapsed = time.time() - t0
                print(f"  [{n}/{len(todo)}] elapsed={elapsed:.0f}s "
                      f"errors={sum(1 for r in results if 'error' in r)}", flush=True)
                # incremental save
                results.sort(key=lambda r: r.get("row_idx", 0))
                out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    results.sort(key=lambda r: r.get("row_idx", 0))
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    # summary
    scored = [r for r in results if "summary_quality" in r]
    print(f"\n=== Audit Summary ===")
    print(f"Total: {len(results)} | Scored: {len(scored)} | Errors: {len(results)-len(scored)}")
    if scored:
        flags = [r.get("flag", "?") for r in scored]
        from collections import Counter
        print(f"Flags: {dict(Counter(flags))}")
        for key in ["summary_quality", "chain_detail_quality", "persona_fit", "schedule_coherence"]:
            vals = [r[key] for r in scored if key in r]
            if vals:
                print(f"  {key}: mean={sum(vals)/len(vals):.2f} "
                      f"<3: {sum(1 for v in vals if v < 3)} ({100*sum(1 for v in vals if v<3)/len(vals):.1f}%)")


def run_prune(args: argparse.Namespace) -> None:
    audit = json.loads(Path(args.prune).read_text())
    df = pd.read_parquet(args.data).reset_index(drop=True)

    scored = {r["row_idx"]: r for r in audit if "summary_quality" in r}
    rejected_flags = {"reject"}
    if args.exclude_warn:
        rejected_flags.add("warn")

    keep_idxs = []
    reject_reasons: list[str] = []
    for i in range(len(df)):
        r = scored.get(i)
        if r is None:
            keep_idxs.append(i)  # not audited → keep
            continue
        avg = sum(r.get(k, 3) for k in
                  ["summary_quality", "chain_detail_quality", "persona_fit", "schedule_coherence"]) / 4
        if r.get("flag") in rejected_flags or avg < args.threshold:
            reject_reasons.append(f"idx={i} flag={r.get('flag')} avg={avg:.2f}: {r.get('reason','')}")
        else:
            keep_idxs.append(i)

    pruned = df.iloc[keep_idxs].reset_index(drop=True)
    out = Path(args.out_pruned)
    out.parent.mkdir(parents=True, exist_ok=True)
    pruned.to_parquet(out, index=False)
    print(f"Pruned: {len(df)} → {len(pruned)} rows (removed {len(df)-len(pruned)})")
    print(f"Version dist: {dict(pruned['version'].value_counts())}")
    if reject_reasons[:10]:
        print("Sample rejections:")
        for r in reject_reasons[:10]:
            print(f"  {r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/scheduler_v9_combined.parquet")
    ap.add_argument("--out", default="outputs/v9/audit_combined.json")
    ap.add_argument("--model", default="Gemini 3.5 Flash (Low)")
    ap.add_argument("--workers", type=int, default=8)
    # prune mode
    ap.add_argument("--prune", default="", help="audit JSON path to apply pruning")
    ap.add_argument("--threshold", type=float, default=3.0, help="min avg score to keep")
    ap.add_argument("--exclude-warn", action="store_true", help="also remove warn rows")
    ap.add_argument("--out-pruned", default="data/scheduler_v9_combined_pruned.parquet")
    args = ap.parse_args()

    if args.prune:
        run_prune(args)
    else:
        run_audit(args)


if __name__ == "__main__":
    main()
