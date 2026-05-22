#!/usr/bin/env python
"""영문 events-scheduling 시드 → Claude/OpenAI/Gemini로 한국어 번역 + 페르소나 확장.

산출물: data/scheduler_ko.parquet  (prompt, chosen, persona 컬럼)

사용:
  uv run python scripts/gen_korean_schedule.py                              # 기본 (gemini)
  uv run python scripts/gen_korean_schedule.py --provider openai            # OpenAI
  uv run python scripts/gen_korean_schedule.py --provider claude            # Claude Haiku
  uv run python scripts/gen_korean_schedule.py --provider gemini --model gemini-3.1-flash-lite
  uv run python scripts/gen_korean_schedule.py --limit 5                    # dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_PERSONAS = ["직장인", "학생", "프리랜서", "부모"]

_TRANSLATE_PROMPT = """\
다음 영문 이벤트/할 일 목록을 한국어 {persona} 페르소나의 일상 할 일 목록으로 자연스럽게 번역하라.
4축(긴급도·중요도·의존성·시간 제약) 기준의 우선순위 정렬 답변도 함께 제공하라.
출력 JSON만: {{"prompt": "...(한국어 할 일 목록)...", "chosen": "1) ... 2) ..."}}

입력: {event_text}"""

_PROVIDER_DEFAULTS: dict[str, str] = {
    "claude": "claude-haiku-4-5",
    "openai": "gpt-5.4-mini",
    "gemini": "gemini-3.1-flash-lite",
}


def _run_cli(cmd: list[str], timeout: int = 120) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"{cmd[0]} CLI failed")
    return result.stdout.strip()


_MODEL_ALIASES = {
    # Gemini
    "gemini-3.5-flash": "gemini-3.5-flash",
    "gemini-3.1-pro": "gemini-3.1-pro",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
    
    # Claude
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude-opus-4-6": "claude-3-opus-20240229",
    "claude-haiku-4-5": "claude-3-5-haiku-20241022",
    
    # OpenAI
    "gpt-5.5": "gpt-5.5",
    "gpt-5.4-mini": "gpt-5.4-mini",
}

def _resolve_model(model_name: str) -> str:
    return _MODEL_ALIASES.get(model_name, model_name)


def _call_model(prompt: str, provider: str, model: str) -> str:
    # API key가 없고, OpenAI API key가 있으면 OpenAI로 폴백
    if provider == "claude" and not os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("OPENAI_API_KEY"):
        fallback_model = "gpt-5.5" if "sonnet" in model or "opus" in model else "gpt-5.4-mini"
        print(f"  [폴백] ANTHROPIC_API_KEY가 없어 {model} 대신 OpenAI {fallback_model}을 사용합니다.")
        return _call_model(prompt, "openai", fallback_model)

    if provider == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
        try:
            print(f"  [시도] GOOGLE_API_KEY가 없어 agy CLI로 {model} 생성을 시도합니다...")
            return _run_cli(["agy", "-p", prompt, "--dangerously-skip-permissions"], timeout=15)
        except Exception as e:
            print(f"  [실패] agy 호출 실패 또는 타임아웃 ({e})")

        if os.environ.get("OPENAI_API_KEY"):
            fallback_model = "gpt-5.5" if "3.5-flash" in model or "pro" in model else "gpt-5.4-mini"
            print(f"  [폴백] agy 사용 불가로 {model} 대신 OpenAI {fallback_model}을 사용합니다.")
            return _call_model(prompt, "openai", fallback_model)

    model = _resolve_model(model)
    if provider == "claude":
        if os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            msg = client.messages.create(
                model=model,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        else:
            return _run_cli(["claude", "-p", prompt, "--model", model])

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=600,
        )
        return resp.choices[0].message.content.strip()

    if provider == "gemini":
        if os.environ.get("GOOGLE_API_KEY"):
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            gmodel = genai.GenerativeModel(model)
            return gmodel.generate_content(prompt).text.strip()
        else:
            return _run_cli(["agy", "-p", prompt, "--model", model])

    raise ValueError(f"지원하지 않는 provider: {provider}")


def translate_scenario(
    event_text: str, persona: str, provider: str, model: str
) -> dict | None:
    prompt = _TRANSLATE_PROMPT.format(persona=persona, event_text=event_text)
    for attempt in range(3):
        try:
            text = _call_model(prompt, provider, model)
            if "```" in text:
                text = text.split("```")[1].lstrip("json").strip()
            return json.loads(text)
        except Exception as e:
            print(f"  [재시도 {attempt+1}] {e}")
            time.sleep(2 ** attempt)
    return None


def _load_checkpoint(ckpt_path: Path) -> tuple[list[dict], set[tuple[int, str]]]:
    """체크포인트 JSONL 로드. (rows, done_set) 반환."""
    rows: list[dict] = []
    done: set[tuple[int, str]] = set()
    if not ckpt_path.exists():
        return rows, done
    with ckpt_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done.add((rec["_seed_idx"], rec["_persona"]))
            # Clean up prompt if it was parsed as a list in checkpoint
            if "prompt" in rec and isinstance(rec["prompt"], list):
                rec["prompt"] = "\n".join([", ".join(map(str, x)) if isinstance(x, list) else str(x) for x in rec["prompt"]])
            rows.append({k: v for k, v in rec.items() if not k.startswith("_")})
    print(f"[체크포인트] {len(done)}개 이미 완료 — 이어서 진행합니다.")
    return rows, done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="처리할 시나리오 수 제한")
    parser.add_argument("--out", default="data/scheduler_ko.parquet")
    parser.add_argument(
        "--provider", default="gemini", choices=list(_PROVIDER_DEFAULTS),
        help="번역 제공자 (기본: gemini)",
    )
    parser.add_argument(
        "--model", default=None,
        help="모델 오버라이드 (미지정 시 provider 기본값 사용)",
    )
    args = parser.parse_args()

    model = args.model or _PROVIDER_DEFAULTS[args.provider]
    print(f"[설정] provider={args.provider}, model={model}")

    from datasets import load_dataset
    seed = load_dataset("anakin87/events-scheduling", split="train")

    if args.limit:
        seed = seed.select(range(min(args.limit, len(seed))))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path.with_suffix(".ckpt.jsonl")

    rows, done = _load_checkpoint(ckpt_path)
    new_count = 0

    with ckpt_path.open("a") as ckpt_f:
        for i, example in enumerate(seed):
            event_text = str(
                example.get("events") or example.get("input") or next(iter(example.values()))
            )
            for persona in _PERSONAS:
                if (i, persona) in done:
                    continue
                print(f"[{i+1}/{len(seed)}] [{persona}] 번역 중...")
                result = translate_scenario(event_text, persona, args.provider, model)
                if result and "prompt" in result and "chosen" in result:
                    prompt_val = result["prompt"]
                    if isinstance(prompt_val, list):
                        prompt_val = "\n".join([", ".join(map(str, x)) if isinstance(x, list) else str(x) for x in prompt_val])
                    row = {
                        "prompt": prompt_val,
                        "chosen": result["chosen"],
                        "persona": persona,
                        "source": "events-scheduling",
                    }
                    rows.append(row)
                    ckpt_f.write(
                        json.dumps({**row, "_seed_idx": i, "_persona": persona}, ensure_ascii=False)
                        + "\n"
                    )
                    ckpt_f.flush()
                    new_count += 1
            time.sleep(0.3)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    print(f"[done] 신규 {new_count}개 추가, 총 {len(df)}개 저장 → {args.out}")


if __name__ == "__main__":
    main()
