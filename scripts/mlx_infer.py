#!/usr/bin/env python
"""MLX-LM 기반 인퍼런스 — Apple Silicon 최적화.

필요 패키지 (Mac에서만):
  pip install mlx-lm

사용:
  # 기본 (SFT v2 어댑터, 직장인 페르소나)
  python scripts/mlx_infer.py \
      --adapter outputs/sft_rtx12g_4b_v2_mlx \
      --prompt '보고서 작성(내일 마감), 점심 약속, 메일 답장'

  # 페르소나 지정
  python scripts/mlx_infer.py \
      --adapter outputs/sft_rtx12g_4b_v2_mlx \
      --persona 학생 \
      --prompt '기말고사 공부, 동아리 회의, 과제 제출(오늘 자정)'

  # 어댑터 없이 베이스 모델만 (빠른 테스트)
  python scripts/mlx_infer.py \
      --prompt '보고서 작성(내일 마감), 점심 약속'
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drl.data.schema import (
    SCHEDULER_SYSTEM_PROMPT_V2,
    parse_or_repair,
    render_system_prompt,
    response_to_text,
)

_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def run(
    prompt: str,
    adapter_path: str | None,
    persona: str,
    max_tokens: int,
    thinking: bool,
) -> None:
    try:
        from mlx_lm import generate, load
    except ImportError:
        print("mlx-lm이 설치되지 않았습니다. Mac에서 실행하세요:")
        print("  pip install mlx-lm")
        sys.exit(1)

    print(f"[MLX] 모델 로드: {_BASE_MODEL}")
    if adapter_path:
        print(f"[MLX] 어댑터: {adapter_path}")

    load_kwargs: dict = {"model_path": _BASE_MODEL}
    if adapter_path and Path(adapter_path).exists():
        load_kwargs["adapter_path"] = adapter_path

    model, tokenizer = load(**load_kwargs)

    system_content = render_system_prompt(SCHEDULER_SYSTEM_PROMPT_V2, persona)

    # Qwen3 thinking mode 지원
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    chat_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )

    print(f"\n[입력] 페르소나={persona}")
    print(f"[할 일] {prompt}\n")
    print("─" * 50)

    raw = generate(
        model,
        tokenizer,
        prompt=chat_input,
        max_tokens=max_tokens,
        verbose=False,
    )

    resp = parse_or_repair(raw)
    print(response_to_text(resp))
    print("─" * 50)

    if resp.refusal_reason:
        print(f"[거부] {resp.refusal_reason}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="MLX 어댑터 디렉토리 (convert_adapter_mlx.py 출력)")
    parser.add_argument("--prompt", required=True, help="할 일 목록 (쉼표 구분)")
    parser.add_argument("--persona", default="직장인")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--thinking", action="store_true", help="Qwen3 thinking mode 활성화")
    args = parser.parse_args()

    run(args.prompt, args.adapter, args.persona, args.max_tokens, args.thinking)


if __name__ == "__main__":
    main()
