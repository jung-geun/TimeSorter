from __future__ import annotations

import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data.schema import (
    SCHEDULER_SYSTEM_PROMPT_V1,
    SCHEDULER_SYSTEM_PROMPT_V2,
    parse_or_repair,
    render_system_prompt,
    response_to_text,
)
from .device import detect


def generate(
    adapter_path: str,
    prompt: str,
    max_new_tokens: int = 512,
    thinking: bool = False,
    persona: str = "직장인",
    schema_version: str = "v1",
) -> str:
    profile = detect()
    base_model_name = _read_base_model(adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=False)

    load_kwargs: dict = {
        "dtype": profile.dtype,
        "trust_remote_code": False,
    }
    if profile.device == "cuda":
        load_kwargs["device_map"] = "auto"

    base = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
    if profile.device == "mps":
        base = base.to("mps")

    model = PeftModel.from_pretrained(base, adapter_path)
    model.train(False)  # inference mode (no dropout / batch-norm tracking)

    system_tmpl = (
        SCHEDULER_SYSTEM_PROMPT_V2 if schema_version == "v2" else SCHEDULER_SYSTEM_PROMPT_V1
    )
    messages = [
        {"role": "system", "content": render_system_prompt(system_tmpl, persona)},
        {"role": "user", "content": prompt},
    ]
    # enable_thinking=False: Qwen3 thinking mode 비활성화 (빠른 직접 응답)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(profile.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)

    if schema_version == "v2":
        resp = parse_or_repair(raw)
        return response_to_text(resp)
    return raw


def _read_base_model(adapter_path: str) -> str:
    with open(f"{adapter_path}/adapter_config.json") as f:
        return json.load(f)["base_model_name_or_path"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="어댑터 저장 경로")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--thinking", action="store_true", help="Qwen3 thinking mode 활성화")
    parser.add_argument("--persona", default="직장인")
    parser.add_argument("--schema-version", default="v1", choices=["v1", "v2"])
    args = parser.parse_args()
    print(generate(args.adapter, args.prompt, args.max_new_tokens, args.thinking,
                   args.persona, args.schema_version))
