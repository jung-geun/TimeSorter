#!/usr/bin/env python
"""TimeSorter 학습 어댑터(LoRA)를 HuggingFace 모델 Hub에 업로드.

업로드 대상 (Qwen3.5-4B 기반 QLoRA 어댑터):
  outputs/sft_q35_4b_v4  → {user}/timesorter-qwen3.5-4b-sft-v4
  outputs/dpo_q35_4b_v5  → {user}/timesorter-qwen3.5-4b-dpo-v5

사용:
  uv run python scripts/upload_hf_model.py
  uv run python scripts/upload_hf_model.py --only dpo
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

USER = "pieroot"
BASE = "Qwen/Qwen3.5-4B"

MODELS = {
    "sft": {
        "local": "outputs/sft_q35_4b_v4",
        "repo": f"{USER}/timesorter-qwen3.5-4b-sft-v4",
        "stage": "SFT v4",
        "data": "sft_v4_train (6,056행, curated)",
        "result": "held-out 통과율 90.0% (n=30)",
    },
    "dpo": {
        "local": "outputs/dpo_q35_4b_v5",
        "repo": f"{USER}/timesorter-qwen3.5-4b-dpo-v5",
        "stage": "SFT v4 + DPO v5",
        "data": "dpo_pairs_v5 (1,368쌍, on-policy 포함)",
        "result": "held-out 통과율 90.0% (n=30), reward_acc 98.9%",
    },
}

# 업로드 제외 (학습 체크포인트·옵티마이저 상태)
IGNORE = ["checkpoint-*", "*.bin", "optimizer*", "scheduler*", "rng_state*", "trainer_state*"]

CARD = """\
---
base_model: {base}
library_name: peft
license: apache-2.0
language: [ko]
tags: [lora, qlora, qwen3.5, scheduling, prioritization, korean, timesorter]
---

# TimeSorter — {stage} (Qwen3.5-4B QLoRA 어댑터)

한국어 할 일 목록을 4축(긴급도·중요도·의존성·시간 제약, 각 1–5점)으로 채점하고
우선순위를 결정하는 일정 정렬 비서. **{base}** 위에 학습한 LoRA 어댑터.

- **학습 단계**: {stage}
- **학습 데이터**: {data}
- **검증**: {result} (골격 규칙 자동 채점)
- **출력**: 4축 점수 JSON (`tasks`/`priority_order`/`scores`/`refusal_reason`)

## 사용법

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("{repo}")
base = AutoModelForCausalLM.from_pretrained("{base}", quantization_config=bnb,
                                            device_map="auto", dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "{repo}").eval()
```

> Qwen3.5는 추론 시 `enable_thinking=False` (chat template) 권장 — 깨끗한 JSON 출력.

학습 코드·데이터셋: https://github.com/jung-geun/TimeSorter
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["sft", "dpo"], help="하나만 업로드")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN 환경변수가 필요합니다.")
    api = HfApi(token=token)

    keys = [args.only] if args.only else list(MODELS)
    for k in keys:
        m = MODELS[k]
        local = Path(m["local"])
        if not (local / "adapter_model.safetensors").exists():
            print(f"  [건너뜀] {m['local']} 어댑터 없음")
            continue
        api.create_repo(m["repo"], repo_type="model", private=args.private, exist_ok=True)
        # 모델 카드 작성
        card = CARD.format(base=BASE, stage=m["stage"], data=m["data"],
                           result=m["result"], repo=m["repo"])
        (local / "README.md").write_text(card, encoding="utf-8")
        api.upload_folder(folder_path=str(local), repo_id=m["repo"], repo_type="model",
                          ignore_patterns=IGNORE)
        print(f"  [업로드] {m['repo']}")
        print(f"           https://huggingface.co/{m['repo']}")


if __name__ == "__main__":
    main()
