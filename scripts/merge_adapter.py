#!/usr/bin/env python
"""PEFT LoRA 어댑터 → 베이스 모델에 병합, 전체 모델 저장.

MLX 2-bit 변환을 위해서는 어댑터만이 아닌 full merged model이 필요하다.
(mlx_lm.convert는 HF 포맷의 전체 모델을 받아 quantize한다)

사용:
  uv run python scripts/merge_adapter.py \
      --adapter outputs/sft_q35_4b_v7 \
      --out outputs/sft_q35_4b_v7_merged

  # DPO 어댑터 병합 (SFT 위에 DPO 체인)
  uv run python scripts/merge_adapter.py \
      --adapter outputs/dpo_q35_4b_v5 \
      --out outputs/dpo_q35_4b_v5_merged

산출물:
  <out>/  — HF AutoModelForCausalLM 로드 가능한 전체 가중치
           (model.safetensors / pytorch_model.bin + tokenizer 파일)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _get_base_model(adapter_dir: Path) -> str:
    cfg_path = adapter_dir / "adapter_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        base = cfg.get("base_model_name_or_path")
        if base:
            return base
    return "Qwen/Qwen3.5-4B"


def merge(adapter_dir: Path, out_dir: Path) -> None:
    base_model_name = _get_base_model(adapter_dir)
    print(f"[merge] 베이스: {base_model_name}")
    print(f"[merge] 어댑터: {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=False)

    print("[merge] 베이스 모델 로드 (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=False,
    )

    print("[merge] LoRA 가중치 병합...")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[merge] 저장: {out_dir}")
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[완료] 파라미터 수: {total_params / 1e9:.2f}B → {out_dir}")
    print("\nMLX 2-bit 변환 (Mac에서 실행):")
    print(f"  python scripts/export_mlx_2bit.py --merged {out_dir} --out {out_dir}_mlx_2bit")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA 어댑터 병합 → 전체 모델 저장")
    parser.add_argument("--adapter", required=True, help="PEFT 어댑터 디렉토리")
    parser.add_argument("--out", required=True, help="병합 모델 출력 디렉토리")
    args = parser.parse_args()

    merge(Path(args.adapter), Path(args.out))


if __name__ == "__main__":
    main()
