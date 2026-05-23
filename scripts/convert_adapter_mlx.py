#!/usr/bin/env python
"""PEFT LoRA 어댑터 → MLX-LM 호환 포맷 변환.

PEFT 키: base_model.model.model.layers.N.xxx.lora_A.weight
MLX 키:  model.layers.N.xxx.lora_A.weight

사용:
  # SFT v2 어댑터 변환
  python scripts/convert_adapter_mlx.py \
      --adapter outputs/sft_rtx12g_4b_v2 \
      --out outputs/sft_rtx12g_4b_v2_mlx

  # DPO v2 어댑터 변환 (학습 완료 후)
  python scripts/convert_adapter_mlx.py \
      --adapter outputs/dpo_rtx12g_4b_v2 \
      --out outputs/dpo_rtx12g_4b_v2_mlx

산출물:
  <out>/adapters.safetensors   — MLX-LM이 --adapter-path로 읽는 가중치
  <out>/adapter_config.json    — lora rank/alpha 설정
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


_PREFIX = "base_model.model."


def convert(adapter_dir: Path, out_dir: Path) -> None:
    src = adapter_dir / "adapter_model.safetensors"
    if not src.exists():
        raise FileNotFoundError(f"어댑터 없음: {src}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 가중치 변환 ──────────────────────────────────────────────────────────
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(src), framework="pt", device="cpu") as f:
        for key in f.keys():
            new_key = key.removeprefix(_PREFIX) if key.startswith(_PREFIX) else key
            tensors[new_key] = f.get_tensor(key)

    print(f"[변환] {len(tensors)}개 텐서")
    print(f"  예시: {next(iter(tensors))}")

    save_file(tensors, str(out_dir / "adapters.safetensors"))
    print(f"[저장] {out_dir}/adapters.safetensors")

    # ── adapter_config.json 생성 ─────────────────────────────────────────────
    src_cfg_path = adapter_dir / "adapter_config.json"
    with open(src_cfg_path) as f:
        peft_cfg = json.load(f)

    mlx_cfg = {
        "lora_parameters": {
            "rank": peft_cfg.get("r", 16),
            "alpha": peft_cfg.get("lora_alpha", 32),
            "dropout": peft_cfg.get("lora_dropout", 0.0),
            "scale": peft_cfg.get("lora_alpha", 32) / peft_cfg.get("r", 16),
        },
        "num_layers": None,
    }
    with open(out_dir / "adapter_config.json", "w") as f:
        json.dump(mlx_cfg, f, indent=2, ensure_ascii=False)
    print(f"[저장] {out_dir}/adapter_config.json")

    # ── 토크나이저 파일 복사 (mlx_lm이 adapter 디렉토리에서 읽을 수 있도록) ──
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                  "chat_template.jinja", "merges.txt", "vocab.json", "added_tokens.json"]:
        src_f = adapter_dir / fname
        if src_f.exists():
            shutil.copy2(src_f, out_dir / fname)

    print(f"\n완료: {out_dir}")
    print("MacBook에서 실행:")
    print(f"  python scripts/mlx_infer.py --adapter {out_dir} --prompt '보고서 작성(내일 마감), 점심 약속, 메일 답장'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="PEFT 어댑터 디렉토리")
    parser.add_argument("--out", required=True, help="MLX 출력 디렉토리")
    args = parser.parse_args()

    convert(Path(args.adapter), Path(args.out))


if __name__ == "__main__":
    main()
