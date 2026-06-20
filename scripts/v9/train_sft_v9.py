#!/usr/bin/env python
"""v9 SFT 학습 (RTX 3080 Ti 12GB) — liger fused-linear-CE로 긴 JSON 시퀀스 수용.

문제: v9 출력 JSON이 길어 full seq p50=3149·p90=4059·max 5211 tok. 표준 SFTTrainer는
[1,seq,248K] logits를 materialize → 12GB OOM. liger arch 패치는 Qwen3.5(qwen3_5) 미지원이나,
LigerFusedLinearCrossEntropyLoss(hidden→loss, logits 미생성)는 아키텍처 무관하게 적용 가능.

전략: decoder(get_decoder)로 hidden state만 계산 → lm_head.weight(4bit면 dequant) + liger
fused CE. logits 텐서를 만들지 않아 seq 4096도 12GB에서 학습 가능.

길이 ≤ max_seq 행만 사용(truncation 금지 — JSON 깨짐 방지).

사용:
  PROBE=1 uv run python scripts/v9/train_sft_v9.py --config configs/sft_rtx12g_q35_4b_v9.yaml  # 3스텝 프로브
  uv run python scripts/v9/train_sft_v9.py --config configs/sft_rtx12g_q35_4b_v9.yaml           # 전체
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, "src")
from timesorter.config import RunConfig  # noqa: E402
from timesorter.data.scheduler import _to_chatml, load_scheduler_dataset  # noqa: E402
from timesorter.device import detect  # noqa: E402
from timesorter.model import load_model_and_tokenizer  # noqa: E402


def _stop_serving_container() -> None:
    try:
        out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                             capture_output=True, text=True, timeout=10)
        if "timesorter-serve" in out.stdout:
            subprocess.run(["docker", "stop", "timesorter-serve"], timeout=30)
            print("[GPU] timesorter-serve 중지")
    except Exception:
        pass


def _dequant_weight(weight: torch.Tensor) -> torch.Tensor:
    """lm_head.weight가 bnb 4bit면 bf16으로 dequantize, 아니면 그대로."""
    try:
        import bitsandbytes as bnb
        if isinstance(weight, bnb.nn.Params4bit):
            return bnb.functional.dequantize_4bit(
                weight.data, weight.quant_state).to(torch.bfloat16)
    except Exception:
        pass
    return weight


class MemEfficientSFTTrainer(SFTTrainer):
    """compute_loss를 liger fused-linear-CE로 대체 — [seq,248K] logits 미생성."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        labels = inputs.pop("labels")
        unwrapped = self.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
        decoder = unwrapped.get_decoder()
        lm_head = unwrapped.get_output_embeddings()

        dec_out = decoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            use_cache=False,
        )
        hidden = dec_out.last_hidden_state            # [B, T, H]
        shift_hidden = hidden[:, :-1, :].reshape(-1, hidden.size(-1))
        shift_labels = labels[:, 1:].reshape(-1).to(shift_hidden.device)

        weight = _dequant_weight(lm_head.weight)
        bias = getattr(lm_head, "bias", None)
        loss_fn = LigerFusedLinearCrossEntropyLoss()
        loss = loss_fn(weight, shift_hidden.float() if weight.dtype == torch.float32
                       else shift_hidden, shift_labels, bias)
        return (loss, dec_out) if return_outputs else loss


def build_v9_dataset(parquet: str, tokenizer, max_seq: int):
    """v9 텍스트 prompt-completion 데이터셋 빌드.

    **enable_thinking=False**로 닫힌 think 블록(`<think>\\n\\n</think>`) 사용 — Qwen3.5가
    추론 모드로 들어가 사고 텍스트를 내는 문제 방지(직접 JSON 출력 학습).
    프롬프트를 미리 문자열로 렌더 → trl이 템플릿 재적용 안 함(마스킹 정합).
    """
    import pandas as pd
    from datasets import Dataset
    from timesorter.data.schema import system_prompt_for
    df = pd.read_parquet(parquet).reset_index(drop=True)
    sysp = system_prompt_for("v9")
    rows, kept = [], 0
    for _, r in df.iterrows():
        msgs = [{"role": "system", "content": sysp},
                {"role": "user", "content": str(r["prompt"])}]
        pstr = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        comp = str(r["chosen"])
        n = (len(tokenizer(pstr, add_special_tokens=False)["input_ids"])
             + len(tokenizer(comp, add_special_tokens=False)["input_ids"]) + 1)
        if n <= max_seq - 16:
            rows.append({"prompt": pstr, "completion": comp})
            kept += 1
    print(f"[v9 data] enable_thinking=False · 길이 ≤{max_seq}: {kept}/{len(df)}행 "
          f"({100*kept/len(df):.1f}%)")
    return Dataset.from_list(rows)


def main(config_path: str) -> None:
    from dotenv import load_dotenv
    load_dotenv()
    _stop_serving_container()
    probe = os.environ.get("PROBE") == "1"

    cfg = RunConfig.from_yaml(config_path)
    profile = detect()
    print(f"[device] {profile.device} | dtype={profile.dtype} | 4bit={profile.supports_4bit}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=cfg.model_name, profile=profile,
        lora_r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
        use_4bit=cfg.lora.use_4bit,
        gradient_checkpointing=cfg.training_args.get("gradient_checkpointing", False),
        sft_adapter_path=cfg.sft_adapter,
    )

    ds = build_v9_dataset(cfg.dataset, tokenizer, cfg.max_seq_length)
    if cfg.max_samples:
        ds = ds.select(range(min(cfg.max_samples, len(ds))))
    print(f"[data] {len(ds)}개 샘플")

    training_kwargs: dict = {
        "output_dir": cfg.output_dir, "bf16": profile.device == "cuda", "fp16": False,
        "max_length": cfg.max_seq_length, "packing": False, "logging_steps": 1,
        "report_to": "none", "remove_unused_columns": False,
    }
    training_kwargs.update(cfg.training_args)
    if probe:
        training_kwargs.update({"max_steps": 3, "save_strategy": "no", "report_to": "none",
                                "num_train_epochs": 1})
        print("[PROBE] max_steps=3 — OOM/VRAM 확인용")
    if training_kwargs.get("report_to") == "wandb" and not probe:
        import wandb
        from timesorter.config import ensure_wandb_mode
        ensure_wandb_mode()
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name)

    trainer = MemEfficientSFTTrainer(
        model=model, processing_class=tokenizer,
        args=SFTConfig(**training_kwargs), train_dataset=ds,
    )
    trainer.train()
    if torch.cuda.is_available():
        print(f"[VRAM] peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    if not probe:
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(out))
        print(f"[done] SFT v9 어댑터 저장: {out}")
    else:
        print("[PROBE] 통과 — 전체 학습 가능")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
