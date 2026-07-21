#!/usr/bin/env python
"""v9 DPO 학습 (RTX 12GB) — liger fused-CE로 긴 시퀀스(~4096) logp 계산.

문제: v9 DPO는 prompt(~1576)+output(~1600)=full p50 3165·p90 4133 tok. 기존
MemEfficientDPOTrainer는 forward마다 [1,T,248K] logits를 만들어(2GB@4096) 12GB OOM.

전략: decoder(get_decoder)로 hidden만 → **LigerFusedLinearCrossEntropyLoss(reduction='sum')**
로 완성부 logp 합을 직접 계산(logits 미생성). chosen/rejected 순차 forward + 표준 autograd.
sft_q35_4b_v9 어댑터에서 이어서 선호학습. precompute_ref_log_probs=True 필수.

사용:
  uv run python scripts/v9/train_dpo_v9.py --config configs/dpo_rtx12g_q35_4b_v9.yaml
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, "src")
from timesorter.config import RunConfig  # noqa: E402
from timesorter.data.loader import _apply_system_to_dpo  # noqa: E402
from timesorter.device import detect  # noqa: E402
from timesorter.model import load_model_and_tokenizer  # noqa: E402


def _stop_serving() -> None:
    try:
        out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                             capture_output=True, text=True, timeout=10)
        if "timesorter-serve" in out.stdout:
            subprocess.run(["docker", "stop", "timesorter-serve"], timeout=30)
            print("[GPU] timesorter-serve 중지")
    except Exception:
        pass


def _dequant(weight):
    try:
        import bitsandbytes as bnb
        if isinstance(weight, bnb.nn.Params4bit):
            return bnb.functional.dequantize_4bit(weight.data, weight.quant_state).to(torch.bfloat16)
    except Exception:
        pass
    return weight


class LigerDPOTrainer(DPOTrainer):
    """liger fused-CE로 logp 계산 — [T,248K] logits 미생성으로 4096 시퀀스 12GB 가능."""

    def _seq_logp(self, raw_model, ids, attn, cmask, *, no_grad: bool, disable_adapter: bool):
        """완성부(cmask=1) 토큰 logp 합 [B] 반환. liger fused CE(reduction=sum)=-Σlogp."""
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        from trl.models.utils import disable_gradient_checkpointing
        decoder = raw_model.get_decoder()
        lm_head = raw_model.get_output_embeddings()
        weight = _dequant(lm_head.weight)
        ce = LigerFusedLinearCrossEntropyLoss(reduction="sum")

        def _run():
            hidden = decoder(input_ids=ids, attention_mask=attn,
                             use_cache=False).last_hidden_state  # [B,T,H]
            B = hidden.shape[0]
            outs = []
            for b in range(B):  # 행별로(B=1 보통) logp 합
                sh = hidden[b, :-1, :]                       # [T-1,H]
                lab = ids[b, 1:].clone()
                lab[cmask[b, 1:] == 0] = -100
                ce_sum = ce(weight, sh, lab)                 # 스칼라 = -Σlogp(완성부)
                outs.append(-ce_sum)
            return torch.stack(outs)                          # [B]

        if no_grad:
            with torch.no_grad():
                if disable_adapter and hasattr(raw_model, "disable_adapter"):
                    with raw_model.disable_adapter(), disable_gradient_checkpointing(
                            self.model, self.args.gradient_checkpointing_kwargs):
                        return _run()
                return _run()
        return _run()

    def compute_ref_log_probs(self, inputs):
        raw = self.accelerator.unwrap_model(self.model, keep_fp32_wrapper=False)
        ids, attn, cm = inputs["input_ids"], inputs["attention_mask"], inputs["completion_mask"]
        half = ids.shape[0] // 2
        rc = self._seq_logp(raw, ids[:half], attn[:half], cm[:half], no_grad=True, disable_adapter=True)
        rr = self._seq_logp(raw, ids[half:], attn[half:], cm[half:], no_grad=True, disable_adapter=True)
        return rc, rr

    def _compute_loss(self, model, inputs, return_outputs=False):  # noqa: ARG002
        raw = self.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
        ids, attn, cm = inputs["input_ids"], inputs["attention_mask"], inputs["completion_mask"]
        half = ids.shape[0] // 2
        ref_c = inputs["ref_chosen_logps"].float()
        ref_r = inputs["ref_rejected_logps"].float()

        logp_c = self._seq_logp(raw, ids[:half], attn[:half], cm[:half], no_grad=False, disable_adapter=False)
        logp_r = self._seq_logp(raw, ids[half:], attn[half:], cm[half:], no_grad=False, disable_adapter=False)

        chosen_logratios = logp_c - ref_c
        rejected_logratios = logp_r - ref_r
        delta = chosen_logratios - rejected_logratios
        loss = -F.logsigmoid(self.beta * delta).mean()

        mode = "train" if model.training else "eval"
        with torch.no_grad():
            cr = self.beta * chosen_logratios
            rr = self.beta * rejected_logratios
            self._metrics[mode]["rewards/chosen"].append(cr.mean().item())
            self._metrics[mode]["rewards/rejected"].append(rr.mean().item())
            self._metrics[mode]["rewards/accuracies"].append((cr > rr).float().mean().item())
            self._metrics[mode]["rewards/margins"].append((cr - rr).mean().item())
            self._metrics[mode]["logps/chosen"].append(logp_c.mean().item())
            self._metrics[mode]["logps/rejected"].append(logp_r.mean().item())
        return loss


def main(config_path: str) -> None:
    from dotenv import load_dotenv
    load_dotenv()
    _stop_serving()
    cfg = RunConfig.from_yaml(config_path)
    profile = detect()
    print(f"[device] {profile.device} | 4bit={profile.supports_4bit}")

    from timesorter.train_dpo import _patch_qwen35_fast_path
    _patch_qwen35_fast_path()
    model, tok = load_model_and_tokenizer(
        model_name=cfg.model_name, profile=profile,
        lora_r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
        use_4bit=cfg.lora.use_4bit,
        gradient_checkpointing=cfg.training_args.get("gradient_checkpointing", False),
        sft_adapter_path=cfg.sft_adapter)

    # 데이터: parquet → system 적용(v9 enable_thinking=False) → 길이 필터
    import pandas as pd
    from datasets import Dataset
    df = pd.read_parquet(cfg.dataset)
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds = _apply_system_to_dpo(ds, tok, schema_version="v9")
    mp = cfg.max_prompt_len
    ml = cfg.training_args["max_length"]

    def _fit(r):
        p = len(tok(r["prompt"], add_special_tokens=False)["input_ids"])
        c = len(tok(r["chosen"], add_special_tokens=False)["input_ids"])
        j = len(tok(r["rejected"], add_special_tokens=False)["input_ids"])
        return p <= mp and max(c, j) <= ml - mp
    before = len(ds)
    ds = ds.filter(_fit, desc="길이 필터")
    if os.environ.get("PROBE") == "1":
        ds = ds.select(range(min(40, len(ds))))  # 프로브: ref precompute 빠르게
    elif cfg.max_samples:
        ds = ds.select(range(min(cfg.max_samples, len(ds))))
    print(f"[data] DPO {len(ds)}/{before}쌍 (prompt≤{mp}, resp≤{ml-mp})")

    targs = dict(output_dir=cfg.output_dir, beta=cfg.training_args.get("beta", 0.1),
                 bf16=profile.device == "cuda", report_to="none", logging_steps=5,
                 remove_unused_columns=False)
    # 주의: trl 1.x DPOConfig에 max_prompt_length 없음 — 길이는 위 _fit 사전필터로 처리.
    for k in ("per_device_train_batch_size", "gradient_accumulation_steps", "num_train_epochs",
              "learning_rate", "gradient_checkpointing", "warmup_ratio", "lr_scheduler_type",
              "save_strategy", "save_steps", "save_total_limit", "max_length",
              "precompute_ref_log_probs", "precompute_ref_batch_size", "optim", "dataloader_num_workers"):
        if k in cfg.training_args:
            targs[k] = cfg.training_args[k]
    targs["precompute_ref_log_probs"] = True

    if os.environ.get("PROBE") == "1":
        targs.update(max_steps=3, save_strategy="no", num_train_epochs=1)
        print("[PROBE] max_steps=3")

    trainer = LigerDPOTrainer(model=model, args=DPOConfig(**targs),
                              processing_class=tok, train_dataset=ds)
    trainer.train()
    if torch.cuda.is_available():
        print(f"[VRAM] peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    if os.environ.get("PROBE") != "1":
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        trainer.save_model(cfg.output_dir)
        print(f"[done] DPO v9 저장: {cfg.output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
