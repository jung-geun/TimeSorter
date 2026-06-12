from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from trl import DPOConfig, DPOTrainer
from trl.trainer.utils import entropy_from_logits

from .config import RunConfig
from .data.loader import load_dpo_dataset, _apply_system_to_dpo
from .device import detect
from .model import load_model_and_tokenizer


class MemEfficientDPOTrainer(DPOTrainer):
    """chosen/rejected를 [1,T] 순차 처리하여 [2,T,248K vocab] logits OOM 방지.

    Qwen3.5 vocab 248K로 인해 [2,T,V] logits = 1.1+ GiB → 12GB GPU OOM.
    각 forward를 [1,T,V] = ~0.56 GiB로 분리, 청킹된 log_softmax로 further reduce.

    한계: sigmoid/ipo/hinge loss만 지원, precompute_ref_log_probs=True 필수, ld_alpha=None.
    """

    _LOGP_CHUNK = 128  # selective_log_softmax 대신 사용할 청크 크기 (토큰 단위)

    @staticmethod
    def _chunked_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """[B, T, V] logits에서 labels 위치의 log-prob을 청킹하여 OOM 없이 계산.

        standard selective_log_softmax([B,T,V])는 [T,V] row마다 0.56GB 임시 할당 → OOM.
        이 함수는 chunk_size 토큰씩 처리해 임시 할당을 ~64MB 이하로 제한.
        """
        B, T, V = logits.shape
        chunk = MemEfficientDPOTrainer._LOGP_CHUNK
        per_token_logps = torch.zeros(B, T, dtype=torch.float32, device=logits.device)
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            c_logits = logits[:, start:end, :].float()         # [B, c, V]
            c_logps = F.log_softmax(c_logits, dim=-1)          # [B, c, V]
            c_labels = labels[:, start:end].clamp(min=0)       # [B, c]
            gathered = c_logps.gather(-1, c_labels.unsqueeze(-1)).squeeze(-1)  # [B, c]
            per_token_logps[:, start:end] = gathered
            del c_logits, c_logps, gathered
        return per_token_logps

    @staticmethod
    def _chunked_entropy(logits: torch.Tensor) -> torch.Tensor:
        """[B, T, V] logits → entropy [B, T], 청킹으로 peak 메모리 64MB 이하."""
        B, T, V = logits.shape
        chunk = MemEfficientDPOTrainer._LOGP_CHUNK
        entropy = torch.zeros(B, T, dtype=torch.float32, device=logits.device)
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            c = logits[:, start:end, :].float()
            p = F.softmax(c, dim=-1)
            lp = F.log_softmax(c, dim=-1)
            entropy[:, start:end] = -(p * lp).sum(dim=-1)
            del c, p, lp
        return entropy

    def _compute_loss(self, model, inputs, return_outputs):  # noqa: ARG002
        mode = "train" if self.model.training else "eval"

        _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
        base_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
        base_kwargs["use_cache"] = False

        completion_mask = inputs["completion_mask"]    # [2, T]
        input_ids = inputs["input_ids"]                # [2, T]
        half = input_ids.shape[0] // 2                 # == 1 for batch_size=1

        def _forward_half(start: int, end: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """[start:end] half forward → (logps [h,T-1], entropy [h,T-1], shift_cmask [h,T-1])."""
            half_kwargs = {k: v[start:end] if isinstance(v, torch.Tensor) else v
                           for k, v in base_kwargs.items()}
            out = model(**half_kwargs)
            logits = out.logits[:, :-1, :]              # view [h, T-1, V] — NO .contiguous()
            shift_labels = input_ids[start:end, 1:]     # [h, T-1]
            shift_cmask = completion_mask[start:end, 1:]  # [h, T-1]

            per_tok_logps = self._chunked_logps(logits, shift_labels)   # [h, T-1], float32
            ent = self._chunked_entropy(logits)                         # [h, T-1], float32
            del logits, out
            torch.cuda.empty_cache()
            per_tok_logps[shift_cmask == 0] = 0.0
            logps = per_tok_logps.sum(dim=1)            # [h]
            return logps, ent, shift_cmask

        # --- chosen forward ---
        chosen_logps, chosen_entropy, chosen_cmask = _forward_half(0, half)
        # --- rejected forward ---
        rejected_logps, rejected_entropy, rejected_cmask = _forward_half(half, 2 * half)

        assert self.precompute_ref_logps, (
            "MemEfficientDPOTrainer requires precompute_ref_log_probs=True"
        )
        ref_chosen_logps = inputs["ref_chosen_logps"].to(chosen_logps)
        ref_rejected_logps = inputs["ref_rejected_logps"].to(rejected_logps)

        # --- DPO loss (sigmoid; ipo/hinge also handled) ---
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        delta = chosen_logratios - rejected_logratios

        loss_type = self.loss_types[0] if hasattr(self, "loss_types") and self.loss_types else "sigmoid"
        if loss_type == "sigmoid":
            per_seq_loss = -F.logsigmoid(self.beta * delta)
        elif loss_type == "ipo":
            chosen_avg = chosen_logratios / chosen_cmask.sum(dim=1).clamp(min=1).float()
            rejected_avg = rejected_logratios / rejected_cmask.sum(dim=1).clamp(min=1).float()
            per_seq_loss = (chosen_avg - rejected_avg - 1.0 / (2.0 * self.beta)) ** 2
        elif loss_type == "hinge":
            per_seq_loss = torch.relu(1 - self.beta * delta)
        else:
            raise NotImplementedError(
                f"MemEfficientDPOTrainer: loss_type='{loss_type}' not supported. "
                "Use sigmoid/ipo/hinge or set use_liger_kernel=false + precompute_ref_log_probs=true."
            )

        loss = per_seq_loss.mean()

        # --- metrics (logging only) ---
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()

        # entropy
        cat_entropy = torch.cat([chosen_entropy, rejected_entropy], dim=0)   # [2, T-1]
        cat_cmask = torch.cat([chosen_cmask, rejected_cmask], dim=0)          # [2, T-1]
        entropy_sum = self.accelerator.gather_for_metrics(
            (cat_entropy * cat_cmask).sum()
        ).sum()
        total_tokens = self.accelerator.gather_for_metrics(cat_cmask.sum()).sum()
        entropy_val = (entropy_sum / total_tokens).item() if total_tokens > 0 else 0.0
        self._metrics[mode]["entropy"].append(entropy_val)

        # token count
        if mode == "train":
            n_tok = self.accelerator.gather_for_metrics(
                inputs["attention_mask"].sum()
            ).sum().item()
            self._total_train_tokens += n_tok
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # logits/logps/rewards
        self._metrics[mode]["logits/chosen"].append(
            self.accelerator.gather_for_metrics(chosen_logratios.detach()).mean().item()
        )
        self._metrics[mode]["logits/rejected"].append(
            self.accelerator.gather_for_metrics(rejected_logratios.detach()).mean().item()
        )
        self._metrics[mode]["logps/chosen"].append(
            self.accelerator.gather_for_metrics(chosen_logps.detach()).mean().item()
        )
        self._metrics[mode]["logps/rejected"].append(
            self.accelerator.gather_for_metrics(rejected_logps.detach()).mean().item()
        )
        self._metrics[mode]["rewards/chosen"].append(
            self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        )
        self._metrics[mode]["rewards/rejected"].append(
            self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        )
        reward_acc = (chosen_rewards > rejected_rewards).float()
        self._metrics[mode]["rewards/accuracies"].append(
            self.accelerator.gather_for_metrics(reward_acc).mean().item()
        )
        margin = chosen_rewards - rejected_rewards
        self._metrics[mode]["rewards/margins"].append(
            self.accelerator.gather_for_metrics(margin).mean().item()
        )

        return loss


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _init_wandb(cfg: RunConfig) -> None:
    import wandb
    from .config import ensure_wandb_mode
    ensure_wandb_mode()
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        config={
            "model_name": cfg.model_name,
            "dataset": cfg.dataset,
            "lora": asdict(cfg.lora),
            **{k: v for k, v in cfg.training_args.items() if k != "report_to"},
        },
    )


def main(config_path: str) -> None:
    _load_dotenv()

    cfg = RunConfig.from_yaml(config_path)
    profile = detect()
    print(f"[device] {profile.device} | dtype={profile.dtype} | 4bit={profile.supports_4bit}")

    _auto: dict = {}
    if cfg.auto_batch and profile.device == "cuda":
        from .device import _estimate_params_b, auto_batch_config, vram_per_gpu_gb
        _vram = vram_per_gpu_gb()
        _params_b = _estimate_params_b(cfg.model_name)
        _auto = auto_batch_config(_vram, mode="dpo", target_eff_batch=cfg.target_eff_batch, model_params_b=_params_b)
        cfg.lora.use_4bit = _auto["use_4bit"]
        import os
        world = int(os.environ.get("WORLD_SIZE", 1))
        print(
            f"[auto-batch] VRAM={_vram:.1f}GB × {world}GPU → "
            f"bs={_auto['per_device_train_batch_size']}, "
            f"grad_accum={_auto['gradient_accumulation_steps']}, "
            f"4bit={_auto['use_4bit']} "
            f"(eff_batch={_auto['per_device_train_batch_size'] * world * _auto['gradient_accumulation_steps']})"
        )

    model, tokenizer = load_model_and_tokenizer(
        model_name=cfg.model_name,
        profile=profile,
        lora_r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        use_4bit=cfg.lora.use_4bit,
        gradient_checkpointing=cfg.training_args.get("gradient_checkpointing", False),
        sft_adapter_path=cfg.sft_adapter,
    )

    max_length = cfg.training_args.get("max_length", 2048)
    max_response_len = max_length - cfg.max_prompt_len

    ds = load_dpo_dataset(
        name=cfg.dataset,
        tokenizer=tokenizer,
        max_samples=cfg.max_samples,
        max_prompt_len=cfg.max_prompt_len,
        max_response_len=max_response_len,
    )
    if cfg.schema_version in ("v2", "v3"):
        ds = _apply_system_to_dpo(ds, tokenizer, schema_version=cfg.schema_version)
    print(f"[data] {len(ds)}개 샘플 로드: {cfg.dataset}")

    # bf16=True는 CUDA mixed-precision 전용.
    # MPS는 torch_dtype=bfloat16으로 모델을 로딩해 처리하므로 Trainer 플래그는 False.
    use_bf16 = profile.device == "cuda"
    training_kwargs: dict = {
        "output_dir": cfg.output_dir,
        "bf16": use_bf16,
        "fp16": False,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "remove_unused_columns": False,
    }
    if _auto:
        training_kwargs["per_device_train_batch_size"] = _auto["per_device_train_batch_size"]
        training_kwargs["gradient_accumulation_steps"] = _auto["gradient_accumulation_steps"]
    training_kwargs.update(cfg.training_args)  # yaml 명시값이 auto 값을 덮어씀

    if training_kwargs.get("report_to") == "wandb":
        _init_wandb(cfg)

    dpo_cfg = DPOConfig(**training_kwargs)

    # MemEfficientDPOTrainer: [2,T,248K vocab] logits OOM → chosen/rejected 순차 [1,T] 처리
    trainer = MemEfficientDPOTrainer(
        model=model,
        ref_model=None,  # PEFT adapter-disable 트릭으로 메모리 절감
        args=dpo_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    trainer.train()

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    print(f"[done] 저장: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML 설정 파일 경로")
    args = parser.parse_args()
    main(args.config)
