from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from trl import DPOConfig, DPOTrainer
from trl.trainer.utils import entropy_from_logits
from trl.models.utils import disable_gradient_checkpointing

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

    def compute_ref_log_probs(self, inputs):
        """Sequential half-batch ref forward to avoid [B,T,248K] fp32 OOM during precompute.

        TRL's default calls self.model (accelerate-wrapped) which auto-converts outputs to fp32,
        causing [2,T,248448]×4B ≈ 2.56 GiB allocation. This override calls the UNWRAPPED model
        directly (bf16 output) and processes chosen/rejected halves one at a time.
        """
        _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
        model_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
        model_kwargs["use_cache"] = False

        raw_model = self.accelerator.unwrap_model(self.model, keep_fp32_wrapper=False)
        input_ids = inputs["input_ids"]
        completion_mask = inputs["completion_mask"]
        half = input_ids.shape[0] // 2

        def _ref_half(start: int, end: int) -> torch.Tensor:
            half_kw = {k: v[start:end] if isinstance(v, torch.Tensor) else v
                       for k, v in model_kwargs.items()}
            with torch.no_grad(), disable_gradient_checkpointing(
                self.model, self.args.gradient_checkpointing_kwargs
            ):
                if hasattr(raw_model, "disable_adapter"):
                    with raw_model.disable_adapter():
                        out = raw_model(**half_kw)
                else:
                    out = raw_model(**half_kw)
            logits = out.logits[:, :-1, :]                      # [1, T-1, V] bf16
            shift_labels = input_ids[start:end, 1:]
            shift_cmask = completion_mask[start:end, 1:]
            per_tok = self._chunked_logps(logits, shift_labels)  # [1, T-1]
            del logits, out
            torch.cuda.empty_cache()
            per_tok[shift_cmask == 0] = 0.0
            return per_tok.sum(dim=1)                            # [1]

        ref_chosen_logps = _ref_half(0, half)
        ref_rejected_logps = _ref_half(half, 2 * half)
        return ref_chosen_logps, ref_rejected_logps

    def _compute_loss(self, model, inputs, return_outputs):  # noqa: ARG002
        """Detach-and-re-run: frees chosen GC graph before rejected forward to avoid OOM.

        Memory timeline (gradient_checkpointing=True, r=8):
          Step 1: chosen forward → chosen_logps_val (scalar leaf), chosen GC graph freed
          Step 2: rejected forward → rejected_logps (with graph, ~0 GC since already freed)
          Step 3: compute DPO gradient factors analytically (no_grad)
          Step 4: proxy backward for rejected (contributes to grad accumulation)
          Step 5: re-run chosen forward → proxy backward for chosen
          Step 6: return dummy loss (zero-grad leaf, Trainer's .backward() is no-op)
        """
        mode = "train" if self.model.training else "eval"
        is_train = model.training

        _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
        base_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
        base_kwargs["use_cache"] = False

        completion_mask = inputs["completion_mask"]    # [2, T]
        input_ids = inputs["input_ids"]                # [2, T]
        half = input_ids.shape[0] // 2                 # == 1 for batch_size=1

        assert self.precompute_ref_logps, (
            "MemEfficientDPOTrainer requires precompute_ref_log_probs=True"
        )
        ref_chosen_logps = inputs["ref_chosen_logps"].float()
        ref_rejected_logps = inputs["ref_rejected_logps"].float()

        # keep_fp32_wrapper=False strips accelerate's convert_to_fp32 hook from forward.
        # [1,T,248K] bf16=566MB; convert_to_fp32 doubles that to 1.13GB → OOM on 12GB.
        raw_model = self.accelerator.unwrap_model(model, keep_fp32_wrapper=False)

        def _forward_half(
            start: int, end: int, compute_entropy: bool = True
        ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
            half_kwargs = {k: v[start:end] if isinstance(v, torch.Tensor) else v
                           for k, v in base_kwargs.items()}
            out = raw_model(**half_kwargs)
            logits = out.logits[:, :-1, :]              # view [h, T-1, V] — NO .contiguous()
            shift_labels = input_ids[start:end, 1:]
            shift_cmask = completion_mask[start:end, 1:]
            per_tok_logps = self._chunked_logps(logits, shift_labels)
            # Free logits (566 MB) before entropy to avoid OOM.
            # Training with GC activations (~263 MB) alive leaves only ~44 MB headroom;
            # entropy chunks (128 × 248K × 4B = 128 MB) won't fit. Skip in train mode.
            if compute_entropy:
                ent = self._chunked_entropy(logits)
            else:
                ent = None
            del logits, out
            torch.cuda.empty_cache()
            per_tok_logps[shift_cmask == 0] = 0.0
            logps = per_tok_logps.sum(dim=1)            # [h], retains grad graph
            return logps, ent, shift_cmask

        loss_type = self.loss_types[0] if hasattr(self, "loss_types") and self.loss_types else "sigmoid"

        if not is_train:
            # Eval: simple sequential forward, no manual backward needed
            chosen_logps, chosen_entropy, chosen_cmask = _forward_half(0, half)
            rejected_logps, rejected_entropy, rejected_cmask = _forward_half(half, 2 * half)

            chosen_logratios = chosen_logps.detach() - ref_chosen_logps
            rejected_logratios = rejected_logps.detach() - ref_rejected_logps
            delta = chosen_logratios - rejected_logratios
            logps_chosen_metric = chosen_logps.detach()
            logps_rejected_metric = rejected_logps.detach()
            if loss_type == "sigmoid":
                loss = -F.logsigmoid(self.beta * delta).mean()
            elif loss_type == "ipo":
                c_avg = chosen_logratios / chosen_cmask.sum(1).clamp(1).float()
                r_avg = rejected_logratios / rejected_cmask.sum(1).clamp(1).float()
                loss = ((c_avg - r_avg - 1.0 / (2.0 * self.beta)) ** 2).mean()
            elif loss_type == "hinge":
                loss = torch.relu(1 - self.beta * delta).mean()
            else:
                raise NotImplementedError(f"loss_type '{loss_type}' not supported")
        else:
            # Train: detach-and-re-run to free chosen GC graph before rejected forward.
            # Entropy skipped (compute_entropy=False): GC activations (~263 MB) + logits
            # (~566 MB) leave <50 MB headroom; 128-tok entropy chunk needs 128 MB → OOM.
            # Step 1: chosen forward — get value, immediately detach + free GC graph
            chosen_logps_tmp, chosen_entropy, chosen_cmask = _forward_half(
                0, half, compute_entropy=False
            )
            chosen_val = chosen_logps_tmp.detach().clone()  # scalar values, no graph
            del chosen_logps_tmp
            torch.cuda.empty_cache()  # frees ~263 MB chosen GC activation graph

            # Step 2: rejected forward — now has enough headroom
            rejected_logps, rejected_entropy, rejected_cmask = _forward_half(
                half, 2 * half, compute_entropy=False
            )
            rejected_val = rejected_logps.detach().clone()

            # Step 3: compute DPO gradient factors analytically (no allocations)
            with torch.no_grad():
                chosen_logratios = chosen_val - ref_chosen_logps
                rejected_logratios = rejected_val - ref_rejected_logps
                delta = chosen_logratios - rejected_logratios

                if loss_type == "sigmoid":
                    # d/d(logps_chosen) of mean(-log σ(β·Δ)) = β/B · σ(-β·Δ)
                    # d/d(logps_rejected) = -β/B · σ(-β·Δ)
                    sigma_neg = torch.sigmoid(-self.beta * delta)           # [B]
                    B = delta.shape[0]
                    scale = 1.0 / max(self.args.gradient_accumulation_steps, 1)
                    g_chosen = scale * (self.beta / B) * sigma_neg          # chosen grad factor
                    g_rejected = scale * (-self.beta / B) * sigma_neg       # rejected grad factor
                    loss_val = -F.logsigmoid(self.beta * delta).mean()
                elif loss_type == "ipo":
                    c_avg = chosen_logratios / chosen_cmask.sum(1).clamp(1).float()
                    r_avg = rejected_logratios / rejected_cmask.sum(1).clamp(1).float()
                    residual = c_avg - r_avg - 1.0 / (2.0 * self.beta)
                    B = delta.shape[0]
                    scale = 1.0 / max(self.args.gradient_accumulation_steps, 1)
                    n_c = chosen_cmask.sum(1).clamp(1).float()
                    n_r = rejected_cmask.sum(1).clamp(1).float()
                    g_chosen = scale * (2.0 * residual / n_c / B)
                    g_rejected = scale * (-2.0 * residual / n_r / B)
                    loss_val = (residual ** 2).mean()
                elif loss_type == "hinge":
                    active = (self.beta * delta < 1).float()
                    B = delta.shape[0]
                    scale = 1.0 / max(self.args.gradient_accumulation_steps, 1)
                    g_chosen = scale * (-self.beta / B) * active
                    g_rejected = scale * (self.beta / B) * active
                    loss_val = torch.relu(1 - self.beta * delta).mean()
                else:
                    raise NotImplementedError(f"loss_type '{loss_type}' not supported")

            # Step 4: backward for rejected via proxy loss (manual gradient injection)
            proxy_rejected = (rejected_logps * g_rejected.detach()).sum()
            proxy_rejected.backward()
            del rejected_logps, proxy_rejected
            torch.cuda.empty_cache()

            # Step 5: re-run chosen forward + backward
            chosen_logps_fresh, _, _ = _forward_half(0, half)
            proxy_chosen = (chosen_logps_fresh * g_chosen.detach()).sum()
            proxy_chosen.backward()
            del chosen_logps_fresh, proxy_chosen
            torch.cuda.empty_cache()

            # Step 6: dummy loss — real value for logging, zero-grad leaf so Trainer's
            # .backward() call adds nothing to the already-accumulated gradients.
            dummy = sum(
                p.sum() * 0 for p in model.parameters()
                if p.requires_grad and p.grad is not None
            )
            loss = loss_val.detach() + dummy
            logps_chosen_metric = chosen_val
            logps_rejected_metric = rejected_val

        # --- metrics (logging only) ---
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()

        # entropy is None in train mode (skipped to avoid OOM; see _forward_half)
        if chosen_entropy is not None and rejected_entropy is not None:
            cat_entropy = torch.cat([chosen_entropy, rejected_entropy], dim=0)
            cat_cmask = torch.cat([chosen_cmask, rejected_cmask], dim=0)
            entropy_sum = self.accelerator.gather_for_metrics(
                (cat_entropy * cat_cmask).sum()
            ).sum()
            total_tokens = self.accelerator.gather_for_metrics(cat_cmask.sum()).sum()
            entropy_val = (entropy_sum / total_tokens).item() if total_tokens > 0 else 0.0
        else:
            entropy_val = 0.0
        self._metrics[mode]["entropy"].append(entropy_val)

        if mode == "train":
            n_tok = self.accelerator.gather_for_metrics(
                inputs["attention_mask"].sum()
            ).sum().item()
            self._total_train_tokens += n_tok
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        self._metrics[mode]["logits/chosen"].append(
            self.accelerator.gather_for_metrics(chosen_logratios.detach()).mean().item()
        )
        self._metrics[mode]["logits/rejected"].append(
            self.accelerator.gather_for_metrics(rejected_logratios.detach()).mean().item()
        )
        self._metrics[mode]["logps/chosen"].append(
            self.accelerator.gather_for_metrics(logps_chosen_metric).mean().item()
        )
        self._metrics[mode]["logps/rejected"].append(
            self.accelerator.gather_for_metrics(logps_rejected_metric).mean().item()
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


def _patch_qwen35_fast_path() -> None:
    """Force Qwen3.5 to use fla's chunk_gated_delta_rule even without causal_conv1d.

    is_fast_path_available requires all four symbols; causal_conv1d won't build
    without nvcc, but the model already has a torch fallback for causal_conv1d_fn.
    Patching to True lets fla handle the delta-rule (fixing the cudaErrorIllegalAddress
    bug in torch_chunk_gated_delta_rule) while the conv still uses the torch path.
    """
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as _qwen35
        if not _qwen35.is_fast_path_available:
            _qwen35.is_fast_path_available = True
            print("[patch] Qwen3.5 is_fast_path_available forced True (fla delta-rule active)")
    except Exception as e:
        print(f"[patch] Could not patch Qwen3.5 fast path: {e}")


def main(config_path: str) -> None:
    _patch_qwen35_fast_path()
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
