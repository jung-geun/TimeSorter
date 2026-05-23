from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from trl import DPOConfig, DPOTrainer

from .config import RunConfig
from .data.loader import load_dpo_dataset, _apply_system_to_dpo
from .device import detect
from .model import load_model_and_tokenizer


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _init_wandb(cfg: RunConfig) -> None:
    import wandb

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
    if cfg.schema_version == "v2":
        ds = _apply_system_to_dpo(ds, tokenizer, schema_version="v2")
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

    trainer = DPOTrainer(
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
