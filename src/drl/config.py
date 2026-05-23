from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class LoraArgs:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    use_4bit: bool = False


@dataclass
class RunConfig:
    model_name: str
    dataset: str
    output_dir: str
    max_samples: int | None = None
    max_prompt_len: int = 1024
    max_seq_length: int = 2048
    sft_adapter: str | None = None
    ko_ultrafeedback_n: int = 0
    schema_version: str = "v1"
    wandb_project: str = "drl-qwen3"
    wandb_run_name: str | None = None
    auto_batch: bool = False
    target_eff_batch: int = 32
    lora: LoraArgs = field(default_factory=LoraArgs)
    training_args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> RunConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        lora = LoraArgs(**data.pop("lora", {}))
        return cls(
            model_name=data["model_name"],
            dataset=data.get("dataset", "maywell/ko_Ultrafeedback_binarized"),
            output_dir=data.get("output_dir", "outputs/run"),
            max_samples=data.get("max_samples"),
            max_prompt_len=data.get("max_prompt_len", 1024),
            max_seq_length=data.get("max_seq_length", 2048),
            sft_adapter=data.get("sft_adapter"),
            ko_ultrafeedback_n=data.get("ko_ultrafeedback_n", 0),
            wandb_project=data.get("wandb_project", "drl-qwen3"),
            wandb_run_name=data.get("wandb_run_name"),
            auto_batch=data.get("auto_batch", False),
            target_eff_batch=data.get("target_eff_batch", 32),
            schema_version=data.get("schema_version", "v1"),
            lora=lora,
            training_args=data.get("training_args", {}),
        )
