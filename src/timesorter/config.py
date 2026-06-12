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
    prompt_completion: bool = False   # TRL prompt-completion 포맷 (프롬프트 loss 마스킹)
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
            prompt_completion=data.get("prompt_completion", False),
            lora=lora,
            training_args=data.get("training_args", {}),
        )


def ensure_wandb_mode() -> None:
    """WANDB_API_KEY가 없으면 오프라인 모드로 전환 — 로컬 wandb/ 디렉토리에만 기록.

    원격 연결 없이도 학습 메트릭이 유실되지 않는다. 이후 `wandb sync wandb/offline-run-*`
    으로 원격 업로드 가능. (README '학습 로깅' 참고)
    """
    import os
    from pathlib import Path
    has_key = bool(os.environ.get("WANDB_API_KEY"))
    has_netrc = any((Path.home() / f).exists() for f in (".netrc", "_netrc"))
    if not (has_key or has_netrc):
        os.environ.setdefault("WANDB_MODE", "offline")
        print("[wandb] API 키 없음 — 오프라인 모드 (로컬 wandb/ 기록)")
