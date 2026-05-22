from __future__ import annotations

import platform
import warnings
from typing import TYPE_CHECKING

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from .device import DeviceProfile


_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_model_and_tokenizer(
    model_name: str,
    profile: DeviceProfile,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_4bit: bool = False,
    gradient_checkpointing: bool = False,
    sft_adapter_path: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    if use_4bit and not profile.supports_4bit:
        warnings.warn(
            f"use_4bit=True이지만 device={profile.device}, arch={platform.machine()}에서 "
            "bitsandbytes가 지원되지 않습니다. bf16 LoRA로 폴백합니다.",
            stacklevel=2,
        )
        use_4bit = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {
        "dtype": profile.dtype,
        "trust_remote_code": False,
        "attn_implementation": profile.attn_impl,
    }

    if use_4bit:
        from transformers import BitsAndBytesConfig

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs["quantization_config"] = bnb_cfg
        load_kwargs["device_map"] = "auto"
    elif profile.device == "cuda":
        # CUDA는 device_map으로 멀티-GPU 자동 배치 지원
        load_kwargs["device_map"] = "auto"
    # MPS/CPU는 device_map 미지원 → 로딩 후 .to(device)

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if profile.device == "mps" and not use_4bit:
        model = model.to("mps")

    if use_4bit:
        # QLoRA: gradient flow를 위해 input_require_grads 설정 + gradient checkpointing 통합
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if sft_adapter_path is not None:
        model = PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=True)
        print(f"[model] SFT 어댑터 로드: {sft_adapter_path}")
    else:
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=_LORA_TARGETS,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer
