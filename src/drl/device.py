from __future__ import annotations

import os
import platform
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceProfile:
    device: str           # "mps" | "cuda" | "cpu"
    dtype: torch.dtype    # bf16 (mps/cuda), fp32 (cpu)
    supports_4bit: bool   # bitsandbytes QLoRA 가용 여부 (CUDA + x86_64만)
    attn_impl: str        # "sdpa" | "flash_attention_2"


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def detect() -> DeviceProfile:
    """현재 환경의 DeviceProfile 반환. CUDA > MPS > CPU 순으로 탐지."""
    if torch.cuda.is_available():
        arch = platform.machine().lower()
        # DGX Spark는 ARM64 — bitsandbytes ARM64 prebuilt wheel 미존재
        supports_4bit = arch in ("x86_64", "amd64")
        # flash-attn 설치 시 자동 활성화 (Blackwell에서 유의미한 속도/메모리 향상)
        attn_impl = "flash_attention_2" if _flash_attn_available() else "sdpa"
        return DeviceProfile(
            device="cuda",
            dtype=torch.bfloat16,
            supports_4bit=supports_4bit,
            attn_impl=attn_impl,
        )
    if torch.backends.mps.is_available():
        return DeviceProfile(
            device="mps",
            dtype=torch.bfloat16,
            supports_4bit=False,
            attn_impl="sdpa",
        )
    return DeviceProfile(
        device="cpu",
        dtype=torch.float32,
        supports_4bit=False,
        attn_impl="sdpa",
    )


def vram_per_gpu_gb() -> float:
    """현재 프로세스가 담당하는 GPU의 전체 VRAM(GB)을 반환."""
    if not torch.cuda.is_available():
        return 0.0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return torch.cuda.get_device_properties(local_rank).total_memory / (1024 ** 3)


# (vram_upper_gb, sft_bs_per_gpu, use_4bit)
# 4B 모델 bf16 + gradient_checkpointing + packing 기준
_VRAM_TABLE = [
    (14,  1, True),   # < 14 GB  : RTX 3060 12GB 등
    (22,  2, False),  # 14–22 GB : RTX 4070 Ti 16GB, RTX 3090 20GB 등
    (30,  4, False),  # 22–30 GB : RTX 4090 / RTX 3090 24GB
    (50,  8, False),  # 30–50 GB : A40 48GB 등
    (90, 16, False),  # 50–90 GB : A100 80GB
]
_VRAM_DEFAULT = (32, False)  # 90 GB 초과 (H100 등)


def _estimate_params_b(model_name: str) -> float:
    """모델 이름에서 파라미터 수(B)를 추정. 예: 'Qwen3.5-9B' → 9.0."""
    import re
    m = re.search(r"(\d+(?:\.\d+)?)[Bb]", model_name)
    return float(m.group(1)) if m else 4.0


def auto_batch_config(
    vram_gb: float,
    mode: str = "sft",
    target_eff_batch: int = 32,
    model_params_b: float = 4.0,
) -> dict:
    """VRAM, GPU 수, 모델 크기로부터 배치 크기·grad_accum·use_4bit를 자동 산출.

    Args:
        vram_gb: GPU 1장의 VRAM (GB).
        mode: "sft" 또는 "dpo". DPO는 절반 배치 사용.
        target_eff_batch: 유지할 유효 배치 크기 (기본 32).
        model_params_b: 모델 파라미터 수(B). 4B 기준으로 VRAM 임계값 스케일.

    Returns:
        per_device_train_batch_size, gradient_accumulation_steps, use_4bit 포함 dict.
    """
    # 4B 기준 테이블 — 모델이 클수록 유효 VRAM이 줄어드는 것으로 환산
    effective_vram = vram_gb * (4.0 / max(model_params_b, 0.1))

    bs, use_4bit = _VRAM_DEFAULT
    for threshold, b, q in _VRAM_TABLE:
        if effective_vram < threshold:
            bs, use_4bit = b, q
            break

    if mode == "dpo":
        bs = max(1, bs // 2)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    grad_accum = max(1, target_eff_batch // (bs * world_size))

    return {
        "per_device_train_batch_size": bs,
        "gradient_accumulation_steps": grad_accum,
        "use_4bit": use_4bit,
    }
