from __future__ import annotations

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
