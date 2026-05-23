from unittest.mock import patch

import torch

from timesorter.device import DeviceProfile, detect


def test_detect_returns_profile():
    profile = detect()
    assert isinstance(profile, DeviceProfile)
    assert profile.device in ("mps", "cuda", "cpu")
    assert profile.dtype in (torch.bfloat16, torch.float32)
    assert isinstance(profile.supports_4bit, bool)
    assert profile.attn_impl == "sdpa"


def test_cpu_fallback():
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        profile = detect()

    assert profile.device == "cpu"
    assert profile.dtype == torch.float32
    assert profile.supports_4bit is False


def test_cuda_x86_supports_4bit():
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("platform.machine", return_value="x86_64"),
    ):
        profile = detect()

    assert profile.device == "cuda"
    assert profile.supports_4bit is True


def test_cuda_arm64_no_4bit():
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("platform.machine", return_value="aarch64"),
    ):
        profile = detect()

    assert profile.device == "cuda"
    assert profile.supports_4bit is False
