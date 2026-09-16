"""Device selection and autocast helpers (XPU > CUDA > CPU)."""

from __future__ import annotations

import contextlib
from typing import Iterator, Optional

import torch


def get_device(preferred: Optional[str] = None) -> str:
    """Return the best available device string."""
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def is_accelerator(device: str) -> bool:
    return device.startswith("xpu") or device.startswith("cuda")


@contextlib.contextmanager
def bf16_autocast(device: str, enabled: bool = True) -> Iterator[None]:
    """bfloat16 autocast on the selected accelerator (no-op elsewhere).

    SAM3 runs in bf16 in its reference implementation; on XPU this is also
    required to avoid materializing fp32 attention maps (OOM).
    """
    if enabled and device.startswith("xpu"):
        with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
            yield
    elif enabled and device.startswith("cuda"):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def synchronize(device: str) -> None:
    if device.startswith("xpu"):
        torch.xpu.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def empty_cache(device: str) -> None:
    if device.startswith("xpu"):
        torch.xpu.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()
