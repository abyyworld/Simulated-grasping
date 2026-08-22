"""Device selection that works the same on a CUDA laptop, an M-series Mac and CI."""

from __future__ import annotations

import os

import torch


def pick_device(preferred: str | None = None) -> torch.device:
    """Return the best available device.

    Order: explicit argument, ``SIMGRASP_DEVICE``, CUDA, Apple MPS, CPU. MPS is
    checked because the whole pipeline is meant to run on an M-series MacBook as
    well as on an NVIDIA laptop.
    """
    name = preferred or os.environ.get("SIMGRASP_DEVICE")
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device: torch.device):
    """Half precision on CUDA only; MPS autocast is still flaky and CPU gains little."""
    if device.type == "cuda":
        return torch.float16
    return None
