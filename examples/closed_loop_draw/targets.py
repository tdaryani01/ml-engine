# examples/closed_loop_draw/targets.py
"""Simple procedural target shapes for the draw demo."""
from __future__ import annotations

import numpy as np


def target_circle(
    *,
    batch_size: int = 1,
    height: int = 28,
    width: int = 28,
    channels: int = 1,
    radius: float = 0.55,
    edge_soft: float = 0.08,
) -> np.ndarray:
    """Filled soft disk target in NCHW [0,1], coords in [-1,1]."""
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, height, dtype=np.float32),
        np.linspace(-1.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    r = np.sqrt(xx * xx + yy * yy)
    disk = (1.0 / (1.0 + np.exp((r - radius) / edge_soft))).astype(np.float32)
    img = disk[None, None, :, :]
    if channels > 1:
        img = np.repeat(img, channels, axis=1)
    return np.repeat(img, batch_size, axis=0)
