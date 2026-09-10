# examples/closed_loop_draw/viz.py
"""Canvas dump helpers (app layer; no heavy deps)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def canvas_to_u8(canvas: np.ndarray, sample: int = 0) -> np.ndarray:
    """NCHW float [0,1] → HxW uint8 grayscale (first channel of sample)."""
    x = np.asarray(canvas)
    if x.ndim != 4:
        raise ValueError(f"expected NCHW, got {x.shape}")
    img = np.clip(x[sample, 0], 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def save_canvas_png(path: str | Path, canvas: np.ndarray, sample: int = 0) -> Path:
    """Write a minimal grayscale PNG (no Pillow required)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = canvas_to_u8(canvas, sample=sample)
    _write_grayscale_png(path, img)
    return path


def save_canvas_npy(path: str | Path, canvas: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(canvas, dtype=np.float32))
    return path


def ascii_preview(canvas: np.ndarray, sample: int = 0, width: int = 28) -> str:
    """Tiny terminal preview using density characters."""
    img = canvas_to_u8(canvas, sample=sample).astype(np.float32) / 255.0
    h, w = img.shape
    # Downsample to ~width columns.
    cols = min(width, w)
    rows = max(1, int(round(h * (cols / w) * 0.5)))  # aspect for terminal cells
    ys = (np.linspace(0, h - 1, rows)).astype(int)
    xs = (np.linspace(0, w - 1, cols)).astype(int)
    small = img[ys][:, xs]
    chars = " .:-=+*#%@"
    lines = []
    for r in range(rows):
        line = "".join(chars[min(len(chars) - 1, int(v * (len(chars) - 1)))] for v in small[r])
        lines.append(line)
    return "\n".join(lines)


def _write_grayscale_png(path: Path, img: np.ndarray) -> None:
    """PNG encoder for 8-bit grayscale (IHDR + IDAT + IEND)."""
    import struct
    import zlib

    if img.dtype != np.uint8 or img.ndim != 2:
        raise ValueError("img must be HxW uint8")
    h, w = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)  # 8-bit grayscale
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", ihdr)
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)
