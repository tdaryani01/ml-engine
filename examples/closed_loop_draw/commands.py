# examples/closed_loop_draw/commands.py
"""Command vocabulary + stock target images (app layer)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from examples.closed_loop_draw.viz import save_canvas_png

ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "commands"

# Canonical command id → display name. Aliases resolve to these ids.
COMMANDS: dict[int, str] = {
    0: "circle",
    1: "line",
    2: "square",
    3: "cross",
    4: "ring",
    5: "sketch",
}

ALIASES: dict[str, int] = {
    "circle": 0,
    "draw circle": 0,
    "disk": 0,
    "filled circle": 0,
    "line": 1,
    "draw line": 1,
    "square": 2,
    "draw square": 2,
    "box": 2,
    "cross": 3,
    "draw cross": 3,
    "x": 3,
    "plus": 3,
    "ring": 4,
    "draw ring": 4,
    "hollow circle": 4,
    "circle outline": 4,
    "sketch": 5,
    "custom": 5,
    "draw sketch": 5,
    "my drawing": 5,
}

# Commands that use procedural/stock PNGs (excludes freehand sketch).
STOCK_COMMAND_IDS: tuple[int, ...] = tuple(
    cid for cid, name in COMMANDS.items() if name != "sketch"
)


def assets_dir() -> Path:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    return ASSETS_DIR


def stock_path(command_id: int) -> Path:
    name = COMMANDS.get(int(command_id))
    if name is None:
        raise KeyError(f"unknown command_id={command_id}")
    return assets_dir() / f"{name}.png"


def resolve_command(text: str) -> tuple[int, str]:
    """Parse user text → (command_id, canonical_name)."""
    key = " ".join(str(text).strip().lower().split())
    if not key:
        raise ValueError("empty command")
    if key.isdigit():
        cid = int(key)
        if cid not in COMMANDS:
            raise ValueError(f"command id {cid} not in {list(COMMANDS)}")
        return cid, COMMANDS[cid]
    if key in ALIASES:
        cid = ALIASES[key]
        return cid, COMMANDS[cid]
    # Prefix match: "draw cir" → circle
    for alias, cid in ALIASES.items():
        if alias.startswith(key) or key in alias:
            return cid, COMMANDS[cid]
    known = ", ".join(COMMANDS[i] for i in sorted(COMMANDS))
    raise ValueError(f"unknown command {text!r}; try: {known}")


def _render_stock_hw(name: str, height: int, width: int) -> np.ndarray:
    """Procedural stock silhouette in [0,1], shape (H, W)."""
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, height, dtype=np.float32),
        np.linspace(-1.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    img = np.zeros((height, width), dtype=np.float32)
    if name == "circle":
        # Filled disk — matches "draw a circle".
        r = np.sqrt(xx * xx + yy * yy)
        soft = 0.08
        img = (1.0 / (1.0 + np.exp((r - 0.55) / soft))).astype(np.float32)
    elif name == "ring":
        # Hollow ring outline.
        r = np.sqrt(xx * xx + yy * yy)
        img = np.exp(-((r - 0.55) ** 2) / (2 * 0.07**2)).astype(np.float32)
    elif name == "sketch":
        # Placeholder only — real target comes from the interactive sketch pad.
        img = np.zeros((height, width), dtype=np.float32)
    elif name == "line":
        # Horizontal stroke through center.
        dist = np.abs(yy)
        along = (np.abs(xx) <= 0.75).astype(np.float32)
        img = (np.exp(-(dist**2) / (2 * 0.06**2)) * along).astype(np.float32)
    elif name == "square":
        ax, ay = np.abs(xx), np.abs(yy)
        # Soft square ring.
        outer = np.maximum(ax, ay)
        img = np.exp(-((outer - 0.55) ** 2) / (2 * 0.06**2)).astype(np.float32)
        img *= ((ax <= 0.72) & (ay <= 0.72)).astype(np.float32)
    elif name == "cross":
        dist_h = np.abs(yy)
        dist_v = np.abs(xx)
        arm = 0.7
        h = np.exp(-(dist_h**2) / (2 * 0.06**2)) * (np.abs(xx) <= arm)
        v = np.exp(-(dist_v**2) / (2 * 0.06**2)) * (np.abs(yy) <= arm)
        img = np.clip(h + v, 0.0, 1.0).astype(np.float32)
    else:
        raise ValueError(f"no stock renderer for {name}")
    return np.clip(img, 0.0, 1.0)


def ensure_stock_images(*, height: int = 28, width: int = 28, force: bool = False) -> list[Path]:
    """Write stock PNGs under assets/commands/ if missing."""
    written: list[Path] = []
    for cid, name in COMMANDS.items():
        path = stock_path(cid)
        if path.exists() and not force:
            written.append(path)
            continue
        hw = _render_stock_hw(name, height, width)
        nchw = hw[None, None, :, :]
        save_canvas_png(path, nchw, sample=0)
        written.append(path)
    return written


def load_png_grayscale(path: Path, *, height: int, width: int) -> np.ndarray:
    """Load PNG → float32 (H,W) in [0,1], resized nearest-neighbor if needed."""
    path = Path(path)
    # Prefer matplotlib (already a project dep via diagnostics) over Pillow.
    import matplotlib.image as mpimg

    raw = mpimg.imread(str(path))
    if raw.ndim == 3:
        # RGB(A) → luma
        rgb = raw[..., :3].astype(np.float64)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(
            np.float32
        )
    else:
        gray = raw.astype(np.float32)
        if gray.max() > 1.5:
            gray = gray / 255.0
    gray = np.clip(gray, 0.0, 1.0)
    if gray.shape != (height, width):
        # Nearest-neighbor resize without scipy dependency on griddata.
        ys = (np.linspace(0, gray.shape[0] - 1, height)).astype(int)
        xs = (np.linspace(0, gray.shape[1] - 1, width)).astype(int)
        gray = gray[ys][:, xs]
    return gray.astype(np.float32)


def load_command_target(
    command_id: int,
    *,
    batch_size: int = 1,
    height: int = 28,
    width: int = 28,
    channels: int = 1,
) -> np.ndarray:
    """Load stock image for command → NCHW float32 batch."""
    ensure_stock_images(height=height, width=width)
    gray = load_png_grayscale(stock_path(command_id), height=height, width=width)
    img = gray[None, None, :, :]
    if channels > 1:
        img = np.repeat(img, channels, axis=1)
    return np.repeat(img, int(batch_size), axis=0)


def command_catalog() -> list[dict[str, Any]]:
    ensure_stock_images()
    return [
        {
            "id": cid,
            "name": name,
            "path": str(stock_path(cid)),
            "aliases": [a for a, i in ALIASES.items() if i == cid],
        }
        for cid, name in COMMANDS.items()
    ]
