# data/generators/cnn/generate_synthetic_shapes.py
"""
Procedural RGB shape images for the CNN path.

NPZ output (X NCHW float32, y int labels). High-contrast silhouettes on a dark
background — easy bench set plus a harder multi-class pack.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import numpy as np

_GEN_ROOT = Path(__file__).resolve().parents[1]
if str(_GEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GEN_ROOT))
from _cli import run_materialize_cli  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Easy bench (legacy 4-class).
CLASS_NAMES_EASY = ("circle", "square", "cross", "diamond")
# Hard pack adds more silhouettes.
CLASS_NAMES_HARD = (
    "circle",
    "square",
    "cross",
    "diamond",
    "triangle",
    "ring",
    "star",
    "l_piece",
)


def draw_circle(grid: np.ndarray, c_y: int, c_x: int, radius: int) -> None:
    h, w = grid.shape[1], grid.shape[2]
    y, x = np.ogrid[:h, :w]
    mask = (x - c_x) ** 2 + (y - c_y) ** 2 <= radius**2
    grid[:, mask] = 1.0


def draw_square(grid: np.ndarray, c_y: int, c_x: int, half_size: int) -> None:
    h, w = grid.shape[1], grid.shape[2]
    y_min, y_max = max(0, c_y - half_size), min(h, c_y + half_size + 1)
    x_min, x_max = max(0, c_x - half_size), min(w, c_x + half_size + 1)
    grid[:, y_min:y_max, x_min:x_max] = 1.0


def draw_cross(grid: np.ndarray, c_y: int, c_x: int, arm_len: int, thickness: int) -> None:
    h, w = grid.shape[1], grid.shape[2]
    x_min, x_max = max(0, c_x - arm_len), min(w, c_x + arm_len + 1)
    y_min, y_max = max(0, c_y - thickness), min(h, c_y + thickness + 1)
    grid[:, y_min:y_max, x_min:x_max] = 1.0
    x_min, x_max = max(0, c_x - thickness), min(w, c_x + thickness + 1)
    y_min, y_max = max(0, c_y - arm_len), min(h, c_y + arm_len + 1)
    grid[:, y_min:y_max, x_min:x_max] = 1.0


def draw_diamond(grid: np.ndarray, c_y: int, c_x: int, radius: int) -> None:
    h, w = grid.shape[1], grid.shape[2]
    y, x = np.ogrid[:h, :w]
    mask = np.abs(x - c_x) + np.abs(y - c_y) <= radius
    grid[:, mask] = 1.0


def draw_triangle(grid: np.ndarray, c_y: int, c_x: int, scale: int) -> None:
    h, w = grid.shape[1], grid.shape[2]
    y, x = np.ogrid[:h, :w]
    # Upward equilateral-ish triangle via half-planes.
    top = c_y - scale
    left = (x - c_x) * scale >= (y - top) * (-scale) - scale * scale
    right = (c_x - x) * scale >= (y - top) * (-scale) - scale * scale
    below = y <= c_y + scale // 2
    mask = left & right & below & (y >= top)
    grid[:, mask] = 1.0


def draw_ring(grid: np.ndarray, c_y: int, c_x: int, radius: int, thickness: int) -> None:
    h, w = grid.shape[1], grid.shape[2]
    y, x = np.ogrid[:h, :w]
    d2 = (x - c_x) ** 2 + (y - c_y) ** 2
    outer = radius**2
    inner = max(1, radius - max(1, thickness)) ** 2
    mask = (d2 <= outer) & (d2 >= inner)
    grid[:, mask] = 1.0


def draw_star(grid: np.ndarray, c_y: int, c_x: int, scale: int) -> None:
    h, w = grid.shape[1], grid.shape[2]
    y, x = np.ogrid[:h, :w]
    # 5-point star via polar angle modulation.
    dy = (y - c_y).astype(np.float64)
    dx = (x - c_x).astype(np.float64)
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)
    # Outer radius oscillates 5 times.
    r_edge = scale * (0.45 + 0.55 * (0.5 + 0.5 * np.cos(5.0 * theta)))
    mask = r <= r_edge
    grid[:, mask] = 1.0


def draw_l_piece(grid: np.ndarray, c_y: int, c_x: int, scale: int, thickness: int) -> None:
    h, w = grid.shape[1], grid.shape[2]
    t = max(1, thickness)
    # Vertical stem.
    y0, y1 = max(0, c_y - scale), min(h, c_y + scale + 1)
    x0, x1 = max(0, c_x - t), min(w, c_x + t + 1)
    grid[:, y0:y1, x0:x1] = 1.0
    # Foot to the right.
    y2, y3 = max(0, c_y + scale - t), min(h, c_y + scale + 1)
    x2, x3 = max(0, c_x - t), min(w, c_x + scale + 1)
    grid[:, y2:y3, x2:x3] = 1.0


def generate_one(
    class_idx: int,
    channels: int,
    height: int,
    width: int,
    rng: np.random.Generator,
    noise_level: float,
    class_names: tuple[str, ...],
) -> np.ndarray:
    img = np.zeros((channels, height, width), dtype=np.float32)

    margin = max(height, width) // 4
    c_y = int(rng.integers(margin, height - margin))
    c_x = int(rng.integers(margin, width - margin))
    scale = int(
        rng.integers(max(4, min(height, width) // 8), max(5, min(height, width) // 3))
    )
    thickness = max(1, scale // 5)
    name = class_names[class_idx]

    if name == "circle":
        draw_circle(img, c_y, c_x, radius=scale)
    elif name == "square":
        draw_square(img, c_y, c_x, half_size=scale)
    elif name == "cross":
        draw_cross(img, c_y, c_x, arm_len=scale, thickness=thickness)
    elif name == "diamond":
        draw_diamond(img, c_y, c_x, radius=scale)
    elif name == "triangle":
        draw_triangle(img, c_y, c_x, scale=scale)
    elif name == "ring":
        draw_ring(img, c_y, c_x, radius=scale, thickness=max(2, thickness))
    elif name == "star":
        draw_star(img, c_y, c_x, scale=scale)
    else:
        draw_l_piece(img, c_y, c_x, scale=scale, thickness=thickness)

    tint = rng.uniform(0.75, 1.0, size=(channels, 1, 1)).astype(np.float32)
    img *= tint

    if noise_level > 0:
        img = img + rng.normal(0.0, noise_level, size=img.shape).astype(np.float32)

    return np.clip(img, 0.0, 1.0).astype(np.float32)


def generate_shapes_dataset(
    output_path: str,
    num_samples_per_class: int = 400,
    channels: int = 3,
    height: int = 128,
    width: int = 128,
    noise_level: float = 0.05,
    seed: int = 42,
    also_csv: bool = False,
    class_names: tuple[str, ...] = CLASS_NAMES_EASY,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    rng = np.random.default_rng(seed)
    num_classes = len(class_names)
    total_samples = num_samples_per_class * num_classes

    logging.info(
        "[Shape Generator] Synthesizing %d samples (%d/class × %d) | shape=(%d,%d,%d)",
        total_samples,
        num_samples_per_class,
        num_classes,
        channels,
        height,
        width,
    )

    images = np.empty((total_samples, channels, height, width), dtype=np.float32)
    labels = np.empty((total_samples,), dtype=np.int32)
    row = 0
    for class_idx in range(num_classes):
        for _ in range(num_samples_per_class):
            images[row] = generate_one(
                class_idx, channels, height, width, rng, noise_level, class_names
            )
            labels[row] = class_idx
            row += 1

    order = rng.permutation(total_samples)
    images = images[order]
    labels = labels[order]

    root, ext = os.path.splitext(output_path)
    npz_path = output_path if ext.lower() == ".npz" else root + ".npz"
    np.savez_compressed(npz_path, X=images, y=labels, class_names=np.array(class_names))
    logging.info(
        "[Shape Generator] Wrote NPZ %s (X=%s, ~%.1f MiB on disk)",
        npz_path,
        images.shape,
        os.path.getsize(npz_path) / (1024 * 1024),
    )

    if also_csv or ext.lower() == ".csv":
        csv_path = output_path if ext.lower() == ".csv" else root + ".csv"
        feature_dim = channels * height * width
        header = [f"px_{i}" for i in range(feature_dim)] + ["target"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            flat = images.reshape(total_samples, feature_dim)
            for i in range(total_samples):
                writer.writerow(np.round(flat[i], 4).tolist() + [int(labels[i])])
        logging.info("[Shape Generator] Wrote CSV %s", csv_path)
    return os.path.abspath(npz_path)


def materialize(
    out_path: str,
    *,
    n_samples: int = 1600,
    height: int = 28,
    width: int = 28,
    channels: int = 3,
    noise: float = 0.05,
    seed: int = 42,
    hard: bool = False,
) -> str:
    """Write NPZ for TM/engine diet materialize."""
    names = CLASS_NAMES_HARD if hard else CLASS_NAMES_EASY
    per = max(1, int(n_samples) // len(names))
    return generate_shapes_dataset(
        out_path,
        num_samples_per_class=per,
        channels=channels,
        height=height,
        width=width,
        noise_level=noise,
        seed=seed,
        also_csv=False,
        class_names=names,
    )


def _default_output(height: int, width: int) -> str:
    return os.path.join("data", "samples", "cnn", f"synthetic_shapes_{height}.npz")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate procedural RGB shape images for CNN training."
    )
    parser.add_argument("--out", "--output", dest="out", default=None)
    parser.add_argument("--n", type=int, default=None, help="Total samples (splits across classes)")
    parser.add_argument("--per-class", type=int, default=400)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard", action="store_true", help="8-class hard pack")
    parser.add_argument("--also-csv", action="store_true")
    parser.add_argument(
        "--both",
        action="store_true",
        help="Write both 28² and 128² size-tagged NPZs",
    )
    args = parser.parse_args()
    names = CLASS_NAMES_HARD if args.hard else CLASS_NAMES_EASY
    per = args.per_class
    if args.n is not None:
        per = max(1, int(args.n) // len(names))

    if args.both:
        for h, w in ((28, 28), (128, 128)):
            generate_shapes_dataset(
                _default_output(h, w),
                num_samples_per_class=per,
                channels=args.channels,
                height=h,
                width=w,
                noise_level=args.noise,
                seed=args.seed,
                also_csv=args.also_csv,
                class_names=names,
            )
        return

    out = args.out if args.out is not None else _default_output(args.height, args.width)
    path = generate_shapes_dataset(
        out,
        num_samples_per_class=per,
        channels=args.channels,
        height=args.height,
        width=args.width,
        noise_level=args.noise,
        seed=args.seed,
        also_csv=args.also_csv,
        class_names=names,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    if "--out" in sys.argv or any(a.startswith("--out=") for a in sys.argv[1:]):

        def _extras(p: argparse.ArgumentParser) -> None:
            p.add_argument("--noise", type=float, default=0.05)
            p.add_argument("--height", type=int, default=28)
            p.add_argument("--width", type=int, default=28)
            p.add_argument("--channels", type=int, default=3)
            p.add_argument("--hard", action="store_true")

        def _mat(out_path: str, **kw):
            hard = bool(kw.pop("hard", False)) or ("--hard" in sys.argv)
            return materialize(out_path, hard=hard, **kw)

        run_materialize_cli(
            __doc__ or "cnn shapes", _mat, default_n=1600, extra_args=_extras
        )
    else:
        main()
