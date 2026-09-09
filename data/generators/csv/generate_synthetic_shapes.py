# data/generators/csv/generate_synthetic_shapes.py
"""
Procedural RGB shape images for the CNN path.

Kept intentionally synthetic (no external downloads), but closer to real photos
than the old 28x28 binary silhouettes: soft edges, color, lighting, clutter,
and mild geometric jitter at a usable spatial size (default 128x128).
"""
from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CLASS_NAMES = ("circle", "square", "cross", "diamond")


def _rot_grid(h: int, w: int, c_y: float, c_x: float, angle_rad: float):
    """Pixel coordinate grids in a frame rotated about (c_y, c_x)."""
    y, x = np.ogrid[:h, :w]
    ys = y.astype(np.float32) - c_y
    xs = x.astype(np.float32) - c_x
    ca, sa = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    yr = ca * ys - sa * xs
    xr = sa * ys + ca * xs
    return yr, xr


def _soft_mask(signed_dist: np.ndarray, edge: float) -> np.ndarray:
    """Smoothstep from solid (dist<=0) to transparent over `edge` pixels."""
    edge = max(float(edge), 1e-3)
    t = np.clip(0.5 - signed_dist / (2.0 * edge), 0.0, 1.0).astype(np.float32)
    return t * t * (3.0 - 2.0 * t)


def _lowfreq_field(h: int, w: int, rng: np.random.Generator, amplitude: float = 1.0) -> np.ndarray:
    """Cheap multi-sine field used for background texture / lighting."""
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    field = np.zeros((h, w), dtype=np.float32)
    for _ in range(4):
        fy = rng.uniform(1.0, 5.0)
        fx = rng.uniform(1.0, 5.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.35, 1.0)
        field += amp * np.sin(2.0 * np.pi * (fy * yy + fx * xx) + phase)
    field -= field.min()
    denom = field.max() - field.min() + 1e-6
    field = field / denom
    return (field * amplitude).astype(np.float32)


def _background(h: int, w: int, channels: int, rng: np.random.Generator) -> np.ndarray:
    base = rng.uniform(0.12, 0.55, size=(channels, 1, 1)).astype(np.float32)
    img = np.broadcast_to(base, (channels, h, w)).copy()
    for c in range(channels):
        tex = _lowfreq_field(h, w, rng, amplitude=rng.uniform(0.08, 0.22))
        img[c] += tex * rng.choice([-1.0, 1.0])
    # Soft vignette / lighting
    yy = (np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]) ** 2
    xx = (np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]) ** 2
    vignette = 1.0 - 0.35 * (yy + xx)
    img *= vignette
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _paint(img: np.ndarray, alpha: np.ndarray, color: np.ndarray) -> None:
    a = alpha[None, :, :]
    img *= (1.0 - a)
    img += a * color[:, None, None]


def _shape_alpha(
    class_idx: int,
    h: int,
    w: int,
    c_y: float,
    c_x: float,
    scale: float,
    angle: float,
    edge: float,
    rng: np.random.Generator,
) -> np.ndarray:
    yr, xr = _rot_grid(h, w, c_y, c_x, angle)
    if class_idx == 0:  # circle
        dist = np.sqrt(xr * xr + yr * yr) - scale
    elif class_idx == 1:  # square (L-inf ball)
        dist = np.maximum(np.abs(xr), np.abs(yr)) - scale
    elif class_idx == 2:  # cross / plus
        thick = max(scale * rng.uniform(0.18, 0.32), 1.5)
        d_h = np.maximum(np.abs(yr) - thick, np.abs(xr) - scale)
        d_v = np.maximum(np.abs(xr) - thick, np.abs(yr) - scale)
        dist = np.minimum(d_h, d_v)
    else:  # diamond (L1 ball)
        dist = (np.abs(xr) + np.abs(yr)) - scale
    return _soft_mask(dist, edge)


def generate_one(
    class_idx: int,
    channels: int,
    height: int,
    width: int,
    rng: np.random.Generator,
    noise_level: float,
) -> np.ndarray:
    img = _background(height, width, channels, rng)

    margin = max(height, width) // 8
    c_y = float(rng.integers(margin, height - margin))
    c_x = float(rng.integers(margin, width - margin))
    scale = float(rng.uniform(0.18, 0.38) * min(height, width))
    angle = float(rng.uniform(-0.55, 0.55))  # ~±31°
    edge = float(rng.uniform(1.2, 3.5) * (min(height, width) / 128.0))

    alpha = _shape_alpha(class_idx, height, width, c_y, c_x, scale, angle, edge, rng)

    # Occasional hollow / ring style for circles & diamonds
    if class_idx in (0, 3) and rng.random() < 0.35:
        inner = scale * rng.uniform(0.35, 0.65)
        yr, xr = _rot_grid(height, width, c_y, c_x, angle)
        if class_idx == 0:
            inner_dist = inner - np.sqrt(xr * xr + yr * yr)
        else:
            inner_dist = inner - (np.abs(xr) + np.abs(yr))
        hole = _soft_mask(inner_dist, edge)
        alpha = np.clip(alpha - hole, 0.0, 1.0)

    color = rng.uniform(0.35, 1.0, size=(channels,)).astype(np.float32)
    # Mild per-channel tint imbalance so shapes are not pure gray
    color *= rng.uniform(0.75, 1.25, size=(channels,)).astype(np.float32)
    color = np.clip(color, 0.0, 1.0)

    _paint(img, alpha, color)

    # Speckle clutter (small distractors)
    n_clutter = int(rng.integers(0, 6))
    for _ in range(n_clutter):
        cy = float(rng.integers(0, height))
        cx = float(rng.integers(0, width))
        rad = float(rng.uniform(1.0, max(2.0, min(height, width) * 0.04)))
        yy = np.arange(height, dtype=np.float32)[:, None] - cy
        xx = np.arange(width, dtype=np.float32)[None, :] - cx
        blob = _soft_mask(np.sqrt(xx * xx + yy * yy) - rad, 1.25)
        blob_color = rng.uniform(0.0, 1.0, size=(channels,)).astype(np.float32)
        _paint(img, blob * rng.uniform(0.15, 0.45), blob_color)

    if noise_level > 0:
        img = img + rng.normal(0.0, noise_level, size=img.shape).astype(np.float32)

    return np.clip(img, 0.0, 1.0).astype(np.float32)


def generate_shapes_dataset(
    output_path: str,
    num_samples_per_class: int = 400,
    channels: int = 3,
    height: int = 128,
    width: int = 128,
    noise_level: float = 0.04,
    seed: int = 42,
    also_csv: bool = False,
) -> None:
    """
    Class 0: Circle
    Class 1: Square
    Class 2: Cross
    Class 3: Diamond

    Default artifact is a float32 NPZ (`X` NCHW + `y` labels). Optional CSV is
    available for debugging but is huge at 128² — prefer NPZ for training.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    rng = np.random.default_rng(seed)
    num_classes = len(CLASS_NAMES)
    total_samples = num_samples_per_class * num_classes

    logging.info(
        "[Shape Generator] Synthesizing %d samples (%d/class) | shape=(%d,%d,%d)",
        total_samples,
        num_samples_per_class,
        channels,
        height,
        width,
    )

    images = np.empty((total_samples, channels, height, width), dtype=np.float32)
    labels = np.empty((total_samples,), dtype=np.int32)
    row = 0
    for class_idx in range(num_classes):
        for _ in range(num_samples_per_class):
            images[row] = generate_one(class_idx, channels, height, width, rng, noise_level)
            labels[row] = class_idx
            row += 1

    order = rng.permutation(total_samples)
    images = images[order]
    labels = labels[order]

    root, ext = os.path.splitext(output_path)
    npz_path = output_path if ext.lower() == ".npz" else root + ".npz"
    np.savez_compressed(npz_path, X=images, y=labels, class_names=np.array(CLASS_NAMES))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate procedural RGB shape images for CNN training.")
    parser.add_argument(
        "--output",
        default=os.path.join("data", "samples", "csv", "synthetic_shapes.npz"),
        help="Output path (.npz preferred; .csv forces text export)",
    )
    parser.add_argument("--per-class", type=int, default=400)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--noise", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--also-csv",
        action="store_true",
        help="Also write a flattened CSV (very large at 128²; not recommended)",
    )
    args = parser.parse_args()

    generate_shapes_dataset(
        args.output,
        num_samples_per_class=args.per_class,
        channels=args.channels,
        height=args.height,
        width=args.width,
        noise_level=args.noise,
        seed=args.seed,
        also_csv=args.also_csv,
    )


if __name__ == "__main__":
    main()
