# data/generators/csv/generate_synthetic_shapes.py
"""
Procedural RGB shape images for the CNN path.

NPZ output (X NCHW float32, y int labels). Shapes are high-contrast silhouettes
on a dark background — easy enough for the tiny bench CNN to learn in a few
epochs at 128², while keeping the NPZ ingest path used by the pipeline.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CLASS_NAMES = ("circle", "square", "cross", "diamond")


def draw_circle(grid: np.ndarray, c_y: int, c_x: int, radius: int) -> None:
    h, w = grid.shape[1], grid.shape[2]
    y, x = np.ogrid[:h, :w]
    mask = (x - c_x) ** 2 + (y - c_y) ** 2 <= radius ** 2
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


def generate_one(
    class_idx: int,
    channels: int,
    height: int,
    width: int,
    rng: np.random.Generator,
    noise_level: float,
) -> np.ndarray:
    img = np.zeros((channels, height, width), dtype=np.float32)

    margin = max(height, width) // 4
    c_y = int(rng.integers(margin, height - margin))
    c_x = int(rng.integers(margin, width - margin))
    scale = int(rng.integers(max(4, min(height, width) // 8), max(5, min(height, width) // 3)))
    thickness = max(1, scale // 5)

    if class_idx == 0:
        draw_circle(img, c_y, c_x, radius=scale)
    elif class_idx == 1:
        draw_square(img, c_y, c_x, half_size=scale)
    elif class_idx == 2:
        draw_cross(img, c_y, c_x, arm_len=scale, thickness=thickness)
    else:
        draw_diamond(img, c_y, c_x, radius=scale)

    # Mild per-channel tint so RGB is not pure gray (still high contrast).
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
) -> None:
    """
    Class 0: Circle
    Class 1: Square
    Class 2: Cross
    Class 3: Diamond

    Default artifact is float32 NPZ (`X` NCHW + `y` labels). Optional CSV is
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


def _default_output(height: int, width: int) -> str:
    """Size-tagged path so 28² and 128² can coexist without clobbering."""
    return os.path.join("data", "samples", "csv", f"synthetic_shapes_{height}.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate procedural RGB shape images for CNN training.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (.npz preferred). Default: synthetic_shapes_{H}.npz",
    )
    parser.add_argument("--per-class", type=int, default=400)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--also-csv",
        action="store_true",
        help="Also write a flattened CSV (very large at 128²; not recommended)",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Write both 28² and 128² size-tagged NPZs (ignores --height/--width/--output)",
    )
    args = parser.parse_args()

    if args.both:
        for h, w in ((28, 28), (128, 128)):
            generate_shapes_dataset(
                _default_output(h, w),
                num_samples_per_class=args.per_class,
                channels=args.channels,
                height=h,
                width=w,
                noise_level=args.noise,
                seed=args.seed,
                also_csv=args.also_csv,
            )
        return

    out = args.output if args.output is not None else _default_output(args.height, args.width)
    generate_shapes_dataset(
        out,
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
