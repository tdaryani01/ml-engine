# data/generators/csv/generate_multiclass_data.py
"""3-arm spiral multiclass CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def generate_spiral_dataset(
    file_path: str,
    samples_per_class: int = 500,
    noise: float = 0.2,
    seed: int = 42,
) -> str:
    num_classes = 3
    X = np.zeros((samples_per_class * num_classes, 2))
    y = np.zeros((samples_per_class * num_classes, 1), dtype=int)
    rng = np.random.default_rng(seed)
    for class_idx in range(num_classes):
        ix = range(samples_per_class * class_idx, samples_per_class * (class_idx + 1))
        r = np.linspace(0.0, 10.0, samples_per_class)
        theta = (
            np.linspace(class_idx * 2.5, (class_idx + 2.5) * 2.5, samples_per_class)
            + rng.normal(0, noise, samples_per_class)
        )
        X[ix] = np.c_[r * np.sin(theta), r * np.cos(theta)]
        y[ix] = class_idx
    dataset_matrix = np.hstack([X, y])
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "Velocity,Movement,Target_Class"
    np.savetxt(path, dataset_matrix, delimiter=",", header=header, comments="")
    return str(path.resolve())


def materialize(
    out_path: str,
    *,
    n_samples: int = 1500,
    noise: float = 0.2,
    seed: int = 42,
) -> str:
    per = max(1, int(n_samples) // 3)
    return generate_spiral_dataset(
        out_path, samples_per_class=per, noise=noise, seed=seed
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise", type=float, default=0.2)
    args = p.parse_args()
    print(f"Wrote {materialize(args.out, n_samples=args.n, seed=args.seed, noise=args.noise)}")


if __name__ == "__main__":
    main()
