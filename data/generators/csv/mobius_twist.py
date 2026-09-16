# data/generators/csv/mobius_twist.py
"""Twin Möbius binary CSV (numpy + csv)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def materialize(
    out_path: str, *, n_samples: int = 2000, noise: float = 0.05, seed: int = 42
) -> str:
    rng = np.random.default_rng(seed)
    n = n_samples // 2
    length = np.linspace(0, 2 * np.pi, n)
    width = np.linspace(-0.5, 0.5, n)
    x0 = (1 + width * np.cos(length / 2)) * np.cos(length)
    y0 = (1 + width * np.cos(length / 2)) * np.sin(length)
    z0 = width * np.sin(length / 2)
    w1 = width + 0.2
    phase = length + np.pi
    x1 = (1 + w1 * np.cos(phase / 2)) * np.cos(phase)
    y1 = (1 + w1 * np.cos(phase / 2)) * np.sin(phase)
    z1 = w1 * np.sin(phase / 2)
    X = np.vstack(
        [
            np.column_stack([x0, y0, z0]),
            np.column_stack([x1, y1, z1]),
        ]
    ) + rng.normal(0, noise, (2 * n, 3))
    y = np.hstack([np.zeros(n), np.ones(n)])
    perm = rng.permutation(len(y))
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["X", "Y", "Z", "Target"])
        for i in perm:
            w.writerow([float(X[i, 0]), float(X[i, 1]), float(X[i, 2]), float(y[i])])
    return str(path.resolve())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    print(f"Wrote {materialize(args.out, n_samples=args.n, seed=args.seed)}")


if __name__ == "__main__":
    main()
