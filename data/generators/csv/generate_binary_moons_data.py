# data/generators/csv/generate_binary_moons_data.py
"""Nested moons binary CSV (numpy-only; no sklearn)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _make_moons(n_samples: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two interlocking half-circles (sklearn make_moons geometry)."""
    rng = np.random.default_rng(seed)
    n_out = n_samples // 2
    n_in = n_samples - n_out
    t_out = np.linspace(0, np.pi, n_out)
    t_in = np.linspace(0, np.pi, n_in)
    x_out = np.cos(t_out)
    y_out = np.sin(t_out)
    x_in = 1 - np.cos(t_in)
    y_in = 1 - np.sin(t_in) - 0.5
    X = np.vstack(
        [np.column_stack([x_out, y_out]), np.column_stack([x_in, y_in])]
    ).astype(np.float64)
    y = np.hstack([np.zeros(n_out), np.ones(n_in)]).astype(np.float64)
    X += rng.normal(0, noise, X.shape)
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def materialize(
    out_path: str,
    *,
    n_samples: int = 1000,
    noise: float = 0.15,
    seed: int = 42,
) -> str:
    X, y = _make_moons(n_samples, noise, seed)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Feature_1", "Feature_2", "Target"])
        for i in range(len(y)):
            w.writerow([float(X[i, 0]), float(X[i, 1]), float(y[i])])
    return str(path.resolve())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    print(f"Wrote {materialize(args.out, n_samples=args.n, seed=args.seed)}")


if __name__ == "__main__":
    main()
