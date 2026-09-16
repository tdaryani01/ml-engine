# data/generators/csv/swiss_hilex.py
"""Double-helix swiss-roll binary CSV (numpy + csv)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def materialize(
    out_path: str, *, n_samples: int = 2000, noise: float = 0.1, seed: int = 42
) -> str:
    rng = np.random.default_rng(seed)
    n = n_samples // 2
    t0 = np.linspace(1.5 * np.pi, 4.5 * np.pi, n)
    x0 = t0 * np.cos(t0)
    y0 = np.linspace(0, 10, n)
    z0 = t0 * np.sin(t0)
    t1 = np.linspace(1.5 * np.pi, 4.5 * np.pi, n)
    x1 = t1 * np.cos(t1 + np.pi)
    y1 = np.linspace(0, 10, n)
    z1 = t1 * np.sin(t1 + np.pi)
    X0 = np.vstack((x0, y0, z0)).T + rng.normal(0, noise, (n, 3))
    X1 = np.vstack((x1, y1, z1)).T + rng.normal(0, noise, (n, 3))
    X = np.vstack([X0, X1])
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
