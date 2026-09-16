# data/generators/csv/swiss_helix.py
"""Double-helix swiss-roll binary CSV (numpy + csv)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_GEN_ROOT = Path(__file__).resolve().parents[1]
if str(_GEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GEN_ROOT))
from _cli import run_materialize_cli  # noqa: E402


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


if __name__ == "__main__":
    run_materialize_cli(__doc__ or "swiss helix", materialize, default_n=2000)
