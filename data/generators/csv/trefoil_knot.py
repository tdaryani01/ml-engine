# data/generators/csv/trefoil_knot.py
"""Trefoil knot ribbon vs thick tube binary CSV (3D)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_GEN_ROOT = Path(__file__).resolve().parents[1]
if str(_GEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GEN_ROOT))
from _cli import run_materialize_cli  # noqa: E402


def _trefoil(t: np.ndarray) -> np.ndarray:
    x = np.sin(t) + 2.0 * np.sin(2.0 * t)
    y = np.cos(t) - 2.0 * np.cos(2.0 * t)
    z = -np.sin(3.0 * t)
    return np.column_stack([x, y, z])


def materialize(
    out_path: str, *, n_samples: int = 2000, noise: float = 0.08, seed: int = 42
) -> str:
    rng = np.random.default_rng(seed)
    n = max(1, n_samples // 2)
    t0 = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Class 0: tight ribbon along the knot.
    ribbon = _trefoil(t0) + rng.normal(0, noise * 0.5, (n, 3))
    # Class 1: same knot with a fatter tube (larger radial noise).
    t1 = t0 + rng.uniform(0, 2 * np.pi / max(n, 1), n)
    tube = _trefoil(t1) + rng.normal(0, noise * 2.5, (n, 3))
    X = np.vstack([ribbon, tube])
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
    run_materialize_cli(__doc__ or "trefoil knot", materialize, default_n=2000)
