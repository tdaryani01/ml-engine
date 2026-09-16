# data/generators/csv/linked_torus.py
"""Linked torus / ring pair binary CSV (3D)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_GEN_ROOT = Path(__file__).resolve().parents[1]
if str(_GEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GEN_ROOT))
from _cli import run_materialize_cli  # noqa: E402


def _torus_points(
    n: int,
    *,
    R: float,
    r: float,
    center: np.ndarray,
    axis: str,
    rng: np.random.Generator,
    noise: float,
) -> np.ndarray:
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    pts = np.column_stack([x, y, z])
    if axis == "x":
        # Rotate so the hole faces along X (links with a YZ torus).
        pts = np.column_stack([pts[:, 2], pts[:, 1], -pts[:, 0]])
    pts = pts + center.reshape(1, 3)
    return pts + rng.normal(0, noise, pts.shape)


def materialize(
    out_path: str, *, n_samples: int = 2000, noise: float = 0.05, seed: int = 42
) -> str:
    rng = np.random.default_rng(seed)
    n = max(1, n_samples // 2)
    # Two linked rings: one in XY, one in YZ, centers offset so they interlock.
    a = _torus_points(
        n, R=1.0, r=0.35, center=np.array([0.0, 0.0, 0.0]), axis="z", rng=rng, noise=noise
    )
    b = _torus_points(
        n, R=1.0, r=0.35, center=np.array([1.0, 0.0, 0.0]), axis="x", rng=rng, noise=noise
    )
    X = np.vstack([a, b])
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
    run_materialize_cli(__doc__ or "linked torus", materialize, default_n=2000)
