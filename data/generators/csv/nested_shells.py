# data/generators/csv/nested_shells.py
"""Concentric spherical shells — 3-class CSV (3D)."""
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
    out_path: str, *, n_samples: int = 2400, noise: float = 0.06, seed: int = 42
) -> str:
    rng = np.random.default_rng(seed)
    radii = (0.6, 1.2, 1.9)
    n_classes = len(radii)
    per = max(1, int(n_samples) // n_classes)
    rows: list[list[float]] = []
    for cls, R in enumerate(radii):
        # Uniform-ish on sphere via Gaussian normalize.
        g = rng.normal(0, 1, (per, 3))
        g /= np.linalg.norm(g, axis=1, keepdims=True).clip(min=1e-8)
        pts = g * R + rng.normal(0, noise, (per, 3))
        for p in pts:
            rows.append([float(p[0]), float(p[1]), float(p[2]), float(cls)])
    rng.shuffle(rows)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["X", "Y", "Z", "Target"])
        w.writerows(rows)
    return str(path.resolve())


if __name__ == "__main__":
    run_materialize_cli(__doc__ or "nested shells", materialize, default_n=2400)
