# data/generators/csv/generate_multiclass_data.py
"""3-arm spiral multiclass CSV."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_GEN_ROOT = Path(__file__).resolve().parents[1]
if str(_GEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GEN_ROOT))
from _cli import run_materialize_cli  # noqa: E402


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


if __name__ == "__main__":
    run_materialize_cli(__doc__ or "spiral multiclass", materialize, default_n=1500)
