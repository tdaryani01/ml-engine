# data/generators/csv/generate_binary_data.py
"""Two-blob Gaussian binary classification CSV (stdlib + numpy)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_GEN_ROOT = Path(__file__).resolve().parents[1]
if str(_GEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GEN_ROOT))
from _cli import run_materialize_cli  # noqa: E402


class BinaryDataGenerator:
    def __init__(
        self,
        n_samples: int = 1000,
        n_features: int = 2,
        separation: float = 4.0,
        noise: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.n_features = n_features
        self.separation = separation
        self.noise = noise
        self.random_state = random_state

    def generate(self) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        n_class_0 = self.n_samples // 2
        n_class_1 = self.n_samples - n_class_0
        center_0 = np.ones(self.n_features) * (-self.separation / 2)
        center_1 = np.ones(self.n_features) * (self.separation / 2)
        X_0 = center_0 + rng.normal(0, self.noise, (n_class_0, self.n_features))
        X_1 = center_1 + rng.normal(0, self.noise, (n_class_1, self.n_features))
        y_0 = np.zeros((n_class_0, 1))
        y_1 = np.ones((n_class_1, 1))
        dataset = np.hstack((np.vstack((X_0, X_1)), np.vstack((y_0, y_1))))
        rng.shuffle(dataset)
        return dataset


def materialize(
    out_path: str,
    *,
    n_samples: int = 1000,
    n_features: int = 2,
    separation: float = 4.0,
    noise: float = 1.0,
    seed: int = 42,
) -> str:
    data = BinaryDataGenerator(
        n_samples=n_samples,
        n_features=n_features,
        separation=separation,
        noise=noise,
        random_state=seed,
    ).generate()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [f"Feature_{i + 1}" for i in range(n_features)] + ["Target"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for row in data:
            w.writerow([float(x) for x in row])
    return str(path.resolve())


if __name__ == "__main__":
    run_materialize_cli(__doc__ or "binary gaussians", materialize, default_n=1000)
