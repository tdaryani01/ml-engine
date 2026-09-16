# data/generators/csv/generate_hard_regression_data.py
"""Hard continuous-manifold regression CSV (numpy + csv)."""
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
    out_path: str,
    *,
    n_samples: int = 2000,
    noise: float = 0.03,
    seed: int = 101,
) -> str:
    rng = np.random.default_rng(seed)
    Time = rng.uniform(0.0, 4.0, size=n_samples)
    Radius = rng.uniform(0.5, 2.5, size=n_samples)
    Angle = rng.uniform(0.0, 2 * np.pi, size=n_samples)
    macro_wave = np.sin(np.exp(Time) * Radius)
    spatial = np.cos(Angle * 3.0) / (Radius + 0.1)
    cliff = 1.0 + np.tanh(((Time * Radius) - 4.5) * 20.0)
    y = macro_wave * spatial + cliff + rng.normal(0, noise, size=n_samples)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Time", "Radius", "Angle", "Outcome"])
        for i in range(n_samples):
            w.writerow(
                [float(Time[i]), float(Radius[i]), float(Angle[i]), float(y[i])]
            )
    return str(path.resolve())


if __name__ == "__main__":
    run_materialize_cli(
        __doc__ or "hard regression", materialize, default_n=2000, default_seed=101
    )
