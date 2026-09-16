# data/generators/csv/generate_regression_data.py
"""Non-linear telemetry regression CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def generate_synthetic_telemetry(
    file_path: str,
    num_samples: int = 1200,
    noise_factor: float = 0.1,
    seed: int = 42,
) -> str:
    rng = np.random.default_rng(seed)
    velocity = rng.uniform(-5.0, 5.0, (num_samples, 1))
    movement = rng.uniform(-5.0, 5.0, (num_samples, 1))
    target_angle = (
        (np.sin(velocity) * np.cos(movement))
        + (0.05 * (velocity**2))
        + rng.normal(0, noise_factor, (num_samples, 1))
    )
    dataset_matrix = np.hstack([velocity, movement, target_angle])
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "Velocity,Movement,Target_Angle"
    np.savetxt(path, dataset_matrix, delimiter=",", header=header, comments="")
    return str(path.resolve())


def materialize(
    out_path: str,
    *,
    n_samples: int = 1200,
    noise: float = 0.1,
    seed: int = 42,
) -> str:
    return generate_synthetic_telemetry(
        out_path, num_samples=n_samples, noise_factor=noise, seed=seed
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise", type=float, default=0.1)
    args = p.parse_args()
    print(f"Wrote {materialize(args.out, n_samples=args.n, seed=args.seed, noise=args.noise)}")


if __name__ == "__main__":
    main()
