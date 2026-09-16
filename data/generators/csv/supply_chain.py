# data/generators/csv/supply_chain.py
"""Industrial supply-chain multiclass CSV (numpy + csv)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_GEN_ROOT = Path(__file__).resolve().parents[1]
if str(_GEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GEN_ROOT))
from _cli import run_materialize_cli  # noqa: E402


def materialize(out_path: str, *, n_samples: int = 3000, seed: int = 101) -> str:
    rng = np.random.default_rng(seed)
    lead = rng.uniform(5, 60, n_samples)
    logistics = rng.exponential(scale=1.5, size=n_samples)
    reserve = rng.beta(a=5, b=2, size=n_samples) * 100
    labor = rng.uniform(0.6, 1.0, n_samples)
    inflation = rng.normal(loc=3.2, scale=1.1, size=n_samples)
    stress = (
        (lead / 15.0) ** 1.8
        + (logistics * 2.5) ** 1.3
        - (reserve / 20.0)
        + (1.0 - labor) * 8.0
        + (inflation * 0.4)
        + rng.normal(0, 1.2, n_samples)
    )
    target = np.zeros(n_samples, dtype=int)
    target[stress > 6.5] = 1
    target[stress > 12.0] = 2
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Supplier_Lead_Time",
                "Logistics_Delay_Index",
                "Resource_Reserve_Percent",
                "Labor_Capacity_Utilization",
                "Macro_Inflation_Rate",
                "Target",
            ]
        )
        for i in range(n_samples):
            w.writerow(
                [
                    float(lead[i]),
                    float(logistics[i]),
                    float(reserve[i]),
                    float(labor[i]),
                    float(inflation[i]),
                    int(target[i]),
                ]
            )
    return str(path.resolve())


if __name__ == "__main__":
    run_materialize_cli(
        __doc__ or "supply chain", materialize, default_n=3000, default_seed=101
    )
