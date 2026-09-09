#!/usr/bin/env python3
"""Fair thread-scaling probe: async reserve OFF, both engines, T=1,2,4.

Diagnostic only — compares OMP-team scaling with the same thread budget.
"""
from __future__ import annotations

import copy
import tempfile
import time
from pathlib import Path

import yaml

from benchmarks.benchmark_cnn import (
    custom_benchmark_child,
    pytorch_benchmark_child,
    run_benchmark_child,
)

# Prior head-to-head reports use ~1024 train samples on synthetic_shapes.
N_TRAIN = 1024


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((root / "config" / "config.yaml").read_text())
    base["optimization"]["epochs_full_dataset"] = 40
    base["optimization"]["early_stopping_enabled"] = False
    base["ledger"]["native_async_submit"] = False
    # Keep ledger off so this measures OMP-team scaling, not async mailbox overlap.
    base["ledger"]["enabled"] = False

    rows: list[tuple[int, dict, dict]] = []
    for T in (1, 2, 4):
        cfg = copy.deepcopy(base)
        cfg["optimization"]["num_threads"] = T
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(cfg, f)
            path = f.name
        print(f"\n===== T={T} (async_reserve=OFF, epochs=40) =====", flush=True)
        t0 = time.perf_counter()
        t_res = run_benchmark_child(
            pytorch_benchmark_child, path, f"scale-torch-{T}"
        )
        c_res = run_benchmark_child(
            custom_benchmark_child, path, f"scale-custom-{T}"
        )
        print(
            f"T={T} wall {time.perf_counter() - t0:.1f}s "
            f"torch_train={t_res['train_time']:.3f}s "
            f"custom_train={c_res['train_time']:.3f}s",
            flush=True,
        )
        rows.append((T, t_res, c_res))
        time.sleep(3)

    print("\n===== SCALE SUMMARY (smp/s = n_train*epochs/train_time) =====")
    print(
        f"{'T':>3}  {'torch_s':>9} {'torch_smp':>10} {'torch_x1':>8}  "
        f"{'cust_s':>9} {'cust_smp':>10} {'cust_x1':>8}  {'cust/torch':>10}"
    )
    base_t = base_c = None
    for T, t_res, c_res in rows:
        te, ce = t_res["epochs_completed"], c_res["epochs_completed"]
        ts = (N_TRAIN * te) / t_res["train_time"]
        cs = (N_TRAIN * ce) / c_res["train_time"]
        if base_t is None:
            base_t, base_c = ts, cs
        print(
            f"{T:3d}  {t_res['train_time']:9.3f} {ts:10.1f} {ts / base_t:8.3f}  "
            f"{c_res['train_time']:9.3f} {cs:10.1f} {cs / base_c:8.3f}  "
            f"{cs / ts:10.3f}"
        )


if __name__ == "__main__":
    main()
