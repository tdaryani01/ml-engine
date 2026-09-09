#!/usr/bin/env python3
"""Attribute custom vs Torch train gap (no kernel changes).

In-process so CONTRACT_OP_PROFILE from release-contract-profile .so hits stderr.

  ./build_native.sh release-contract-profile
  PYTHONPATH=. .venv/bin/python scripts/train_gap_attribute.py 2> /tmp/contract_profile.txt
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OMP_THREAD_LIMIT", "4")


def write_cfg(epochs: int, ledger_enabled: bool) -> str:
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    cfg["optimization"]["epochs_full_dataset"] = epochs
    cfg["meta"]["suppress_logging"] = True
    cfg["ledger"]["enabled"] = ledger_enabled
    cfg["ledger"]["store_backend"] = "noop"
    td = Path(tempfile.mkdtemp(prefix="gap_"))
    path = td / "cfg.yaml"
    path.write_text(yaml.dump(cfg))
    return str(path)


def run_custom(epochs: int, ledger_enabled: bool, label: str):
    from benchmarks.benchmark_cnn import (
        _benchmark_common_from_config,
        run_custom_engine_benchmark,
    )
    from config.constants import EngineBackend
    from config.schema import LedgerSettings

    cfg_path = write_cfg(epochs, ledger_enabled)
    c = _benchmark_common_from_config(cfg_path)
    # Force ledger flag from this run (common reloads yaml — already set in file)
    ledger = LedgerSettings(**(yaml.safe_load(Path(cfg_path).read_text()).get("ledger") or {}))
    print(f"\n=== {label} epochs={epochs} ledger.enabled={ledger.enabled} ===", flush=True)
    res = run_custom_engine_benchmark(
        data_provider=c["data_provider"],
        X_train=c["X_train"],
        y_train=c["y_train"],
        X_val=c["X_val"],
        y_val=c["y_val"],
        y_val_classes=c["y_val_classes"],
        cnn_dict=c["cnn_dict"],
        num_classes=c["num_classes"],
        task_type=c["task_type"],
        epochs=epochs,
        lr_init=c["lr_init"],
        lam_l1=c["lam_l1"],
        lam_l2=c["lam_l2"],
        early_stopping_enabled=c["early_stopping_enabled"],
        patience=c["patience"],
        min_delta=c["min_delta"],
        backend=EngineBackend.NATIVE,
        num_threads=c["num_threads"],
        config_path=cfg_path,
        ledger_settings=ledger,
        output_dir=c["output_dir"],
    )
    print(
        f"[{label}] train_time={res['train_time']:.3f}s "
        f"epochs={res['epochs_completed']} fwd={res['forward_counts']} bwd={res['backward_counts']}",
        flush=True,
    )
    return res


def kernel_budget():
    import torch
    import torch.nn.functional as F
    from config.constants import EngineBackend
    from utils.conv_dispatch import (
        conv2d_backward_fused,
        conv2d_forward,
        init_engine_backend,
        sync_native_thread_policy,
    )

    init_engine_backend(EngineBackend.NATIVE)
    sync_native_thread_policy(4)
    torch.set_num_threads(4)

    def med(fn, reps=30, warm=6):
        for _ in range(warm):
            fn()
        xs = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            xs.append((time.perf_counter() - t0) / reps * 1e3)
        xs.sort()
        return xs[len(xs) // 2]

    rows = []
    # Match live config/config.yaml (pad=1): L0 128→124; pool→62 for L1.
    for name, N, Cin, Cout, H, K, pad in [
        ("L0", 32, 3, 8, 128, 7, 1),
        ("L1", 32, 8, 16, 62, 7, 1),
    ]:
        oh = H + 2 * pad - K + 1
        rng = np.random.default_rng(0)
        x = np.ascontiguousarray(rng.standard_normal((N, Cin, H, H), np.float32))
        Wn = np.ascontiguousarray(rng.standard_normal((Cout, Cin, K, K), np.float32))
        b = np.zeros(Cout, np.float32)
        out = np.empty((N, Cout, oh, oh), np.float32)
        dy = np.ascontiguousarray(rng.standard_normal((N, Cout, oh, oh), np.float32))
        dx = np.empty_like(x)
        dW = np.empty_like(Wn)
        xt = torch.from_numpy(x.copy()).to(memory_format=torch.channels_last).contiguous()
        Wt = torch.from_numpy(Wn.copy()).to(memory_format=torch.channels_last).contiguous()
        dyt = torch.from_numpy(dy.copy()).to(memory_format=torch.channels_last).contiguous()
        xt.requires_grad_(True)
        Wt.requires_grad_(True)

        nf = med(
            lambda: conv2d_forward(
                x, Wn, b, out_buf=out, stride=1, pad=pad, backend=EngineBackend.NATIVE
            )
        )
        nb = med(
            lambda: conv2d_backward_fused(
                dy, x, Wn, dx, dW, stride=1, pad=pad, inv_m=1.0 / N, backend=EngineBackend.NATIVE
            )
        )
        tf = med(lambda: F.conv2d(xt, Wt, None, 1, pad))

        def tb_fn():
            y = F.conv2d(xt, Wt, None, 1, pad)
            (y * dyt).sum().backward()
            xt.grad = None
            Wt.grad = None

        tb = max(med(tb_fn) - tf, 0.0)
        rows.append((name, nf, nb, tf, tb))
        print(
            f"kernel {name}: fwd {nf:.2f}/{tf:.2f}ms ({nf/tf:.2f}x)  "
            f"bwd {nb:.2f}/{tb:.2f}ms ({nb/tb:.2f}x)  "
            f"sum {nf+nb:.2f}/{tf+tb:.2f}ms ({(nf+nb)/(tf+tb):.2f}x)",
            flush=True,
        )
    n_sum = sum(r[1] + r[2] for r in rows)
    t_sum = sum(r[3] + r[4] for r in rows)
    print(
        f"kernel L0+L1 step budget: native {n_sum:.1f}ms  torch {t_sum:.1f}ms  ({n_sum/t_sum:.2f}x)",
        flush=True,
    )
    # Absolute excess ms that must be explained by train ratio ~1.2
    print(
        f"absolute excess per step (native-torch) L0+L1: {n_sum - t_sum:.1f}ms "
        f"(L0 {rows[0][1]+rows[0][2]-rows[0][3]-rows[0][4]:+.1f}  "
        f"L1 {rows[1][1]+rows[1][2]-rows[1][3]-rows[1][4]:+.1f})",
        flush=True,
    )


def main():
    print("=== B) Isolated kernel budget @128 pipeline ===", flush=True)
    kernel_budget()

    print("\n=== C) Custom fit ledger on vs off (2 epochs) ===", flush=True)
    run_custom(2, True, "ledger_on")
    run_custom(2, False, "ledger_off")


if __name__ == "__main__":
    main()
