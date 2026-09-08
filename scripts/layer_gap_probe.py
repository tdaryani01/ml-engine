#!/usr/bin/env python3
"""Diagnostic: time native vs Torch per CNN layer (fwd + fused bwd).

Ledger-free kernel probe — isolates whether L0 (Cin=3) or L1 (Cin%8==0)
still loses to oneDNN on the current K. Not a training-loop bench.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OMP_THREAD_LIMIT", "4")

from config.constants import EngineBackend
from utils.conv_dispatch import (
    conv2d_backward_fused,
    conv2d_forward,
    init_engine_backend,
    sync_native_thread_policy,
)


def _out_hw(H, W, K, stride, pad):
    oh = (H + 2 * pad - K) // stride + 1
    ow = (W + 2 * pad - K) // stride + 1
    return oh, ow


def bench_layer(name: str, N: int, Cin: int, Cout: int, H: int, W: int, K: int, *, pad: int = 2, reps: int = 80, warmup: int = 10):
    stride = 1
    oh, ow = _out_hw(H, W, K, stride, pad)
    rng = np.random.default_rng(0)
    x = np.ascontiguousarray(rng.standard_normal((N, Cin, H, W), dtype=np.float32))
    W_n = np.ascontiguousarray(rng.standard_normal((Cout, Cin, K, K), dtype=np.float32))
    b = np.zeros((Cout,), dtype=np.float32)
    dy = np.ascontiguousarray(rng.standard_normal((N, Cout, oh, ow), dtype=np.float32))
    out = np.empty((N, Cout, oh, ow), dtype=np.float32)
    dx = np.empty_like(x)
    dW = np.empty_like(W_n)

    xt = torch.from_numpy(x.copy()).to(memory_format=torch.channels_last).contiguous()
    Wt = torch.from_numpy(W_n.copy()).to(memory_format=torch.channels_last).contiguous()
    dyt = torch.from_numpy(dy.copy()).to(memory_format=torch.channels_last).contiguous()
    xt.requires_grad_(True)
    Wt.requires_grad_(True)

    torch.set_num_threads(4)
    sync_native_thread_policy(4)

    # warmup
    for _ in range(warmup):
        conv2d_forward(x, W_n, b, out_buf=out, stride=stride, pad=pad, backend=EngineBackend.NATIVE)
        conv2d_backward_fused(
            dy, x, W_n, dx, dW, stride=stride, pad=pad, inv_m=1.0 / N, backend=EngineBackend.NATIVE
        )
        y = F.conv2d(xt, Wt, None, stride=stride, padding=pad)
        (y * dyt).sum().backward()
        xt.grad = None
        Wt.grad = None

    def time_native_fwd():
        t0 = time.perf_counter()
        for _ in range(reps):
            conv2d_forward(x, W_n, b, out_buf=out, stride=stride, pad=pad, backend=EngineBackend.NATIVE)
        return (time.perf_counter() - t0) / reps * 1e3

    def time_native_bwd():
        t0 = time.perf_counter()
        for _ in range(reps):
            conv2d_backward_fused(
                dy, x, W_n, dx, dW, stride=stride, pad=pad, inv_m=1.0 / N, backend=EngineBackend.NATIVE
            )
        return (time.perf_counter() - t0) / reps * 1e3

    def time_torch_fwd():
        t0 = time.perf_counter()
        for _ in range(reps):
            F.conv2d(xt, Wt, None, stride=stride, padding=pad)
        return (time.perf_counter() - t0) / reps * 1e3

    def time_torch_bwd():
        t0 = time.perf_counter()
        for _ in range(reps):
            y = F.conv2d(xt, Wt, None, stride=stride, padding=pad)
            (y * dyt).sum().backward()
            xt.grad = None
            Wt.grad = None
        return (time.perf_counter() - t0) / reps * 1e3

    nf, nb = time_native_fwd(), time_native_bwd()
    tf, tb = time_torch_fwd(), time_torch_bwd()
    print(
        f"{name:8s} N={N} {Cin}->{Cout} {H}x{W} K={K}  "
        f"fwd native {nf:6.3f}ms torch {tf:6.3f}ms ({nf/tf:5.2f}x)  "
        f"bwd native {nb:6.3f}ms torch {tb:6.3f}ms ({nb/tb:5.2f}x)"
    )
    return nf, nb, tf, tb


def main():
    init_engine_backend(EngineBackend.NATIVE)
    sync_native_thread_policy(4)
    print("layer gap probe (fused conv only; no ledger/pool/dense)")
    print("--- K=7 pad=2 (bench geom) ---")
    # L0: 28x28 pad2 K7 -> 26x26; after pool2 -> 13x13
    bench_layer("L0", 32, 3, 8, 28, 28, 7, pad=2)
    bench_layer("L1", 32, 8, 16, 13, 13, 7, pad=2)


if __name__ == "__main__":
    main()
