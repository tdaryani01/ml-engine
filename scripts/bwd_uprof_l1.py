#!/usr/bin/env python3
"""Tight L1 fused-bwd loop for uProf Memory/Cache (N=32, 8->16 @63, K=7 pad=2)."""
from __future__ import annotations

import ctypes
import os
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OMP_THREAD_LIMIT", "4")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SO = os.path.join(ROOT, "bin", "conv_kernels.so")

N, Cin, Cout, H, K, pad = 32, 8, 16, 63, 7, 2
oh = H + 2 * pad - K + 1
REPS = int(os.environ.get("BWD_UPROF_REPS", "800"))
WARM = 20

lib = ctypes.CDLL(SO)
lib.configure_native_threads.argtypes = [ctypes.c_int32]
lib.configure_native_threads(4)
lib.direct_conv2d_backward_fused_avx2.restype = ctypes.c_int32
lib.direct_conv2d_backward_fused_avx2.argtypes = (
    [ctypes.c_void_p] * 6 + [ctypes.c_int64] * 11 + [ctypes.c_float, ctypes.c_int32]
)

rng = np.random.default_rng(7)
dY = np.ascontiguousarray(rng.uniform(-1, 1, (N, Cout, oh, oh)).astype(np.float32))
X = np.ascontiguousarray(rng.uniform(-1, 1, (N, Cin, H, H)).astype(np.float32))
W = np.ascontiguousarray(rng.uniform(-1, 1, (Cout, Cin, K, K)).astype(np.float32))
dx = np.zeros_like(X)
dW = np.zeros_like(W)
yp, xp, wp, dxp, dwp = (a.ctypes.data_as(ctypes.c_void_p) for a in (dY, X, W, dx, dW))


def once():
    lib.direct_conv2d_backward_fused_avx2(
        yp, xp, wp, None, dxp, dwp,
        N, Cin, H, H, H, Cout, K, K, 1, pad, oh, 1.0, 0,
    )


for _ in range(WARM):
    once()

print(f"[bwd_uprof_l1] start reps={REPS}", flush=True)
t0 = time.perf_counter()
for _ in range(REPS):
    once()
dt = time.perf_counter() - t0
print(f"[bwd_uprof_l1] done wall={dt:.3f}s ms/call={dt/REPS*1e3:.2f}", flush=True)
