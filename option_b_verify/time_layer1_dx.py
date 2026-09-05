"""Read-only diagnostic: isolate wall-time cost of forward/backward-dX/backward-dW
for layer 1's actual dims (C_in=3, C_out=8, K=7, 28x28 -> 24x24), which cannot
use Option B (cin-blocked) since C_in=3 fails the C_in % 8 == 0 gate and falls
back to the crawl tile-queue path (stride1_bwd_dx_tile_c8_k7, C_out==8 specialist).

No source changes. Purpose: get a concrete before-number for layer 1's dX crawl
cost, matching config.yaml's actual layer 1 shape, prior to considering any
kernel change.
"""
import ctypes
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
SO = os.path.join(ROOT, "bin", "conv_kernels.so")

N, C_in, C_out, K, PAD, STRIDE = 32, 3, 8, 7, 1, 1
H_in = W_in = 28
H_out = W_out = (H_in + 2 * PAD - K) // STRIDE + 1  # 24
REPS = 60
WARMUP = 5

os.environ.setdefault("OMP_NUM_THREADS", "4")

lib = ctypes.CDLL(SO)
if hasattr(lib, "configure_native_threads"):
    lib.configure_native_threads.argtypes = [ctypes.c_int32]
    lib.configure_native_threads(4)

lib.direct_conv2d_forward_avx2.restype = ctypes.c_int32
lib.direct_conv2d_forward_avx2.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_int64, ctypes.c_int32,
]
lib.direct_conv2d_backward_input_avx2.restype = ctypes.c_int32
lib.direct_conv2d_backward_input_avx2.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_int32,
]
lib.direct_conv2d_backward_weight_avx2.restype = ctypes.c_int32
lib.direct_conv2d_backward_weight_avx2.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_float,
]

rng = np.random.default_rng(11)
X = rng.uniform(-1, 1, size=(N, C_in, H_in, W_in)).astype(np.float32)
Wt = rng.uniform(-1, 1, size=(C_out, C_in, K, K)).astype(np.float32)
b = np.zeros(C_out, dtype=np.float32)
out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
dY = rng.uniform(-1, 1, size=(N, C_out, H_out, W_out)).astype(np.float32)
dx = np.zeros((N, C_in, H_in, W_in), dtype=np.float32)
dW = np.zeros((C_out, C_in, K, K), dtype=np.float32)

Xp, Wp, bp, op, dYp, dxp, dWp = (
    a.ctypes.data_as(ctypes.c_void_p) for a in (X, Wt, b, out, dY, dx, dW)
)


def time_fwd(reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        lib.direct_conv2d_forward_avx2(
            Xp, Wp, bp, op, N, C_in, H_in, W_in, W_in, C_out, K, K, STRIDE, PAD, W_out, 0
        )
    return time.perf_counter() - t0


def time_dx(reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        lib.direct_conv2d_backward_input_avx2(
            dYp, Wp, None, dxp,
            N, C_in, H_in, W_in, W_in, C_out, K, K, STRIDE, PAD, W_out, 0
        )
    return time.perf_counter() - t0


def time_dw(reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        lib.direct_conv2d_backward_weight_avx2(
            dYp, Xp, dWp,
            N, C_in, H_in, W_in, W_in, C_out, K, K, STRIDE, PAD, W_out, 1.0
        )
    return time.perf_counter() - t0


def run_sustained_dx(seconds: float):
    """Loop dX calls continuously for `seconds` wall time so a sampling
    profiler (uProf) gets enough samples inside stride1_bwd_dx_tile_c8_k7 to
    attribute instruction-level time. No Python/ledger overhead in the loop.
    """
    time_dx(WARMUP)
    t_end = time.perf_counter() + seconds
    n = 0
    while time.perf_counter() < t_end:
        time_dx(50)
        n += 50
    print(f"sustained: dx_calls={n} over {seconds}s")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "sustained":
        secs = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
        print(f"dims: N={N} C_in={C_in} C_out={C_out} K={K} H_in={H_in} W_in={W_in} -> H_out={H_out} W_out={W_out}")
        run_sustained_dx(secs)
        return
    print(f"dims: N={N} C_in={C_in} C_out={C_out} K={K} H_in={H_in} W_in={W_in} -> H_out={H_out} W_out={W_out}")
    time_fwd(WARMUP)
    time_dx(WARMUP)
    time_dw(WARMUP)
    fwd_t = time_fwd(REPS) / REPS * 1e3
    dx_t = time_dx(REPS) / REPS * 1e3
    dw_t = time_dw(REPS) / REPS * 1e3
    print(f"layer1: fwd={fwd_t:.4f} dx={dx_t:.4f} dw={dw_t:.4f} sum_bwd={dx_t+dw_t:.4f} ms/call")


if __name__ == "__main__":
    main()
