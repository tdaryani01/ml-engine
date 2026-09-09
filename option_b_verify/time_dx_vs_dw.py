"""Read-only diagnostic: isolate wall-time cost of backward-dX vs backward-dW
for the real second-conv-layer dims, using the actual exported native entry
points (no source changes). Purpose: decide whether dW is worth attacking,
and how large the dx/dw interleaving loss plausibly is, before writing any
new kernel code.
"""
import ctypes
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
SO = os.path.join(ROOT, "bin", "conv_kernels.so")

N, C_in, C_out, K, PAD, STRIDE = 32, 8, 16, 7, 1, 1
H_in = W_in = 12
H_out = W_out = (H_in + 2 * PAD - K) // STRIDE + 1  # 8
REPS = 60
WARMUP = 5

os.environ.setdefault("OMP_NUM_THREADS", "4")

lib = ctypes.CDLL(SO)
if hasattr(lib, "configure_native_threads"):
    lib.configure_native_threads.argtypes = [ctypes.c_int32]
    lib.configure_native_threads(4)

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
lib.direct_conv2d_backward_fused_avx2.restype = ctypes.c_int32
lib.direct_conv2d_backward_fused_avx2.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_float, ctypes.c_int32,
]

rng = np.random.default_rng(7)
dY = rng.uniform(-1, 1, size=(N, C_out, H_out, W_out)).astype(np.float32)
X = rng.uniform(-1, 1, size=(N, C_in, H_in, W_in)).astype(np.float32)
W = rng.uniform(-1, 1, size=(C_out, C_in, K, K)).astype(np.float32)
dx = np.zeros((N, C_in, H_in, W_in), dtype=np.float32)
dW = np.zeros((C_out, C_in, K, K), dtype=np.float32)

dY_p, X_p, W_p, dx_p, dW_p = (a.ctypes.data_as(ctypes.c_void_p) for a in (dY, X, W, dx, dW))


def time_dx(reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        lib.direct_conv2d_backward_input_avx2(
            dY_p, W_p, None, dx_p,
            N, C_in, H_in, W_in, W_in, C_out, K, K, STRIDE, PAD, W_out, 0
        )
    return time.perf_counter() - t0


def time_dw(reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        lib.direct_conv2d_backward_weight_avx2(
            dY_p, X_p, dW_p,
            N, C_in, H_in, W_in, W_in, C_out, K, K, STRIDE, PAD, W_out, 1.0
        )
    return time.perf_counter() - t0


def time_fused(reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        lib.direct_conv2d_backward_fused_avx2(
            dY_p, X_p, W_p, None, dx_p, dW_p,
            N, C_in, H_in, W_in, W_in, C_out, K, K, STRIDE, PAD, W_out,
            1.0, 0
        )
    return time.perf_counter() - t0


def run_sustained(seconds: float):
    """Loop dx/dw/fused calls continuously for `seconds` wall time, so a
    sampling profiler (uProf) gets enough samples inside dw_nci/dx tile code
    to attribute instruction-level time. No Python/ledger overhead in the loop.
    """
    time_dx(WARMUP)
    time_dw(WARMUP)
    time_fused(WARMUP)
    t_end = time.perf_counter() + seconds
    n_dx = n_dw = n_fused = 0
    while time.perf_counter() < t_end:
        time_dx(50); n_dx += 50
        time_dw(50); n_dw += 50
        time_fused(50); n_fused += 50
    print(f"sustained: dx_calls={n_dx} dw_calls={n_dw} fused_calls={n_fused} over {seconds}s")


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "default"
    if label == "sustained":
        secs = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
        print(f"dims: N={N} C_in={C_in} C_out={C_out} K={K} H_in={H_in} W_in={W_in} -> H_out={H_out} W_out={W_out}")
        run_sustained(secs)
        return
    print(f"dims: N={N} C_in={C_in} C_out={C_out} K={K} H_in={H_in} W_in={W_in} -> H_out={H_out} W_out={W_out}")
    time_dx(WARMUP)
    time_dw(WARMUP)
    time_fused(WARMUP)
    dx_t = time_dx(REPS) / REPS * 1e3
    dw_t = time_dw(REPS) / REPS * 1e3
    fused_t = time_fused(REPS) / REPS * 1e3
    print(f"{label}: dx={dx_t:.4f} dw={dw_t:.4f} sum={dx_t+dw_t:.4f} fused={fused_t:.4f} ms/call  "
          f"(fused vs sum: {(fused_t/(dx_t+dw_t)-1)*100:+.1f}%)")


if __name__ == "__main__":
    # NOTE: ML_ENGINE_FORCE_DX_CRAWL is read once and cached per-process
    # (matches production behavior), so crawl vs cin-blocked are run as two
    # separate process invocations by the shell wrapper, not two calls here.
    main()
