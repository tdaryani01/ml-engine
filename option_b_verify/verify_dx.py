"""Diagnostic-only correctness check for the Option B cin-blocked backward-dX
experiment in conv_fallback.cpp. Calls the REAL exported native entry point
`direct_conv2d_backward_input_avx2` (already used in production by
utils/conv_dispatch.py) with identical, fixed-seed inputs, once with the new
cin-blocked path (default) and once with ML_ENGINE_FORCE_DX_CRAWL=1 (forces
the old, unmodified sliding-window path). Compares the two dx outputs, and
also compares both against an independent NumPy reference.

This does not touch or import any training code; it is a standalone check.
"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
SCRATCH = os.path.dirname(__file__)

# Matches the second conv layer dims used in the earlier microbenchmark.
N, C_in, C_out, PAD, STRIDE = 32, 8, 16, 1, 1
H_in = W_in = 12


def numpy_reference(dY: np.ndarray, W: np.ndarray, K: int, H_out: int, W_out: int) -> np.ndarray:
    dX = np.zeros((N, C_in, H_in, W_in), dtype=np.float64)
    dYd = dY.astype(np.float64)
    Wd = W.astype(np.float64)
    for cout in range(C_out):
        for kh in range(K):
            for kw in range(K):
                for ih in range(H_in):
                    oh = ih + PAD - kh
                    if oh < 0 or oh >= H_out:
                        continue
                    for iw in range(W_in):
                        ow = iw + PAD - kw
                        if ow < 0 or ow >= W_out:
                            continue
                        dX[:, :, ih, iw] += (
                            dYd[:, cout, oh, ow][:, None] * Wd[cout, :, kh, kw][None, :]
                        )
    return dX.astype(np.float32)


def run_native(K: int, H_out: int, W_out: int, force_crawl: bool) -> np.ndarray:
    env = os.environ.copy()
    if force_crawl:
        env["ML_ENGINE_FORCE_DX_CRAWL"] = "1"
    else:
        env.pop("ML_ENGINE_FORCE_DX_CRAWL", None)

    so_path = os.path.join(ROOT, "bin", "conv_kernels.so")
    dy_path = os.path.join(SCRATCH, "_dy.npy")
    w_path = os.path.join(SCRATCH, "_w.npy")
    out_path = os.path.join(SCRATCH, "_dx_out.npy")

    script = f"""
import ctypes, numpy as np
lib = ctypes.CDLL({so_path!r})
lib.direct_conv2d_backward_input_avx2.restype = ctypes.c_int32
lib.direct_conv2d_backward_input_avx2.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_int32,
]
dY = np.load({dy_path!r})
W = np.load({w_path!r})
dx = np.zeros(({N}, {C_in}, {H_in}, {W_in}), dtype=np.float32)
status = lib.direct_conv2d_backward_input_avx2(
    dY.ctypes.data_as(ctypes.c_void_p), W.ctypes.data_as(ctypes.c_void_p), None,
    dx.ctypes.data_as(ctypes.c_void_p),
    {N}, {C_in}, {H_in}, {W_in}, {W_in}, {C_out}, {K}, {K}, {STRIDE}, {PAD}, {W_out},
    0
)
assert status == 0, f"native call failed: {{status}}"
np.save({out_path!r}, dx)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("native subprocess failed")
    return np.load(out_path)


def check_one_k(K: int, seed: int) -> bool:
    H_out = W_out = (H_in + 2 * PAD - K) // STRIDE + 1
    rng = np.random.default_rng(seed)
    dY = rng.uniform(-1, 1, size=(N, C_out, H_out, W_out)).astype(np.float32)
    W = rng.uniform(-1, 1, size=(C_out, C_in, K, K)).astype(np.float32)
    np.save(os.path.join(SCRATCH, "_dy.npy"), dY)
    np.save(os.path.join(SCRATCH, "_w.npy"), W)

    dx_ref = numpy_reference(dY, W, K, H_out, W_out)
    dx_new = run_native(K, H_out, W_out, force_crawl=False)  # cin-blocked (Option B)
    dx_old = run_native(K, H_out, W_out, force_crawl=True)   # original crawl path

    diff_new_ref = np.max(np.abs(dx_new - dx_ref))
    diff_old_ref = np.max(np.abs(dx_old - dx_ref))
    diff_new_old = np.max(np.abs(dx_new - dx_old))

    print(f"K={K}: max|new-ref|={diff_new_ref:.3g}  max|old-ref|={diff_old_ref:.3g}  max|new-old|={diff_new_old:.3g}")

    tol = 1e-2
    return diff_new_ref < tol and diff_old_ref < tol and diff_new_old < tol


def main() -> None:
    results = {K: check_one_k(K, seed=42 + K) for K in (5, 6, 7)}
    for K, ok in results.items():
        print(f"K={K}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
