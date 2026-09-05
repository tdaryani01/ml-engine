"""Read-only correctness check: Option D (C_out-blocked backward-dX) vs a
NumPy reference and vs the old crawl path, for layer 1's exact dims
(C_in=3, C_out=8, K=7, 28x28 -> 24x24, N=32).

No production behavior is trusted from this file alone -- it just verifies
gradients match before any timing claim is made.
"""
import ctypes
import os

import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
SO = os.path.join(ROOT, "bin", "conv_kernels.so")

N, C_in, C_out, K, PAD, STRIDE = 32, 3, 8, 7, 1, 1
H_in = W_in = 28
H_out = W_out = (H_in + 2 * PAD - K) // STRIDE + 1  # 24

os.environ.setdefault("OMP_NUM_THREADS", "4")

lib = ctypes.CDLL(SO)
lib.direct_conv2d_backward_input_avx2.restype = ctypes.c_int32
lib.direct_conv2d_backward_input_avx2.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_int32,
]


def numpy_reference_dx(dY, W):
    dX = np.zeros((N, C_in, H_in, W_in), dtype=np.float64)
    dYd = dY.astype(np.float64)
    Wd = W.astype(np.float64)
    for kh in range(K):
        for kw in range(K):
            # oh = ih - kh + PAD  =>  ih = oh + kh - PAD
            for oh in range(H_out):
                ih = oh + kh - PAD
                if ih < 0 or ih >= H_in:
                    continue
                for ow in range(W_out):
                    iw = ow + kw - PAD
                    if iw < 0 or iw >= W_in:
                        continue
                    # dX[n,cin,ih,iw] += sum_cout dY[n,cout,oh,ow] * W[cout,cin,kh,kw]
                    dX[:, :, ih, iw] += np.einsum(
                        "nc,cd->nd", dYd[:, :, oh, ow], Wd[:, :, kh, kw]
                    )
    return dX


def run_native(dY, W, force_crawl):
    dx = np.zeros((N, C_in, H_in, W_in), dtype=np.float32)
    env_key = "ML_ENGINE_FORCE_DX_CRAWL"
    old = os.environ.get(env_key)
    if force_crawl:
        os.environ[env_key] = "1"
    elif env_key in os.environ:
        del os.environ[env_key]
    try:
        dY_p = dY.ctypes.data_as(ctypes.c_void_p)
        W_p = W.ctypes.data_as(ctypes.c_void_p)
        dx_p = dx.ctypes.data_as(ctypes.c_void_p)
        rc = lib.direct_conv2d_backward_input_avx2(
            dY_p, W_p, None, dx_p,
            N, C_in, H_in, W_in, W_in, C_out, K, K, STRIDE, PAD, W_out, 0
        )
        assert rc == 0, f"native call failed rc={rc}"
    finally:
        if old is not None:
            os.environ[env_key] = old
        elif env_key in os.environ:
            del os.environ[env_key]
    return dx


def main():
    rng = np.random.default_rng(123)
    dY = rng.uniform(-1, 1, size=(N, C_out, H_out, W_out)).astype(np.float32)
    W = rng.uniform(-1, 1, size=(C_out, C_in, K, K)).astype(np.float32)

    print("Computing NumPy reference (slow, one-time)...")
    dx_ref = numpy_reference_dx(dY, W)

    dx_crawl = run_native(dY, W, force_crawl=True)
    dx_optd = run_native(dY, W, force_crawl=False)

    err_crawl = np.max(np.abs(dx_ref - dx_crawl))
    err_optd = np.max(np.abs(dx_ref - dx_optd))
    err_crawl_vs_optd = np.max(np.abs(dx_crawl - dx_optd))

    print(f"crawl   vs numpy ref: max abs err = {err_crawl:.6e}")
    print(f"Opt D   vs numpy ref: max abs err = {err_optd:.6e}")
    print(f"crawl   vs Opt D:     max abs err = {err_crawl_vs_optd:.6e}")

    ok = err_crawl < 1e-2 and err_optd < 1e-2 and err_crawl_vs_optd < 1e-2
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
