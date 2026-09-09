"""Correctness check for the batched-reduce dw_nci change (K=7): calls the
real exported direct_conv2d_backward_weight_avx2 and diffs against an
independent NumPy reference dW computation.
"""
import ctypes
import os

import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
SO = os.path.join(ROOT, "bin", "conv_kernels.so")

N, C_in, C_out, K, PAD, STRIDE = 32, 8, 16, 7, 1, 1
H_in = W_in = 12
H_out = W_out = (H_in + 2 * PAD - K) // STRIDE + 1  # 8


def numpy_dw_reference(dY: np.ndarray, X: np.ndarray) -> np.ndarray:
    dW = np.zeros((C_out, C_in, K, K), dtype=np.float64)
    dYd = dY.astype(np.float64)
    Xd = X.astype(np.float64)
    for kh in range(K):
        for kw in range(K):
            for oh in range(H_out):
                ih = oh - PAD + kh
                if ih < 0 or ih >= H_in:
                    continue
                for ow in range(W_out):
                    iw = ow - PAD + kw
                    if iw < 0 or iw >= W_in:
                        continue
                    # dY[:, cout, oh, ow] (N,) x X[:, cin, ih, iw] (N,) -> (cout, cin)
                    dW[:, :, kh, kw] += np.einsum(
                        "nc,nd->cd", dYd[:, :, oh, ow], Xd[:, :, ih, iw]
                    )
    return dW.astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(11)
    dY = rng.uniform(-1, 1, size=(N, C_out, H_out, W_out)).astype(np.float32)
    X = rng.uniform(-1, 1, size=(N, C_in, H_in, W_in)).astype(np.float32)
    dW = np.zeros((C_out, C_in, K, K), dtype=np.float32)

    lib = ctypes.CDLL(SO)
    lib.direct_conv2d_backward_weight_avx2.restype = ctypes.c_int32
    lib.direct_conv2d_backward_weight_avx2.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_float,
    ]
    status = lib.direct_conv2d_backward_weight_avx2(
        dY.ctypes.data_as(ctypes.c_void_p), X.ctypes.data_as(ctypes.c_void_p),
        dW.ctypes.data_as(ctypes.c_void_p),
        N, C_in, H_in, W_in, W_in, C_out, K, K, STRIDE, PAD, W_out, 1.0
    )
    assert status == 0, f"native call failed: {status}"

    dw_ref = numpy_dw_reference(dY, X)
    diff = np.max(np.abs(dW - dw_ref))
    rel = diff / (np.max(np.abs(dw_ref)) + 1e-8)
    print(f"max|native - ref| = {diff:.6g}  (rel {rel:.3g})")
    tol = 5e-2  # float32 accumulation across N*H_out*W_out ~ 2048 terms per (cout,cin,kh,kw)
    print("CORRECTNESS:", "PASS" if diff < tol else "FAIL")


if __name__ == "__main__":
    main()
