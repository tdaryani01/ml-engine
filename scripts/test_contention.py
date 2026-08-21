# test_contention.py
import time
import numpy as np
import numba
import threadpoolctl
from utils.im2col_fast import _col2im_numba_kernel, _fuse_dout_transpose_and_bias

N, C, H, W = 32, 3, 28, 28
k_h, k_w, stride, pad = 3, 3, 1, 1
out_h = (H + 2 * pad - k_h) // stride + 1
out_w = (W + 2 * pad - k_w) // stride + 1
total_rows = N * out_h * out_w
total_cols = C * k_h * k_w
out_channels = 16

dout = np.random.randn(N, out_channels, out_h, out_w).astype(np.float32)
dout_trans = np.empty((total_rows, out_channels), dtype=np.float32)
db = np.zeros((1, out_channels), dtype=np.float32)

col = np.random.randn(total_rows, total_cols).astype(np.float32)
dW_flat = np.empty((out_channels, total_cols), dtype=np.float32)
W_2d = np.random.randn(out_channels, total_cols).astype(np.float32)
dcol = np.empty((total_rows, total_cols), dtype=np.float32)
dx = np.zeros((N, C, H, W), dtype=np.float32)

# Warmup
_fuse_dout_transpose_and_bias(dout, dout_trans, db)
np.dot(dout_trans.T, col, out=dW_flat)
np.dot(dout_trans, W_2d, out=dcol)
_col2im_numba_kernel(dcol, N, C, H, W, k_h, k_w, stride, pad, dx)

def benchmark_bwd(n_iters=500):
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _fuse_dout_transpose_and_bias(dout, dout_trans, db)
        np.dot(dout_trans.T, col, out=dW_flat)
        np.dot(dout_trans, W_2d, out=dcol)
        _col2im_numba_kernel(dcol, N, C, H, W, k_h, k_w, stride, pad, dx)
    return (time.perf_counter() - t0) * 1000.0 / n_iters

print("=" * 70)
print("  BACKWARD CONV2D SWEEP: fuse_dout + 2x GEMM + col2im")
print("=" * 70)
print(f"{'Numba Threads':<16} | {'BLAS Threads':<14} | {'Latency per Iter (ms)':<22}")
print("-" * 70)

for numba_t in [1, 2, 4, 8]:
    numba.set_num_threads(numba_t)
    for blas_t in [1, 2, 4, 8]:
        with threadpoolctl.threadpool_limits(limits=blas_t, user_api='blas'):
            latency = benchmark_bwd()
            print(f"{numba_t:<16} | {blas_t:<14} | {latency:.4f} ms")
print("=" * 70)