# utils/im2col.py
import os
import numpy as np

BACKEND = os.getenv("ENGINE_BACKEND", "fast").lower()

_USE_FAST = False
try:
    if BACKEND in ("fast", "numba"):
        from utils.im2col_fast import (
            im2col_fast as _im2col_impl,
            col2im_fast as _col2im_impl,
            _maxpool_forward_kernel,
            _maxpool_backward_kernel,
            fuse_dout_transpose_bias_fast as _fuse_dout_impl
        )
        _USE_FAST = True
        print("[Engine Backend] Initialized with Numba JIT C-Kernels.")
    else:
        raise ImportError("Forced pure NumPy backend.")
except Exception as e:
    BACKEND = "numpy"
    _USE_FAST = False
    print(f"[Engine Backend] Using Pure NumPy backend. (Reason: {e})")


def im2col(x: np.ndarray, k_h: int, k_w: int, stride: int = 1, pad: int = 0, out_buf: np.ndarray = None) -> np.ndarray:
    if _USE_FAST:
        return _im2col_impl(x, k_h, k_w, stride, pad, out_buf=out_buf)

    N, C, H, W = x.shape
    out_h = (H + 2 * pad - k_h) // stride + 1
    out_w = (W + 2 * pad - k_w) // stride + 1
    img = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant') if pad > 0 else x

    total_rows = N * out_h * out_w
    total_cols = C * k_h * k_w
    col = out_buf if (out_buf is not None and out_buf.shape == (total_rows, total_cols) and out_buf.dtype == x.dtype) else np.empty((total_rows, total_cols), dtype=x.dtype)

    col_idx = 0
    for c in range(C):
        for ky in range(k_h):
            for kx in range(k_w):
                y_max = ky + stride * out_h
                x_max = kx + stride * out_w
                col[:, col_idx] = img[:, c, ky:y_max:stride, kx:x_max:stride].reshape(-1)
                col_idx += 1
    return col


def col2im(col: np.ndarray, input_shape: tuple, k_h: int, k_w: int, stride: int = 1, pad: int = 0, out_buf: np.ndarray = None) -> np.ndarray:
    if _USE_FAST:
        return _col2im_impl(col, input_shape, k_h, k_w, stride, pad, out_buf=out_buf)

    N, C, H, W = input_shape
    out_h = (H + 2 * pad - k_h) // stride + 1
    out_w = (W + 2 * pad - k_w) // stride + 1
    pad_h, pad_w = H + 2 * pad, W + 2 * pad

    if out_buf is not None and out_buf.shape == (N, C, pad_h, pad_w) and out_buf.dtype == col.dtype:
        img = out_buf
        img.fill(0)
    else:
        img = np.zeros((N, C, pad_h, pad_w), dtype=col.dtype)

    col_idx = 0
    for c in range(C):
        for ky in range(k_h):
            for kx in range(k_w):
                y_max = ky + stride * out_h
                x_max = kx + stride * out_w
                img[:, c, ky:y_max:stride, kx:x_max:stride] += col[:, col_idx].reshape(N, out_h, out_w)
                col_idx += 1

    return img if pad == 0 else img[:, :, pad:-pad, pad:-pad]


def maxpool_forward(x: np.ndarray, pool_size: int, stride: int, out_buf: np.ndarray = None, argmax_buf: np.ndarray = None):
    N, C, H, W = x.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1

    if _USE_FAST:
        if out_buf is None or out_buf.shape != (N, C, out_h, out_w) or out_buf.dtype != x.dtype:
            out_buf = np.empty((N, C, out_h, out_w), dtype=x.dtype)
        if argmax_buf is None or argmax_buf.shape != (N, C, out_h, out_w, 2):
            argmax_buf = np.empty((N, C, out_h, out_w, 2), dtype=np.int64)

        _maxpool_forward_kernel(x, pool_size, stride, out_buf, argmax_buf)
        return out_buf, argmax_buf

    x_reshaped = x[:, :, :out_h * stride, :out_w * stride].reshape(N, C, out_h, stride, out_w, stride)
    out = x_reshaped.max(axis=(3, 5))
    mask = (x_reshaped == out[:, :, :, None, :, None])
    return out, mask


def maxpool_backward(dout: np.ndarray, cache, x_shape: tuple, pool_size: int, stride: int, dx_buf: np.ndarray = None):
    if _USE_FAST:
        if dx_buf is None or dx_buf.shape != x_shape or dx_buf.dtype != dout.dtype:
            dx_buf = np.zeros(x_shape, dtype=dout.dtype)
        _maxpool_backward_kernel(dout, cache, dx_buf)
        return dx_buf

    mask = cache
    dx_reshaped = mask * dout[:, :, :, None, :, None]
    return np.ascontiguousarray(dx_reshaped.reshape(x_shape))


def fuse_dout_transpose_and_bias(dout: np.ndarray, dout_trans_buf: np.ndarray, db_buf: np.ndarray):
    if _USE_FAST:
        _fuse_dout_impl(dout, dout_trans_buf, db_buf)
        return

    m = dout.shape[0]
    dout_trans_buf[:] = dout.transpose(0, 2, 3, 1).reshape(-1, dout.shape[1])
    db_buf[:] = np.sum(dout_trans_buf, axis=0, keepdims=True) / m