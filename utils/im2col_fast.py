# utils/im2col_fast.py
import numpy as np
from numba import njit, float32, float64, int64, void

_im2col_sig = [
    void(float32[:, :, :, :], int64, int64, int64, int64, float32[:, :]),
    void(float64[:, :, :, :], int64, int64, int64, int64, float64[:, :]),
]

_col2im_sig = [
    void(float32[:, :], int64, int64, int64, int64, int64, int64, int64, int64, float32[:, :, :, :]),
    void(float64[:, :], int64, int64, int64, int64, int64, int64, int64, int64, float64[:, :, :, :]),
]

_maxpool_fwd_sig = [
    void(float32[:, :, :, :], int64, int64, float32[:, :, :, :], int64[:, :, :, :, :]),
    void(float64[:, :, :, :], int64, int64, float64[:, :, :, :], int64[:, :, :, :, :]),
]

_maxpool_bwd_sig = [
    void(float32[:, :, :, :], int64[:, :, :, :, :], float32[:, :, :, :]),
    void(float64[:, :, :, :], int64[:, :, :, :, :], float64[:, :, :, :]),
]

_fuse_dout_sig = [
    void(float32[:, :, :, :], float32[:, :], float32[:, :]),
    void(float64[:, :, :, :], float64[:, :], float64[:, :]),
]


@njit(_im2col_sig, fastmath=True, nogil=True)
def _im2col_numba_kernel(x, k_h, k_w, stride, pad, out_buf):
    N, C, H, W = x.shape
    out_h = (H + 2 * pad - k_h) // stride + 1
    out_w = (W + 2 * pad - k_w) // stride + 1
    spatial_k = k_h * k_w

    if pad == 0:
        for n in range(N):
            n_row_base = n * (out_h * out_w)
            for c in range(C):
                col_c_base = c * spatial_k
                for ky in range(k_h):
                    col_ky_base = col_c_base + ky * k_w
                    for kx in range(k_w):
                        col_idx = col_ky_base + kx
                        for out_y in range(out_h):
                            in_y = out_y * stride + ky
                            row_y_base = n_row_base + out_y * out_w
                            for out_x in range(out_w):
                                in_x = out_x * stride + kx
                                out_buf[row_y_base + out_x, col_idx] = x[n, c, in_y, in_x]
    else:
        for n in range(N):
            n_row_base = n * (out_h * out_w)
            for c in range(C):
                col_c_base = c * spatial_k
                for ky in range(k_h):
                    col_ky_base = col_c_base + ky * k_w
                    for kx in range(k_w):
                        col_idx = col_ky_base + kx
                        for out_y in range(out_h):
                            in_y = out_y * stride - pad + ky
                            row_y_base = n_row_base + out_y * out_w
                            for out_x in range(out_w):
                                in_x = out_x * stride - pad + kx
                                row_idx = row_y_base + out_x
                                if 0 <= in_y < H and 0 <= in_x < W:
                                    out_buf[row_idx, col_idx] = x[n, c, in_y, in_x]
                                else:
                                    out_buf[row_idx, col_idx] = 0.0


@njit(_col2im_sig, fastmath=True, nogil=True)
def _col2im_numba_kernel(col, N, C, H, W, k_h, k_w, stride, pad, dx_out):
    out_h = (H + 2 * pad - k_h) // stride + 1
    out_w = (W + 2 * pad - k_w) // stride + 1
    spatial_k = k_h * k_w

    dx_out.fill(0.0)

    if pad == 0:
        for n in range(N):
            n_row_base = n * (out_h * out_w)
            for c in range(C):
                col_c_base = c * spatial_k
                for ky in range(k_h):
                    col_ky_base = col_c_base + ky * k_w
                    for kx in range(k_w):
                        col_idx = col_ky_base + kx
                        for out_y in range(out_h):
                            in_y = out_y * stride + ky
                            row_y_base = n_row_base + out_y * out_w
                            for out_x in range(out_w):
                                in_x = out_x * stride + kx
                                dx_out[n, c, in_y, in_x] += col[row_y_base + out_x, col_idx]
    else:
        for n in range(N):
            n_row_base = n * (out_h * out_w)
            for c in range(C):
                col_c_base = c * spatial_k
                for ky in range(k_h):
                    col_ky_base = col_c_base + ky * k_w
                    for kx in range(k_w):
                        col_idx = col_ky_base + kx
                        for out_y in range(out_h):
                            in_y = out_y * stride - pad + ky
                            row_y_base = n_row_base + out_y * out_w
                            for out_x in range(out_w):
                                in_x = out_x * stride - pad + kx
                                if 0 <= in_y < H and 0 <= in_x < W:
                                    dx_out[n, c, in_y, in_x] += col[row_y_base + out_x, col_idx]


@njit(_maxpool_fwd_sig, fastmath=True, nogil=True)
def _maxpool_forward_kernel(x, pool_size, stride, out, argmax):
    N, C, H, W = x.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1

    for n in range(N):
        for c in range(C):
            for oh in range(out_h):
                h_start = oh * stride
                for ow in range(out_w):
                    w_start = ow * stride
                    
                    max_val = -1e30
                    max_h = 0
                    max_w = 0
                    
                    for kh in range(pool_size):
                        for kw in range(pool_size):
                            val = x[n, c, h_start + kh, w_start + kw]
                            if val > max_val:
                                max_val = val
                                max_h = h_start + kh
                                max_w = w_start + kw
                                
                    out[n, c, oh, ow] = max_val
                    argmax[n, c, oh, ow, 0] = max_h
                    argmax[n, c, oh, ow, 1] = max_w


@njit(_maxpool_bwd_sig, fastmath=True, nogil=True)
def _maxpool_backward_kernel(dout, argmax, dx):
    dx.fill(0.0)
    N, C, out_h, out_w = dout.shape
    for n in range(N):
        for c in range(C):
            for oh in range(out_h):
                for ow in range(out_w):
                    h_idx = argmax[n, c, oh, ow, 0]
                    w_idx = argmax[n, c, oh, ow, 1]
                    dx[n, c, h_idx, w_idx] += dout[n, c, oh, ow]


@njit(_fuse_dout_sig, fastmath=True, nogil=True)
def _fuse_dout_transpose_and_bias(dout, dout_trans_out, db_out):
    N, OutC, H_out, W_out = dout.shape
    spatial_size = H_out * W_out
    inv_m = 1.0 / float(N)

    for n in range(N):
        row_n_offset = n * spatial_size
        for h in range(H_out):
            row_h_offset = row_n_offset + h * W_out
            for w in range(W_out):
                row_idx = row_h_offset + w
                for c in range(OutC):
                    dout_trans_out[row_idx, c] = dout[n, c, h, w]

    for c in range(OutC):
        sum_val = 0.0
        for n in range(N):
            for h in range(H_out):
                for w in range(W_out):
                    sum_val += dout[n, c, h, w]
        db_out[0, c] = sum_val * inv_m


def im2col_fast(x: np.ndarray, k_h: int, k_w: int, stride: int = 1, pad: int = 0, out_buf: np.ndarray = None) -> np.ndarray:
    if not x.flags['C_CONTIGUOUS']:
        x = np.ascontiguousarray(x)

    N, C, H, W = x.shape
    out_h = (H + 2 * pad - k_h) // stride + 1
    out_w = (W + 2 * pad - k_w) // stride + 1
    total_rows = N * out_h * out_w
    total_cols = C * k_h * k_w

    if out_buf is None or out_buf.shape != (total_rows, total_cols) or out_buf.dtype != x.dtype:
        out_buf = np.empty((total_rows, total_cols), dtype=x.dtype)

    _im2col_numba_kernel(x, k_h, k_w, stride, pad, out_buf)
    return out_buf


def col2im_fast(col: np.ndarray, input_shape: tuple, k_h: int, k_w: int, stride: int = 1, pad: int = 0, out_buf: np.ndarray = None) -> np.ndarray:
    if not col.flags['C_CONTIGUOUS']:
        col = np.ascontiguousarray(col)

    N, C, H, W = input_shape
    if out_buf is None or out_buf.shape != (N, C, H, W) or out_buf.dtype != col.dtype:
        out_buf = np.zeros((N, C, H, W), dtype=col.dtype)

    _col2im_numba_kernel(col, N, C, H, W, k_h, k_w, stride, pad, out_buf)
    return out_buf


def fuse_dout_transpose_bias_fast(dout: np.ndarray, dout_trans_buf: np.ndarray, db_buf: np.ndarray):
    if not dout.flags['C_CONTIGUOUS']:
        dout = np.ascontiguousarray(dout)
    _fuse_dout_transpose_and_bias(dout, dout_trans_buf, db_buf)

