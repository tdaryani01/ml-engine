# utils/im2col_fast.py
import ctypes
import numpy as np
from numba import njit, prange, float32, float64, int64, void
from scipy.linalg import cython_blas

# -----------------------------------------------------------------------------
# Direct Native BLAS C-Function Pointers via Scipy Cython Interface
# -----------------------------------------------------------------------------
ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]

ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

_sgemm_capsule = cython_blas.__pyx_capi__['sgemm']
_dgemm_capsule = cython_blas.__pyx_capi__['dgemm']

_sgemm_name = ctypes.pythonapi.PyCapsule_GetName(_sgemm_capsule)
_dgemm_name = ctypes.pythonapi.PyCapsule_GetName(_dgemm_capsule)

_sgemm_addr = ctypes.pythonapi.PyCapsule_GetPointer(_sgemm_capsule, _sgemm_name)
_dgemm_addr = ctypes.pythonapi.PyCapsule_GetPointer(_dgemm_capsule, _dgemm_name)

_BLAS_FUNC_TYPE_S = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_char),
    ctypes.POINTER(ctypes.c_char),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int)
)

_BLAS_FUNC_TYPE_D = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_char),
    ctypes.POINTER(ctypes.c_char),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int)
)

_raw_sgemm = _BLAS_FUNC_TYPE_S(_sgemm_addr)
_raw_dgemm = _BLAS_FUNC_TYPE_D(_dgemm_addr)


def _sgemm(trans_a: bytes, trans_b: bytes, m: int, n: int, k: int,
           alpha: float, a_ptr, lda: int, b_ptr, ldb: int,
           beta: float, c_ptr, ldc: int) -> None:
    """Thread-safe SGEMM: all BLAS scalar args are stack-local per call."""
    ta = ctypes.c_char(trans_a)
    tb = ctypes.c_char(trans_b)
    mi = ctypes.c_int(m)
    ni = ctypes.c_int(n)
    ki = ctypes.c_int(k)
    ldai = ctypes.c_int(lda)
    ldbi = ctypes.c_int(ldb)
    ldci = ctypes.c_int(ldc)
    alphaf = ctypes.c_float(alpha)
    betaf = ctypes.c_float(beta)
    _raw_sgemm(
        ctypes.byref(ta), ctypes.byref(tb),
        ctypes.byref(mi), ctypes.byref(ni), ctypes.byref(ki),
        ctypes.byref(alphaf),
        a_ptr, ctypes.byref(ldai),
        b_ptr, ctypes.byref(ldbi),
        ctypes.byref(betaf),
        c_ptr, ctypes.byref(ldci),
    )


def _dgemm(trans_a: bytes, trans_b: bytes, m: int, n: int, k: int,
           alpha: float, a_ptr, lda: int, b_ptr, ldb: int,
           beta: float, c_ptr, ldc: int) -> None:
    """Thread-safe DGEMM: all BLAS scalar args are stack-local per call."""
    ta = ctypes.c_char(trans_a)
    tb = ctypes.c_char(trans_b)
    mi = ctypes.c_int(m)
    ni = ctypes.c_int(n)
    ki = ctypes.c_int(k)
    ldai = ctypes.c_int(lda)
    ldbi = ctypes.c_int(ldb)
    ldci = ctypes.c_int(ldc)
    alphad = ctypes.c_double(alpha)
    betad = ctypes.c_double(beta)
    _raw_dgemm(
        ctypes.byref(ta), ctypes.byref(tb),
        ctypes.byref(mi), ctypes.byref(ni), ctypes.byref(ki),
        ctypes.byref(alphad),
        a_ptr, ctypes.byref(ldai),
        b_ptr, ctypes.byref(ldbi),
        ctypes.byref(betad),
        c_ptr, ctypes.byref(ldci),
    )


def gemm_forward_fast(col: np.ndarray, W_fwd: np.ndarray, out_gemm: np.ndarray):
    """Computes out_gemm = col @ W_fwd; W_fwd is [K, C_out] contiguous."""
    m_dim = col.shape[0]
    k_dim = col.shape[1]
    n_dim = W_fwd.shape[1]

    if col.dtype == np.float32:
        _sgemm(
            b'N', b'N',
            n_dim, m_dim, k_dim,
            1.0,
            W_fwd.ctypes.data, n_dim,
            col.ctypes.data, k_dim,
            0.0,
            out_gemm.ctypes.data, n_dim,
        )
    else:
        _dgemm(
            b'N', b'N',
            n_dim, m_dim, k_dim,
            1.0,
            W_fwd.ctypes.data, n_dim,
            col.ctypes.data, k_dim,
            0.0,
            out_gemm.ctypes.data, n_dim,
        )


def gemm_param_grad_fast(dout_trans: np.ndarray, col: np.ndarray, dW_flat: np.ndarray, inv_m: float = 1.0):
    """Computes dW_flat = inv_m * (dout_trans.T @ col)"""
    if not col.flags["C_CONTIGUOUS"]:
        col = np.ascontiguousarray(col)
    if not dout_trans.flags["C_CONTIGUOUS"]:
        dout_trans = np.ascontiguousarray(dout_trans)

    n_cols = dW_flat.shape[1]
    n_rows = dW_flat.shape[0]
    k_dim = dout_trans.shape[0]

    if dout_trans.dtype == np.float32:
        _sgemm(
            b"N", b"T",
            n_cols, n_rows, k_dim,
            inv_m,
            col.ctypes.data, n_cols,
            dout_trans.ctypes.data, n_rows,
            0.0,
            dW_flat.ctypes.data, n_cols,
        )
    else:
        _dgemm(
            b"N", b"T",
            n_cols, n_rows, k_dim,
            inv_m,
            col.ctypes.data, n_cols,
            dout_trans.ctypes.data, n_rows,
            0.0,
            dW_flat.ctypes.data, n_cols,
        )


def gemm_backward_input_fast(dout_trans: np.ndarray, W_2d: np.ndarray, dcol_out: np.ndarray):
    """Computes dcol_out = dout_trans @ W_2d (input-gradient GEMM)."""
    if not dout_trans.flags["C_CONTIGUOUS"]:
        dout_trans = np.ascontiguousarray(dout_trans)
    if not W_2d.flags["C_CONTIGUOUS"]:
        W_2d = np.ascontiguousarray(W_2d)

    m_dim = dout_trans.shape[0]
    k_dim = W_2d.shape[0]
    n_dim = W_2d.shape[1]

    if dout_trans.dtype == np.float32:
        _sgemm(
            b"N", b"N",
            n_dim, m_dim, k_dim,
            1.0,
            W_2d.ctypes.data, n_dim,
            dout_trans.ctypes.data, k_dim,
            0.0,
            dcol_out.ctypes.data, n_dim,
        )
    else:
        _dgemm(
            b"N", b"N",
            n_dim, m_dim, k_dim,
            1.0,
            W_2d.ctypes.data, n_dim,
            dout_trans.ctypes.data, k_dim,
            0.0,
            dcol_out.ctypes.data, n_dim,
        )


# -----------------------------------------------------------------------------
# Numba Static Signatures
# -----------------------------------------------------------------------------
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

_fwd_trans_sig = [
    void(float32[:, :], float32[:, :], float32[:, :, :, :]),
    void(float64[:, :], float64[:, :], float64[:, :, :, :]),
]


@njit(_im2col_sig, parallel=True, fastmath=True, nogil=True, cache=True)
def _im2col_numba_kernel(x, k_h, k_w, stride, pad, out_buf):
    N, C, H, W = x.shape
    out_h = (H + 2 * pad - k_h) // stride + 1
    out_w = (W + 2 * pad - k_w) // stride + 1
    spatial_k = k_h * k_w
    spatial_out = out_h * out_w

    if pad == 0:
        for n in prange(N):
            n_row_base = n * spatial_out
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
        for n in prange(N):
            n_row_base = n * spatial_out
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


@njit(_col2im_sig, parallel=True, fastmath=True, nogil=True, cache=True)
def _col2im_numba_kernel(col, N, C, H, W, k_h, k_w, stride, pad, dx_out):
    out_h = (H + 2 * pad - k_h) // stride + 1
    out_w = (W + 2 * pad - k_w) // stride + 1
    spatial_k = k_h * k_w
    spatial_out = out_h * out_w

    dx_out.fill(0.0)

    if pad == 0:
        for n in prange(N):
            n_row_base = n * spatial_out
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
        for n in prange(N):
            n_row_base = n * spatial_out
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


@njit(_maxpool_fwd_sig, parallel=True, fastmath=True, nogil=True, cache=True)
def _maxpool_forward_kernel(x, pool_size, stride, out, argmax):
    N, C, H, W = x.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1

    for n in prange(N):
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


@njit(_maxpool_bwd_sig, parallel=True, fastmath=True, nogil=True, cache=True)
def _maxpool_backward_kernel(dout, argmax, dx):
    dx.fill(0.0)
    N, C, out_h, out_w = dout.shape
    for n in prange(N):
        for c in range(C):
            for oh in range(out_h):
                for ow in range(out_w):
                    h_idx = argmax[n, c, oh, ow, 0]
                    w_idx = argmax[n, c, oh, ow, 1]
                    dx[n, c, h_idx, w_idx] += dout[n, c, oh, ow]


@njit(_fuse_dout_sig, parallel=True, fastmath=True, nogil=True, cache=True)
def _fuse_dout_transpose_and_bias(dout, dout_trans_out, db_out):
    N, OutC, H_out, W_out = dout.shape
    spatial_size = H_out * W_out
    inv_m = 1.0 / float(N)

    for n in prange(N):
        row_n_offset = n * spatial_size
        for h in range(H_out):
            row_h_offset = row_n_offset + h * W_out
            for w in range(W_out):
                row_idx = row_h_offset + w
                for c in range(OutC):
                    dout_trans_out[row_idx, c] = dout[n, c, h, w]

    for c in prange(OutC):
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


@njit(_fwd_trans_sig, parallel=True, fastmath=True, nogil=True, cache=True)
def _fuse_forward_transpose_and_bias(gemm_out, bias, out_nchw):
    N, OutC, H_out, W_out = out_nchw.shape
    spatial_size = H_out * W_out

    for n in prange(N):
        row_n_base = n * spatial_size
        for h in range(H_out):
            row_h_base = row_n_base + h * W_out
            for w in range(W_out):
                row_idx = row_h_base + w
                for c in range(OutC):
                    out_nchw[n, c, h, w] = gemm_out[row_idx, c] + bias[0, c]


_relu_fwd_sig = [
    void(float32[:, :, :, :]),
    void(float64[:, :, :, :]),
]

_relu_bwd_sig = [
    void(float32[:, :, :, :], float32[:, :, :, :]),
    void(float64[:, :, :, :], float64[:, :, :, :]),
]


@njit(_relu_fwd_sig, parallel=True, fastmath=True, nogil=True, cache=True)
def _relu_fwd_inplace_kernel(x):
    N, C, H, W = x.shape
    for n in prange(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    if x[n, c, h, w] < 0.0:
                        x[n, c, h, w] = 0.0


@njit(_relu_bwd_sig, parallel=True, fastmath=True, nogil=True, cache=True)
def _relu_bwd_inplace_kernel(dout, in_act):
    N, C, H, W = dout.shape
    for n in prange(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    if in_act[n, c, h, w] <= 0.0:
                        dout[n, c, h, w] = 0.0