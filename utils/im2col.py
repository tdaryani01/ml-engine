# utils/im2col.py
import os
import ctypes
import numpy as np
from config.constants import EngineBackend

_active_backend: EngineBackend = None
_native_lib = None
_is_initialized: bool = False


def _check_status(status: int, func_name: str):
    if status != 0:
        raise RuntimeError(f"[Native Engine Error] {func_name} failed with native return code: {status}")


# -----------------------------------------------------------------------------
# Engine Backend Initialization
# -----------------------------------------------------------------------------
def init_engine_backend(backend: EngineBackend = EngineBackend.NATIVE):
    """Initializes the active execution backend and loads native binaries if required."""
    global _active_backend, _native_lib, _is_initialized

    if _is_initialized and _active_backend == backend:
        return

    _active_backend = backend

    if _active_backend == EngineBackend.NATIVE:
        lib_name = "conv_kernels.dll" if os.name == "nt" else "conv_kernels.so"
        
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        possible_paths = [
            os.path.join(root_dir, "bin", lib_name),
            os.path.join(os.path.dirname(__file__), "bin", lib_name),
            os.path.join(os.path.dirname(__file__), "..", "bin", lib_name),
            os.path.join(os.path.dirname(__file__), "..", "native", lib_name),
            os.path.join(os.path.dirname(__file__), "..", "src", "native", lib_name),
            os.path.join(os.path.dirname(__file__), "native", lib_name),
        ]

        lib_path = None
        for p in possible_paths:
            if os.path.exists(os.path.abspath(p)):
                lib_path = os.path.abspath(p)
                break

        if lib_path is None:
            raise FileNotFoundError(f"Could not locate {lib_name}. Checked: {[os.path.abspath(p) for p in possible_paths]}")

        lib = ctypes.CDLL(lib_path)

        lib.get_omp_threads.restype = ctypes.c_int
        lib.get_omp_threads.argtypes = []

        lib.log_engine_runtime_diagnostics.restype = None
        lib.log_engine_runtime_diagnostics.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
        ]

        # 1. Direct Composite Forward (Conv + ReLU + Pool)
        lib.direct_conv_block_forward_avx2.restype = ctypes.c_int32
        lib.direct_conv_block_forward_avx2.argtypes = [
            ctypes.c_void_p,  # 1. x
            ctypes.c_void_p,  # 2. W
            ctypes.c_void_p,  # 3. bias
            ctypes.c_void_p,  # 4. out_conv_buf
            ctypes.c_void_p,  # 5. out_pool_buf
            ctypes.c_void_p,  # 6. argmax_buf (uint8_t)
            ctypes.c_int64,   # 7. N
            ctypes.c_int64,   # 8. C_in
            ctypes.c_int64,   # 9. H
            ctypes.c_int64,   # 10. W_in
            ctypes.c_int64,   # 11. W_in_stride
            ctypes.c_int64,   # 12. C_out
            ctypes.c_int64,   # 13. k_h
            ctypes.c_int64,   # 14. k_w
            ctypes.c_int64,   # 15. conv_stride
            ctypes.c_int64,   # 16. conv_pad
            ctypes.c_int64,   # 17. conv_out_w_stride
            ctypes.c_int64,   # 18. pool_size
            ctypes.c_int64    # 19. pool_stride
        ]

        # 2. Direct Composite Backward (Unpool + ReLU Gate + Bias Acc + dx + dW)
        lib.direct_conv_block_backward_avx2.restype = ctypes.c_int32
        lib.direct_conv_block_backward_avx2.argtypes = [
            ctypes.c_void_p,  # 1. dout_pool
            ctypes.c_void_p,  # 2. argmax_buf (uint8_t)
            ctypes.c_void_p,  # 3. x
            ctypes.c_void_p,  # 4. W
            ctypes.c_void_p,  # 5. conv_act
            ctypes.c_void_p,  # 6. d_conv_buf
            ctypes.c_void_p,  # 7. dx_buf
            ctypes.c_void_p,  # 8. dW_buf
            ctypes.c_void_p,  # 9. db_buf
            ctypes.c_int64,   # 10. N
            ctypes.c_int64,   # 11. C_in
            ctypes.c_int64,   # 12. H
            ctypes.c_int64,   # 13. W_in
            ctypes.c_int64,   # 14. W_in_stride
            ctypes.c_int64,   # 15. C_out
            ctypes.c_int64,   # 16. k_h
            ctypes.c_int64,   # 17. k_w
            ctypes.c_int64,   # 18. conv_stride
            ctypes.c_int64,   # 19. conv_pad
            ctypes.c_int64,   # 20. conv_out_w_stride
            ctypes.c_int64,   # 21. pool_size
            ctypes.c_int64,   # 22. pool_stride
            ctypes.c_int64,   # 23. pool_out_h
            ctypes.c_int64,   # 24. pool_out_w
            ctypes.c_float    # 25. inv_m
        ]

        # 3. Direct Conv2D Standalone Primitives
        lib.direct_conv2d_forward_avx2.restype = ctypes.c_int32
        lib.direct_conv2d_forward_avx2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int32
        ]

        lib.direct_conv2d_backward_fused_avx2.restype = ctypes.c_int32
        lib.direct_conv2d_backward_fused_avx2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_int32
        ]

        lib.direct_conv2d_backward_weight_avx2.restype = ctypes.c_int32
        lib.direct_conv2d_backward_weight_avx2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_float
        ]

        lib.direct_conv2d_backward_input_avx2.restype = ctypes.c_int32
        lib.direct_conv2d_backward_input_avx2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int32
        ]

        lib.direct_relu_forward_avx2.restype = ctypes.c_int32
        lib.direct_relu_forward_avx2.argtypes = [ctypes.c_void_p, ctypes.c_int64]

        lib.direct_relu_backward_avx2.restype = ctypes.c_int32
        lib.direct_relu_backward_avx2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]

        lib.direct_maxpool_forward_avx2.restype = ctypes.c_int32
        lib.direct_maxpool_forward_avx2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64
        ]

        lib.direct_maxpool_backward_avx2.restype = ctypes.c_int32
        lib.direct_maxpool_backward_avx2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
        ]

        lib.direct_bias_backward_avx2.restype = ctypes.c_int32
        lib.direct_bias_backward_avx2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_float
        ]

        _native_lib = lib

    _is_initialized = True


def _ensure_initialized():
    if not _is_initialized:
        init_engine_backend(EngineBackend.NATIVE)


from utils.im2col_fast import (
    im2col_fast as _im2col_impl,
    col2im_fast as _col2im_impl,
    gemm_forward_fast as _gemm_forward_impl,
    gemm_param_grad_fast as _gemm_param_grad_impl,
    _maxpool_forward_kernel,
    _maxpool_backward_kernel,
    fuse_dout_transpose_bias_fast as _fuse_dout_impl,
    _fuse_forward_transpose_and_bias as fuse_forward_impl,
    _relu_fwd_inplace_kernel,
    _relu_bwd_inplace_kernel
)


def _get_c_ptr(arr: np.ndarray):
    """Safely extracts a C data pointer from a NumPy array."""
    if arr is None or arr.size == 0:
        return None
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    return arr.ctypes.data


# -----------------------------------------------------------------------------
# Composite Block Routines
# -----------------------------------------------------------------------------
def conv_block_forward(x: np.ndarray, W: np.ndarray, bias: np.ndarray,
                       out_conv_buf: np.ndarray, out_pool_buf: np.ndarray, argmax_buf: np.ndarray,
                       conv_stride: int = 1, conv_pad: int = 1,
                       pool_size: int = 2, pool_stride: int = 2,
                       col_buf: np.ndarray = None, gemm_buf: np.ndarray = None,
                       W_logical: int = None, out_w_logical: int = None) -> tuple:
    _ensure_initialized()

    if _active_backend == EngineBackend.NATIVE and _native_lib is not None and x.dtype == np.float32:
        N, C_in, H, W_in_stride = x.shape
        C_out, _, k_h, k_w = W.shape
        W_in = W_logical if W_logical is not None else W_in_stride
        conv_out_w_stride = out_conv_buf.shape[3]
        
        status = _native_lib.direct_conv_block_forward_avx2(
            _get_c_ptr(x),
            _get_c_ptr(W),
            _get_c_ptr(bias),
            _get_c_ptr(out_conv_buf),
            _get_c_ptr(out_pool_buf),
            _get_c_ptr(argmax_buf),
            int(N), int(C_in), int(H), int(W_in), int(W_in_stride),
            int(C_out), int(k_h), int(k_w),
            int(conv_stride), int(conv_pad), int(conv_out_w_stride),
            int(pool_size), int(pool_stride)
        )
        _check_status(status, "direct_conv_block_forward_avx2")
        return out_pool_buf, out_conv_buf, argmax_buf, None

    out_conv, col = conv2d_forward(
        x=x, W=W, bias=bias, stride=conv_stride, pad=conv_pad,
        out_buf=out_conv_buf, col_buf=col_buf, gemm_buf=gemm_buf,
        fuse_relu=True, W_logical=W_logical
    )
    
    # Extract logical slice before pooling on fallback backends
    out_h = (x.shape[2] + 2 * conv_pad - W.shape[2]) // conv_stride + 1
    out_w = ((W_logical if W_logical is not None else x.shape[3]) + 2 * conv_pad - W.shape[3]) // conv_stride + 1
    valid_conv = out_conv[:, :, :out_h, :out_w] if out_conv.shape[3] != out_w else out_conv

    out_pool, argmax = maxpool_forward(
        valid_conv, pool_size, pool_stride, out_buf=out_pool_buf,
        argmax_buf=argmax_buf
    )
    return out_pool, out_conv, argmax, col


def conv_block_backward(dout_pool: np.ndarray, argmax_buf: np.ndarray,
                        x: np.ndarray, W: np.ndarray, conv_act: np.ndarray,
                        d_conv_buf: np.ndarray, dx_buf: np.ndarray, dW_buf: np.ndarray, db_buf: np.ndarray,
                        conv_stride: int = 1, conv_pad: int = 1,
                        pool_size: int = 2, pool_stride: int = 2,
                        inv_m: float = 1.0,
                        col: np.ndarray = None, dout_trans: np.ndarray = None, dcol_buf: np.ndarray = None,
                        W_logical: int = None, out_w_logical: int = None) -> tuple:
    _ensure_initialized()

    if _active_backend == EngineBackend.NATIVE and _native_lib is not None and dout_pool.dtype == np.float32:
        N, C_in, H, W_in_stride = x.shape
        C_out, _, k_h, k_w = W.shape
        W_in = W_logical if W_logical is not None else W_in_stride
        conv_out_w_stride = d_conv_buf.shape[3]
        pool_out_h, pool_out_w = dout_pool.shape[2], dout_pool.shape[3]

        status = _native_lib.direct_conv_block_backward_avx2(
            _get_c_ptr(dout_pool),
            _get_c_ptr(argmax_buf),
            _get_c_ptr(x),
            _get_c_ptr(W),
            _get_c_ptr(conv_act),
            _get_c_ptr(d_conv_buf),
            _get_c_ptr(dx_buf),
            _get_c_ptr(dW_buf),
            _get_c_ptr(db_buf),
            int(N), int(C_in), int(H), int(W_in), int(W_in_stride),
            int(C_out), int(k_h), int(k_w),
            int(conv_stride), int(conv_pad), int(conv_out_w_stride),
            int(pool_size), int(pool_stride),
            int(pool_out_h), int(pool_out_w),
            ctypes.c_float(inv_m)
        )
        _check_status(status, "direct_conv_block_backward_avx2")
        return dx_buf, dW_buf, db_buf

    out_h = (x.shape[2] + 2 * conv_pad - W.shape[2]) // conv_stride + 1
    out_w = ((W_logical if W_logical is not None else x.shape[3]) + 2 * conv_pad - W.shape[3]) // conv_stride + 1
    conv_act_logical = conv_act[:, :, :out_h, :out_w] if conv_act.shape[3] != out_w else conv_act

    d_conv_logical = maxpool_backward(
        dout_pool, argmax_buf, conv_act_logical.shape, pool_size, pool_stride
    )
    _relu_bwd_inplace_kernel(d_conv_logical, conv_act_logical)

    d_conv_buf.fill(0.0)
    d_conv_buf[:, :, :out_h, :out_w] = d_conv_logical

    if db_buf is not None:
        fuse_dout_transpose_and_bias(d_conv_logical, dout_trans, db_buf)
        
    dx, dW = conv2d_backward_fused(
        d_conv_buf, x, W, dx_buf, dW_buf,
        stride=conv_stride, pad=conv_pad, inv_m=inv_m,
        in_act=None, fuse_relu=False, col=col, dout_trans=dout_trans,
        dcol_buf=dcol_buf, W_logical=W_logical
    )
    return dx, dW, db_buf


# -----------------------------------------------------------------------------
# Standalone Layer Routines
# -----------------------------------------------------------------------------
def conv2d_forward(x: np.ndarray, W: np.ndarray, bias: np.ndarray,
                   stride: int, pad: int, out_buf: np.ndarray,
                   col_buf: np.ndarray = None, gemm_buf: np.ndarray = None,
                   fuse_relu: bool = False, W_logical: int = None) -> tuple:
    _ensure_initialized()
    N, C_in, H, W_in_stride = x.shape
    C_out, _, k_h, k_w = W.shape
    W_in = W_logical if W_logical is not None else W_in_stride
    out_w_stride = out_buf.shape[3]

    if _active_backend == EngineBackend.NATIVE and _native_lib is not None and x.dtype == np.float32:
        status = _native_lib.direct_conv2d_forward_avx2(
            _get_c_ptr(x),
            _get_c_ptr(W),
            _get_c_ptr(bias),
            _get_c_ptr(out_buf),
            int(N), int(C_in), int(H), int(W_in), int(W_in_stride),
            int(C_out), int(k_h), int(k_w),
            int(stride), int(pad), int(out_w_stride),
            1 if fuse_relu else 0
        )
        _check_status(status, "direct_conv2d_forward_avx2")
        return out_buf, None

    out_h = (H + 2 * pad - k_h) // stride + 1
    out_w = (W_in + 2 * pad - k_w) // stride + 1
    total_rows = N * out_h * out_w

    x_logical = x[:, :, :, :W_in] if x.shape[3] != W_in else x

    if _active_backend in (EngineBackend.NATIVE, EngineBackend.IM2COL_GEMM):
        active_col = col_buf[:total_rows] if col_buf is not None else np.empty((total_rows, C_in * k_h * k_w), dtype=x.dtype)
        active_gemm = gemm_buf[:total_rows] if gemm_buf is not None else np.empty((total_rows, C_out), dtype=x.dtype)

        _im2col_impl(x_logical, k_h, k_w, stride, pad, out_buf=active_col)
        _gemm_forward_impl(active_col, W.reshape(C_out, -1), active_gemm)
        
        # Temporary buffer for fused transpose & bias if out_buf is stride-padded
        if out_buf.shape[3] != out_w:
            temp_out = np.empty((N, C_out, out_h, out_w), dtype=x.dtype)
            fuse_forward_impl(active_gemm, bias, temp_out)
            out_buf.fill(0.0)
            out_buf[:, :, :out_h, :out_w] = temp_out
        else:
            fuse_forward_impl(active_gemm, bias, out_buf)

        if fuse_relu:
            _relu_fwd_inplace_kernel(out_buf)
        return out_buf, active_col

    # Reference pure NumPy path
    col = im2col(x_logical, k_h, k_w, stride, pad, out_buf=col_buf[:total_rows] if col_buf is not None else None)
    gemm_out = np.dot(col, W.reshape(C_out, -1).T)
    out_reshaped = (gemm_out + bias).reshape(N, out_h, out_w, C_out).transpose(0, 3, 1, 2)
    out_buf.fill(0.0)
    out_buf[:, :, :out_h, :out_w] = out_reshaped
    if fuse_relu:
        np.maximum(0.0, out_buf, out=out_buf)
    return out_buf, col


def conv2d_backward_fused(dout: np.ndarray, x: np.ndarray, W: np.ndarray,
                          dx_buf: np.ndarray, dW_buf: np.ndarray,
                          stride: int, pad: int, inv_m: float,
                          in_act: np.ndarray = None, fuse_relu: bool = False,
                          col: np.ndarray = None, dout_trans: np.ndarray = None,
                          dcol_buf: np.ndarray = None, W_logical: int = None) -> tuple:
    _ensure_initialized()
    N, C_in, H, W_in_stride = x.shape
    C_out, _, k_h, k_w = W.shape
    W_in = W_logical if W_logical is not None else W_in_stride
    conv_out_w_stride = dout.shape[3]

    if _active_backend == EngineBackend.NATIVE and _native_lib is not None and dout.dtype == np.float32:
        status = _native_lib.direct_conv2d_backward_fused_avx2(
            _get_c_ptr(dout),
            _get_c_ptr(x),
            _get_c_ptr(W),
            _get_c_ptr(in_act) if (fuse_relu and in_act is not None) else None,
            _get_c_ptr(dx_buf),
            _get_c_ptr(dW_buf),
            int(N), int(C_in), int(H), int(W_in), int(W_in_stride),
            int(C_out), int(k_h), int(k_w),
            int(stride), int(pad), int(conv_out_w_stride),
            ctypes.c_float(inv_m),
            1 if (fuse_relu and in_act is not None) else 0
        )
        _check_status(status, "direct_conv2d_backward_fused_avx2")
        return dx_buf, dW_buf

    dx = conv2d_backward_input(dout, W, dx_buf, stride, pad, dout_trans=dout_trans, dcol_buf=dcol_buf, in_act=in_act, fuse_relu=fuse_relu, W_logical=W_logical)
    dW = conv2d_backward_weight(dout, x, dW_buf, col, dout_trans, stride, pad, inv_m, W_logical=W_logical)
    return dx, dW


def conv2d_backward_weight(dout: np.ndarray, x: np.ndarray, dW: np.ndarray,
                           col: np.ndarray, dout_trans: np.ndarray,
                           stride: int, pad: int, inv_m: float, W_logical: int = None) -> np.ndarray:
    _ensure_initialized()
    if _active_backend == EngineBackend.NATIVE and _native_lib is not None and dout.dtype == np.float32:
        N, C_in, H, W_in_stride = x.shape
        C_out, _, k_h, k_w = dW.shape
        W_in = W_logical if W_logical is not None else W_in_stride
        conv_out_w_stride = dout.shape[3]
        status = _native_lib.direct_conv2d_backward_weight_avx2(
            _get_c_ptr(dout), _get_c_ptr(x), _get_c_ptr(dW),
            int(N), int(C_in), int(H), int(W_in), int(W_in_stride), int(C_out), int(k_h), int(k_w), int(stride), int(pad), int(conv_out_w_stride), ctypes.c_float(inv_m)
        )
        _check_status(status, "direct_conv2d_backward_weight_avx2")
        return dW

    N = dout.shape[0]
    C_out, C_in, k_h, k_w = dW.shape
    H = x.shape[2]
    W_in = W_logical if W_logical is not None else x.shape[3]
    
    out_h = (H + 2 * pad - k_h) // stride + 1
    out_w = (W_in + 2 * pad - k_w) // stride + 1
    total_rows = N * out_h * out_w

    # Reconstruct/slice active dout_trans strictly over valid logical features
    active_dout_trans = (
        dout[:, :, :out_h, :out_w].transpose(0, 2, 3, 1).reshape(total_rows, C_out)
        if dout.shape[3] != out_w or dout.shape[2] != out_h
        else dout.transpose(0, 2, 3, 1).reshape(total_rows, C_out)
    )

    active_col = col[:total_rows] if col is not None else None
    if active_col is None:
        x_logical = x[:, :, :, :W_in] if x.shape[3] != W_in else x
        active_col = im2col(x_logical, k_h, k_w, stride, pad)

    if _active_backend in (EngineBackend.NATIVE, EngineBackend.IM2COL_GEMM):
        orig_shape = dW.shape
        dW_flat = np.empty((orig_shape[0], int(np.prod(orig_shape[1:]))), dtype=active_dout_trans.dtype)
        gemm_param_grad(active_dout_trans, active_col, dW_flat, inv_m)
        dW[...] = dW_flat.reshape(orig_shape).astype(dW.dtype)
        return dW

    dW_flat = np.dot(active_dout_trans.T, active_col) * inv_m
    dW[...] = dW_flat.reshape(dW.shape).astype(dW.dtype)
    return dW


def conv2d_backward_input(dout: np.ndarray, W: np.ndarray, dx_buf: np.ndarray,
                          stride: int, pad: int,
                          dout_trans: np.ndarray = None, dcol_buf: np.ndarray = None,
                          in_act: np.ndarray = None, fuse_relu: bool = False, W_logical: int = None) -> np.ndarray:
    _ensure_initialized()
    if _active_backend == EngineBackend.NATIVE and _native_lib is not None and dout.dtype == np.float32:
        N, C_in, H, W_in_stride = dx_buf.shape
        C_out, _, k_h, k_w = W.shape
        W_in = W_logical if W_logical is not None else W_in_stride
        conv_out_w_stride = dout.shape[3]
        status = _native_lib.direct_conv2d_backward_input_avx2(
            _get_c_ptr(dout), _get_c_ptr(W),
            _get_c_ptr(in_act) if (fuse_relu and in_act is not None) else None,
            _get_c_ptr(dx_buf),
            int(N), int(C_in), int(H), int(W_in), int(W_in_stride), int(C_out), int(k_h), int(k_w), int(stride), int(pad), int(conv_out_w_stride),
            1 if (fuse_relu and in_act is not None) else 0
        )
        _check_status(status, "direct_conv2d_backward_input_avx2")
        return dx_buf

    N = dout.shape[0]
    C_in = dx_buf.shape[1]
    H = dx_buf.shape[2]
    W_in = W_logical if W_logical is not None else dx_buf.shape[3]
    k_h, k_w = W.shape[2], W.shape[3]

    out_h = (H + 2 * pad - k_h) // stride + 1
    out_w = (W_in + 2 * pad - k_w) // stride + 1
    total_rows = N * out_h * out_w

    W_2d = W.reshape(W.shape[0], -1)
    
    # Extract only active logical rows for backprop GEMM
    active_dout_trans = dout[:, :, :out_h, :out_w].transpose(0, 2, 3, 1).reshape(total_rows, W.shape[0])
    active_dcol = dcol_buf[:total_rows] if dcol_buf is not None else np.empty((total_rows, W_2d.shape[1]), dtype=dout.dtype)
    np.dot(active_dout_trans, W_2d, out=active_dcol)
    
    logical_shape = (N, C_in, H, W_in)
    dx_logical = col2im(active_dcol, logical_shape, k_h, k_w, stride, pad)
    
    dx_buf.fill(0.0)
    dx_buf[:, :, :, :W_in] = dx_logical
    if fuse_relu and in_act is not None:
        _relu_bwd_inplace_kernel(dx_buf, in_act)
    return dx_buf


# -----------------------------------------------------------------------------
# Primitives, Pooling & Activations
# -----------------------------------------------------------------------------
def im2col(x: np.ndarray, k_h: int, k_w: int, stride: int = 1, pad: int = 0, out_buf: np.ndarray = None) -> np.ndarray:
    _ensure_initialized()
    if _active_backend in (EngineBackend.NATIVE, EngineBackend.IM2COL_GEMM):
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
    _ensure_initialized()
    if _active_backend in (EngineBackend.NATIVE, EngineBackend.IM2COL_GEMM):
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
    _ensure_initialized()
    N, C, H, W = x.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1

    if _active_backend == EngineBackend.NATIVE and _native_lib is not None and x.dtype == np.float32:
        if out_buf is None or out_buf.shape != (N, C, out_h, out_w) or out_buf.dtype != x.dtype:
            out_buf = np.empty((N, C, out_h, out_w), dtype=x.dtype)
        if argmax_buf is None or argmax_buf.shape != (N, C, out_h, out_w) or argmax_buf.dtype != np.uint8:
            argmax_buf = np.empty((N, C, out_h, out_w), dtype=np.uint8)

        status = _native_lib.direct_maxpool_forward_avx2(
            _get_c_ptr(x), _get_c_ptr(out_buf), _get_c_ptr(argmax_buf),
            int(N), int(C), int(H), int(W), int(pool_size), int(stride)
        )
        _check_status(status, "direct_maxpool_forward_avx2")
        return out_buf, argmax_buf

    if _active_backend in (EngineBackend.NATIVE, EngineBackend.IM2COL_GEMM):
        if out_buf is None or out_buf.shape != (N, C, out_h, out_w) or out_buf.dtype != x.dtype:
            out_buf = np.empty((N, C, out_h, out_w), dtype=x.dtype)
        if argmax_buf is None or argmax_buf.shape != (N, C, out_h, out_w, 2) or argmax_buf.dtype != np.int64:
            argmax_buf = np.empty((N, C, out_h, out_w, 2), dtype=np.int64)

        _maxpool_forward_kernel(x, pool_size, stride, out_buf, argmax_buf)
        return out_buf, argmax_buf

    if out_buf is None or out_buf.shape != (N, C, out_h, out_w) or out_buf.dtype != x.dtype:
        out_buf = np.empty((N, C, out_h, out_w), dtype=x.dtype)
    x_reshaped = x[:, :, :out_h * stride, :out_w * stride].reshape(N, C, out_h, stride, out_w, stride)
    out_buf[:] = x_reshaped.max(axis=(3, 5))
    mask = (x_reshaped == out_buf[:, :, :, None, :, None])
    return out_buf, mask


def maxpool_backward(dout: np.ndarray, cache: np.ndarray, x_shape: tuple, pool_size: int, stride: int, dx_buf: np.ndarray = None):
    _ensure_initialized()
    if dx_buf is None or dx_buf.shape != x_shape or dx_buf.dtype != dout.dtype:
        dx_buf = np.zeros(x_shape, dtype=dout.dtype)
    else:
        dx_buf.fill(0.0)

    N, C, in_h, in_w = x_shape
    out_h, out_w = dout.shape[2], dout.shape[3]

    if _active_backend == EngineBackend.NATIVE and _native_lib is not None and dout.dtype == np.float32:
        status = _native_lib.direct_maxpool_backward_avx2(
            _get_c_ptr(dout), _get_c_ptr(cache), _get_c_ptr(dx_buf),
            int(N), int(C), int(out_h), int(out_w), int(in_h), int(in_w), int(pool_size), int(stride)
        )
        _check_status(status, "direct_maxpool_backward_avx2")
        return dx_buf

    if _active_backend in (EngineBackend.NATIVE, EngineBackend.IM2COL_GEMM):
        _maxpool_backward_kernel(dout, cache, dx_buf)
        return dx_buf

    mask = cache
    dx_reshaped = mask * dout[:, :, :, None, :, None]
    dx_buf[:] = dx_reshaped.reshape(x_shape)
    return dx_buf


def fuse_dout_transpose_and_bias(dout: np.ndarray, dout_trans_buf: np.ndarray, db_buf: np.ndarray):
    _ensure_initialized()
    if _active_backend == EngineBackend.NATIVE and _native_lib is not None and dout.dtype == np.float32:
        N, C_out, out_h, out_w = dout.shape
        inv_m = 1.0 / float(N)
        status = _native_lib.direct_bias_backward_avx2(
            _get_c_ptr(dout), _get_c_ptr(db_buf),
            int(N), int(C_out), int(out_h), int(out_w), ctypes.c_float(inv_m)
        )
        _check_status(status, "direct_bias_backward_avx2")
        if dout_trans_buf is not None:
            dout_trans_buf[:N * out_h * out_w] = np.transpose(dout, (0, 2, 3, 1)).reshape(-1, C_out)
        return

    _fuse_dout_impl(dout, dout_trans_buf, db_buf)


def gemm_param_grad(dout_trans: np.ndarray, col: np.ndarray, dW_flat: np.ndarray, inv_m: float):
    _gemm_param_grad_impl(dout_trans, col, dW_flat, inv_m)


def relu_spatial_forward(x: np.ndarray) -> np.ndarray:
    _ensure_initialized()
    if _active_backend == EngineBackend.NATIVE and _native_lib is not None and x.dtype == np.float32:
        status = _native_lib.direct_relu_forward_avx2(_get_c_ptr(x), x.size)
        _check_status(status, "direct_relu_forward_avx2")
        return x
    _relu_fwd_inplace_kernel(x)
    return x


def relu_spatial_backward(dout: np.ndarray, in_act: np.ndarray) -> np.ndarray:
    _ensure_initialized()
    if _active_backend == EngineBackend.NATIVE and _native_lib is not None and dout.dtype == np.float32:
        status = _native_lib.direct_relu_backward_avx2(_get_c_ptr(dout), _get_c_ptr(in_act), dout.size)
        _check_status(status, "direct_relu_backward_avx2")
        return dout
    _relu_bwd_inplace_kernel(dout, in_act)
    return dout