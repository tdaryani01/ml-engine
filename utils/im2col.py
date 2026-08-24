import os
import ctypes
import time
import numpy as np

os.environ["NUMBA_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

BACKEND = os.getenv("ENGINE_BACKEND", "native").lower()

_USE_NATIVE = False
_USE_FAST = False
_native_lib = None
_diagnostics_logged = True

# -----------------------------------------------------------------------------
# 1. Native C++ AVX2 Initialization
# -----------------------------------------------------------------------------
if BACKEND in ("native", "cpp", "c++"):
    try:
        lib_name = "conv_kernels.dll" if os.name == "nt" else "conv_kernels.so"
        possible_paths = [
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
            raise FileNotFoundError(f"Could not locate {lib_name}")

        _native_lib = ctypes.CDLL(lib_path)

        _native_lib.get_omp_threads.restype = ctypes.c_int
        _native_lib.get_omp_threads.argtypes = []

        _native_lib.log_engine_runtime_diagnostics.restype = None
        _native_lib.log_engine_runtime_diagnostics.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
        ]

        _native_lib.direct_conv2d_forward_avx2.restype = None
        _native_lib.direct_conv2d_forward_avx2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int32
        ]

        _native_lib.direct_conv2d_backward_weight_avx2.restype = None
        _native_lib.direct_conv2d_backward_weight_avx2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_float
        ]

        _native_lib.direct_conv2d_backward_input_avx2.restype = None
        _native_lib.direct_conv2d_backward_input_avx2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int32
        ]

        _native_lib.direct_conv2d_backward_fused_avx2.restype = None
        _native_lib.direct_conv2d_backward_fused_avx2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_int32
        ]

        _native_lib.direct_conv_block_forward_avx2.restype = None
        _native_lib.direct_conv_block_forward_avx2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64
        ]

        _native_lib.direct_conv_block_backward_avx2.restype = None
        _native_lib.direct_conv_block_backward_avx2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64,
            ctypes.c_float
        ]

        _native_lib.direct_relu_forward_avx2.restype = None
        _native_lib.direct_relu_forward_avx2.argtypes = [ctypes.c_void_p, ctypes.c_int64]

        _native_lib.direct_relu_backward_avx2.restype = None
        _native_lib.direct_relu_backward_avx2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]

        _native_lib.direct_maxpool_forward_avx2.restype = None
        _native_lib.direct_maxpool_forward_avx2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64
        ]

        _native_lib.direct_maxpool_backward_avx2.restype = None
        _native_lib.direct_maxpool_backward_avx2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64
        ]

        _native_lib.direct_bias_backward_avx2.restype = None
        _native_lib.direct_bias_backward_avx2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_float
        ]

        _USE_NATIVE = True
    except Exception as err:
        raise RuntimeError(f"Native DLL failed to load: {err}")

from utils.im2col_fast import (
    im2col_fast as _im2col_impl,
    col2im_fast as _col2im_impl,
    gemm_forward_fast as _gemm_forward_impl,
    gemm_param_grad_fast as _gemm_param_grad_impl,
    fuse_dout_transpose_bias_fast as _fuse_dout_impl,
    _fuse_forward_transpose_and_bias as fuse_forward_impl,
    _relu_fwd_inplace_kernel,
    _relu_bwd_inplace_kernel
)
_USE_FAST = True


def _verify_native_args(func_name: str, **arrays):
    if not _USE_NATIVE or _native_lib is None:
        raise RuntimeError(f"[{func_name}] Native backend requested, but _USE_NATIVE is False or library is not loaded.")
    for name, arr in arrays.items():
        if arr is not None:
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"[{func_name}] Expected numpy.ndarray for '{name}', got {type(arr)}")
            if arr.dtype != np.float32:
                raise TypeError(f"[{func_name}] Array '{name}' must be float32, but got {arr.dtype}")
            if not arr.flags['C_CONTIGUOUS']:
                raise ValueError(f"[{func_name}] Array '{name}' must be C-contiguous for native memory pointers.")


def conv_block_forward(x: np.ndarray, W: np.ndarray, bias: np.ndarray,
                       out_conv_buf: np.ndarray, out_pool_buf: np.ndarray, argmax_buf: np.ndarray,
                       conv_stride: int = 1, conv_pad: int = 1,
                       pool_size: int = 2, pool_stride: int = 2,
                       col_buf: np.ndarray = None, gemm_buf: np.ndarray = None) -> tuple:
    if _USE_NATIVE:
        _verify_native_args("conv_block_forward", x=x, W=W, bias=bias, out_conv_buf=out_conv_buf, out_pool_buf=out_pool_buf)
        N, C_in, H, W_in = x.shape
        C_out, _, k_h, k_w = W.shape
        
        _native_lib.direct_conv_block_forward_avx2(
            x.ctypes.data,
            W.ctypes.data,
            bias.ctypes.data if bias is not None else None,
            out_conv_buf.ctypes.data,
            out_pool_buf.ctypes.data,
            argmax_buf.ctypes.data,
            N, C_in, H, W_in,
            C_out, k_h, k_w,
            conv_stride, conv_pad,
            pool_size, pool_stride
        )
        return out_pool_buf, out_conv_buf, argmax_buf, None

    out_conv, col = conv2d_forward(
        x=x, W=W, bias=bias, stride=conv_stride, pad=conv_pad,
        out_buf=out_conv_buf, col_buf=col_buf, gemm_buf=gemm_buf, fuse_relu=True
    )
    out_pool, argmax = maxpool_forward(
        out_conv, pool_size, pool_stride, out_buf=out_pool_buf, argmax_buf=argmax_buf
    )
    return out_pool, out_conv, argmax, col


def conv_block_backward(dout_pool: np.ndarray, argmax_buf: np.ndarray,
                        x: np.ndarray, W: np.ndarray, conv_act: np.ndarray,
                        d_conv_buf: np.ndarray, dx_buf: np.ndarray, dW_buf: np.ndarray, db_buf: np.ndarray,
                        conv_stride: int = 1, conv_pad: int = 1,
                        pool_size: int = 2, pool_stride: int = 2,
                        inv_m: float = 1.0,
                        col: np.ndarray = None, dout_trans: np.ndarray = None, dcol_buf: np.ndarray = None) -> tuple:
    if _USE_NATIVE:
        _verify_native_args("conv_block_backward", dout_pool=dout_pool, x=x, W=W, conv_act=conv_act,
                            d_conv_buf=d_conv_buf, dx_buf=dx_buf, dW_buf=dW_buf, db_buf=db_buf)
        N, C_in, H, W_in = x.shape
        C_out, _, k_h, k_w = W.shape
        pool_out_h, pool_out_w = dout_pool.shape[2], dout_pool.shape[3]

        _native_lib.direct_conv_block_backward_avx2(
            dout_pool.ctypes.data,
            argmax_buf.ctypes.data,
            x.ctypes.data,
            W.ctypes.data,
            conv_act.ctypes.data if conv_act is not None else None,
            d_conv_buf.ctypes.data,
            dx_buf.ctypes.data,
            dW_buf.ctypes.data,
            db_buf.ctypes.data if db_buf is not None else None,
            N, C_in, H, W_in,
            C_out, k_h, k_w,
            conv_stride, conv_pad,
            pool_size, pool_stride,
            pool_out_h, pool_out_w,
            inv_m
        )
        return dx_buf, dW_buf, db_buf

    d_conv = maxpool_backward(
        dout_pool, argmax_buf, conv_act.shape, pool_size, pool_stride, dx_buf=d_conv_buf
    )
    _relu_bwd_inplace_kernel(d_conv, conv_act)
    if db_buf is not None:
        fuse_dout_transpose_and_bias(d_conv, dout_trans, db_buf)
    dx, dW = conv2d_backward_fused(
        d_conv, x, W, dx_buf, dW_buf,
        stride=conv_stride, pad=conv_pad, inv_m=inv_m,
        in_act=None, fuse_relu=False, col=col, dout_trans=dout_trans, dcol_buf=dcol_buf
    )
    return dx, dW, db_buf


def conv2d_forward(x: np.ndarray, W: np.ndarray, bias: np.ndarray,
                   stride: int, pad: int, out_buf: np.ndarray,
                   col_buf: np.ndarray = None, gemm_buf: np.ndarray = None,
                   fuse_relu: bool = False) -> tuple:
    if _USE_NATIVE:
        _verify_native_args("conv2d_forward", x=x, W=W, bias=bias, out_buf=out_buf)
        N, C_in, H, W_in = x.shape
        C_out, _, k_h, k_w = W.shape

        _native_lib.direct_conv2d_forward_avx2(
            x.ctypes.data,
            W.ctypes.data,
            bias.ctypes.data if bias is not None else None,
            out_buf.ctypes.data,
            N, C_in, H, W_in,
            C_out, k_h, k_w,
            stride, pad,
            1 if fuse_relu else 0
        )
        return out_buf, None

    N, C_in, H, W_in = x.shape
    C_out, _, k_h, k_w = W.shape
    out_h = (H + 2 * pad - k_h) // stride + 1
    out_w = (W_in + 2 * pad - k_w) // stride + 1
    total_rows = N * out_h * out_w

    active_col = col_buf[:total_rows] if col_buf is not None else np.empty((total_rows, C_in * k_h * k_w), dtype=x.dtype)
    active_gemm = gemm_buf[:total_rows] if gemm_buf is not None else np.empty((total_rows, C_out), dtype=x.dtype)

    _im2col_impl(x, k_h, k_w, stride, pad, out_buf=active_col)
    _gemm_forward_impl(active_col, W.reshape(C_out, -1), active_gemm)
    fuse_forward_impl(active_gemm, bias, out_buf)
    if fuse_relu:
        _relu_fwd_inplace_kernel(out_buf)
    return out_buf, active_col


def conv2d_backward_fused(dout: np.ndarray, x: np.ndarray, W: np.ndarray,
                          dx_buf: np.ndarray, dW_buf: np.ndarray,
                          stride: int, pad: int, inv_m: float,
                          in_act: np.ndarray = None, fuse_relu: bool = False,
                          col: np.ndarray = None, dout_trans: np.ndarray = None,
                          dcol_buf: np.ndarray = None) -> tuple:
    if _USE_NATIVE:
        _verify_native_args("conv2d_backward_fused", dout=dout, x=x, W=W, dx_buf=dx_buf, dW_buf=dW_buf, in_act=in_act)
        N, C_in, H, W_in = x.shape
        C_out, _, k_h, k_w = W.shape

        _native_lib.direct_conv2d_backward_fused_avx2(
            dout.ctypes.data,
            x.ctypes.data,
            W.ctypes.data,
            in_act.ctypes.data if (fuse_relu and in_act is not None) else None,
            dx_buf.ctypes.data,
            dW_buf.ctypes.data,
            N, C_in, H, W_in,
            C_out, k_h, k_w,
            stride, pad,
            inv_m,
            1 if (fuse_relu and in_act is not None) else 0
        )
        return dx_buf, dW_buf

    dx = conv2d_backward_input(dout, W, dx_buf, stride, pad, dout_trans=dout_trans, dcol_buf=dcol_buf, in_act=in_act, fuse_relu=fuse_relu)
    dW = conv2d_backward_weight(dout, x, dW_buf, col, dout_trans, stride, pad, inv_m)
    return dx, dW


def conv2d_backward_weight(dout: np.ndarray, x: np.ndarray, dW: np.ndarray,
                           col: np.ndarray, dout_trans: np.ndarray,
                           stride: int, pad: int, inv_m: float) -> np.ndarray:
    if _USE_NATIVE:
        _verify_native_args("conv2d_backward_weight", dout=dout, x=x, dW=dW)
        N, C_in, H, W_in = x.shape
        C_out, _, k_h, k_w = dW.shape
        _native_lib.direct_conv2d_backward_weight_avx2(
            dout.ctypes.data, x.ctypes.data, dW.ctypes.data,
            N, C_in, H, W_in, C_out, k_h, k_w, stride, pad, inv_m
        )
        return dW

    orig_shape = dW.shape
    dW_flat = np.empty((orig_shape[0], int(np.prod(orig_shape[1:]))), dtype=dout_trans.dtype)
    gemm_param_grad(dout_trans, col, dW_flat, inv_m)
    dW[...] = dW_flat.reshape(orig_shape).astype(dW.dtype)
    return dW


def conv2d_backward_input(dout: np.ndarray, W: np.ndarray, dx_buf: np.ndarray,
                          stride: int, pad: int,
                          dout_trans: np.ndarray = None, dcol_buf: np.ndarray = None,
                          in_act: np.ndarray = None, fuse_relu: bool = False) -> np.ndarray:
    if _USE_NATIVE:
        _verify_native_args("conv2d_backward_input", dout=dout, W=W, dx_buf=dx_buf, in_act=in_act)
        N, C_in, H, W_in = dx_buf.shape
        C_out, _, k_h, k_w = W.shape
        _native_lib.direct_conv2d_backward_input_avx2(
            dout.ctypes.data, W.ctypes.data,
            in_act.ctypes.data if (fuse_relu and in_act is not None) else None,
            dx_buf.ctypes.data,
            N, C_in, H, W_in, C_out, k_h, k_w, stride, pad,
            1 if (fuse_relu and in_act is not None) else 0
        )
        return dx_buf

    W_2d = W.reshape(W.shape[0], -1)
    total_rows = dout_trans.shape[0]
    active_dcol = dcol_buf[:total_rows]
    np.dot(dout_trans, W_2d, out=active_dcol)
    dx = col2im(active_dcol, dx_buf.shape, W.shape[2], W.shape[3], stride, pad, out_buf=dx_buf)
    if fuse_relu and in_act is not None:
        _relu_bwd_inplace_kernel(dx, in_act)
    return dx


def maxpool_forward(x: np.ndarray, pool_size: int, stride: int, out_buf: np.ndarray = None, argmax_buf: np.ndarray = None):
    N, C, H, W = x.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1

    if out_buf is None:
        out_buf = np.empty((N, C, out_h, out_w), dtype=x.dtype)
    if argmax_buf is None:
        argmax_buf = np.empty((N, C, out_h, out_w), dtype=np.uint8)

    if _USE_NATIVE:
        _verify_native_args("maxpool_forward", x=x, out_buf=out_buf)
        _native_lib.direct_maxpool_forward_avx2(
            x.ctypes.data, out_buf.ctypes.data, argmax_buf.ctypes.data,
            N, C, H, W, pool_size, stride
        )
        return out_buf, argmax_buf

    for oh in range(out_h):
        ih_base = oh * stride
        for ow in range(out_w):
            iw_base = ow * stride
            window = x[:, :, ih_base:ih_base+pool_size, iw_base:iw_base+pool_size]
            win_reshaped = window.reshape(N, C, -1)
            out_buf[:, :, oh, ow] = np.max(win_reshaped, axis=-1)
            argmax_buf[:, :, oh, ow] = np.argmax(win_reshaped, axis=-1).astype(np.uint8)
    return out_buf, argmax_buf


def maxpool_backward(dout: np.ndarray, cache: np.ndarray, x_shape: tuple, pool_size: int, stride: int, dx_buf: np.ndarray = None):
    if dx_buf is None:
        dx_buf = np.zeros(x_shape, dtype=dout.dtype)
    else:
        dx_buf.fill(0.0)

    N, C, in_h, in_w = x_shape
    out_h, out_w = dout.shape[2], dout.shape[3]

    if _USE_NATIVE:
        _verify_native_args("maxpool_backward", dout=dout, dx_buf=dx_buf)
        _native_lib.direct_maxpool_backward_avx2(
            dout.ctypes.data, cache.ctypes.data, dx_buf.ctypes.data,
            N, C, out_h, out_w, in_h, in_w, pool_size, stride
        )
        return dx_buf

    for oh in range(out_h):
        ih_base = oh * stride
        for ow in range(out_w):
            for n in range(N):
                for c in range(C):
                    idx = cache[n, c, oh, ow]
                    r = idx // pool_size
                    col = idx % pool_size
                    dx_buf[n, c, ih_base + r, ow * stride + col] += dout[n, c, oh, ow]
    return dx_buf


def fuse_dout_transpose_and_bias(dout: np.ndarray, dout_trans_buf: np.ndarray, db_buf: np.ndarray):
    if _USE_NATIVE:
        _verify_native_args("fuse_dout_transpose_and_bias", dout=dout, db_buf=db_buf)
        N, C_out, out_h, out_w = dout.shape
        inv_m = 1.0 / float(N)
        _native_lib.direct_bias_backward_avx2(
            dout.ctypes.data, db_buf.ctypes.data,
            N, C_out, out_h, out_w, inv_m
        )
        return

    _fuse_dout_impl(dout, dout_trans_buf, db_buf)


def gemm_param_grad(dout_trans: np.ndarray, col: np.ndarray, dW_flat: np.ndarray, inv_m: float):
    _gemm_param_grad_impl(dout_trans, col, dW_flat, inv_m)


def relu_spatial_forward(x: np.ndarray) -> np.ndarray:
    if _USE_NATIVE:
        _verify_native_args("relu_spatial_forward", x=x)
        _native_lib.direct_relu_forward_avx2(x.ctypes.data, x.size)
        return x
    _relu_fwd_inplace_kernel(x)
    return x


def relu_spatial_backward(dout: np.ndarray, in_act: np.ndarray) -> np.ndarray:
    if _USE_NATIVE:
        _verify_native_args("relu_spatial_backward", dout=dout, in_act=in_act)
        _native_lib.direct_relu_backward_avx2(dout.ctypes.data, in_act.ctypes.data, dout.size)
        return dout
    _relu_bwd_inplace_kernel(dout, in_act)
    return dout


def im2col(x: np.ndarray, k_h: int, k_w: int, stride: int = 1, pad: int = 0, out_buf: np.ndarray = None) -> np.ndarray:
    return _im2col_impl(x, k_h, k_w, stride, pad, out_buf=out_buf)


def col2im(col: np.ndarray, input_shape: tuple, k_h: int, k_w: int, stride: int = 1, pad: int = 0, out_buf: np.ndarray = None) -> np.ndarray:
    return _col2im_impl(col, input_shape, k_h, k_w, stride, pad, out_buf=out_buf)