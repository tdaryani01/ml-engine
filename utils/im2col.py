import os
import ctypes
import time
import numpy as np

# Enforce thread configuration before importing C runtimes
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
_diagnostics_logged = False

# -----------------------------------------------------------------------------
# Timing Diagnostic Collector
# -----------------------------------------------------------------------------
_stats = {
    "fwd_count": 0, "fwd_l0": 0.0, "fwd_l1": 0.0,
    "dx_count": 0,  "dx_l0": 0.0,  "dx_l1": 0.0,
    "dw_count": 0,  "dw_l0": 0.0,  "dw_l1": 0.0,
}

def _log_timing(phase: str, Cin: int, dt: float):
    count_key = f"{phase}_count"
    l0_key = f"{phase}_l0"
    l1_key = f"{phase}_l1"
    
    if _stats[count_key] < 100:
        _stats[count_key] += 1
        if Cin == 3:
            _stats[l0_key] += dt
        else:
            _stats[l1_key] += dt

        if _stats["fwd_count"] >= 100 and _stats["dx_count"] >= 100 and _stats["dw_count"] >= 100:
            print("\n" + "=" * 65)
            print("  EXACT ENGINE TIME BREAKDOWN (Average ms per call)")
            print("=" * 65)
            print(f" Layer 0 (3 -> 8, 32x32):")
            print(f"   -> Forward : {_stats['fwd_l0'] / 50.0:.3f} ms")
            print(f"   -> dx      : {_stats['dx_l0'] / 50.0:.3f} ms")
            print(f"   -> dW      : {_stats['dw_l0'] / 50.0:.3f} ms")
            print(f"   -> Total   : {(_stats['fwd_l0'] + _stats['dx_l0'] + _stats['dw_l0']) / 50.0:.3f} ms")
            print("-" * 65)
            print(f" Layer 3 (8 -> 16, 16x16):")
            print(f"   -> Forward : {_stats['fwd_l1'] / 50.0:.3f} ms")
            print(f"   -> dx      : {_stats['dx_l1'] / 50.0:.3f} ms")
            print(f"   -> dW      : {_stats['dw_l1'] / 50.0:.3f} ms")
            print(f"   -> Total   : {(_stats['fwd_l1'] + _stats['dx_l1'] + _stats['dw_l1']) / 50.0:.3f} ms")
            print("=" * 65 + "\n")
            _stats["fwd_count"] = 1000

# -----------------------------------------------------------------------------
# Memory Alignment Helpers
# -----------------------------------------------------------------------------
def as_aligned_array(arr: np.ndarray, alignment: int = 32) -> np.ndarray:
    """Guarantees contiguous 32-byte aligned float32 array for AVX2 loads/stores."""
    if arr.ctypes.data % alignment == 0 and arr.flags['C_CONTIGUOUS'] and arr.dtype == np.float32:
        return arr
    buf = np.empty(arr.size * 4 + alignment, dtype=np.uint8)
    offset = (alignment - (buf.ctypes.data % alignment)) % alignment
    aligned = np.frombuffer(buf.data, dtype=np.float32, offset=offset, count=arr.size).reshape(arr.shape)
    np.copyto(aligned, arr.astype(np.float32, copy=False))
    return aligned

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
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64
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
            ctypes.c_int64, ctypes.c_int64
        ]

        _native_lib.direct_bias_backward_avx2.restype = None
        _native_lib.direct_bias_backward_avx2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_float
        ]

        _USE_NATIVE = True
    except Exception:
        BACKEND = "fast"

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
_USE_FAST = True


# -----------------------------------------------------------------------------
# Convolution Routing with Strict float32 Guarantees
# -----------------------------------------------------------------------------
def conv2d_forward(x: np.ndarray, W: np.ndarray, bias: np.ndarray,
                   stride: int, pad: int, out_buf: np.ndarray,
                   col_buf: np.ndarray = None, gemm_buf: np.ndarray = None,
                   fuse_relu: bool = False) -> tuple:
    global _diagnostics_logged

    if _USE_NATIVE and x.dtype == np.float32 and W.dtype == np.float32:
        N, C_in, H, W_in = x.shape
        C_out, _, k_h, k_w = W.shape

        if not _diagnostics_logged:
            print("\n--- Corrected Tensor Audit ---")
            print(f"  x       contiguous: {x.flags['C_CONTIGUOUS']} | dtype: {x.dtype} | strides: {x.strides}")
            print(f"  W       contiguous: {W.flags['C_CONTIGUOUS']} | dtype: {W.dtype} | strides: {W.strides}")
            print(f"  out_buf contiguous: {out_buf.flags['C_CONTIGUOUS']} | dtype: {out_buf.dtype} | strides: {out_buf.strides}")
            
            _native_lib.log_engine_runtime_diagnostics(
                x.ctypes.data, W.ctypes.data, out_buf.ctypes.data,
                N, C_in, H, W_in, C_out, k_h, k_w
            )
            _diagnostics_logged = True

        t0 = time.perf_counter()
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
        dt = (time.perf_counter() - t0) * 1000.0
        _log_timing("fwd", C_in, dt)
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


def conv2d_backward_weight(dout: np.ndarray, x: np.ndarray, dW: np.ndarray,
                           col: np.ndarray, dout_trans: np.ndarray,
                           stride: int, pad: int, inv_m: float) -> np.ndarray:
    if _USE_NATIVE and dout.dtype == np.float32 and x.dtype == np.float32:
        N, C_in, H, W_in = x.shape
        C_out, _, k_h, k_w = dW.shape
        
        t0 = time.perf_counter()
        _native_lib.direct_conv2d_backward_weight_avx2(
            dout.ctypes.data,
            x.ctypes.data,
            dW.ctypes.data,
            N, C_in, H, W_in,
            C_out, k_h, k_w,
            stride, pad,
            inv_m
        )
        dt = (time.perf_counter() - t0) * 1000.0
        _log_timing("dw", C_in, dt)
        return dW

    dW_flat = dW.reshape(dW.shape[0], -1)
    gemm_param_grad(dout_trans, col, dW_flat, inv_m)
    return dW


def conv2d_backward_input(dout: np.ndarray, W: np.ndarray, dx_buf: np.ndarray,
                          stride: int, pad: int,
                          dout_trans: np.ndarray = None, dcol_buf: np.ndarray = None) -> np.ndarray:
    if _USE_NATIVE and dout.dtype == np.float32 and W.dtype == np.float32:
        N, C_in, H, W_in = dx_buf.shape
        C_out, _, k_h, k_w = W.shape
        
        t0 = time.perf_counter()
        _native_lib.direct_conv2d_backward_input_avx2(
            dout.ctypes.data,
            W.ctypes.data,
            dx_buf.ctypes.data,
            N, C_in, H, W_in,
            C_out, k_h, k_w,
            stride, pad
        )
        dt = (time.perf_counter() - t0) * 1000.0
        _log_timing("dx", C_in, dt)
        return dx_buf

    W_2d = W.reshape(W.shape[0], -1)
    total_rows = dout_trans.shape[0]
    active_dcol = dcol_buf[:total_rows]
    np.dot(dout_trans, W_2d, out=active_dcol)
    return col2im(active_dcol, dx_buf.shape, W.shape[2], W.shape[3], stride, pad, out_buf=dx_buf)


# -----------------------------------------------------------------------------
# Primitives & Activations
# -----------------------------------------------------------------------------
def im2col(x: np.ndarray, k_h: int, k_w: int, stride: int = 1, pad: int = 0, out_buf: np.ndarray = None) -> np.ndarray:
    return _im2col_impl(x, k_h, k_w, stride, pad, out_buf=out_buf)


def col2im(col: np.ndarray, input_shape: tuple, k_h: int, k_w: int, stride: int = 1, pad: int = 0, out_buf: np.ndarray = None) -> np.ndarray:
    return _col2im_impl(col, input_shape, k_h, k_w, stride, pad, out_buf=out_buf)


def maxpool_forward(x: np.ndarray, pool_size: int, stride: int, out_buf: np.ndarray = None, argmax_buf: np.ndarray = None):
    N, C, H, W = x.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1

    if _USE_NATIVE and x.dtype == np.float32:
        _native_lib.direct_maxpool_forward_avx2(
            x.ctypes.data, out_buf.ctypes.data, argmax_buf.ctypes.data,
            N, C, H, W, pool_size, stride
        )
        return out_buf, argmax_buf

    _maxpool_forward_kernel(x, pool_size, stride, out_buf, argmax_buf)
    return out_buf, argmax_buf


def maxpool_backward(dout: np.ndarray, cache, x_shape: tuple, pool_size: int, stride: int, dx_buf: np.ndarray = None):
    if _USE_NATIVE and dout.dtype == np.float32:
        N, C, in_h, in_w = x_shape
        out_h, out_w = dout.shape[2], dout.shape[3]
        _native_lib.direct_maxpool_backward_avx2(
            dout.ctypes.data, cache.ctypes.data, dx_buf.ctypes.data,
            N, C, out_h, out_w, in_h, in_w
        )
        return dx_buf

    _maxpool_backward_kernel(dout, cache, dx_buf)
    return dx_buf


def fuse_dout_transpose_and_bias(dout: np.ndarray, dout_trans_buf: np.ndarray, db_buf: np.ndarray):
    if _USE_NATIVE and dout.dtype == np.float32:
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
    if _USE_NATIVE and x.dtype == np.float32:
        _native_lib.direct_relu_forward_avx2(x.ctypes.data, x.size)
        return x
    _relu_fwd_inplace_kernel(x)
    return x


def relu_spatial_backward(dout: np.ndarray, in_act: np.ndarray) -> np.ndarray:
    if _USE_NATIVE and dout.dtype == np.float32:
        _native_lib.direct_relu_backward_avx2(dout.ctypes.data, in_act.ctypes.data, dout.size)
        return dout
    _relu_bwd_inplace_kernel(dout, in_act)
    return dout