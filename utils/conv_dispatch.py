# utils/conv_dispatch.py
# Convolution backend dispatch: native ctypes, im2col+GEMM, and NumPy reference paths.
import os
import ctypes
import glob
import logging
from typing import TYPE_CHECKING

import numpy as np
from config.constants import EngineBackend
from utils.perf_experiments import skip_dx_zero, step_dx_zero

if TYPE_CHECKING:
    from utils.engine_ops import EngineContext

logger = logging.getLogger(__name__)

_active_backend: EngineBackend = None
_native_lib = None
_primitives_lib = None
_is_initialized: bool = False


def _locate_native_dll() -> str | None:
    lib_name = "conv_kernels.dll" if os.name == "nt" else "conv_kernels.so"
    stage_name = "conv_kernels_stage.dll" if os.name == "nt" else "conv_kernels_stage.so"
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if os.environ.get("ML_ENGINE_USE_STAGE_DLL", "").strip().lower() in ("1", "true", "yes"):
        stage_path = os.path.join(root_dir, "bin", stage_name)
        if os.path.exists(stage_path):
            return os.path.abspath(stage_path)
    possible_paths = [
        os.path.join(root_dir, "bin", lib_name),
        os.path.join(os.path.dirname(__file__), "bin", lib_name),
        os.path.join(os.path.dirname(__file__), "..", "bin", lib_name),
        os.path.join(os.path.dirname(__file__), "..", "native", lib_name),
        os.path.join(os.path.dirname(__file__), "..", "src", "native", lib_name),
        os.path.join(os.path.dirname(__file__), "native", lib_name),
    ]
    for p in possible_paths:
        abspath = os.path.abspath(p)
        if os.path.exists(abspath):
            return abspath
    return None


def _bind_thread_config(lib) -> bool:
    """Bind DLL OpenMP policy exports (configure_native_threads / get_omp_threads)."""
    try:
        lib.configure_native_threads.restype = ctypes.c_int32
        lib.configure_native_threads.argtypes = [ctypes.c_int32]
        lib.get_omp_threads.restype = ctypes.c_int32
        lib.get_omp_threads.argtypes = []
        lib.configure_im2col_parallel_cap.restype = ctypes.c_int32
        lib.configure_im2col_parallel_cap.argtypes = [ctypes.c_int32]
        lib.get_im2col_parallel_cap.restype = ctypes.c_int32
        lib.get_im2col_parallel_cap.argtypes = []
        lib.configure_openblas_threads.restype = ctypes.c_int32
        lib.configure_openblas_threads.argtypes = [ctypes.c_int32]
        return True
    except AttributeError:
        return False


def sync_native_thread_policy(omp_threads: int) -> int:
    """Pin conv_kernels.dll LLVM OpenMP (im2col/col2im/fuse).

    threadpoolctl does not reach the DLL's LLVM OpenMP — this call does.
    """
    omp_threads = max(1, int(omp_threads))
    lib = _load_conv_dll()
    if lib is None or not hasattr(lib, "configure_native_threads"):
        return omp_threads
    lib.configure_native_threads(omp_threads)
    if hasattr(lib, "get_omp_threads"):
        return int(lib.get_omp_threads())
    return omp_threads


def sync_openblas_thread_policy(blas_threads: int) -> int:
    """Pin native OpenBLAS GEMM threads (USE_OPENMP=1 → openblas_set_num_threads)."""
    blas_threads = max(1, int(blas_threads))
    lib = _load_conv_dll()
    if lib is None or not hasattr(lib, "configure_openblas_threads"):
        return blas_threads
    lib.configure_openblas_threads(blas_threads)
    return blas_threads


def sync_im2col_parallel_cap(omp_threads: int) -> int:
    """Mirror omp_during_fit for telemetry (im2col uses get_omp_threads())."""
    omp_threads = max(1, int(omp_threads))
    lib = _load_conv_dll()
    if lib is None or not hasattr(lib, "configure_im2col_parallel_cap"):
        return omp_threads
    lib.configure_im2col_parallel_cap(omp_threads)
    if hasattr(lib, "get_im2col_parallel_cap"):
        return int(lib.get_im2col_parallel_cap())
    return omp_threads


def _bind_im2col_primitives(lib) -> bool:
    """Bind im2col/col2im exports if present in the DLL."""
    try:
        lib.im2col_avx2.restype = ctypes.c_int32
        lib.im2col_avx2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ]
        lib.col2im_avx2.restype = ctypes.c_int32
        lib.col2im_avx2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ]
        _bind_im2col_telemetry(lib)
        return True
    except AttributeError:
        return False


class _Im2ColTelemetry(ctypes.Structure):
    _fields_ = [
        ("im2col_calls", ctypes.c_uint64),
        ("im2col_tile_fast", ctypes.c_uint64),
        ("im2col_tile_padded", ctypes.c_uint64),
        ("col2im_calls", ctypes.c_uint64),
        ("col2im_tile_fast", ctypes.c_uint64),
        ("col2im_tile_xclip", ctypes.c_uint64),
        ("col2im_tile_yclip", ctypes.c_uint64),
        ("col2im_tile_corner", ctypes.c_uint64),
        ("col2im_memset_bytes", ctypes.c_uint64),
        ("col2im_memset_calls", ctypes.c_uint64),
        ("misalign_x_ptr", ctypes.c_uint64),
        ("misalign_col_ptr", ctypes.c_uint64),
        ("misalign_dx_ptr", ctypes.c_uint64),
        ("misalign_out_ptr", ctypes.c_uint64),
        ("gemm_fwd_calls", ctypes.c_uint64),
        ("gemm_bwd_w_calls", ctypes.c_uint64),
        ("gemm_bwd_x_calls", ctypes.c_uint64),
        ("fuse_dout_transpose_calls", ctypes.c_uint64),
    ]


def _bind_im2col_telemetry(lib) -> None:
    try:
        lib.reset_im2col_telemetry.restype = None
        lib.reset_im2col_telemetry.argtypes = []
        lib.get_im2col_telemetry.restype = None
        lib.get_im2col_telemetry.argtypes = [ctypes.POINTER(_Im2ColTelemetry)]
        lib.log_im2col_telemetry.restype = None
        lib.log_im2col_telemetry.argtypes = []
    except AttributeError:
        pass


def _format_im2col_telemetry(t: _Im2ColTelemetry) -> str:
    im2col_tiles = t.im2col_tile_fast + t.im2col_tile_padded
    col2im_tiles = (
        t.col2im_tile_fast + t.col2im_tile_xclip + t.col2im_tile_yclip + t.col2im_tile_corner
    )
    gemm_total = t.gemm_fwd_calls + t.gemm_bwd_w_calls + t.gemm_bwd_x_calls
    im2col_fast_pct = (100.0 * t.im2col_tile_fast / im2col_tiles) if im2col_tiles else 0.0
    im2col_pad_pct = (100.0 * t.im2col_tile_padded / im2col_tiles) if im2col_tiles else 0.0
    col2im_fast_pct = (100.0 * t.col2im_tile_fast / col2im_tiles) if col2im_tiles else 0.0
    col2im_xclip_pct = (100.0 * t.col2im_tile_xclip / col2im_tiles) if col2im_tiles else 0.0
    col2im_yclip_pct = (100.0 * t.col2im_tile_yclip / col2im_tiles) if col2im_tiles else 0.0
    col2im_corner_pct = (100.0 * t.col2im_tile_corner / col2im_tiles) if col2im_tiles else 0.0
    memset_mib = t.col2im_memset_bytes / (1024.0 * 1024.0)
    memset_avg_kib = (
        t.col2im_memset_bytes / (1024.0 * t.col2im_memset_calls)
        if t.col2im_memset_calls
        else 0.0
    )
    gemm_split = ""
    if gemm_total > 0:
        gemm_split = (
            f" (fwd={100.0 * t.gemm_fwd_calls / gemm_total:.1f}% "
            f"bwd_w={100.0 * t.gemm_bwd_w_calls / gemm_total:.1f}% "
            f"bwd_x={100.0 * t.gemm_bwd_x_calls / gemm_total:.1f}%)"
        )
    misalign_warn = ""
    if t.misalign_x_ptr + t.misalign_col_ptr + t.misalign_dx_ptr + t.misalign_out_ptr > 0:
        misalign_warn = "  [WARN: unaligned buffers can hurt AVX/memcpy]"
    return (
        "\n=== im2col+gemm telemetry (cumulative) ===\n"
        f"im2col calls={t.im2col_calls} tiles={im2col_tiles} "
        f"(fast={im2col_fast_pct:.1f}% padded={im2col_pad_pct:.1f}%)\n"
        f"col2im calls={t.col2im_calls} tiles={col2im_tiles} "
        f"(fast={col2im_fast_pct:.1f}% xclip={col2im_xclip_pct:.1f}% "
        f"yclip={col2im_yclip_pct:.1f}% corner={col2im_corner_pct:.1f}%)\n"
        f"col2im memset: calls={t.col2im_memset_calls} total_bytes={t.col2im_memset_bytes} "
        f"({memset_mib:.2f} MiB) avg={memset_avg_kib:.0f} KiB/call\n"
        f"GEMM calls: fwd={t.gemm_fwd_calls} bwd_w={t.gemm_bwd_w_calls} "
        f"bwd_x={t.gemm_bwd_x_calls} total={gemm_total}{gemm_split}\n"
        f"fuse_dout_transpose calls={t.fuse_dout_transpose_calls}\n"
        f"ptr misalign (32B): x={t.misalign_x_ptr} col/out={t.misalign_col_ptr} "
        f"dx={t.misalign_dx_ptr} out_only={t.misalign_out_ptr}{misalign_warn}\n"
        f"hint: C col2im memset off by default; Python step_dx_zero uses ndarray.fill "
        f"(shows as memset_repstos in uProf). ML_ENGINE_SKIP_DX_ZERO=1 to mock.\n"
        "=========================================="
    )


def reset_im2col_telemetry() -> None:
    lib = _load_conv_dll()
    if lib is not None and hasattr(lib, "reset_im2col_telemetry"):
        lib.reset_im2col_telemetry()


def log_im2col_telemetry() -> None:
    """Emit cumulative im2col/col2im/GEMM path counters from conv_kernels.dll."""
    lib = _load_conv_dll()
    if lib is None:
        logger.warning("im2col telemetry: conv_kernels.dll not loaded")
        return
    if hasattr(lib, "get_im2col_telemetry"):
        snapshot = _Im2ColTelemetry()
        lib.get_im2col_telemetry(ctypes.byref(snapshot))
        logger.warning("%s", _format_im2col_telemetry(snapshot))
    elif hasattr(lib, "log_im2col_telemetry"):
        lib.log_im2col_telemetry()
    else:
        logger.warning("im2col telemetry: DLL missing get_im2col_telemetry/log_im2col_telemetry exports")


def _discover_openblas_dll() -> str | None:
    """Locate OpenBLAS DLL shipped with NumPy (Windows)."""
    try:
        import numpy
    except ImportError:
        return None
    libs_dir = os.path.abspath(os.path.join(os.path.dirname(numpy.__file__), "..", "numpy.libs"))
    for pattern in ("*openblas*.dll", "*OpenBLAS*.dll"):
        hits = sorted(glob.glob(os.path.join(libs_dir, pattern)))
        if hits:
            return hits[0]
    return None


def _bind_im2col_gemm_exports(lib) -> bool:
    """Bind fused im2col+GEMM conv exports and OpenBLAS runtime init."""
    try:
        lib.init_openblas_runtime.restype = ctypes.c_int32
        lib.init_openblas_runtime.argtypes = [ctypes.c_char_p]
        lib.init_blas_sgemm_ptr.restype = ctypes.c_int32
        lib.init_blas_sgemm_ptr.argtypes = [ctypes.c_void_p]
        lib.blas_runtime_ready.restype = ctypes.c_int32
        lib.blas_runtime_ready.argtypes = []
        lib.blas_uses_unified_omp.restype = ctypes.c_int32
        lib.blas_uses_unified_omp.argtypes = []

        _conv_fwd_sig = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int32,
        ]
        lib.conv2d_forward_im2col_gemm_avx2.restype = ctypes.c_int32
        lib.conv2d_forward_im2col_gemm_avx2.argtypes = _conv_fwd_sig

        _bwd_w_sig = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_float, ctypes.c_int32,
            ctypes.c_int32,
        ]
        lib.conv2d_backward_weight_im2col_gemm_avx2.restype = ctypes.c_int32
        lib.conv2d_backward_weight_im2col_gemm_avx2.argtypes = _bwd_w_sig

        _bwd_x_sig = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int32,
            ctypes.c_int32,
        ]
        lib.conv2d_backward_input_im2col_gemm_avx2.restype = ctypes.c_int32
        lib.conv2d_backward_input_im2col_gemm_avx2.argtypes = _bwd_x_sig
        lib.fuse_dout_transpose_bias_avx2.restype = ctypes.c_int32
        lib.fuse_dout_transpose_bias_avx2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_float,
        ]
        return True
    except AttributeError:
        return False


def native_blas_unified_omp() -> bool:
    """True when bin/libopenblas.dll was built with USE_OPENMP=1 (OpenMP inside GEMM)."""
    lib = _load_conv_dll()
    if lib is None or not hasattr(lib, "blas_uses_unified_omp"):
        return False
    try:
        return int(lib.blas_uses_unified_omp()) == 1
    except Exception:
        return False


def bootstrap_im2col_gemm_runtime() -> bool:
    """Load conv_kernels + bin/libopenblas before runtime thread policy (idempotent)."""
    lib = _ensure_primitives_lib()
    if lib is None:
        return False
    return native_blas_unified_omp()


def _openblas_build_marker_path() -> str:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(root_dir, "bin", "openblas_build.json")


def _apply_openblas_build_marker() -> bool:
    """Set ML_ENGINE_OPENBLAS_USE_OPENMP before first native OpenBLAS load."""
    raw = os.environ.get("ML_ENGINE_OPENBLAS_USE_OPENMP")
    if raw is not None:
        return raw.strip() == "1"
    marker_path = _openblas_build_marker_path()
    if os.path.exists(marker_path):
        try:
            import json
            with open(marker_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if data.get("use_openmp"):
                os.environ["ML_ENGINE_OPENBLAS_USE_OPENMP"] = "1"
                return True
        except Exception as exc:
            logger.debug("openblas_build.json read failed: %s", exc)
    if _local_openblas_dll() is not None:
        os.environ["ML_ENGINE_OPENBLAS_USE_OPENMP"] = "1"
        return True
    return False


def _local_openblas_dll() -> str | None:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    name = "libopenblas.dll" if os.name == "nt" else "libopenblas.so"
    path = os.path.join(root_dir, "bin", name)
    return path if os.path.exists(path) else None


def _init_native_openblas(lib) -> bool:
    _apply_openblas_build_marker()
    local = _local_openblas_dll()
    if local is not None:
        status = lib.init_openblas_runtime(local.encode("utf-8"))
        if status == 0 and lib.blas_runtime_ready() == 1:
            unified = native_blas_unified_omp()
            logger.info(
                "OpenBLAS runtime ready: %s (use_openmp=%s)",
                local, unified,
            )
            return True
        logger.warning("bin OpenBLAS init failed (status=%s path=%s)", status, local)

    dll_path = os.environ.get("ML_ENGINE_OPENBLAS_DLL")
    if dll_path:
        status = lib.init_openblas_runtime(dll_path.encode("utf-8"))
        if status == 0 and lib.blas_runtime_ready() == 1:
            logger.info("OpenBLAS runtime ready: %s (use_openmp=%s)", dll_path, native_blas_unified_omp())
            return True
        logger.warning("OpenBLAS env init failed (status=%s path=%s)", status, dll_path)

    try:
        from scipy.linalg import cython_blas

        ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
        ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

        cap = cython_blas.__pyx_capi__["sgemm"]
        cap_name = ctypes.pythonapi.PyCapsule_GetName(cap)
        sgemm_ptr = ctypes.pythonapi.PyCapsule_GetPointer(cap, cap_name)
        lib.init_blas_sgemm_ptr.restype = ctypes.c_int32
        lib.init_blas_sgemm_ptr.argtypes = [ctypes.c_void_p]
        if lib.init_blas_sgemm_ptr(sgemm_ptr) == 0 and lib.blas_runtime_ready() == 1:
            logger.warning(
                "OpenBLAS via scipy capsule (pthreads — not USE_OPENMP=1). "
                "Run scripts/build_openblas.ps1 for bin/libopenblas.dll."
            )
            return True
    except Exception as exc:
        logger.debug("scipy BLAS capsule init failed: %s", exc)

    dll_path = _discover_openblas_dll()
    path_arg = dll_path.encode("utf-8") if dll_path else None
    status = lib.init_openblas_runtime(path_arg)
    if status != 0:
        logger.warning("OpenBLAS runtime init failed (status=%s, path=%s)", status, dll_path)
        return False
    if lib.blas_runtime_ready() != 1:
        logger.warning("OpenBLAS runtime not ready after init (path=%s)", dll_path)
        return False
    logger.warning(
        "OpenBLAS from numpy.libs (pthreads — not USE_OPENMP=1). "
        "Run scripts/build_openblas.ps1 for bin/libopenblas.dll."
    )
    return True


_fast_kernels_loaded = False
_native_im2col_available = False
_native_im2col_gemm_available = False
_im2col_gemm_ctypes_bound = False


def _load_conv_dll():
    """Load conv_kernels once; shared by NATIVE and IM2COL_GEMM."""
    global _primitives_lib
    if _primitives_lib is not None:
        return _primitives_lib
    lib_path = _locate_native_dll()
    if lib_path is None:
        return None
    lib = ctypes.CDLL(lib_path)
    if not _bind_im2col_primitives(lib):
        return None
    _bind_thread_config(lib)
    _primitives_lib = lib
    return _primitives_lib


def _ensure_im2col_gemm_ctypes(lib) -> None:
    """Bind fused im2col+GEMM ctypes exports once per DLL handle."""
    global _im2col_gemm_ctypes_bound, _native_im2col_gemm_available
    if _im2col_gemm_ctypes_bound:
        return
    if _bind_im2col_gemm_exports(lib) and _init_native_openblas(lib):
        _native_im2col_gemm_available = True
    _im2col_gemm_ctypes_bound = True


def _ensure_primitives_lib():
    """Return shared DLL handle with im2col/col2im (+ optional GEMM) bindings."""
    lib = _load_conv_dll()
    if lib is not None:
        _ensure_im2col_gemm_ctypes(lib)
    return lib


def _native_allow_im2col_fallback() -> bool:
    """Legacy dev escape hatch; default is strict native-only (Phase D2)."""
    for key in ("ML_ENGINE_NATIVE_FALLBACK", "ML_ENGINE_FORCE_FALLBACK"):
        if os.environ.get(key, "").lower() in ("1", "true", "yes"):
            return True
    return False


def _fail_native(func_name: str, reason: str) -> None:
    raise RuntimeError(
        f"[Native Engine Error] {func_name} failed: {reason}. "
        "Set ML_ENGINE_NATIVE_FALLBACK=1 to allow im2col+GEMM fallback."
    )


def _use_im2col_gemm_fast_path(be: EngineBackend) -> bool:
    return be == EngineBackend.IM2COL_GEMM


def _resolve_backend(
    ctx: "EngineContext | None" = None,
    backend: EngineBackend | None = None,
) -> EngineBackend:
    """Prefer explicit backend enum; then context; then legacy module global."""
    if backend is not None:
        return backend
    if ctx is not None:
        return ctx.backend
    _ensure_initialized()
    return _active_backend


def get_native_lib():
    """Return the loaded native DLL handle, or ``None`` if not on NATIVE backend."""
    _ensure_initialized()
    return _native_lib


def _check_status(status: int, func_name: str):
    if status != 0:
        raise RuntimeError(f"[Native Engine Error] {func_name} failed with native return code: {status}")


def _log_fallback(func_name: str, reason: str):
    logger.warning(f"[Engine Backend Fallback] {func_name} bypassed NATIVE execution. Reason: {reason}")


def _native_miss(func_name: str, reason: str) -> EngineBackend:
    """Strict native failure or explicit downgrade to im2col+GEMM."""
    if not _native_allow_im2col_fallback():
        _fail_native(func_name, reason)
    _log_fallback(func_name, reason)
    return EngineBackend.IM2COL_GEMM


def _native_unsupported_dtype(func_name: str, dtype) -> EngineBackend:
    """Native DLL is float32-only; route to im2col+GEMM without treating as failure."""
    logger.debug(
        "[%s] dtype %s unsupported on NATIVE; using im2col+GEMM path",
        func_name,
        dtype,
    )
    return EngineBackend.IM2COL_GEMM


# -----------------------------------------------------------------------------
# Engine Backend Initialization
# -----------------------------------------------------------------------------
def init_engine_backend(backend: EngineBackend = EngineBackend.NATIVE):
    """Initialize global backend shim (tests/legacy). Prefer ``create_engine_context()``."""
    global _active_backend, _native_lib, _is_initialized

    if _is_initialized and _active_backend == backend:
        return

    _active_backend = backend

    if _active_backend == EngineBackend.NATIVE:
        lib = _load_conv_dll()
        if lib is None:
            lib_name = "conv_kernels.dll" if os.name == "nt" else "conv_kernels.so"
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            raise FileNotFoundError(
                f"Could not locate {lib_name}. Checked bin/ and native search paths under {root_dir}"
            )

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

    elif _active_backend == EngineBackend.IM2COL_GEMM:
        _native_lib = None
        _ensure_primitives_lib()
        _ensure_fast_kernels()

    else:
        _native_lib = None

    _is_initialized = True
    _sync_backend_thread_policy(backend)


def _sync_backend_thread_policy(backend: EngineBackend) -> None:
    """Pin shared LLVM OMP pool (configure_native_threads syncs OpenBLAS when unified)."""
    try:
        from utils.runtime import load_runtime_settings

        settings = load_runtime_settings()
        omp = settings.omp_threads_for(backend)
        sync_im2col_parallel_cap(omp)
        effective = sync_native_thread_policy(omp)
        logger.debug(
            "DLL OMP pinned: backend=%s fit_omp=%d effective=%d unified=%s",
            backend.value, omp, effective, native_blas_unified_omp(),
        )
    except Exception as exc:
        logger.debug("DLL thread sync skipped: %s", exc)


def _ensure_initialized():
    if not _is_initialized:
        init_engine_backend(EngineBackend.NATIVE)


_im2col_impl = None
_col2im_impl = None
_gemm_forward_impl = None
_gemm_param_grad_impl = None
_gemm_backward_input_impl = None
_maxpool_forward_kernel = None
_maxpool_backward_kernel = None
_fuse_dout_impl = None
fuse_forward_impl = None
_relu_fwd_inplace_kernel = None
_relu_bwd_inplace_kernel = None


def _im2col_dispatch(x: np.ndarray, k_h: int, k_w: int, stride: int = 1, pad: int = 0, out_buf: np.ndarray = None) -> np.ndarray:
    from utils import im2col_fast as fast
    if x.dtype == np.float32 and _native_im2col_available:
        return _im2col_native(x, k_h, k_w, stride, pad, out_buf=out_buf)
    return fast.im2col_fast(x, k_h, k_w, stride, pad, out_buf=out_buf)


def _col2im_dispatch(col: np.ndarray, input_shape: tuple, k_h: int, k_w: int, stride: int = 1, pad: int = 0, out_buf: np.ndarray = None) -> np.ndarray:
    from utils import im2col_fast as fast
    if col.dtype == np.float32 and _native_im2col_available:
        return _col2im_native(col, input_shape, k_h, k_w, stride, pad, out_buf=out_buf)
    return fast.col2im_fast(col, input_shape, k_h, k_w, stride, pad, out_buf=out_buf)


def _im2col_native(x: np.ndarray, k_h: int, k_w: int, stride: int = 1, pad: int = 0, out_buf: np.ndarray = None) -> np.ndarray:
    lib = _ensure_primitives_lib()
    if lib is None:
        raise RuntimeError("Native im2col requested but conv_kernels.dll not loaded")
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    if x.dtype != np.float32:
        raise TypeError("Native im2col supports float32 only")
    N, C, H, W = x.shape
    out_h = (H + 2 * pad - k_h) // stride + 1
    out_w = (W + 2 * pad - k_w) // stride + 1
    total_rows = N * out_h * out_w
    total_cols = C * k_h * k_w
    if out_buf is None or out_buf.shape != (total_rows, total_cols) or out_buf.dtype != x.dtype:
        out_buf = np.empty((total_rows, total_cols), dtype=x.dtype)
    status = lib.im2col_avx2(
        _get_c_ptr(x), _get_c_ptr(out_buf),
        int(N), int(C), int(H), int(W), int(W), int(k_h), int(k_w), int(stride), int(pad),
    )
    if status != 0:
        raise RuntimeError(f"im2col_avx2 failed with code {status}")
    return out_buf


def _col2im_native(col: np.ndarray, input_shape: tuple, k_h: int, k_w: int, stride: int = 1, pad: int = 0, out_buf: np.ndarray = None) -> np.ndarray:
    lib = _ensure_primitives_lib()
    if lib is None:
        raise RuntimeError("Native col2im requested but conv_kernels.dll not loaded")
    if not col.flags["C_CONTIGUOUS"]:
        col = np.ascontiguousarray(col)
    N, C, H, W_logical = input_shape
    W_stride = out_buf.shape[3] if out_buf is not None else W_logical
    if out_buf is None or out_buf.shape != (N, C, H, W_stride) or out_buf.dtype != col.dtype:
        out_buf = np.zeros((N, C, H, W_stride), dtype=col.dtype)
    status = lib.col2im_avx2(
        _get_c_ptr(col), _get_c_ptr(out_buf),
        int(N), int(C), int(H), int(W_logical), int(W_stride),
        int(k_h), int(k_w), int(stride), int(pad),
    )
    if status != 0:
        raise RuntimeError(f"col2im_avx2 failed with code {status}")
    return out_buf


def _ensure_fast_kernels():
    global _im2col_impl, _col2im_impl, _gemm_forward_impl, _gemm_param_grad_impl
    global _gemm_backward_input_impl
    global _maxpool_forward_kernel, _maxpool_backward_kernel
    global _fuse_dout_impl, fuse_forward_impl
    global _relu_fwd_inplace_kernel, _relu_bwd_inplace_kernel, _fast_kernels_loaded
    global _native_im2col_available
    if _fast_kernels_loaded:
        return
    from utils import im2col_fast as fast
    lib = _ensure_primitives_lib()
    if lib is not None:
        _im2col_impl = _im2col_dispatch
        _col2im_impl = _col2im_dispatch
        _native_im2col_available = True
    else:
        _im2col_impl = fast.im2col_fast
        _col2im_impl = fast.col2im_fast
        _native_im2col_available = False
    _gemm_forward_impl = fast.gemm_forward_fast
    _gemm_param_grad_impl = fast.gemm_param_grad_fast
    _gemm_backward_input_impl = fast.gemm_backward_input_fast
    _maxpool_forward_kernel = fast._maxpool_forward_kernel
    _maxpool_backward_kernel = fast._maxpool_backward_kernel
    _fuse_dout_impl = fast.fuse_dout_transpose_bias_fast
    fuse_forward_impl = fast._fuse_forward_transpose_and_bias
    _relu_fwd_inplace_kernel = fast._relu_fwd_inplace_kernel
    _relu_bwd_inplace_kernel = fast._relu_bwd_inplace_kernel
    _fast_kernels_loaded = True


def _get_c_ptr(arr: np.ndarray):
    """Safely extracts a C data pointer from a NumPy array."""
    if arr is None or arr.size == 0:
        return None
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    return ctypes.c_void_p(int(arr.ctypes.data))


def _pack_w_gemm_fwd(W: np.ndarray, buf: np.ndarray | None = None) -> np.ndarray:
    """Pack [C_out,K] filters into contiguous [K,C_out] for SGEMM('N','N') forward."""
    c_out = W.shape[0]
    k_dim = int(np.prod(W.shape[1:]))
    if buf is None or buf.shape != (k_dim, c_out) or buf.dtype != W.dtype:
        buf = np.empty((k_dim, c_out), dtype=W.dtype)
    np.copyto(buf, W.reshape(c_out, k_dim).T)
    return buf


# -----------------------------------------------------------------------------
# Composite Block Routines
# -----------------------------------------------------------------------------
# @profile
def conv_block_forward(x: np.ndarray, W: np.ndarray, bias: np.ndarray,
                       out_conv_buf: np.ndarray, out_pool_buf: np.ndarray, argmax_buf: np.ndarray,
                       conv_stride: int = 1, conv_pad: int = 1,
                       pool_size: int = 2, pool_stride: int = 2,
                       col_buf: np.ndarray = None, gemm_buf: np.ndarray = None,
                       w_gemm_fwd_buf: np.ndarray = None,
                       W_logical: int = None, out_w_logical: int = None,
                       ctx: "EngineContext | None" = None,
                       backend: EngineBackend | None = None) -> tuple:
    _ensure_initialized()
    be = _resolve_backend(ctx, backend)

    if be == EngineBackend.NATIVE:
        if _native_lib is None:
            be = _native_miss("conv_block_forward", "Native library not loaded")
        elif x.dtype != np.float32:
            be = _native_unsupported_dtype("conv_block_forward", x.dtype)
        else:
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
            if status == 0:
                return out_pool_buf, out_conv_buf, argmax_buf, None
            be = _native_miss("conv_block_forward", f"Native DLL returned error code {status}")

    out_conv, col = conv2d_forward(
        x=x, W=W, bias=bias, stride=conv_stride, pad=conv_pad,
        out_buf=out_conv_buf, col_buf=col_buf, gemm_buf=gemm_buf,
        w_gemm_fwd_buf=w_gemm_fwd_buf,
        fuse_relu=True, W_logical=W_logical, backend=be
    )
    
    # Extract logical slice before pooling on fallback backends
    out_h = (x.shape[2] + 2 * conv_pad - W.shape[2]) // conv_stride + 1
    out_w = ((W_logical if W_logical is not None else x.shape[3]) + 2 * conv_pad - W.shape[3]) // conv_stride + 1
    valid_conv = out_conv[:, :, :out_h, :out_w] if out_conv.shape[3] != out_w else out_conv

    out_pool, argmax = maxpool_forward(
        valid_conv, pool_size, pool_stride, out_buf=out_pool_buf,
        argmax_buf=argmax_buf, backend=be
    )
    return out_pool, out_conv, argmax, col

# @profile
def conv_block_backward(dout_pool: np.ndarray, argmax_buf: np.ndarray,
                        x: np.ndarray, W: np.ndarray, conv_act: np.ndarray,
                        d_conv_buf: np.ndarray, dx_buf: np.ndarray, dW_buf: np.ndarray, db_buf: np.ndarray,
                        conv_stride: int = 1, conv_pad: int = 1,
                        pool_size: int = 2, pool_stride: int = 2,
                        inv_m: float = 1.0,
                        col: np.ndarray = None, dout_trans: np.ndarray = None, dcol_buf: np.ndarray = None,
                        W_logical: int = None, out_w_logical: int = None,
                        ctx: "EngineContext | None" = None,
                        backend: EngineBackend | None = None) -> tuple:
    _ensure_initialized()
    be = _resolve_backend(ctx, backend)

    if be == EngineBackend.NATIVE:
        if _native_lib is None:
            be = _native_miss("conv_block_backward", "Native library not loaded")
        elif dout_pool.dtype != np.float32:
            be = _native_unsupported_dtype("conv_block_backward", dout_pool.dtype)
        else:
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
            if status == 0:
                return dx_buf, dW_buf, db_buf
            be = _native_miss("conv_block_backward", f"Native DLL returned error code {status}")

    out_h = (x.shape[2] + 2 * conv_pad - W.shape[2]) // conv_stride + 1
    out_w = ((W_logical if W_logical is not None else x.shape[3]) + 2 * conv_pad - W.shape[3]) // conv_stride + 1
    N = x.shape[0]
    conv_act_logical = conv_act[:, :, :out_h, :out_w] if conv_act.shape[3] != out_w else conv_act

    d_conv_buf.fill(0.0)
    if d_conv_buf.shape[3] == out_w:
        d_conv_target = d_conv_buf[:N]
    else:
        d_conv_target = d_conv_buf[:N, :, :out_h, :out_w]
    maxpool_backward(
        dout_pool, argmax_buf, conv_act_logical.shape, pool_size, pool_stride,
        dx_buf=d_conv_target, backend=be,
    )
    _ensure_fast_kernels()
    _relu_bwd_inplace_kernel(d_conv_target, conv_act_logical)

    if db_buf is not None:
        fuse_dout_transpose_and_bias(d_conv_target, dout_trans, db_buf, backend=be)

    dx, dW = conv2d_backward_fused(
        d_conv_buf, x, W, dx_buf, dW_buf,
        stride=conv_stride, pad=conv_pad, inv_m=inv_m,
        in_act=None, fuse_relu=False, col=col, dout_trans=dout_trans,
        dcol_buf=dcol_buf, W_logical=W_logical, backend=be
    )
    return dx, dW, db_buf


# -----------------------------------------------------------------------------
# Standalone Layer Routines
# -----------------------------------------------------------------------------
def conv2d_forward(x: np.ndarray, W: np.ndarray, bias: np.ndarray,
                   stride: int, pad: int, out_buf: np.ndarray,
                   col_buf: np.ndarray = None, gemm_buf: np.ndarray = None,
                   w_gemm_fwd_buf: np.ndarray = None,
                   fuse_relu: bool = False, W_logical: int = None,
                   ctx: "EngineContext | None" = None,
                   backend: EngineBackend | None = None) -> tuple:
    _ensure_initialized()
    be = _resolve_backend(ctx, backend)
    N, C_in, H, W_in_stride = x.shape
    C_out, _, k_h, k_w = W.shape
    W_in = W_logical if W_logical is not None else W_in_stride
    out_w_stride = out_buf.shape[3]

    if be == EngineBackend.NATIVE:
        if _native_lib is None:
            be = _native_miss("conv2d_forward", "Native library not loaded")
        elif x.dtype != np.float32:
            be = _native_unsupported_dtype("conv2d_forward", x.dtype)
        else:
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
            if status == 0:
                return out_buf, None
            be = _native_miss("conv2d_forward", f"Native DLL returned error code {status}")

    out_h = (H + 2 * pad - k_h) // stride + 1
    out_w = (W_in + 2 * pad - k_w) // stride + 1
    total_rows = N * out_h * out_w

    x_logical = x[:, :, :, :W_in] if x.shape[3] != W_in else x

    if _use_im2col_gemm_fast_path(be):
        _ensure_fast_kernels()
        active_col = col_buf[:total_rows] if col_buf is not None else np.empty((total_rows, C_in * k_h * k_w), dtype=x.dtype)
        active_gemm = gemm_buf[:total_rows] if gemm_buf is not None else np.empty((total_rows, C_out), dtype=x.dtype)
        w_fwd = _pack_w_gemm_fwd(W, w_gemm_fwd_buf)

        if _native_im2col_gemm_available and x.dtype == np.float32:
            lib = _ensure_primitives_lib()
            x_for_conv = x_logical
            if not x_for_conv.flags["C_CONTIGUOUS"]:
                x_for_conv = np.ascontiguousarray(x_for_conv)
            x_w_stride = x_for_conv.shape[3]
            status = lib.conv2d_forward_im2col_gemm_avx2(
                _get_c_ptr(x_for_conv), _get_c_ptr(w_fwd), _get_c_ptr(bias), _get_c_ptr(out_buf),
                _get_c_ptr(active_col), _get_c_ptr(active_gemm),
                int(N), int(C_in), int(H), int(W_in), int(x_w_stride),
                int(C_out), int(k_h), int(k_w), int(stride), int(pad), int(out_w_stride),
                1 if fuse_relu else 0,
            )
            if status == 0:
                return out_buf, active_col
            logger.warning("conv2d_forward_im2col_gemm_avx2 failed (%s); falling back to Python GEMM", status)

        _im2col_impl(x_logical, k_h, k_w, stride, pad, out_buf=active_col)
        _gemm_forward_impl(active_col, w_fwd, active_gemm)
        
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
                          dcol_buf: np.ndarray = None, W_logical: int = None,
                          ctx: "EngineContext | None" = None,
                          backend: EngineBackend | None = None) -> tuple:
    _ensure_initialized()
    be = _resolve_backend(ctx, backend)
    N, C_in, H, W_in_stride = x.shape
    C_out, _, k_h, k_w = W.shape
    W_in = W_logical if W_logical is not None else W_in_stride
    conv_out_w_stride = dout.shape[3]

    if be == EngineBackend.NATIVE:
        if _native_lib is None:
            be = _native_miss("conv2d_backward_fused", "Native library not loaded")
        elif dout.dtype != np.float32:
            be = _native_unsupported_dtype("conv2d_backward_fused", dout.dtype)
        else:
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
            if status == 0:
                return dx_buf, dW_buf
            be = _native_miss("conv2d_backward_fused", f"Native DLL returned error code {status}")

    dx = conv2d_backward_input(
        dout, W, dx_buf, stride, pad, dout_trans=dout_trans, dcol_buf=dcol_buf,
        in_act=in_act, fuse_relu=fuse_relu, W_logical=W_logical, backend=be
    )
    dW = conv2d_backward_weight(
        dout, x, dW_buf, col, dout_trans, stride, pad, inv_m,
        W_logical=W_logical, backend=be
    )
    return dx, dW


def conv2d_backward_weight(dout: np.ndarray, x: np.ndarray, dW: np.ndarray,
                           col: np.ndarray, dout_trans: np.ndarray,
                           stride: int, pad: int, inv_m: float, W_logical: int = None,
                           ctx: "EngineContext | None" = None,
                           backend: EngineBackend | None = None) -> np.ndarray:
    _ensure_initialized()
    be = _resolve_backend(ctx, backend)
    if be == EngineBackend.NATIVE:
        if _native_lib is None:
            be = _native_miss("conv2d_backward_weight", "Native library not loaded")
        elif dout.dtype != np.float32:
            be = _native_unsupported_dtype("conv2d_backward_weight", dout.dtype)
        else:
            N, C_in, H, W_in_stride = x.shape
            C_out, _, k_h, k_w = dW.shape
            W_in = W_logical if W_logical is not None else W_in_stride
            conv_out_w_stride = dout.shape[3]
            status = _native_lib.direct_conv2d_backward_weight_avx2(
                _get_c_ptr(dout), _get_c_ptr(x), _get_c_ptr(dW),
                int(N), int(C_in), int(H), int(W_in), int(W_in_stride), int(C_out), int(k_h), int(k_w), int(stride), int(pad), int(conv_out_w_stride), ctypes.c_float(inv_m)
            )
            if status == 0:
                return dW
            be = _native_miss("conv2d_backward_weight", f"Native DLL returned error code {status}")

    N = dout.shape[0]
    C_out, C_in, k_h, k_w = dW.shape
    H = x.shape[2]
    W_in_stride = x.shape[3]
    W_in = W_logical if W_logical is not None else W_in_stride
    
    out_h = (H + 2 * pad - k_h) // stride + 1
    out_w = (W_in + 2 * pad - k_w) // stride + 1
    total_rows = N * out_h * out_w

    active_col = col[:total_rows] if col is not None else None
    if active_col is None:
        x_logical = x[:, :, :, :W_in] if x.shape[3] != W_in else x
        active_col = im2col(x_logical, k_h, k_w, stride, pad, ctx=ctx, backend=be)

    if _use_im2col_gemm_fast_path(be):
        _ensure_fast_kernels()

        if _native_im2col_gemm_available and dout.dtype == np.float32:
            lib = _ensure_primitives_lib()
            if dout_trans is not None:
                active_dout_trans = dout_trans[:total_rows]
            else:
                active_dout_trans = np.empty((total_rows, C_out), dtype=dout.dtype)
            status = lib.conv2d_backward_weight_im2col_gemm_avx2(
                _get_c_ptr(dout), _get_c_ptr(x), _get_c_ptr(dW),
                _get_c_ptr(active_col), _get_c_ptr(active_dout_trans),
                int(N), int(C_in), int(H), int(W_in), int(W_in_stride),
                int(C_out), int(k_h), int(k_w), int(stride), int(pad), int(dout.shape[3]),
                ctypes.c_float(inv_m),
                1 if col is not None else 0,
                1 if dout_trans is not None else 0,
            )
            if status == 0:
                return dW
            logger.warning("conv2d_backward_weight_im2col_gemm_avx2 failed (%s); falling back to Python GEMM", status)

        if dout_trans is not None:
            active_dout_trans = dout_trans[:total_rows]
        elif dout.shape[3] != out_w or dout.shape[2] != out_h:
            active_dout_trans = np.ascontiguousarray(
                dout[:, :, :out_h, :out_w].transpose(0, 2, 3, 1).reshape(total_rows, C_out)
            )
        else:
            active_dout_trans = np.ascontiguousarray(
                dout.transpose(0, 2, 3, 1).reshape(total_rows, C_out)
            )
        orig_shape = dW.shape
        dW_flat = np.empty((orig_shape[0], int(np.prod(orig_shape[1:]))), dtype=active_dout_trans.dtype)
        gemm_param_grad(active_dout_trans, active_col, dW_flat, inv_m)
        dW[...] = dW_flat.reshape(orig_shape).astype(dW.dtype)
        return dW

    if dout_trans is not None:
        active_dout_trans = dout_trans[:total_rows]
    elif dout.shape[3] != out_w or dout.shape[2] != out_h:
        active_dout_trans = np.ascontiguousarray(
            dout[:, :, :out_h, :out_w].transpose(0, 2, 3, 1).reshape(total_rows, C_out)
        )
    else:
        active_dout_trans = np.ascontiguousarray(
            dout.transpose(0, 2, 3, 1).reshape(total_rows, C_out)
        )

    if active_col is None:
        x_logical = x[:, :, :, :W_in] if x.shape[3] != W_in else x
        active_col = im2col(x_logical, k_h, k_w, stride, pad, ctx=ctx, backend=be)

    dW_flat = np.dot(active_dout_trans.T, active_col) * inv_m
    dW[...] = dW_flat.reshape(dW.shape).astype(dW.dtype)
    return dW


def conv2d_backward_input(dout: np.ndarray, W: np.ndarray, dx_buf: np.ndarray,
                          stride: int, pad: int,
                          dout_trans: np.ndarray = None, dcol_buf: np.ndarray = None,
                          in_act: np.ndarray = None, fuse_relu: bool = False, W_logical: int = None,
                          ctx: "EngineContext | None" = None,
                          backend: EngineBackend | None = None) -> np.ndarray:
    _ensure_initialized()
    be = _resolve_backend(ctx, backend)
    if be == EngineBackend.NATIVE:
        if _native_lib is None:
            be = _native_miss("conv2d_backward_input", "Native library not loaded")
        elif dout.dtype != np.float32:
            be = _native_unsupported_dtype("conv2d_backward_input", dout.dtype)
        else:
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
            if status == 0:
                return dx_buf
            be = _native_miss("conv2d_backward_input", f"Native DLL returned error code {status}")

    N = dout.shape[0]
    C_in = dx_buf.shape[1]
    H = dx_buf.shape[2]
    W_in = W_logical if W_logical is not None else dx_buf.shape[3]
    k_h, k_w = W.shape[2], W.shape[3]

    out_h = (H + 2 * pad - k_h) // stride + 1
    out_w = (W_in + 2 * pad - k_w) // stride + 1
    total_rows = N * out_h * out_w

    W_2d = W.reshape(W.shape[0], -1)

    active_dcol = dcol_buf[:total_rows] if dcol_buf is not None else np.empty((total_rows, W_2d.shape[1]), dtype=dout.dtype)
    if _use_im2col_gemm_fast_path(be):
        _ensure_fast_kernels()

        if _native_im2col_gemm_available and dout.dtype == np.float32:
            lib = _ensure_primitives_lib()
            if dout_trans is not None:
                active_dout_trans = dout_trans[:total_rows]
            else:
                active_dout_trans = np.empty((total_rows, W.shape[0]), dtype=dout.dtype)
            status = lib.conv2d_backward_input_im2col_gemm_avx2(
                _get_c_ptr(dout), _get_c_ptr(W),
                _get_c_ptr(in_act) if (fuse_relu and in_act is not None) else None,
                _get_c_ptr(dx_buf),
                _get_c_ptr(active_dout_trans), _get_c_ptr(active_dcol),
                int(N), int(C_in), int(H), int(W_in), int(dx_buf.shape[3]),
                int(W.shape[0]), int(k_h), int(k_w), int(stride), int(pad), int(dout.shape[3]),
                1 if (fuse_relu and in_act is not None) else 0,
                1 if dout_trans is not None else 0,
            )
            if status == 0:
                return dx_buf
            logger.warning("conv2d_backward_input_im2col_gemm_avx2 failed (%s); falling back to Python GEMM", status)

        if dout_trans is not None:
            active_dout_trans = dout_trans[:total_rows]
        else:
            dout_logical = dout[:, :, :out_h, :out_w]
            active_dout_trans = np.ascontiguousarray(
                dout_logical.transpose(0, 2, 3, 1).reshape(total_rows, W.shape[0])
            )
        _gemm_backward_input_impl(active_dout_trans, W_2d, active_dcol)
    else:
        if dout_trans is not None:
            active_dout_trans = dout_trans[:total_rows]
        else:
            dout_logical = dout[:, :, :out_h, :out_w]
            active_dout_trans = np.ascontiguousarray(
                dout_logical.transpose(0, 2, 3, 1).reshape(total_rows, W.shape[0])
            )
        np.dot(active_dout_trans, W_2d, out=active_dcol)

    logical_shape = (N, C_in, H, W_in)
    if skip_dx_zero():
        pass
    elif not _use_im2col_gemm_fast_path(be):
        dx_buf.fill(0.0)
    elif not step_dx_zero():
        dx_buf.fill(0.0)
    col2im(active_dcol, logical_shape, k_h, k_w, stride, pad, out_buf=dx_buf, backend=be)
    if fuse_relu and in_act is not None:
        _ensure_fast_kernels()
        _relu_bwd_inplace_kernel(dx_buf, in_act)
    return dx_buf


# -----------------------------------------------------------------------------
# Primitives, Pooling & Activations
# -----------------------------------------------------------------------------
def im2col(x: np.ndarray, k_h: int, k_w: int, stride: int = 1, pad: int = 0, out_buf: np.ndarray = None,
           ctx: "EngineContext | None" = None, backend: EngineBackend | None = None) -> np.ndarray:
    _ensure_initialized()
    if _use_im2col_gemm_fast_path(_resolve_backend(ctx, backend)):
        _ensure_fast_kernels()
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


def col2im(col: np.ndarray, input_shape: tuple, k_h: int, k_w: int, stride: int = 1, pad: int = 0, out_buf: np.ndarray = None,
           ctx: "EngineContext | None" = None, backend: EngineBackend | None = None) -> np.ndarray:
    _ensure_initialized()
    if _use_im2col_gemm_fast_path(_resolve_backend(ctx, backend)):
        _ensure_fast_kernels()
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

    result = img if pad == 0 else img[:, :, pad:-pad, pad:-pad]
    if out_buf is not None and out_buf.shape == result.shape and out_buf.dtype == col.dtype:
        out_buf[:] = result
        return out_buf
    return result


def maxpool_forward(x: np.ndarray, pool_size: int, stride: int, out_buf: np.ndarray = None, argmax_buf: np.ndarray = None,
                    ctx: "EngineContext | None" = None, backend: EngineBackend | None = None):
    _ensure_initialized()
    be = _resolve_backend(ctx, backend)
    N, C, H, W = x.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1

    if be == EngineBackend.NATIVE:
        if _native_lib is None:
            be = _native_miss("maxpool_forward", "Native library not loaded")
        elif x.dtype != np.float32:
            be = _native_unsupported_dtype("maxpool_forward", x.dtype)
        else:
            if out_buf is None or out_buf.shape != (N, C, out_h, out_w) or out_buf.dtype != x.dtype:
                out_buf = np.empty((N, C, out_h, out_w), dtype=x.dtype)
            if argmax_buf is None or argmax_buf.shape != (N, C, out_h, out_w) or argmax_buf.dtype != np.uint8:
                argmax_buf = np.empty((N, C, out_h, out_w), dtype=np.uint8)

            status = _native_lib.direct_maxpool_forward_avx2(
                _get_c_ptr(x), _get_c_ptr(out_buf), _get_c_ptr(argmax_buf),
                int(N), int(C), int(H), int(W), int(pool_size), int(stride)
            )
            if status == 0:
                return out_buf, argmax_buf
            be = _native_miss("maxpool_forward", f"Native DLL returned error code {status}")

    if _use_im2col_gemm_fast_path(be):
        _ensure_fast_kernels()
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


def maxpool_backward(dout: np.ndarray, cache: np.ndarray, x_shape: tuple, pool_size: int, stride: int, dx_buf: np.ndarray = None,
                     ctx: "EngineContext | None" = None, backend: EngineBackend | None = None):
    _ensure_initialized()
    be = _resolve_backend(ctx, backend)
    if dx_buf is None or dx_buf.shape != x_shape or dx_buf.dtype != dout.dtype:
        dx_buf = np.zeros(x_shape, dtype=dout.dtype)
    else:
        dx_buf.fill(0.0)

    N, C, in_h, in_w = x_shape
    out_h, out_w = dout.shape[2], dout.shape[3]

    if be == EngineBackend.NATIVE:
        if _native_lib is None:
            be = _native_miss("maxpool_backward", "Native library not loaded")
        elif dout.dtype != np.float32:
            be = _native_unsupported_dtype("maxpool_backward", dout.dtype)
        else:
            status = _native_lib.direct_maxpool_backward_avx2(
                _get_c_ptr(dout), _get_c_ptr(cache), _get_c_ptr(dx_buf),
                int(N), int(C), int(out_h), int(out_w), int(in_h), int(in_w), int(pool_size), int(stride)
            )
            if status == 0:
                return dx_buf
            be = _native_miss("maxpool_backward", f"Native DLL returned error code {status}")

    if _use_im2col_gemm_fast_path(be):
        _ensure_fast_kernels()
        _maxpool_backward_kernel(dout, cache, dx_buf)
        return dx_buf

    mask = cache
    dx_reshaped = mask * dout[:, :, :, None, :, None]
    dx_buf[:] = dx_reshaped.reshape(x_shape)
    return dx_buf


def fuse_dout_transpose_and_bias(dout: np.ndarray, dout_trans_buf: np.ndarray, db_buf: np.ndarray,
                               ctx: "EngineContext | None" = None,
                               backend: EngineBackend | None = None):
    _ensure_initialized()
    be = _resolve_backend(ctx, backend)
    if be == EngineBackend.NATIVE:
        if _native_lib is None:
            be = _native_miss("fuse_dout_transpose_and_bias", "Native library not loaded")
        elif dout.dtype != np.float32:
            be = _native_unsupported_dtype("fuse_dout_transpose_and_bias", dout.dtype)
        else:
            N, C_out, out_h, out_w = dout.shape
            inv_m = 1.0 / float(N)
            status = _native_lib.direct_bias_backward_avx2(
                _get_c_ptr(dout), _get_c_ptr(db_buf),
                int(N), int(C_out), int(out_h), int(out_w), ctypes.c_float(inv_m)
            )
            if status == 0:
                if dout_trans_buf is not None:
                    dout_trans_buf[:N * out_h * out_w] = np.transpose(dout, (0, 2, 3, 1)).reshape(-1, C_out)
                return
            be = _native_miss("fuse_dout_transpose_and_bias", f"Native DLL returned error code {status}")

    if _use_im2col_gemm_fast_path(be):
        _ensure_fast_kernels()
        _fuse_dout_impl(dout, dout_trans_buf, db_buf)
        return

    N, C_out, out_h, out_w = dout.shape
    inv_m = 1.0 / float(N)
    if dout_trans_buf is not None:
        dout_trans_buf[: N * out_h * out_w] = np.transpose(dout, (0, 2, 3, 1)).reshape(-1, C_out)
    np.sum(dout, axis=(0, 2, 3), out=db_buf[0])
    db_buf *= inv_m


def gemm_param_grad(dout_trans: np.ndarray, col: np.ndarray, dW_flat: np.ndarray, inv_m: float):
    _ensure_fast_kernels()
    _gemm_param_grad_impl(dout_trans, col, dW_flat, inv_m)


def gemm_backward_input(dout_trans: np.ndarray, W_2d: np.ndarray, dcol_out: np.ndarray):
    _ensure_fast_kernels()
    _gemm_backward_input_impl(dout_trans, W_2d, dcol_out)


def relu_spatial_forward(x: np.ndarray, ctx: "EngineContext | None" = None,
                         backend: EngineBackend | None = None) -> np.ndarray:
    _ensure_initialized()
    be = _resolve_backend(ctx, backend)
    if be == EngineBackend.NATIVE:
        if _native_lib is None:
            be = _native_miss("relu_spatial_forward", "Native library not loaded")
        elif x.dtype != np.float32:
            be = _native_unsupported_dtype("relu_spatial_forward", x.dtype)
        else:
            status = _native_lib.direct_relu_forward_avx2(_get_c_ptr(x), x.size)
            if status == 0:
                return x
            be = _native_miss("relu_spatial_forward", f"Native DLL returned error code {status}")
    if _use_im2col_gemm_fast_path(be):
        _ensure_fast_kernels()
        _relu_fwd_inplace_kernel(x)
        return x
    return np.maximum(0.0, x, out=x)


def relu_spatial_backward(dout: np.ndarray, in_act: np.ndarray, ctx: "EngineContext | None" = None,
                          backend: EngineBackend | None = None) -> np.ndarray:
    _ensure_initialized()
    be = _resolve_backend(ctx, backend)
    if be == EngineBackend.NATIVE:
        if _native_lib is None:
            be = _native_miss("relu_spatial_backward", "Native library not loaded")
        elif dout.dtype != np.float32:
            be = _native_unsupported_dtype("relu_spatial_backward", dout.dtype)
        else:
            status = _native_lib.direct_relu_backward_avx2(_get_c_ptr(dout), _get_c_ptr(in_act), dout.size)
            if status == 0:
                return dout
            be = _native_miss("relu_spatial_backward", f"Native DLL returned error code {status}")
    if _use_im2col_gemm_fast_path(be):
        _ensure_fast_kernels()
        _relu_bwd_inplace_kernel(dout, in_act)
        return dout
    dout *= (in_act > 0.0)
    return dout