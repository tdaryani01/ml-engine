# testing/test_runtime.py
"""Runtime threading policy: per-backend OMP vs BLAS split."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import EngineBackend
from utils.runtime import RuntimeSettings, training_threadpool


def _settings(num_threads: int = 4) -> RuntimeSettings:
    return RuntimeSettings(
        num_threads=num_threads,
        platform="windows",
        env={},
        blas_threads={"native": None, "numpy": None, "im2col_gemm": None},
    )


def test_conv_backends_shared_omp():
    s = _settings(4)
    for backend in (EngineBackend.NATIVE, EngineBackend.IM2COL_GEMM):
        assert s.omp_threads_for(backend) == 4
        assert s.blas_threads_for(backend) == 4
    print("[PASSED] native + im2col+gemm: shared omp=4")


def test_conv_backends_dll_omp_pinned():
    from utils.conv_dispatch import (
        sync_im2col_parallel_cap,
        sync_native_thread_policy,
        _load_conv_dll,
    )

    lib = _load_conv_dll()
    if lib is None or not hasattr(lib, "configure_native_threads"):
        print("[SKIPPED] dll_omp: rebuild conv_kernels.dll (configure_native_threads export missing)")
        return
    sync_im2col_parallel_cap(4)
    sync_native_thread_policy(4)
    assert lib.get_omp_threads() == 4, f"expected dll_omp=4, got {lib.get_omp_threads()}"
    print("[PASSED] DLL shared LLVM OMP pool: dll_omp=4")


def test_d9_unified_openblas_when_dll_present():
    """D9: bin/libopenblas.dll present => unified LLVM OMP (not scipy capsule)."""
    from pathlib import Path

    from utils.conv_dispatch import bootstrap_im2col_gemm_runtime, native_blas_unified_omp

    root = Path(__file__).resolve().parents[1]
    dll = root / "bin" / "libopenblas.dll"
    if not dll.exists():
        print("[SKIPPED] D9: bin/libopenblas.dll not present")
        return
    bootstrap_im2col_gemm_runtime()
    assert native_blas_unified_omp(), "bin/libopenblas.dll present but unified_omp=False"
    print("[PASSED] D9: unified OpenBLAS runtime active")


def test_conv_backends_numba_pinned_during_fit():
    import numba

    s = _settings(4)
    prev = numba.get_num_threads()
    expected = 1
    try:
        numba.set_num_threads(8)
        for backend in (EngineBackend.NATIVE, EngineBackend.IM2COL_GEMM):
            with training_threadpool(s, backend):
                assert numba.get_num_threads() == expected
        assert numba.get_num_threads() == 8
    finally:
        numba.set_num_threads(prev)
    print(f"[PASSED] native + im2col+gemm: numba pinned to {expected} during fit")


RUNTIME_TESTS = [
    test_conv_backends_shared_omp,
    test_conv_backends_dll_omp_pinned,
    test_d9_unified_openblas_when_dll_present,
    test_conv_backends_numba_pinned_during_fit,
]


if __name__ == "__main__":
    failed = []
    for fn in RUNTIME_TESTS:
        try:
            fn()
        except Exception as exc:
            failed.append((fn.__name__, exc))
            print(f"[FAILED] {fn.__name__}: {exc}")
    if failed:
        sys.exit(1)
    print(f"[SUCCESS] All {len(RUNTIME_TESTS)} runtime tests passed.")
