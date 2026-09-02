# testing/test_im2col_gemm.py
"""Phase D gates: thread-safe GEMM, im2col+GEMM backend parity, strict dispatch."""
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import EngineBackend
from utils.engine_ops import Im2colGemmConvOps, create_engine_context
from utils.conv_dispatch import conv2d_forward, init_engine_backend
from utils.im2col_fast import gemm_forward_fast, gemm_param_grad_fast, gemm_backward_input_fast


def _gemm_once(seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    col = rng.standard_normal((64, 27), dtype=np.float32)
    w = rng.standard_normal((16, 27), dtype=np.float32)
    w_fwd = np.ascontiguousarray(w.T)
    out = np.empty((64, 16), dtype=np.float32)
    d_w = np.empty((16, 27), dtype=np.float32)
    dout = rng.standard_normal((64, 16), dtype=np.float32)
    for _ in range(40):
        gemm_forward_fast(col, w_fwd, out)
        gemm_param_grad_fast(dout, col, d_w, 0.1)
    return float(out.sum()), float(d_w.sum())


def test_d1_gemm_backward_input_matches_numpy():
    """D1: input-gradient GEMM matches np.dot reference."""
    rng = np.random.default_rng(13)
    m_dim, n_dim, k_dim = 64, 27, 16
    dout = rng.standard_normal((m_dim, n_dim), dtype=np.float32)
    w = rng.standard_normal((n_dim, k_dim), dtype=np.float32)
    ref = np.dot(dout, w)
    out = np.empty((m_dim, k_dim), dtype=np.float32)
    gemm_backward_input_fast(dout, w, out)
    assert np.allclose(out, ref, rtol=1e-5, atol=1e-6), f"max diff {np.max(np.abs(out - ref)):.2e}"
    print("[PASSED] D1: backward-input GEMM matches NumPy reference")


def test_d1_gemm_param_grad_matches_numpy():
    """D1: param-grad GEMM matches np.dot reference."""
    rng = np.random.default_rng(11)
    m_dim, k_dim, n_dim = 64, 27, 16
    col = rng.standard_normal((m_dim, k_dim), dtype=np.float32)
    dout = rng.standard_normal((m_dim, n_dim), dtype=np.float32)
    inv_m = 0.1
    ref = np.dot(dout.T, col) * inv_m
    out = np.empty((n_dim, k_dim), dtype=np.float32)
    gemm_param_grad_fast(dout, col, out, inv_m)
    assert np.allclose(out, ref, rtol=1e-5, atol=1e-6), f"max diff {np.max(np.abs(out - ref)):.2e}"
    print("[PASSED] D1: param-grad GEMM matches NumPy reference")


def test_d_gemm_thread_safe():
    """D1: concurrent GEMM calls with stack-local ctypes scalars."""
    ref = _gemm_once(0)
    results: list[tuple[float, float] | None] = [None] * 8
    errors: list[Exception] = []

    def worker(idx: int) -> None:
        try:
            results[idx] = _gemm_once(0)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"GEMM threads raised: {errors}"
    for i, val in enumerate(results):
        assert val is not None, f"thread {i} produced no result"
        assert np.allclose(val, ref, rtol=1e-4, atol=1e-3), f"thread {i} mismatch vs single-thread"
    print("[PASSED] D1: thread-safe GEMM smoke test")


def test_d2_im2col_gemm_explicit_backend():
    """D2: IM2COL_GEMM is a peer backend with no native handle."""
    ctx = create_engine_context(EngineBackend.IM2COL_GEMM)
    assert isinstance(ctx.conv, Im2colGemmConvOps)
    assert ctx.native_lib is None
    assert ctx.conv.backend == EngineBackend.IM2COL_GEMM

    x, w, b, out = _tiny_fixture()
    out_buf = np.zeros_like(out)
    ctx.conv.conv2d_forward(x=x, W=w, bias=b, stride=1, pad=1, out_buf=out_buf)
    assert np.isfinite(out_buf).all()
    print("[PASSED] D2: Im2colGemmConvOps explicit peer backend")


def test_d2_im2col_gemm_matches_numpy_reference():
    """D2/D3: im2col+GEMM forward matches NumPy reference dispatch."""
    x, w, b, out = _tiny_fixture()
    gemm_ctx = create_engine_context(EngineBackend.IM2COL_GEMM)
    numpy_ctx = create_engine_context(EngineBackend.NUMPY)

    out_gemm = np.zeros_like(out)
    out_numpy = np.zeros_like(out)
    gemm_ctx.conv.conv2d_forward(x=x, W=w, bias=b, stride=1, pad=1, out_buf=out_gemm)
    numpy_ctx.conv.conv2d_forward(x=x, W=w, bias=b, stride=1, pad=1, out_buf=out_numpy)
    assert np.allclose(out_gemm, out_numpy, rtol=1e-4, atol=1e-4)
    print("[PASSED] D2/D3: im2col+GEMM matches NumPy reference forward")


def test_d_three_backend_forward_parity():
    """D3: native vs im2col+GEMM vs numpy on one geometry (when native available)."""
    x, w, b, out = _tiny_fixture()
    numpy_ctx = create_engine_context(EngineBackend.NUMPY)
    gemm_ctx = create_engine_context(EngineBackend.IM2COL_GEMM)

    out_numpy = np.zeros_like(out)
    out_gemm = np.zeros_like(out)
    numpy_ctx.conv.conv2d_forward(x=x, W=w, bias=b, stride=1, pad=1, out_buf=out_numpy)
    gemm_ctx.conv.conv2d_forward(x=x, W=w, bias=b, stride=1, pad=1, out_buf=out_gemm)
    assert np.allclose(out_gemm, out_numpy, rtol=1e-4, atol=1e-4)

    try:
        native_ctx = create_engine_context(EngineBackend.NATIVE)
        out_native = np.zeros_like(out)
        native_ctx.conv.conv2d_forward(x=x, W=w, bias=b, stride=1, pad=1, out_buf=out_native)
        diff = float(np.max(np.abs(out_native - out_numpy)))
        assert diff < 1e-3, f"native vs numpy forward max diff {diff:.2e}"
        print(f"[PASSED] D3: three-backend forward parity (native max diff {diff:.2e})")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[SKIPPED] D3 native forward parity: {exc}")


def test_d8_native_im2col_gemm_k6_fwd_bwd():
    """D8: C++ im2col+OpenBLAS SGEMM fwd/bwd vs NumPy oracle (k6/s1/p1, tiny)."""
    from utils.conv_dispatch import (
        bootstrap_im2col_gemm_runtime,
        conv2d_backward_fused,
        conv2d_forward,
        init_engine_backend,
        native_im2col_gemm_available,
    )

    init_engine_backend(EngineBackend.IM2COL_GEMM)
    bootstrap_im2col_gemm_runtime()
    if not native_im2col_gemm_available():
        print("[SKIPPED] D8: native im2col+GEMM unavailable (build_native + bin/libopenblas.dll)")
        return

    rng = np.random.default_rng(42)
    n, c_in, h, w = 2, 3, 8, 8
    c_out, k, stride, pad = 8, 6, 1, 1
    x = (rng.standard_normal((n, c_in, h, w), dtype=np.float32) * 0.1).copy()
    wf = (rng.standard_normal((c_out, c_in, k, k), dtype=np.float32) * 0.1).copy()
    bias = np.zeros((1, c_out), dtype=np.float32)
    out_h = (h + 2 * pad - k) // stride + 1
    out_w = (w + 2 * pad - k) // stride + 1

    out_gemm = np.zeros((n, c_out, out_h, out_w), dtype=np.float32)
    out_ref = np.zeros_like(out_gemm)
    _, col_g = conv2d_forward(
        x, wf, bias, stride, pad, out_buf=out_gemm, backend=EngineBackend.IM2COL_GEMM,
    )
    _, col_r = conv2d_forward(
        x, wf, bias, stride, pad, out_buf=out_ref, backend=EngineBackend.NUMPY,
    )
    assert np.allclose(out_gemm, out_ref, rtol=1e-4, atol=1e-4)

    dout = rng.standard_normal(out_gemm.shape, dtype=np.float32)
    dx_g = np.zeros_like(x)
    dx_r = np.zeros_like(x)
    dw_g = np.zeros_like(wf)
    dw_r = np.zeros_like(wf)
    inv_m = 1.0 / float(n)
    conv2d_backward_fused(
        dout, x, wf, dx_g, dw_g, stride, pad, inv_m,
        col=col_g, backend=EngineBackend.IM2COL_GEMM,
    )
    conv2d_backward_fused(
        dout, x, wf, dx_r, dw_r, stride, pad, inv_m,
        col=col_r, backend=EngineBackend.NUMPY,
    )
    assert np.allclose(dx_g, dx_r, rtol=1e-4, atol=1e-4)
    assert np.allclose(dw_g, dw_r, rtol=1e-4, atol=1e-4)
    print("[PASSED] D8: native im2col+GEMM k6/s1/p1 fwd+bwd vs NumPy")


def _tiny_fixture():
    rng = np.random.default_rng(7)
    h, w = 8, 8
    x = (rng.standard_normal((2, 2, h, w), dtype=np.float32) * 0.1).copy()
    weight = (rng.standard_normal((3, 2, 3, 3), dtype=np.float32) * 0.1).copy()
    bias = np.zeros((1, 3), dtype=np.float32)
    out_h = (h + 2 - 3) // 1 + 1
    out_w = (w + 2 - 3) // 1 + 1
    out = np.zeros((2, 3, out_h, out_w), dtype=np.float32)
    return x, weight, bias, out


PHASE_D_TESTS = [
    test_d1_gemm_param_grad_matches_numpy,
    test_d1_gemm_backward_input_matches_numpy,
    test_d_gemm_thread_safe,
    test_d2_im2col_gemm_explicit_backend,
    test_d2_im2col_gemm_matches_numpy_reference,
    test_d_three_backend_forward_parity,
    test_d8_native_im2col_gemm_k6_fwd_bwd,
]


if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING PHASE D IM2COL+GEMM TESTS ")
    print("=" * 60)
    failed = []
    for fn in PHASE_D_TESTS:
        try:
            fn()
        except Exception as exc:
            failed.append((fn.__name__, exc))
            print(f"[FAILED] {fn.__name__}: {exc}")
    print("=" * 60)
    if failed:
        print(f"[FAILURE] {len(failed)} test(s) failed.")
        sys.exit(1)
    print(f"[SUCCESS] All {len(PHASE_D_TESTS)} Phase D tests passed.")
    print("=" * 60)
