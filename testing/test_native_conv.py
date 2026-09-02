# testing/test_native_conv.py
"""Native conv parity and dispatch regression tests."""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import EngineBackend
from utils.engine_ops import create_engine_context
from utils.conv_dispatch import (
    conv2d_forward,
    conv2d_backward_fused,
    init_engine_backend,
)

# Reuse native-vs-NumPy dX/dW matrix from gradient harness (run via test_gradient_check.py).
from testing.test_gradient_check import CONV_DX_KERNEL_PAD_CASES  # noqa: F401 — re-export for callers

RTOL = 1e-4
ATOL = 1e-4

FWD_KERNEL_PAD_CASES = [
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 1),
    (5, 1),
    (5, 2),
    (6, 1),
    (7, 1),
]


def _make_conv_tensors(k: int, pad: int, *, n: int = 2, c_in: int = 3, h: int = 14, w: int = 14, c_out: int = 8):
    rng = np.random.default_rng(42 + k * 11 + pad)
    x = rng.standard_normal((n, c_in, h, w), dtype=np.float32) * 0.1
    weight = rng.standard_normal((c_out, c_in, k, k), dtype=np.float32) * 0.1
    bias = rng.standard_normal((1, c_out), dtype=np.float32) * 0.01
    out_h = (h + 2 * pad - k) + 1
    out_w = (w + 2 * pad - k) + 1
    return x, weight, bias, out_h, out_w


def test_native_forward_matches_numpy_reference(k: int, pad: int) -> None:
    x, weight, bias, out_h, out_w = _make_conv_tensors(k, pad)
    out_ref = np.zeros((x.shape[0], weight.shape[0], out_h, out_w), dtype=np.float32)
    out_native = np.zeros_like(out_ref)

    numpy_ctx = create_engine_context(EngineBackend.NUMPY)
    native_ctx = create_engine_context(EngineBackend.NATIVE)
    conv2d_forward(
        x, weight, bias, stride=1, pad=pad, out_buf=out_ref, fuse_relu=True, ctx=numpy_ctx
    )
    conv2d_forward(
        x, weight, bias, stride=1, pad=pad, out_buf=out_native, fuse_relu=True, ctx=native_ctx
    )

    np.testing.assert_allclose(out_native, out_ref, rtol=RTOL, atol=ATOL)
    print(f"[PASSED] native forward matches numpy (k={k}, pad={pad})")


def test_native_dx_covered_by_gradient_matrix() -> None:
    """dX/dW parity (k=1..7, stride/pad matrix) lives in test_gradient_check.py."""
    assert len(CONV_DX_KERNEL_PAD_CASES) == 14
    print(
        "[PASSED] conv dX/dW native-vs-numpy matrix delegated to test_gradient_check.py "
        f"({len(CONV_DX_KERNEL_PAD_CASES)} stride-1 cases)"
    )


def test_native_conv_uses_generic_fallback_dispatch() -> None:
    """All native conv forward passes route through GENERIC_FALLBACK."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    code = """
import numpy as np
from config.constants import EngineBackend
from utils.conv_dispatch import init_engine_backend, conv2d_forward

init_engine_backend(EngineBackend.NATIVE)
x = np.zeros((1, 3, 14, 14), dtype=np.float32)
W = np.zeros((4, 3, 3, 3), dtype=np.float32)
b = np.zeros((1, 4), dtype=np.float32)
out = np.zeros((1, 4, 14, 14), dtype=np.float32)
conv2d_forward(x, W, b, stride=1, pad=1, out_buf=out, fuse_relu=False)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=project_root,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"dispatch probe failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    assert "GENERIC_FALLBACK" in result.stdout, (
        "Expected FWD -> GENERIC_FALLBACK in dispatch log, got:\n" + result.stdout
    )
    assert "3x3_SPECIALIZED" not in result.stdout
    print("[PASSED] native conv forward uses GENERIC_FALLBACK dispatch")


def test_conv_block_forward_padded_output_matches_numpy(k: int = 3) -> None:
    """Regression: SIMD-padded conv output must not use dense-layout specialists."""
    from utils.conv_dispatch import conv_block_forward

    rng = np.random.default_rng(17)
    n, cin, cout = 2, 8, 16
    h, w_log, w_stride = 14, 14, 14
    conv_out_w = (w_log + 2 - k) + 1
    conv_out_ws = ((conv_out_w + 7) // 8) * 8
    conv_out_h = (h + 2 - k) + 1
    pool_out_h = (conv_out_h - 2) // 2 + 1
    pool_out_w = (conv_out_w - 2) // 2 + 1

    x = rng.standard_normal((n, cin, h, w_stride), dtype=np.float32) * 0.1
    W = rng.standard_normal((cout, cin, k, k), dtype=np.float32) * 0.1
    b = rng.standard_normal((1, cout), dtype=np.float32) * 0.01
    out_conv_ref = np.zeros((n, cout, conv_out_h, conv_out_ws), dtype=np.float32)
    out_pool_ref = np.zeros((n, cout, pool_out_h, pool_out_w), dtype=np.float32)
    argmax_ref = np.zeros((n, cout, pool_out_h, pool_out_w), dtype=np.uint8)
    out_conv_native = np.zeros_like(out_conv_ref)
    out_pool_native = np.zeros_like(out_pool_ref)
    argmax_native = np.zeros_like(argmax_ref)

    init_engine_backend(EngineBackend.NUMPY)
    conv_block_forward(
        x, W, b, out_conv_ref, out_pool_ref, argmax_ref, W_logical=w_log
    )
    init_engine_backend(EngineBackend.NATIVE)
    conv_block_forward(
        x, W, b, out_conv_native, out_pool_native, argmax_native, W_logical=w_log
    )

    np.testing.assert_allclose(out_pool_native, out_pool_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        out_conv_native[:, :, :, :conv_out_w],
        out_conv_ref[:, :, :, :conv_out_w],
        rtol=RTOL,
        atol=ATOL,
    )
    print(f"[PASSED] conv_block padded output matches numpy (k={k})")


def test_fallback_conv_block_backward_padded_dw(k: int, pad: int, h: int = 28, w_log: int = 28) -> None:
    """Regression: generic fallback dW on SIMD-padded rows (28/32 geometry)."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    code = f"""
import os
import numpy as np
from config.constants import EngineBackend
from utils.conv_dispatch import init_engine_backend, conv_block_forward, conv_block_backward

k, pad, h, w_log = {k}, {pad}, {h}, {w_log}
rng = np.random.default_rng(31 + k)
n, cin, cout = 32, 3, 8
w_stride = ((w_log + 7) // 8) * 8
conv_out_w = (w_log + 2 * pad - k) + 1
conv_out_ws = ((conv_out_w + 7) // 8) * 8
conv_out_h = (h + 2 * pad - k) + 1
pool_out_h = (conv_out_h - 2) // 2 + 1
pool_out_w = (conv_out_w - 2) // 2 + 1

x = rng.standard_normal((n, cin, h, w_stride), dtype=np.float32) * 0.1
x[:, :, :, w_log:] = 0.0
W = rng.standard_normal((cout, cin, k, k), dtype=np.float32) * 0.1
b = rng.standard_normal((1, cout), dtype=np.float32) * 0.01
oc = np.zeros((n, cout, conv_out_h, conv_out_ws), dtype=np.float32)
op = np.zeros((n, cout, pool_out_h, pool_out_w), dtype=np.float32)
am = np.zeros((n, cout, pool_out_h, pool_out_w), dtype=np.uint8)

init_engine_backend(EngineBackend.NUMPY)
_, oc2, am2, col = conv_block_forward(x, W, b, oc, op, am, conv_pad=pad, W_logical=w_log)
dout = rng.standard_normal(op.shape, dtype=np.float32) * 0.1
dc2 = np.zeros_like(oc2)
dxr = np.zeros((n, cin, h, w_stride), dtype=np.float32)
dWr = np.zeros_like(W)
dbr = np.zeros_like(b)
dt = np.empty((n * conv_out_h * conv_out_ws, cout), dtype=np.float32)
dcol = np.empty_like(col)
conv_block_backward(
    dout, am2, x, W, oc2, dc2, dxr, dWr, dbr,
    conv_pad=pad, inv_m=1.0/n, col=col, dout_trans=dt, dcol_buf=dcol, W_logical=w_log,
)

os.environ["ML_ENGINE_FORCE_FALLBACK"] = "1"
init_engine_backend(EngineBackend.NATIVE)
_, ocn, amn, _ = conv_block_forward(x, W, b, oc, op, am, conv_pad=pad, W_logical=w_log)
dc = np.zeros_like(ocn)
dxn = np.zeros((n, cin, h, w_stride), dtype=np.float32)
dWn = np.zeros_like(W)
dbn = np.zeros_like(b)
conv_block_backward(dout, amn, x, W, ocn, dc, dxn, dWn, dbn, conv_pad=pad, inv_m=1.0/n, W_logical=w_log)
if not np.all(np.isfinite(dWn)):
    raise AssertionError(f"non-finite fallback dW k={{k}} pad={{pad}}")
np.testing.assert_allclose(dWn, dWr, rtol={RTOL}, atol={ATOL})
print("ok")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("ML_ENGINE_FORCE_FALLBACK", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=project_root,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"fallback dW k={k} pad={pad} failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    print(f"[PASSED] fallback conv_block dW matches numpy (k={k}, pad={pad}, {w_log}/{((w_log + 7) // 8) * 8})")


if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING NATIVE CONV REGRESSION TESTS ")
    print("=" * 60)
    for kernel, pad in FWD_KERNEL_PAD_CASES:
        test_native_forward_matches_numpy_reference(kernel, pad)
    test_native_dx_covered_by_gradient_matrix()
    test_native_conv_uses_generic_fallback_dispatch()
    test_conv_block_forward_padded_output_matches_numpy(3)
    for kernel, pad in ((3, 1), (5, 1), (5, 2), (7, 1)):
        test_fallback_conv_block_backward_padded_dw(kernel, pad)
    print("=" * 60)
    print("[SUCCESS] All native conv regression tests passed.")
    print("=" * 60)
