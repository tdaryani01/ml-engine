# testing/test_native_conv.py
"""Native conv parity and dispatch regression tests."""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import EngineBackend
from utils.im2col import (
    conv2d_forward,
    conv2d_backward_fused,
    init_engine_backend,
)

# Reuse finite-difference dX checker from gradient harness.
from testing.test_gradient_check import check_conv2d_input_gradient, CONV_DX_KERNEL_PAD_CASES

RTOL = 1e-4
ATOL = 1e-4

FWD_KERNEL_PAD_CASES = [
    (1, 1),
    (3, 1),
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

    init_engine_backend(EngineBackend.NUMPY)
    conv2d_forward(x, weight, bias, stride=1, pad=pad, out_buf=out_ref, fuse_relu=True)

    init_engine_backend(EngineBackend.NATIVE)
    conv2d_forward(x, weight, bias, stride=1, pad=pad, out_buf=out_native, fuse_relu=True)

    np.testing.assert_allclose(out_native, out_ref, rtol=RTOL, atol=ATOL)
    print(f"[PASSED] native forward matches numpy (k={k}, pad={pad})")


def test_native_dx_finite_difference(k: int, pad: int) -> None:
    assert check_conv2d_input_gradient(EngineBackend.NATIVE, k, pad, fuse_relu=False)


def test_3x3_forward_uses_specialized_dispatch() -> None:
    """Regression: k=3 pad=1 forward should route to 3x3_SPECIALIZED (not generic-only)."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    code = """
import numpy as np
from config.constants import EngineBackend
from utils.im2col import init_engine_backend, conv2d_forward

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
            f"3x3 dispatch probe failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    assert "3x3_SPECIALIZED" in result.stdout, (
        "Expected FWD -> 3x3_SPECIALIZED in dispatch log, got:\n" + result.stdout
    )
    print("[PASSED] 3x3 forward dispatch uses 3x3_SPECIALIZED")


if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING NATIVE CONV REGRESSION TESTS ")
    print("=" * 60)
    for kernel, pad in FWD_KERNEL_PAD_CASES:
        test_native_forward_matches_numpy_reference(kernel, pad)
    for kernel, pad in CONV_DX_KERNEL_PAD_CASES:
        test_native_dx_finite_difference(kernel, pad)
    test_3x3_forward_uses_specialized_dispatch()
    print("=" * 60)
    print("[SUCCESS] All native conv regression tests passed.")
    print("=" * 60)
