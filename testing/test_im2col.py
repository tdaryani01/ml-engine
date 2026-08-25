# testing/test_im2col.py
import sys
import os
import numpy as np

# Ensure project root is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import EngineBackend
from utils.im2col import im2col, col2im, init_engine_backend


def test_im2col_output_shape(backend: EngineBackend):
    """Verify output 2D matrix shape from 4D input tensor."""
    init_engine_backend(backend)
    N, C, H, W = 2, 3, 6, 6
    k_h, k_w = 3, 3
    stride, pad = 1, 0
    x = np.random.randn(N, C, H, W).astype(np.float32)
    
    col = im2col(x, k_h, k_w, stride=stride, pad=pad)
    
    out_h = (H + 2 * pad - k_h) // stride + 1  # 4
    out_w = (W + 2 * pad - k_w) // stride + 1  # 4
    expected_rows = N * out_h * out_w          # 32
    expected_cols = C * k_h * k_w              # 27
    
    assert col.shape == (expected_rows, expected_cols), f"[{backend.value}] Expected {(expected_rows, expected_cols)}, got {col.shape}"
    print(f"[PASSED] [{backend.value}] im2col output shape verified.")


def test_im2col_col2im_roundtrip_gradient(backend: EngineBackend):
    """Verify col2im correctly accumulates gradients across sliding window patches."""
    init_engine_backend(backend)
    N, C, H, W = 1, 1, 4, 4
    k_h, k_w = 2, 2
    stride = 1
    pad = 0
    
    x = np.ones((N, C, H, W), dtype=np.float32)
    col = im2col(x, k_h, k_w, stride=stride, pad=pad)
    dcol = np.ones_like(col, dtype=np.float32)
    
    dx = col2im(dcol, x.shape, k_h, k_w, stride=stride, pad=pad)
    
    assert dx[0, 0, 0, 0] == 1.0, f"[{backend.value}] Expected 1.0, got {dx[0, 0, 0, 0]}"
    assert dx[0, 0, 1, 1] == 4.0, f"[{backend.value}] Expected 4.0, got {dx[0, 0, 1, 1]}"
    assert dx.shape == (N, C, H, W), f"[{backend.value}] Expected {(N, C, H, W)}, got {dx.shape}"
    print(f"[PASSED] [{backend.value}] im2col/col2im roundtrip gradient accumulation verified.")


if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING IM2COL / COL2IM OPS UNIT TESTS ")
    print("=" * 60)
    for backend in [EngineBackend.NATIVE, EngineBackend.IM2COL_GEMM, EngineBackend.NUMPY]:
        print(f"\n--- Backend: {backend.value} ---")
        test_im2col_output_shape(backend)
        test_im2col_col2im_roundtrip_gradient(backend)
    print("\n" + "=" * 60)
    print("[SUCCESS] All im2col ops tests passed cleanly across all backends!")
    print("=" * 60)