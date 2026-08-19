# testing/test_im2col.py
import numpy as np
from utils.im2col import im2col, col2im

def test_im2col_output_shape():
    """Verify output 2D matrix shape from 4D input tensor."""
    N, C, H, W = 2, 3, 6, 6
    k_h, k_w = 3, 3
    stride, pad = 1, 0
    x = np.random.randn(N, C, H, W)
    
    col = im2col(x, k_h, k_w, stride=stride, pad=pad)
    
    out_h = (H + 2 * pad - k_h) // stride + 1  # 4
    out_w = (W + 2 * pad - k_w) // stride + 1  # 4
    expected_rows = N * out_h * out_w          # 32
    expected_cols = C * k_h * k_w              # 27
    
    assert col.shape == (expected_rows, expected_cols), f"Expected {(expected_rows, expected_cols)}, got {col.shape}"
    print("[PASSED] im2col output shape verified.")

def test_im2col_col2im_roundtrip_gradient():
    """Verify col2im correctly accumulates gradients across sliding window patches."""
    N, C, H, W = 1, 1, 4, 4
    k_h, k_w = 2, 2
    stride = 1
    pad = 0
    
    x = np.ones((N, C, H, W))
    col = im2col(x, k_h, k_w, stride=stride, pad=pad)
    dcol = np.ones_like(col)
    
    dx = col2im(dcol, x.shape, k_h, k_w, stride=stride, pad=pad)
    
    assert dx[0, 0, 0, 0] == 1.0, f"Expected 1.0, got {dx[0, 0, 0, 0]}"
    assert dx[0, 0, 1, 1] == 4.0, f"Expected 4.0, got {dx[0, 0, 1, 1]}"
    assert dx.shape == (N, C, H, W), f"Expected {(N, C, H, W)}, got {dx.shape}"
    print("[PASSED] im2col/col2im roundtrip gradient accumulation verified.")

if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING IM2COL / COL2IM OPS UNIT TESTS ")
    print("=" * 60)
    test_im2col_output_shape()
    test_im2col_col2im_roundtrip_gradient()
    print("=" * 60)
    print("[SUCCESS] All im2col ops tests passed cleanly!")
    print("=" * 60)