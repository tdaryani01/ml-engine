# testing/test_spatial_layers.py
import numpy as np
from src.spatial_layers import Conv2D, MaxPool2D, Flatten

def test_conv2d_forward_shape():
    layer = Conv2D(in_channels=3, out_channels=8, kernel_size=3, stride=1, pad=1)
    x = np.random.randn(4, 3, 16, 16)
    out = layer.forward(x)
    assert out.shape == (4, 8, 16, 16), f"Expected (4, 8, 16, 16), got {out.shape}"
    print("[PASSED] Conv2D forward output shape verified.")

def test_conv2d_numerical_gradient_check():
    """Finite difference numerical gradient check for Conv2D weights."""
    np.random.seed(42)
    layer = Conv2D(in_channels=2, out_channels=2, kernel_size=3, stride=1, pad=0)
    x = np.random.randn(2, 2, 6, 6)
    
    # Forward & Analytical Backward
    out = layer.forward(x)
    dout = np.random.randn(*out.shape)
    _ = layer.backward(dout)
    analytical_dW = layer.dW.copy()
    
    # Numerical Gradient Approximation on W
    numerical_dW = np.zeros_like(layer.W)
    h = 1e-5
    
    for idx, val in np.ndenumerate(layer.W):
        layer.W[idx] = val + h
        out_plus = layer.forward(x)
        cost_plus = np.sum(out_plus * dout) / x.shape[0]
        
        layer.W[idx] = val - h
        out_minus = layer.forward(x)
        cost_minus = np.sum(out_minus * dout) / x.shape[0]
        
        numerical_dW[idx] = (cost_plus - cost_minus) / (2 * h)
        layer.W[idx] = val
        
    rel_error = np.max(np.abs(analytical_dW - numerical_dW) / (np.abs(analytical_dW) + np.abs(numerical_dW) + 1e-15))
    assert rel_error < 1e-4, f"Conv2D gradient check failed with relative error {rel_error}"
    print(f"[PASSED] Conv2D weight numerical gradient verified (Rel Error: {rel_error:.2e}).")

def test_maxpool2d_forward_backward():
    pool = MaxPool2D(pool_size=2, stride=2)
    x = np.array([[[[1.0, 2.0],
                    [3.0, 4.0]]]])  # (1, 1, 2, 2)
    out = pool.forward(x)
    assert out.shape == (1, 1, 1, 1)
    assert out[0, 0, 0, 0] == 4.0
    
    dout = np.array([[[[10.0]]]])
    dx = pool.backward(dout)
    
    expected_dx = np.array([[[[0.0, 0.0],
                             [0.0, 10.0]]]])
    np.testing.assert_array_almost_equal(dx, expected_dx)
    print("[PASSED] MaxPool2D forward and selective backprop routing verified.")

def test_flatten_layer():
    flat = Flatten()
    x = np.random.randn(4, 3, 8, 8)
    out = flat.forward(x)
    assert out.shape == (4, 3 * 8 * 8)
    
    dout = np.random.randn(*out.shape)
    dx = flat.backward(dout)
    assert dx.shape == x.shape
    np.testing.assert_array_equal(dx, dout.reshape(x.shape))
    print("[PASSED] Flatten forward and backward tensor reshape verified.")

if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING SPATIAL LAYERS & CONV GRADIENT UNIT TESTS ")
    print("=" * 60)
    test_conv2d_forward_shape()
    test_conv2d_numerical_gradient_check()
    test_maxpool2d_forward_backward()
    test_flatten_layer()
    print("=" * 60)
    print("[SUCCESS] All spatial layers unit tests passed cleanly!")
    print("=" * 60)