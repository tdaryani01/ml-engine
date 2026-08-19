# utils/im2col.py
import numpy as np

def im2col(x: np.ndarray, k_h: int, k_w: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """
    Extracts sliding local spatial blocks from a 4D input tensor (N, C, H, W)
    into a 2D matrix so convolution can be executed via a single GEMM dot product.
    """
    N, C, H, W = x.shape
    out_h = (H + 2 * pad - k_h) // stride + 1
    out_w = (W + 2 * pad - k_w) // stride + 1

    # Apply zero padding to spatial dimensions (Height, Width) if specified
    img = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')
    col = np.zeros((N, C, k_h, k_w, out_h, out_w), dtype=x.dtype)

    for y in range(k_h):
        y_max = y + stride * out_h
        for x_idx in range(k_w):
            x_max = x_idx + stride * out_w
            col[:, :, y, x_idx, :, :] = img[:, :, y:y_max:stride, x_idx:x_max:stride]

    # Transpose and flatten into 2D table: (N * out_h * out_w, C * k_h * k_w)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col: np.ndarray, input_shape: tuple, k_h: int, k_w: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """
    Folds the 2D gradient matrix back into the original 4D input gradient shape (N, C, H, W).
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - k_h) // stride + 1
    out_w = (W + 2 * pad - k_w) // stride + 1

    col = col.reshape(N, out_h, out_w, C, k_h, k_w).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=col.dtype)

    for y in range(k_h):
        y_max = y + stride * out_h
        for x_idx in range(k_w):
            x_max = x_idx + stride * out_w
            img[:, :, y:y_max:stride, x_idx:x_max:stride] += col[:, :, y, x_idx, :, :]

    if pad == 0:
        return img
    return img[:, :, pad:-pad, pad:-pad]