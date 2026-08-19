# src/spatial_layers.py
import numpy as np
from utils.im2col import im2col, col2im

class Conv2D:
    """
    2D Convolutional Layer using im2col vectorization.
    Stores weights with shape (out_channels, in_channels, k_h, k_w) and biases with shape (1, out_channels).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, pad: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_h = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.k_w = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
        self.stride = stride
        self.pad = pad

        # He / Kaiming Uniform Initialization
        fan_in = in_channels * self.k_h * self.k_w
        limit = np.sqrt(6.0 / fan_in)
        self.W = np.random.uniform(-limit, limit, (out_channels, in_channels, self.k_h, self.k_w))
        self.b = np.zeros((1, out_channels))

        # Gradient placeholders
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        # Forward cache
        self.x = None
        self.col = None
        self.out_h = 0
        self.out_w = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Input:  (N, C, H, W)
        Output: (N, out_channels, out_h, out_w)
        """
        self.x = x
        N, C, H, W = x.shape
        self.out_h = (H + 2 * self.pad - self.k_h) // self.stride + 1
        self.out_w = (W + 2 * self.pad - self.k_w) // self.stride + 1

        # Flatten patches into 2D matrix
        self.col = im2col(x, self.k_h, self.k_w, self.stride, self.pad)
        
        # Reshape weights: (out_channels, in_channels * k_h * k_w) -> transpose for dot product
        w_row = self.W.reshape(self.out_channels, -1).T

        # Dot product + bias broadcast
        out = np.dot(self.col, w_row) + self.b
        
        # Reshape back to 4D tensor: (N, out_channels, out_h, out_w)
        out = out.reshape(N, self.out_h, self.out_w, self.out_channels).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        dout: (N, out_channels, out_h, out_w)
        Returns: dx (N, in_channels, H, W)
        """
        m = self.x.shape[0]
        # Reshape incoming error: (N * out_h * out_w, out_channels)
        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        # Gradients normalized by batch size m
        self.db = np.sum(dout_reshaped, axis=0, keepdims=True) / m
        self.dW = (np.dot(self.col.T, dout_reshaped).T).reshape(self.W.shape) / m

        # Propagate error back through patches
        w_row = self.W.reshape(self.out_channels, -1).T
        dcol = np.dot(dout_reshaped, w_row.T)
        
        # Reconstruct 4D tensor gradient
        dx = col2im(dcol, self.x.shape, self.k_h, self.k_w, self.stride, self.pad)
        return dx


class MaxPool2D:
    """
    2D Max-Pooling Layer.
    Downsamples spatial dimensions by tracking local maximum indices during forward
    and routing gradients exclusively to winning indices during backward.
    """
    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = pool_size
        self.stride = stride
        self.x = None
        self.max_idx = None
        self.out_h = 0
        self.out_w = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Input:  (N, C, H, W)
        Output: (N, C, out_h, out_w)
        """
        self.x = x
        N, C, H, W = x.shape
        self.out_h = (H - self.pool_size) // self.stride + 1
        self.out_w = (W - self.pool_size) // self.stride + 1

        # Flatten into pooling windows
        col = im2col(x, self.pool_size, self.pool_size, self.stride, pad=0)
        col = col.reshape(-1, self.pool_size * self.pool_size)

        # Cache winning indices and compute max values
        self.max_idx = np.argmax(col, axis=1)
        out = np.max(col, axis=1)

        # Reshape to (N, C, out_h, out_w)
        out = out.reshape(N, self.out_h, self.out_w, C).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        dout: (N, C, out_h, out_w)
        Returns: dx (N, C, H, W)
        """
        N, C, H, W = self.x.shape
        dout_reshaped = dout.transpose(0, 2, 3, 1).ravel()

        # Route error signal strictly to winning coordinates
        dcol = np.zeros((dout_reshaped.size, self.pool_size * self.pool_size), dtype=dout.dtype)
        dcol[np.arange(self.max_idx.size), self.max_idx] = dout_reshaped

        dcol = dcol.reshape(-1, C * self.pool_size * self.pool_size)
        dx = col2im(dcol, self.x.shape, self.pool_size, self.pool_size, self.stride, pad=0)
        return dx


class Flatten:
    """
    Bridges 4D spatial tensors (N, C, H, W) to 2D feature matrices (N, C * H * W).
    """
    def __init__(self):
        self.orig_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self.orig_shape)