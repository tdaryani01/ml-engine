# src/spatial_layers.py
import builtins
import numpy as np
from config.constants import EngineBackend
from utils.im2col import (
    init_engine_backend,
    conv2d_forward,
    conv2d_backward_fused,
    conv_block_forward,
    conv_block_backward,
    col2im,
    maxpool_forward,
    maxpool_backward,
    fuse_dout_transpose_and_bias
)

if 'profile' not in builtins.__dict__:
    builtins.__dict__['profile'] = lambda x: x


class ConvBlock:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 conv_stride: int = 1, conv_pad: int = 0,
                 pool_size: int = 2, pool_stride: int = 2,
                 backend: EngineBackend = EngineBackend.NATIVE):
        init_engine_backend(backend)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_h = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.k_w = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
        self.conv_stride = conv_stride
        self.conv_pad = conv_pad
        self.pool_size = pool_size
        self.pool_stride = pool_stride

        fan_in = in_channels * self.k_h * self.k_w
        limit = np.sqrt(6.0 / fan_in)
        self.W = np.random.uniform(-limit, limit, (out_channels, in_channels, self.k_h, self.k_w)).astype(np.float32)
        self.b = np.zeros((1, out_channels), dtype=np.float32)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.x_cached = None
        self.conv_act_cached = None
        self.argmax_cached = None
        self.col = None

        self._cached_batch_size = 0
        self._cached_dtype = None
        self._out_conv_buffer = None
        self._out_pool_buffer = None
        self._argmax_buffer = None
        self._d_conv_buffer = None
        self._dx_buffer = None
        self._dout_trans_buffer = None
        self._col_buffer = None
        self._dcol_buffer = None
        self._fwd_gemm_buffer = None

    def _ensure_buffers(self, N: int, C: int, H: int, W: int, dtype):
        conv_out_h = (H + 2 * self.conv_pad - self.k_h) // self.conv_stride + 1
        conv_out_w = (W + 2 * self.conv_pad - self.k_w) // self.conv_stride + 1
        pool_out_h = (conv_out_h - self.pool_size) // self.pool_stride + 1
        pool_out_w = (conv_out_w - self.pool_size) // self.pool_stride + 1

        if self.W.dtype != dtype:
            self.W = self.W.astype(dtype)
            self.b = self.b.astype(dtype)
            self.dW = np.zeros_like(self.W)
            self.db = np.zeros_like(self.b)

        if self._cached_batch_size == N and self._cached_dtype == dtype and self._dx_buffer is not None:
            return

        self._cached_batch_size = N
        self._cached_dtype = dtype
        total_rows = N * conv_out_h * conv_out_w
        total_cols = C * self.k_h * self.k_w

        self._out_conv_buffer = np.empty((N, self.out_channels, conv_out_h, conv_out_w), dtype=dtype)
        self._out_pool_buffer = np.empty((N, self.out_channels, pool_out_h, pool_out_w), dtype=dtype)
        self._argmax_buffer   = np.empty((N, self.out_channels, pool_out_h, pool_out_w), dtype=np.uint8)
        self._d_conv_buffer   = np.empty((N, self.out_channels, conv_out_h, conv_out_w), dtype=dtype)
        self._dx_buffer       = np.zeros((N, C, H, W), dtype=dtype)
        self._dout_trans_buffer = np.empty((total_rows, self.out_channels), dtype=dtype)
        self._col_buffer      = np.empty((total_rows, total_cols), dtype=dtype)
        self._dcol_buffer     = np.empty((total_rows, total_cols), dtype=dtype)
        self._fwd_gemm_buffer = np.empty((total_rows, self.out_channels), dtype=dtype)

    @profile
    def forward(self, x: np.ndarray) -> np.ndarray:
        if not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x)

        N, C, H, W = x.shape
        self._ensure_buffers(N, C, H, W, x.dtype)
        self.x_cached = x

        out_pool, out_conv, argmax, self.col = conv_block_forward(
            x=x, W=self.W, bias=self.b,
            out_conv_buf=self._out_conv_buffer[:N],
            out_pool_buf=self._out_pool_buffer[:N],
            argmax_buf=self._argmax_buffer[:N],
            conv_stride=self.conv_stride, conv_pad=self.conv_pad,
            pool_size=self.pool_size, pool_stride=self.pool_stride,
            col_buf=self._col_buffer, gemm_buf=self._fwd_gemm_buffer
        )

        self.conv_act_cached = out_conv
        self.argmax_cached = argmax
        return out_pool

    @profile
    def backward(self, dout: np.ndarray) -> np.ndarray:
        if not dout.flags['C_CONTIGUOUS']:
            dout = np.ascontiguousarray(dout)

        N, C, H, W = self.x_cached.shape
        inv_m = 1.0 / float(N)
        self._ensure_buffers(N, C, H, W, dout.dtype)

        dx, self.dW, self.db = conv_block_backward(
            dout_pool=dout,
            argmax_buf=self.argmax_cached,
            x=self.x_cached,
            W=self.W,
            conv_act=self.conv_act_cached,
            d_conv_buf=self._d_conv_buffer[:N],
            dx_buf=self._dx_buffer[:N],
            dW_buf=self.dW,
            db_buf=self.db,
            conv_stride=self.conv_stride,
            conv_pad=self.conv_pad,
            pool_size=self.pool_size,
            pool_stride=self.pool_stride,
            inv_m=inv_m,
            col=self.col,
            dout_trans=self._dout_trans_buffer,
            dcol_buf=self._dcol_buffer
        )
        return dx


class Conv2D:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, pad: int = 0,
                 backend: EngineBackend = EngineBackend.NATIVE):
        init_engine_backend(backend)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_h = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.k_w = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
        self.stride = stride
        self.pad = pad

        fan_in = in_channels * self.k_h * self.k_w
        limit = np.sqrt(6.0 / fan_in)
        self.W = np.random.uniform(-limit, limit, (out_channels, in_channels, self.k_h, self.k_w)).astype(np.float32)
        self.b = np.zeros((1, out_channels), dtype=np.float32)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.x_shape = None
        self.x_cached = None
        self.col = None
        self.out_h = 0
        self.out_w = 0

        self._col_cap = 0
        self._col_buffer = None
        self._dx_buffer = None
        self._dout_trans_buffer = None
        self._dcol_buffer = None
        self._fwd_gemm_buffer = None
        self._fwd_out_buffer = None
        self._cached_dtype = None

    def _ensure_buffers(self, N: int, C: int, H: int, W: int, out_h: int, out_w: int, dtype):
        total_rows = N * out_h * out_w

        if self.W.dtype != dtype:
            self.W = self.W.astype(dtype)
            self.b = self.b.astype(dtype)
            self.dW = np.zeros_like(self.W, dtype=dtype)
            self.db = np.zeros_like(self.b, dtype=dtype)

        if self.db is None or self.db.dtype != dtype:
            self.db = np.zeros_like(self.b, dtype=dtype)
        if self.dW is None or self.dW.dtype != dtype:
            self.dW = np.zeros_like(self.W, dtype=dtype)

        if self._cached_dtype == dtype and total_rows <= self._col_cap and self._dx_buffer is not None:
            return

        total_cols = C * self.k_h * self.k_w
        self._col_cap = total_rows
        self._cached_dtype = dtype
        self._col_buffer = np.empty((total_rows, total_cols), dtype=dtype)
        self._dcol_buffer = np.empty((total_rows, total_cols), dtype=dtype)
        self._dout_trans_buffer = np.empty((total_rows, self.out_channels), dtype=dtype)
        self._fwd_gemm_buffer = np.empty((total_rows, self.out_channels), dtype=dtype)
        self._fwd_out_buffer = np.empty((N, self.out_channels, out_h, out_w), dtype=dtype)
        self._dx_buffer = np.zeros((N, C, H, W), dtype=dtype)

    @profile
    def forward(self, x: np.ndarray) -> np.ndarray:
        if not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x)

        self.x_shape = x.shape
        self.x_cached = x
        N, C, H, W = self.x_shape
        self.out_h = (H + 2 * self.pad - self.k_h) // self.stride + 1
        self.out_w = (W + 2 * self.pad - self.k_w) // self.stride + 1

        self._ensure_buffers(N, C, H, W, self.out_h, self.out_w, x.dtype)
        active_out = self._fwd_out_buffer[:N]

        active_out, self.col = conv2d_forward(
            x=x, W=self.W, bias=self.b,
            stride=self.stride, pad=self.pad,
            out_buf=active_out,
            col_buf=self._col_buffer,
            gemm_buf=self._fwd_gemm_buffer,
            fuse_relu=False
        )
        return active_out

    @profile
    def backward(self, dout: np.ndarray, in_act: np.ndarray = None, fuse_relu: bool = False) -> np.ndarray:
        if not dout.flags['C_CONTIGUOUS']:
            dout = np.ascontiguousarray(dout)

        N, C, H, W = self.x_shape
        inv_m = 1.0 / float(N)
        total_rows = N * self.out_h * self.out_w

        self._ensure_buffers(N, C, H, W, self.out_h, self.out_w, dout.dtype)

        active_dout_trans = self._dout_trans_buffer[:total_rows]

        fuse_dout_transpose_and_bias(dout, active_dout_trans, self.db)

        active_dx = self._dx_buffer[:N]
        dx, self.dW = conv2d_backward_fused(
            dout=dout,
            x=self.x_cached,
            W=self.W,
            dx_buf=active_dx,
            dW_buf=self.dW,
            stride=self.stride,
            pad=self.pad,
            inv_m=inv_m,
            in_act=in_act,
            fuse_relu=fuse_relu,
            col=self.col,
            dout_trans=active_dout_trans,
            dcol_buf=self._dcol_buffer
        )

        return dx


class MaxPool2D:
    def __init__(self, pool_size: int = 2, stride: int = 2, backend: EngineBackend = EngineBackend.NATIVE):
        init_engine_backend(backend)
        self.pool_size = pool_size
        self.stride = stride
        self.x_shape = None
        self._cache = None
        self._out_buf = None
        self._dx_buf = None
        self._argmax_buf = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x)

        self.x_shape = x.shape
        N, C, H, W = self.x_shape
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1

        if self._out_buf is None or self._out_buf.shape != (N, C, out_h, out_w) or self._out_buf.dtype != x.dtype:
            self._out_buf = np.empty((N, C, out_h, out_w), dtype=x.dtype)
            self._argmax_buf = np.empty((N, C, out_h, out_w), dtype=np.uint8)

        out, self._cache = maxpool_forward(
            x, self.pool_size, self.stride,
            out_buf=self._out_buf,
            argmax_buf=self._argmax_buf
        )
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if not dout.flags['C_CONTIGUOUS']:
            dout = np.ascontiguousarray(dout)

        if self._dx_buf is None or self._dx_buf.shape != self.x_shape or self._dx_buf.dtype != dout.dtype:
            self._dx_buf = np.zeros(self.x_shape, dtype=dout.dtype)

        return maxpool_backward(
            dout, self._cache, self.x_shape,
            self.pool_size, self.stride,
            dx_buf=self._dx_buf
        )


class Flatten:
    def __init__(self):
        self.orig_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self.orig_shape)