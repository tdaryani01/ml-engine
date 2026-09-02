# src/spatial_layers.py
import builtins
import logging
import numpy as np
from config.constants import EngineBackend
from src.scratch_arena import ScratchArena
from src.training_cache import (
    Conv2DStepCache,
    ConvBlockStepCache,
    FlattenStepCache,
    ForwardCache,
    MaxPoolStepCache,
)
from utils.engine_ops import EngineContext, resolve_engine_context
from utils.conv_dispatch import col2im

logger = logging.getLogger(__name__)

if 'profile' not in builtins.__dict__:
    builtins.__dict__['profile'] = lambda x: x


def _round_up_simd(w: int, align: int = 8) -> int:
    """Rounds a spatial width dimension up to the nearest SIMD vector boundary."""
    return (w + align - 1) & ~(align - 1)


class ConvBlock:
    """
    Fused spatial block executing Conv2D -> ReLU -> MaxPool2D in a single C++ dispatch.
    Holds parameters only; per-step state lives in ForwardCache + ScratchArena.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 conv_stride: int = 1, conv_pad: int = 0,
                 pool_size: int = 2, pool_stride: int = 2,
                 engine_ctx: EngineContext | None = None,
                 backend: EngineBackend | None = None):
        self._ctx = resolve_engine_context(engine_ctx, backend)
        self._conv = self._ctx.conv
        self.backend = self._ctx.backend
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
        self.out_h = 0
        self.out_w = 0

    def _compute_geometry(self, C: int, H: int, W_stride: int, W_logical: int):
        conv_out_h = (H + 2 * self.conv_pad - self.k_h) // self.conv_stride + 1
        conv_out_w = (W_logical + 2 * self.conv_pad - self.k_w) // self.conv_stride + 1
        pool_out_h = (conv_out_h - self.pool_size) // self.pool_stride + 1
        pool_out_w = (conv_out_w - self.pool_size) // self.pool_stride + 1
        self.out_h = pool_out_h
        self.out_w = pool_out_w

    def _sync_param_dtypes(self, dtype):
        if self.W.dtype != dtype:
            self.W = self.W.astype(dtype)
            self.b = self.b.astype(dtype)
            self.dW = np.zeros_like(self.W, dtype=dtype)
            self.db = np.zeros_like(self.b, dtype=dtype)

    def forward(
        self,
        x: np.ndarray,
        W_logical: int = None,
        inference: bool = False,
        cache: ForwardCache | None = None,
        arena: ScratchArena | None = None,
        layer_idx: int = 0,
    ) -> np.ndarray:
        if not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x)

        N, C, H, W_stride = x.shape
        w_log = W_logical if W_logical is not None else W_stride
        self._compute_geometry(C, H, W_stride, w_log)
        self._sync_param_dtypes(x.dtype)

        if arena is None:
            raise ValueError("ConvBlock.forward requires a ScratchArena")

        if inference:
            scratch = arena.ensure_conv_block_eval(
                layer_idx,
                out_channels=self.out_channels,
                k_h=self.k_h,
                k_w=self.k_w,
                conv_stride=self.conv_stride,
                conv_pad=self.conv_pad,
                pool_size=self.pool_size,
                pool_stride=self.pool_stride,
                N=N,
                C=C,
                H=H,
                W_logical=w_log,
                dtype=x.dtype,
            )
            out_conv_buf = scratch.eval_out_conv_buffer
            out_pool_buf = scratch.eval_out_pool_buffer
            argmax_buf = scratch.eval_argmax_buffer
            col_buf = None
            gemm_buf = None
        else:
            scratch = arena.ensure_conv_block_train(
                layer_idx,
                out_channels=self.out_channels,
                in_channels=self.in_channels,
                k_h=self.k_h,
                k_w=self.k_w,
                conv_stride=self.conv_stride,
                conv_pad=self.conv_pad,
                pool_size=self.pool_size,
                pool_stride=self.pool_stride,
                N=N,
                C=C,
                H=H,
                W_stride=W_stride,
                W_logical=w_log,
                dtype=x.dtype,
            )
            out_conv_buf = scratch.out_conv_buffer
            out_pool_buf = scratch.out_pool_buffer
            argmax_buf = scratch.argmax_buffer
            col_buf = scratch.col_buffer
            gemm_buf = scratch.fwd_gemm_buffer

        out_pool, out_conv, argmax, col = self._conv.conv_block_forward(
            x=x, W=self.W, bias=self.b,
            out_conv_buf=out_conv_buf[:N],
            out_pool_buf=out_pool_buf[:N],
            argmax_buf=argmax_buf[:N],
            conv_stride=self.conv_stride, conv_pad=self.conv_pad,
            pool_size=self.pool_size, pool_stride=self.pool_stride,
            col_buf=col_buf, gemm_buf=gemm_buf,
            w_gemm_fwd_buf=scratch.w_gemm_fwd_buffer,
            W_logical=w_log,
        )

        if cache is not None and not inference:
            cache.conv_blocks[layer_idx] = ConvBlockStepCache(
                x=x, conv_act=out_conv, argmax=argmax, col=col, scratch=scratch,
            )
        return out_pool

    def backward(
        self,
        dout: np.ndarray,
        W_logical: int = None,
        cache: ForwardCache | None = None,
        arena: ScratchArena | None = None,
        layer_idx: int = 0,
    ) -> np.ndarray:
        if cache is None or arena is None:
            raise ValueError("ConvBlock.backward requires ForwardCache and ScratchArena")
        step = cache.conv_blocks.get(layer_idx)
        if step is None:
            raise KeyError(f"ConvBlock backward missing step cache for layer {layer_idx}")

        if not dout.flags['C_CONTIGUOUS']:
            dout = np.ascontiguousarray(dout)

        N, C, H, W_stride = step.x.shape
        w_log = W_logical if W_logical is not None else W_stride
        inv_m = 1.0 / float(N)

        scratch = step.scratch
        if scratch is None:
            scratch = arena.ensure_conv_block_train(
                layer_idx,
                out_channels=self.out_channels,
                in_channels=self.in_channels,
                k_h=self.k_h,
                k_w=self.k_w,
                conv_stride=self.conv_stride,
                conv_pad=self.conv_pad,
                pool_size=self.pool_size,
                pool_stride=self.pool_stride,
                N=N,
                C=C,
                H=H,
                W_stride=W_stride,
                W_logical=w_log,
                dtype=dout.dtype,
            )
        self._sync_param_dtypes(dout.dtype)

        dx, self.dW, self.db = self._conv.conv_block_backward(
            dout_pool=dout,
            argmax_buf=step.argmax,
            x=step.x,
            W=self.W,
            conv_act=step.conv_act,
            d_conv_buf=scratch.d_conv_buffer[:N],
            dx_buf=scratch.dx_buffer[:N],
            dW_buf=self.dW,
            db_buf=self.db,
            conv_stride=self.conv_stride,
            conv_pad=self.conv_pad,
            pool_size=self.pool_size,
            pool_stride=self.pool_stride,
            inv_m=inv_m,
            col=step.col,
            dout_trans=scratch.dout_trans_buffer,
            dcol_buf=scratch.dcol_buffer,
            W_logical=w_log,
        )
        return dx


class Conv2D:
    """Standalone 2D convolution; parameters only, step state in cache/arena."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, pad: int = 0,
                 engine_ctx: EngineContext | None = None,
                 backend: EngineBackend | None = None):
        self._ctx = resolve_engine_context(engine_ctx, backend)
        self._conv = self._ctx.conv
        self.backend = self._ctx.backend
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
        self.out_h = 0
        self.out_w = 0
        self.out_w_stride = 0
        self._bound_scratch = None

    def forward(
        self,
        x: np.ndarray,
        W_logical: int = None,
        cache: ForwardCache | None = None,
        arena: ScratchArena | None = None,
        layer_idx: int = 0,
    ) -> np.ndarray:
        if not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x)
        if arena is None:
            raise ValueError("Conv2D.forward requires a ScratchArena")

        N, C, H, W_stride = x.shape
        w_log = W_logical if W_logical is not None else W_stride
        if self.W.dtype != x.dtype:
            self.W = self.W.astype(x.dtype)
            self.b = self.b.astype(x.dtype)
        if self.dW is None or self.dW.dtype != x.dtype:
            self.dW = np.zeros_like(self.W, dtype=x.dtype)
        if self.db is None or self.db.dtype != x.dtype:
            self.db = np.zeros_like(self.b, dtype=x.dtype)
        scratch = arena.ensure_conv2d(
            layer_idx,
            out_channels=self.out_channels,
            in_channels=self.in_channels,
            k_h=self.k_h,
            k_w=self.k_w,
            stride=self.stride,
            pad=self.pad,
            N=N,
            C=C,
            H=H,
            W_stride=W_stride,
            W_logical=w_log,
            dtype=x.dtype,
        )
        self.out_h = (H + 2 * self.pad - self.k_h) // self.stride + 1
        self.out_w = (w_log + 2 * self.pad - self.k_w) // self.stride + 1
        self.out_w_stride = _round_up_simd(self.out_w)

        active_out, col = self._conv.conv2d_forward(
            x=x, W=self.W, bias=self.b,
            stride=self.stride, pad=self.pad,
            out_buf=scratch.fwd_out_buffer[:N],
            col_buf=scratch.col_buffer,
            gemm_buf=scratch.fwd_gemm_buffer,
            w_gemm_fwd_buf=scratch.w_gemm_fwd_buffer,
            fuse_relu=False,
            W_logical=w_log,
        )
        if cache is not None:
            cache.conv2d[layer_idx] = Conv2DStepCache(x=x, col=col)
            self._bound_scratch = scratch
        return active_out

    def backward(
        self,
        dout: np.ndarray,
        in_act: np.ndarray = None,
        fuse_relu: bool = False,
        W_logical: int = None,
        cache: ForwardCache | None = None,
        arena: ScratchArena | None = None,
        layer_idx: int = 0,
    ) -> np.ndarray:
        if cache is None or arena is None:
            raise ValueError("Conv2D.backward requires ForwardCache and ScratchArena")
        step = cache.conv2d.get(layer_idx)
        if step is None:
            raise KeyError(f"Conv2D backward missing step cache for layer {layer_idx}")

        if not dout.flags['C_CONTIGUOUS']:
            dout = np.ascontiguousarray(dout)

        N, C, H, W_stride = step.x.shape
        w_log = W_logical if W_logical is not None else W_stride
        inv_m = 1.0 / float(N)
        total_rows = N * self.out_h * self.out_w
        dout_logical = dout[:, :, :self.out_h, :self.out_w]

        scratch = self._bound_scratch
        if scratch is None:
            scratch = arena.ensure_conv2d(
                layer_idx,
                out_channels=self.out_channels,
                in_channels=self.in_channels,
                k_h=self.k_h,
                k_w=self.k_w,
                stride=self.stride,
                pad=self.pad,
                N=N,
                C=C,
                H=H,
                W_stride=W_stride,
                W_logical=w_log,
                dtype=dout.dtype,
            )
        active_dout_trans = scratch.dout_trans_buffer[:total_rows]
        self._conv.fuse_dout_transpose_and_bias(dout_logical, active_dout_trans, self.db)

        dx, self.dW = self._conv.conv2d_backward_fused(
            dout=dout_logical,
            x=step.x,
            W=self.W,
            dx_buf=scratch.dx_buffer[:N],
            dW_buf=self.dW,
            stride=self.stride,
            pad=self.pad,
            inv_m=inv_m,
            in_act=in_act,
            fuse_relu=fuse_relu,
            col=step.col,
            dout_trans=active_dout_trans,
            dcol_buf=scratch.dcol_buffer,
            W_logical=w_log,
        )
        self._bound_scratch = None
        return dx


class MaxPool2D:
    """Spatial max-pooling; parameters only, step state in cache/arena."""

    def __init__(self, pool_size: int = 2, stride: int = 2,
                 engine_ctx: EngineContext | None = None,
                 backend: EngineBackend | None = None):
        self._ctx = resolve_engine_context(engine_ctx, backend)
        self._conv = self._ctx.conv
        self.backend = self._ctx.backend
        self.pool_size = pool_size
        self.stride = stride
        self.out_h = 0
        self.out_w = 0
        self._bound_scratch = None

    def forward(
        self,
        x: np.ndarray,
        W_logical: int = None,
        cache: ForwardCache | None = None,
        arena: ScratchArena | None = None,
        layer_idx: int = 0,
    ) -> np.ndarray:
        if not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x)
        if arena is None:
            raise ValueError("MaxPool2D.forward requires a ScratchArena")

        N, C, H, W_stride = x.shape
        w_log = W_logical if W_logical is not None else W_stride
        self.out_h = (H - self.pool_size) // self.stride + 1
        self.out_w = (w_log - self.pool_size) // self.stride + 1

        scratch = arena.ensure_maxpool(
            layer_idx,
            N=N,
            C=C,
            H=H,
            W_stride=W_stride,
            w_log=w_log,
            pool_size=self.pool_size,
            pool_stride=self.stride,
            dtype=x.dtype,
        )
        out, pool_cache = self._conv.maxpool_forward(
            x, self.pool_size, self.stride,
            out_buf=scratch.out_buf[:N],
            argmax_buf=scratch.argmax_buf[:N],
        )
        if cache is not None:
            cache.maxpool[layer_idx] = MaxPoolStepCache(x_shape=x.shape, pool_cache=pool_cache)
            self._bound_scratch = scratch
        return out

    def backward(
        self,
        dout: np.ndarray,
        cache: ForwardCache | None = None,
        arena: ScratchArena | None = None,
        layer_idx: int = 0,
    ) -> np.ndarray:
        if cache is None or arena is None:
            raise ValueError("MaxPool2D.backward requires ForwardCache and ScratchArena")
        step = cache.maxpool.get(layer_idx)
        if step is None:
            raise KeyError(f"MaxPool backward missing step cache for layer {layer_idx}")

        if not dout.flags['C_CONTIGUOUS']:
            dout = np.ascontiguousarray(dout)

        scratch = self._bound_scratch if self._bound_scratch is not None else arena.maxpool_layer(layer_idx)
        if scratch.dx_buf is None or scratch.dx_buf.shape != step.x_shape or scratch.dx_buf.dtype != dout.dtype:
            scratch.dx_buf = np.zeros(step.x_shape, dtype=dout.dtype)

        result = self._conv.maxpool_backward(
            dout, step.pool_cache, step.x_shape,
            self.pool_size, self.stride,
            dx_buf=scratch.dx_buf,
        )
        self._bound_scratch = None
        return result


class Flatten:
    """Zero-overhead spatial-to-dense adapter; shape state in ForwardCache."""

    def forward(
        self,
        x: np.ndarray,
        logical_w: int = None,
        cache: ForwardCache | None = None,
        layer_idx: int = 0,
    ) -> np.ndarray:
        if not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x)

        if x.ndim == 4:
            N, C, H, W_stride = x.shape
            lw = logical_w if logical_w is not None else W_stride
            logical_shape = (N, C, H, lw)
            padded_shape = (N, C, H, W_stride)
            if cache is not None:
                cache.flatten[layer_idx] = FlattenStepCache(
                    logical_shape=logical_shape,
                    padded_shape=padded_shape,
                )
            if W_stride != lw:
                return np.ascontiguousarray(x[:, :, :, :lw].reshape(N, -1))
            return x.reshape(N, -1)

        logical_shape = x.shape
        if cache is not None:
            cache.flatten[layer_idx] = FlattenStepCache(
                logical_shape=logical_shape,
                padded_shape=logical_shape,
            )
        return x.reshape(x.shape[0], -1)

    def backward(
        self,
        dout: np.ndarray,
        cache: ForwardCache | None = None,
        layer_idx: int = 0,
    ) -> np.ndarray:
        if cache is None:
            raise ValueError("Flatten.backward requires ForwardCache")
        step = cache.flatten.get(layer_idx)
        if step is None:
            raise KeyError(f"Flatten backward missing step cache for layer {layer_idx}")

        if not dout.flags['C_CONTIGUOUS']:
            dout = np.ascontiguousarray(dout)

        if step.padded_shape is not None and len(step.padded_shape) == 4:
            N, C, H, W_stride = step.padded_shape
            _, _, _, lw = step.logical_shape
            dout_spatial = dout.reshape(N, C, H, lw)
            if W_stride != lw:
                dx_padded = np.zeros((N, C, H, W_stride), dtype=dout.dtype)
                dx_padded[:, :, :, :lw] = dout_spatial
                return dx_padded
            return np.ascontiguousarray(dout_spatial)

        return dout.reshape(step.logical_shape)
