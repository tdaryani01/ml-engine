# src/scratch_arena.py
"""Per-session/per-step scratch buffers sized by batch cap (model-agnostic)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from config.constants import EngineBackend
from utils.perf_experiments import aligned_scratch, skip_dx_zero, step_dx_zero

logger = logging.getLogger(__name__)

_AVX_ALIGN = 32


def _alloc_aligned(shape, dtype=np.float32, *, align: int = _AVX_ALIGN, zero: bool = False):
    """Return a C-contiguous ndarray with ``align``-byte base pointer."""
    shape = tuple(int(x) for x in shape)
    count = int(np.prod(shape))
    itemsize = np.dtype(dtype).itemsize
    nbytes = count * itemsize
    slab = np.empty(nbytes + align, dtype=np.uint8)
    start = (-int(slab.ctypes.data)) % align
    view = np.frombuffer(slab, dtype=dtype, offset=start, count=count)
    arr = view.reshape(shape)
    if zero:
        arr.fill(0.0)
    return arr, slab


def aligned_zeros(shape, dtype=np.float32, align: int = _AVX_ALIGN):
    arr, slab = _alloc_aligned(shape, dtype=dtype, align=align, zero=True)
    return arr, slab


def aligned_empty(shape, dtype=np.float32, align: int = _AVX_ALIGN):
    arr, slab = _alloc_aligned(shape, dtype=dtype, align=align, zero=False)
    return arr, slab


def _np_zeros(shape, dtype=np.float32, **kw):
    return np.zeros(shape, dtype=dtype), None


def _np_empty(shape, dtype=np.float32, **kw):
    return np.empty(shape, dtype=dtype), None


def _round_up_simd(w: int, align: int = 8) -> int:
    return (w + align - 1) & ~(align - 1)


@dataclass
class ConvBlockScratch:
    max_n: int = 0
    eval_max_n: int = 0
    cached_dtype: object = None
    eval_cached_dtype: object = None
    col_cap: int = 0
    geom_key: tuple | None = None  # (C, H, W_stride, W_logical) — skip geometry recompute
    out_conv_buffer: np.ndarray | None = None
    out_pool_buffer: np.ndarray | None = None
    argmax_buffer: np.ndarray | None = None
    d_conv_buffer: np.ndarray | None = None
    dx_buffer: np.ndarray | None = None
    dout_trans_buffer: np.ndarray | None = None
    col_buffer: np.ndarray | None = None
    dcol_buffer: np.ndarray | None = None
    fwd_gemm_buffer: np.ndarray | None = None
    w_gemm_fwd_buffer: np.ndarray | None = None
    eval_out_conv_buffer: np.ndarray | None = None
    eval_out_pool_buffer: np.ndarray | None = None
    eval_argmax_buffer: np.ndarray | None = None
    _slabs: list | None = None


@dataclass
class Conv2DScratch:
    col_cap: int = 0
    cached_dtype: object = None
    geom_key: tuple | None = None
    col_buffer: np.ndarray | None = None
    dcol_buffer: np.ndarray | None = None
    dout_trans_buffer: np.ndarray | None = None
    fwd_gemm_buffer: np.ndarray | None = None
    w_gemm_fwd_buffer: np.ndarray | None = None
    fwd_out_buffer: np.ndarray | None = None
    dx_buffer: np.ndarray | None = None
    max_n: int = 0
    _slabs: list | None = None


@dataclass
class MaxPoolScratch:
    max_n: int = 0
    cached_dtype: object = None
    out_buf: np.ndarray | None = None
    argmax_buf: np.ndarray | None = None
    dx_buf: np.ndarray | None = None


class ScratchArena:
    """Owns reusable numpy buffers for spatial layers; keyed by layer index."""

    def __init__(self, backend: EngineBackend = EngineBackend.NATIVE):
        self.backend = backend
        self.train_batch_cap = 0
        self.conv_blocks: dict[int, ConvBlockScratch] = {}
        self.conv2d: dict[int, Conv2DScratch] = {}
        self.maxpool: dict[int, MaxPoolScratch] = {}

    def set_train_batch_cap(self, cap: int) -> None:
        self.train_batch_cap = int(cap)

    def zero_dx_buffers(self, batch_n: int) -> None:
        """Zero all conv dx scratch once per backward (replaces per-col2im memset)."""
        if skip_dx_zero() or not step_dx_zero():
            return
        n = int(batch_n)
        for scratch in self.conv_blocks.values():
            if scratch.dx_buffer is not None:
                scratch.dx_buffer[:n].fill(0.0)
        for scratch in self.conv2d.values():
            if scratch.dx_buffer is not None:
                scratch.dx_buffer[:n].fill(0.0)

    
    def conv_block(self, layer_idx: int) -> ConvBlockScratch:
        if layer_idx not in self.conv_blocks:
            self.conv_blocks[layer_idx] = ConvBlockScratch()
        return self.conv_blocks[layer_idx]

    def conv2d_layer(self, layer_idx: int) -> Conv2DScratch:
        if layer_idx not in self.conv2d:
            self.conv2d[layer_idx] = Conv2DScratch()
        return self.conv2d[layer_idx]

    def maxpool_layer(self, layer_idx: int) -> MaxPoolScratch:
        if layer_idx not in self.maxpool:
            self.maxpool[layer_idx] = MaxPoolScratch()
        return self.maxpool[layer_idx]

    
    def ensure_conv_block_train(
        self,
        layer_idx: int,
        *,
        out_channels: int,
        in_channels: int,
        k_h: int,
        k_w: int,
        conv_stride: int,
        conv_pad: int,
        pool_size: int,
        pool_stride: int,
        N: int,
        C: int,
        H: int,
        W_stride: int,
        W_logical: int,
        dtype,
    ) -> ConvBlockScratch:
        scratch = self.conv_block(layer_idx)
        geom_key = (C, H, W_stride, W_logical)
        if (
            scratch.cached_dtype == dtype
            and N <= scratch.max_n
            and scratch.dx_buffer is not None
            and scratch.geom_key == geom_key
        ):
            return scratch

        conv_out_h = (H + 2 * conv_pad - k_h) // conv_stride + 1
        conv_out_w = (W_logical + 2 * conv_pad - k_w) // conv_stride + 1
        conv_out_w_stride = _round_up_simd(conv_out_w)
        pool_out_h = (conv_out_h - pool_size) // pool_stride + 1
        pool_out_w = (conv_out_w - pool_size) // pool_stride + 1
        total_cols = C * k_h * k_w

        target_n = max(scratch.max_n, N)
        if self.train_batch_cap > 0:
            if scratch.max_n > self.train_batch_cap:
                target_n = self.train_batch_cap
            else:
                target_n = min(target_n, self.train_batch_cap)

        scratch.max_n = target_n
        scratch.col_cap = max(scratch.col_cap, scratch.max_n * conv_out_h * conv_out_w_stride)
        scratch.cached_dtype = dtype
        scratch.geom_key = geom_key
        slabs: list = []
        alloc_z = aligned_zeros if aligned_scratch() else _np_zeros
        alloc_e = aligned_empty if aligned_scratch() else _np_empty

        scratch.out_conv_buffer, s = alloc_z(
            (scratch.max_n, out_channels, conv_out_h, conv_out_w_stride), dtype=dtype
        )
        slabs.append(s)
        scratch.out_pool_buffer, s = alloc_e(
            (scratch.max_n, out_channels, pool_out_h, pool_out_w), dtype=dtype
        )
        slabs.append(s)
        scratch.argmax_buffer, s = alloc_e(
            (scratch.max_n, out_channels, pool_out_h, pool_out_w), dtype=np.uint8, align=32
        )
        slabs.append(s)
        scratch.d_conv_buffer, s = alloc_z(
            (scratch.max_n, out_channels, conv_out_h, conv_out_w_stride), dtype=dtype
        )
        slabs.append(s)
        scratch.dx_buffer, s = alloc_z((scratch.max_n, C, H, W_stride), dtype=dtype)
        slabs.append(s)

        if self.backend != EngineBackend.NATIVE:
            scratch.dout_trans_buffer, s = alloc_e((scratch.col_cap, out_channels), dtype=dtype)
            slabs.append(s)
            scratch.col_buffer, s = alloc_e((scratch.col_cap, total_cols), dtype=dtype)
            slabs.append(s)
            scratch.dcol_buffer, s = alloc_e((scratch.col_cap, total_cols), dtype=dtype)
            slabs.append(s)
            scratch.fwd_gemm_buffer, s = alloc_e((scratch.col_cap, out_channels), dtype=dtype)
            slabs.append(s)
            scratch.w_gemm_fwd_buffer, s = alloc_e((total_cols, out_channels), dtype=dtype)
            slabs.append(s)
        else:
            scratch.dout_trans_buffer = None
            scratch.col_buffer = None
            scratch.dcol_buffer = None
            scratch.fwd_gemm_buffer = None
            scratch.w_gemm_fwd_buffer = None
        scratch._slabs = slabs
        return scratch

    def ensure_conv_block_eval(
        self,
        layer_idx: int,
        *,
        out_channels: int,
        k_h: int,
        k_w: int,
        conv_stride: int,
        conv_pad: int,
        pool_size: int,
        pool_stride: int,
        N: int,
        C: int,
        H: int,
        W_logical: int,
        dtype,
    ) -> ConvBlockScratch:
        scratch = self.conv_block(layer_idx)
        conv_out_h = (H + 2 * conv_pad - k_h) // conv_stride + 1
        conv_out_w = (W_logical + 2 * conv_pad - k_w) // conv_stride + 1
        conv_out_w_stride = _round_up_simd(conv_out_w)
        pool_out_h = (conv_out_h - pool_size) // pool_stride + 1
        pool_out_w = (conv_out_w - pool_size) // pool_stride + 1

        target_n = max(scratch.eval_max_n, N)
        if (
            scratch.eval_cached_dtype == dtype
            and N <= scratch.eval_max_n
            and scratch.eval_out_conv_buffer is not None
        ):
            return scratch

        scratch.eval_max_n = target_n
        scratch.eval_cached_dtype = dtype
        scratch.eval_out_conv_buffer = np.zeros(
            (scratch.eval_max_n, out_channels, conv_out_h, conv_out_w_stride), dtype=dtype
        )
        scratch.eval_out_pool_buffer = np.empty(
            (scratch.eval_max_n, out_channels, pool_out_h, pool_out_w), dtype=dtype
        )
        scratch.eval_argmax_buffer = np.empty(
            (scratch.eval_max_n, out_channels, pool_out_h, pool_out_w), dtype=np.uint8
        )
        return scratch

    def ensure_conv2d(
        self,
        layer_idx: int,
        *,
        out_channels: int,
        in_channels: int,
        k_h: int,
        k_w: int,
        stride: int,
        pad: int,
        N: int,
        C: int,
        H: int,
        W_stride: int,
        W_logical: int,
        dtype,
    ) -> Conv2DScratch:
        scratch = self.conv2d_layer(layer_idx)
        geom_key = (C, H, W_stride, W_logical)
        if (
            scratch.cached_dtype == dtype
            and N <= scratch.max_n
            and scratch.dx_buffer is not None
            and scratch.fwd_out_buffer is not None
            and scratch.geom_key == geom_key
        ):
            return scratch

        out_h = (H + 2 * pad - k_h) // stride + 1
        out_w = (W_logical + 2 * pad - k_w) // stride + 1
        out_w_stride = _round_up_simd(out_w)
        total_rows = N * out_h * out_w_stride
        total_cols = C * k_h * k_w

        target_n = max(scratch.max_n, N)
        if self.train_batch_cap > 0:
            target_n = min(target_n, self.train_batch_cap) if scratch.max_n <= self.train_batch_cap else self.train_batch_cap

        scratch.col_cap = max(scratch.col_cap, total_rows)
        scratch.max_n = max(scratch.max_n, target_n)
        scratch.cached_dtype = dtype
        scratch.geom_key = geom_key
        slabs: list = []
        alloc_z = aligned_zeros if aligned_scratch() else _np_zeros
        alloc_e = aligned_empty if aligned_scratch() else _np_empty
        scratch.col_buffer, s = alloc_e((scratch.col_cap, total_cols), dtype=dtype)
        slabs.append(s)
        scratch.dcol_buffer, s = alloc_e((scratch.col_cap, total_cols), dtype=dtype)
        slabs.append(s)
        scratch.dout_trans_buffer, s = alloc_e((scratch.col_cap, out_channels), dtype=dtype)
        slabs.append(s)
        scratch.fwd_gemm_buffer, s = alloc_e((scratch.col_cap, out_channels), dtype=dtype)
        slabs.append(s)
        scratch.w_gemm_fwd_buffer, s = alloc_e((total_cols, out_channels), dtype=dtype)
        slabs.append(s)
        scratch.fwd_out_buffer, s = alloc_z(
            (scratch.max_n, out_channels, out_h, out_w_stride), dtype=dtype
        )
        slabs.append(s)
        scratch.dx_buffer, s = alloc_z((scratch.max_n, C, H, W_stride), dtype=dtype)
        slabs.append(s)
        scratch._slabs = slabs
        return scratch

    def ensure_maxpool(
        self,
        layer_idx: int,
        *,
        N: int,
        C: int,
        H: int,
        W_stride: int,
        w_log: int,
        pool_size: int,
        pool_stride: int,
        dtype,
    ) -> MaxPoolScratch:
        scratch = self.maxpool_layer(layer_idx)
        out_h = (H - pool_size) // pool_stride + 1
        out_w = (w_log - pool_size) // pool_stride + 1
        out_w_stride = (W_stride - pool_size) // pool_stride + 1

        if (
            scratch.cached_dtype == dtype
            and N <= scratch.max_n
            and scratch.out_buf is not None
            and scratch.out_buf.shape[1:] == (C, out_h, out_w_stride)
        ):
            return scratch

        scratch.max_n = max(scratch.max_n, N)
        scratch.cached_dtype = dtype
        scratch.out_buf = np.empty((scratch.max_n, C, out_h, out_w_stride), dtype=dtype)
        scratch.argmax_buf = np.empty((scratch.max_n, C, out_h, out_w_stride), dtype=np.uint8)
        return scratch
