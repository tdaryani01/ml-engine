# src/scratch_arena.py
"""Per-session/per-step scratch buffers sized by batch cap (model-agnostic)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from config.constants import EngineBackend

logger = logging.getLogger(__name__)


def _round_up_simd(w: int, align: int = 8) -> int:
    return (w + align - 1) & ~(align - 1)


@dataclass
class ConvBlockScratch:
    max_n: int = 0
    eval_max_n: int = 0
    cached_dtype: object = None
    eval_cached_dtype: object = None
    col_cap: int = 0
    out_conv_buffer: np.ndarray | None = None
    out_pool_buffer: np.ndarray | None = None
    argmax_buffer: np.ndarray | None = None
    d_conv_buffer: np.ndarray | None = None
    dx_buffer: np.ndarray | None = None
    dout_trans_buffer: np.ndarray | None = None
    col_buffer: np.ndarray | None = None
    dcol_buffer: np.ndarray | None = None
    fwd_gemm_buffer: np.ndarray | None = None
    eval_out_conv_buffer: np.ndarray | None = None
    eval_out_pool_buffer: np.ndarray | None = None
    eval_argmax_buffer: np.ndarray | None = None


@dataclass
class Conv2DScratch:
    col_cap: int = 0
    cached_dtype: object = None
    col_buffer: np.ndarray | None = None
    dcol_buffer: np.ndarray | None = None
    dout_trans_buffer: np.ndarray | None = None
    fwd_gemm_buffer: np.ndarray | None = None
    fwd_out_buffer: np.ndarray | None = None
    dx_buffer: np.ndarray | None = None
    max_n: int = 0


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

        if (
            scratch.cached_dtype == dtype
            and N <= scratch.max_n
            and scratch.max_n == target_n
            and scratch.dx_buffer is not None
        ):
            return scratch

        scratch.max_n = target_n
        scratch.col_cap = max(scratch.col_cap, scratch.max_n * conv_out_h * conv_out_w_stride)
        scratch.cached_dtype = dtype

        scratch.out_conv_buffer = np.zeros(
            (scratch.max_n, out_channels, conv_out_h, conv_out_w_stride), dtype=dtype
        )
        scratch.out_pool_buffer = np.empty(
            (scratch.max_n, out_channels, pool_out_h, pool_out_w), dtype=dtype
        )
        scratch.argmax_buffer = np.empty(
            (scratch.max_n, out_channels, pool_out_h, pool_out_w), dtype=np.uint8
        )
        scratch.d_conv_buffer = np.zeros(
            (scratch.max_n, out_channels, conv_out_h, conv_out_w_stride), dtype=dtype
        )
        scratch.dx_buffer = np.zeros((scratch.max_n, C, H, W_stride), dtype=dtype)

        if self.backend != EngineBackend.NATIVE:
            scratch.dout_trans_buffer = np.empty((scratch.col_cap, out_channels), dtype=dtype)
            scratch.col_buffer = np.empty((scratch.col_cap, total_cols), dtype=dtype)
            scratch.dcol_buffer = np.empty((scratch.col_cap, total_cols), dtype=dtype)
            scratch.fwd_gemm_buffer = np.empty((scratch.col_cap, out_channels), dtype=dtype)
        else:
            scratch.dout_trans_buffer = None
            scratch.col_buffer = None
            scratch.dcol_buffer = None
            scratch.fwd_gemm_buffer = None
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
        out_h = (H + 2 * pad - k_h) // stride + 1
        out_w = (W_logical + 2 * pad - k_w) // stride + 1
        out_w_stride = _round_up_simd(out_w)
        total_rows = N * out_h * out_w_stride
        total_cols = C * k_h * k_w

        target_n = max(scratch.max_n, N)
        if self.train_batch_cap > 0:
            target_n = min(target_n, self.train_batch_cap) if scratch.max_n <= self.train_batch_cap else self.train_batch_cap

        if scratch.cached_dtype == dtype and total_rows <= scratch.col_cap and scratch.dx_buffer is not None:
            if scratch.fwd_out_buffer is not None and scratch.fwd_out_buffer.shape[0] >= N:
                return scratch

        scratch.col_cap = max(scratch.col_cap, total_rows)
        scratch.max_n = max(scratch.max_n, target_n)
        scratch.cached_dtype = dtype
        scratch.col_buffer = np.empty((scratch.col_cap, total_cols), dtype=dtype)
        scratch.dcol_buffer = np.empty((scratch.col_cap, total_cols), dtype=dtype)
        scratch.dout_trans_buffer = np.empty((scratch.col_cap, out_channels), dtype=dtype)
        scratch.fwd_gemm_buffer = np.empty((scratch.col_cap, out_channels), dtype=dtype)
        scratch.fwd_out_buffer = np.zeros(
            (scratch.max_n, out_channels, out_h, out_w_stride), dtype=dtype
        )
        scratch.dx_buffer = np.zeros((scratch.max_n, C, H, W_stride), dtype=dtype)
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
