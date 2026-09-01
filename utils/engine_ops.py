# utils/engine_ops.py
"""
Model-agnostic execution context and ops backends.

``EngineContext`` is shared by every layer in a model session (CNN today,
attention/transformer later). Each *ops family* (conv, attention, …) is a
separate bundle registered on the context and accessed via ``ctx.conv`` or
``ctx.ops("conv")``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from config.constants import EngineBackend
from utils import im2col


@runtime_checkable
class EngineOps(Protocol):
    """Base protocol for a kernel family (conv, attention, dense, …)."""

    @property
    def backend(self) -> EngineBackend: ...


@runtime_checkable
class ConvOps(EngineOps, Protocol):
    """Convolution + fused block + pooling dispatch."""

    def conv2d_forward(
        self,
        x: np.ndarray,
        W: np.ndarray,
        bias: np.ndarray,
        stride: int,
        pad: int,
        out_buf: np.ndarray,
        col_buf: np.ndarray | None = None,
        gemm_buf: np.ndarray | None = None,
        fuse_relu: bool = False,
        W_logical: int | None = None,
    ) -> tuple: ...

    def conv2d_backward_fused(
        self,
        dout: np.ndarray,
        x: np.ndarray,
        W: np.ndarray,
        dx_buf: np.ndarray,
        dW_buf: np.ndarray,
        stride: int,
        pad: int,
        inv_m: float,
        in_act: np.ndarray | None = None,
        fuse_relu: bool = False,
        col: np.ndarray | None = None,
        dout_trans: np.ndarray | None = None,
        dcol_buf: np.ndarray | None = None,
        W_logical: int | None = None,
    ) -> tuple: ...

    def conv_block_forward(
        self,
        x: np.ndarray,
        W: np.ndarray,
        bias: np.ndarray,
        out_conv_buf: np.ndarray,
        out_pool_buf: np.ndarray,
        argmax_buf: np.ndarray,
        conv_stride: int = 1,
        conv_pad: int = 1,
        pool_size: int = 2,
        pool_stride: int = 2,
        col_buf: np.ndarray | None = None,
        gemm_buf: np.ndarray | None = None,
        W_logical: int | None = None,
        out_w_logical: int | None = None,
    ) -> tuple: ...

    def conv_block_backward(
        self,
        dout_pool: np.ndarray,
        argmax_buf: np.ndarray,
        x: np.ndarray,
        W: np.ndarray,
        conv_act: np.ndarray,
        d_conv_buf: np.ndarray,
        dx_buf: np.ndarray,
        dW_buf: np.ndarray,
        db_buf: np.ndarray,
        conv_stride: int = 1,
        conv_pad: int = 1,
        pool_size: int = 2,
        pool_stride: int = 2,
        inv_m: float = 1.0,
        col: np.ndarray | None = None,
        dout_trans: np.ndarray | None = None,
        dcol_buf: np.ndarray | None = None,
        W_logical: int | None = None,
        out_w_logical: int | None = None,
    ) -> tuple: ...

    def maxpool_forward(
        self,
        x: np.ndarray,
        pool_size: int,
        stride: int,
        out_buf: np.ndarray | None = None,
        argmax_buf: np.ndarray | None = None,
    ) -> tuple: ...

    def maxpool_backward(
        self,
        dout: np.ndarray,
        cache: np.ndarray,
        x_shape: tuple,
        pool_size: int,
        stride: int,
        dx_buf: np.ndarray | None = None,
    ) -> np.ndarray: ...

    def fuse_dout_transpose_and_bias(
        self,
        dout: np.ndarray,
        dout_trans_buf: np.ndarray,
        db_buf: np.ndarray,
    ) -> None: ...


class _Im2colConvOpsBase:
    """Delegates to ``utils.im2col`` with a bound backend enum (no ctx lookup)."""

    __slots__ = ("_backend",)

    def __init__(self, backend: EngineBackend) -> None:
        self._backend = backend

    @property
    def backend(self) -> EngineBackend:
        return self._backend

    def conv2d_forward(self, x, W, bias, stride, pad, out_buf, col_buf=None, gemm_buf=None,
                       fuse_relu=False, W_logical=None):
        return im2col.conv2d_forward(
            x=x, W=W, bias=bias, stride=stride, pad=pad, out_buf=out_buf,
            col_buf=col_buf, gemm_buf=gemm_buf, fuse_relu=fuse_relu,
            W_logical=W_logical, backend=self._backend,
        )

    def conv2d_backward_fused(self, dout, x, W, dx_buf, dW_buf, stride, pad, inv_m,
                              in_act=None, fuse_relu=False, col=None, dout_trans=None,
                              dcol_buf=None, W_logical=None):
        return im2col.conv2d_backward_fused(
            dout=dout, x=x, W=W, dx_buf=dx_buf, dW_buf=dW_buf,
            stride=stride, pad=pad, inv_m=inv_m, in_act=in_act, fuse_relu=fuse_relu,
            col=col, dout_trans=dout_trans, dcol_buf=dcol_buf, W_logical=W_logical,
            backend=self._backend,
        )

    def conv_block_forward(self, x, W, bias, out_conv_buf, out_pool_buf, argmax_buf,
                           conv_stride=1, conv_pad=1, pool_size=2, pool_stride=2,
                           col_buf=None, gemm_buf=None, W_logical=None, out_w_logical=None):
        return im2col.conv_block_forward(
            x=x, W=W, bias=bias, out_conv_buf=out_conv_buf, out_pool_buf=out_pool_buf,
            argmax_buf=argmax_buf, conv_stride=conv_stride, conv_pad=conv_pad,
            pool_size=pool_size, pool_stride=pool_stride, col_buf=col_buf,
            gemm_buf=gemm_buf, W_logical=W_logical, out_w_logical=out_w_logical,
            backend=self._backend,
        )

    def conv_block_backward(self, dout_pool, argmax_buf, x, W, conv_act, d_conv_buf,
                            dx_buf, dW_buf, db_buf, conv_stride=1, conv_pad=1,
                            pool_size=2, pool_stride=2, inv_m=1.0, col=None,
                            dout_trans=None, dcol_buf=None, W_logical=None, out_w_logical=None):
        return im2col.conv_block_backward(
            dout_pool=dout_pool, argmax_buf=argmax_buf, x=x, W=W, conv_act=conv_act,
            d_conv_buf=d_conv_buf, dx_buf=dx_buf, dW_buf=dW_buf, db_buf=db_buf,
            conv_stride=conv_stride, conv_pad=conv_pad, pool_size=pool_size,
            pool_stride=pool_stride, inv_m=inv_m, col=col, dout_trans=dout_trans,
            dcol_buf=dcol_buf, W_logical=W_logical, out_w_logical=out_w_logical,
            backend=self._backend,
        )

    def maxpool_forward(self, x, pool_size, stride, out_buf=None, argmax_buf=None):
        return im2col.maxpool_forward(
            x, pool_size, stride, out_buf=out_buf, argmax_buf=argmax_buf,
            backend=self._backend,
        )

    def maxpool_backward(self, dout, cache, x_shape, pool_size, stride, dx_buf=None):
        return im2col.maxpool_backward(
            dout, cache, x_shape, pool_size, stride, dx_buf=dx_buf,
            backend=self._backend,
        )

    def fuse_dout_transpose_and_bias(self, dout, dout_trans_buf, db_buf):
        return im2col.fuse_dout_transpose_and_bias(
            dout, dout_trans_buf, db_buf, backend=self._backend,
        )


class NativeConvOps(_Im2colConvOpsBase):
    def __init__(self) -> None:
        super().__init__(EngineBackend.NATIVE)


class NumpyConvOps(_Im2colConvOpsBase):
    def __init__(self) -> None:
        super().__init__(EngineBackend.NUMPY)


class Im2colGemmConvOps(_Im2colConvOpsBase):
    def __init__(self) -> None:
        super().__init__(EngineBackend.IM2COL_GEMM)


_CONV_OPS_BY_BACKEND: Mapping[EngineBackend, type[_Im2colConvOpsBase]] = {
    EngineBackend.NATIVE: NativeConvOps,
    EngineBackend.NUMPY: NumpyConvOps,
    EngineBackend.IM2COL_GEMM: Im2colGemmConvOps,
}


@dataclass
class EngineContext:
    """
    Session-scoped execution context for any model architecture.

    Holds the selected backend and registered ops bundles. CNN layers use
    ``ctx.conv``; future attention blocks will register ``ctx.ops("attention")``.
    """

    backend: EngineBackend
    conv: ConvOps
    _registry: dict[str, EngineOps] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._registry.setdefault("conv", self.conv)

    @property
    def native_lib(self):
        """Loaded native DLL handle, or ``None`` when backend is not NATIVE."""
        if self.backend != EngineBackend.NATIVE:
            return None
        return im2col.get_native_lib()

    def ops(self, name: str) -> EngineOps:
        """Return a registered ops bundle by name (extensible for new model types)."""
        try:
            return self._registry[name]
        except KeyError as exc:
            registered = sorted(self._registry)
            raise KeyError(
                f"No ops bundle {name!r}. Registered: {registered}"
            ) from exc

    def register(self, name: str, bundle: EngineOps) -> None:
        """Attach an additional ops family (e.g. attention) to this session."""
        self._registry[name] = bundle


def create_engine_context(backend: EngineBackend = EngineBackend.NATIVE) -> EngineContext:
    """Create a model-agnostic execution context with conv ops initialized."""
    im2col.init_engine_backend(backend)
    conv_cls = _CONV_OPS_BY_BACKEND[backend]
    conv = conv_cls()
    return EngineContext(backend=backend, conv=conv)


def resolve_engine_context(
    engine_ctx: EngineContext | None = None,
    backend: EngineBackend | None = None,
) -> EngineContext:
    """Resolve or create a context (used by layers when ctx is not injected)."""
    if engine_ctx is not None:
        return engine_ctx
    return create_engine_context(backend or EngineBackend.NATIVE)
