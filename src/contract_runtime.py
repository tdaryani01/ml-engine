# src/contract_runtime.py
"""Phase F: ctypes bindings for native contract-list executor."""
from __future__ import annotations

import ctypes
import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from src.contract import ContractList
from utils.conv_dispatch import _load_conv_dll


def _round_up_simd(w: int, align: int = 8) -> int:
    return (w + align - 1) & ~(align - 1)



def _conv_block_output_geom(
    c: int, h: int, w_log: int, layer
) -> tuple[int, int, int, int]:
    """Return (C_out, pool_h, pool_w, pool_w_stride) after ConvBlock."""
    conv_out_h = (h + 2 * layer.conv_pad - layer.k_h) // layer.conv_stride + 1
    conv_out_w = (w_log + 2 * layer.conv_pad - layer.k_w) // layer.conv_stride + 1
    pool_out_h = (conv_out_h - layer.pool_size) // layer.pool_stride + 1
    pool_out_w = (conv_out_w - layer.pool_size) // layer.pool_stride + 1
    return layer.out_channels, pool_out_h, pool_out_w, pool_out_w


class ContractOpRow(ctypes.Structure):
    _fields_ = [
        ("opcode", ctypes.c_int32),
        ("layer_idx", ctypes.c_int32),
        ("param_idx", ctypes.c_int32),
        ("flags", ctypes.c_int32),
        ("i0", ctypes.c_int32),
        ("i1", ctypes.c_int32),
        ("i2", ctypes.c_int32),
    ]


class LayerBinding(ctypes.Structure):
    _fields_ = [
        ("W", ctypes.c_void_p),
        ("b", ctypes.c_void_p),
        ("dW", ctypes.c_void_p),
        ("db", ctypes.c_void_p),
        ("out_conv", ctypes.c_void_p),
        ("out_pool", ctypes.c_void_p),
        ("argmax", ctypes.c_void_p),
        ("dx", ctypes.c_void_p),
        ("d_conv", ctypes.c_void_p),
        ("x_cache", ctypes.c_void_p),
        ("conv_act_cache", ctypes.c_void_p),
        ("ms_w", ctypes.c_void_p),
        ("vs_w", ctypes.c_void_p),
        ("ms_b", ctypes.c_void_p),
        ("vs_b", ctypes.c_void_p),
        ("w_count", ctypes.c_int64),
        ("b_count", ctypes.c_int64),
        ("C_in", ctypes.c_int64),
        ("C_out", ctypes.c_int64),
        ("H", ctypes.c_int64),
        ("W_in", ctypes.c_int64),
        ("W_stride", ctypes.c_int64),
        ("k_h", ctypes.c_int64),
        ("k_w", ctypes.c_int64),
        ("conv_stride", ctypes.c_int64),
        ("conv_pad", ctypes.c_int64),
        ("pool_size", ctypes.c_int64),
        ("pool_stride", ctypes.c_int64),
        ("pool_out_h", ctypes.c_int64),
        ("pool_out_w", ctypes.c_int64),
        ("conv_out_w_stride", ctypes.c_int64),
    ]


class DenseBinding(ctypes.Structure):
    _fields_ = [
        ("W", ctypes.c_void_p),
        ("b", ctypes.c_void_p),
        ("dW", ctypes.c_void_p),
        ("db", ctypes.c_void_p),
        ("z", ctypes.c_void_p),
        ("output", ctypes.c_void_p),
        ("delta", ctypes.c_void_p),
        ("input_cache", ctypes.c_void_p),
        ("dx_flat", ctypes.c_void_p),
        ("ms_w", ctypes.c_void_p),
        ("vs_w", ctypes.c_void_p),
        ("ms_b", ctypes.c_void_p),
        ("vs_b", ctypes.c_void_p),
        ("fan_in", ctypes.c_int64),
        ("fan_out", ctypes.c_int64),
    ]


class AdamBinding(ctypes.Structure):
    _fields_ = [
        ("beta1", ctypes.c_float),
        ("beta2", ctypes.c_float),
        ("eps", ctypes.c_float),
        ("t", ctypes.c_int32),
    ]


class ContractExecCtx(ctypes.Structure):
    _fields_ = [
        ("N", ctypes.c_int64),
        ("lr", ctypes.c_float),
        ("lam_l2", ctypes.c_float),
        ("max_norm", ctypes.c_float),
        ("skip_adam", ctypes.c_int32),
        ("X", ctypes.c_void_p),
        ("y", ctypes.c_void_p),
        ("act", ctypes.c_void_p),
        ("flat_dim", ctypes.c_int64),
        ("num_layers", ctypes.c_int32),
        ("layers", LayerBinding * 8),
        ("dense", DenseBinding),
        ("adam", AdamBinding),
        ("loss_out", ctypes.c_void_p),
    ]



def _ptr(arr: np.ndarray) -> int:
    return int(arr.ctypes.data)


def _bind_runner(lib) -> None:
    lib.run_contract_training_step.restype = ctypes.c_int32
    lib.run_contract_training_step.argtypes = [
        ctypes.POINTER(ContractOpRow),
        ctypes.c_int32,
        ctypes.POINTER(ContractExecCtx),
    ]
    if hasattr(lib, "submit_contract_training_step"):
        lib.submit_contract_training_step.restype = ctypes.c_int32
        lib.submit_contract_training_step.argtypes = [
            ctypes.POINTER(ContractOpRow),
            ctypes.c_int32,
            ctypes.POINTER(ContractExecCtx),
            ctypes.c_int64,
        ]
        lib.try_reap_contract_completion.restype = ctypes.c_int32
        lib.try_reap_contract_completion.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.contract_async_in_flight.restype = ctypes.c_int32
        lib.contract_async_in_flight.argtypes = []
        lib.contract_async_shutdown.restype = None
        lib.contract_async_shutdown.argtypes = []
        if hasattr(lib, "contract_register_completion_callback"):
            lib.contract_register_completion_callback.restype = None
            lib.contract_register_completion_callback.argtypes = [
                ctypes.CFUNCTYPE(None, ctypes.c_int64, ctypes.c_int32),
            ]


@dataclass
class ContractBuffers:
    dense_z: np.ndarray
    dense_output: np.ndarray
    dense_delta: np.ndarray
    dense_dW: np.ndarray
    dense_db: np.ndarray
    dense_dx_flat: np.ndarray


@dataclass
class _SubmittedStep:
    m: int
    dtype: Any
    y: np.ndarray
    bound: list[tuple[int, Any]]
    apply_adam: bool


def shutdown_contract_async() -> None:
    """Stop native contract worker thread (call at end of fit / tests)."""
    lib = _load_conv_dll()
    if lib is not None and hasattr(lib, "contract_async_shutdown"):
        lib.contract_async_shutdown()


class ContractRuntime:
    """Binds a compiled ContractList to live CNN weights + scratch arena."""

    def __init__(self, model: Any, contract: ContractList):
        self.model = model
        self.contract = contract
        self._lib = _load_conv_dll()
        if self._lib is None or not hasattr(self._lib, "run_contract_training_step"):
            raise RuntimeError("conv_kernels.dll missing run_contract_training_step — rebuild native")
        _bind_runner(self._lib)

        if len(model._dense_w_indices) != 1:
            raise ValueError("Contract path requires exactly one dense head layer")

        self._ops = (ContractOpRow * contract.op_count)(*self._build_op_rows())
        self._loss_scalar = np.zeros(1, dtype=np.float32)
        self._buffers: ContractBuffers | None = None
        self._bound_layers: list[tuple[int, Any]] = []
        self._ctx = ContractExecCtx()
        self._bindings_ready = False
        self._conv_bindings_ready = False
        self._dense_bindings_ready = False
        self._input_logical_w: int | None = getattr(model, "input_logical_w", None)
        self._async_enabled = hasattr(self._lib, "submit_contract_training_step")
        self._pending_token: int | None = None
        self._engine_driven = False
        self._submitted: _SubmittedStep | None = None
        self._completed: tuple[float, list[np.ndarray], list[np.ndarray], int] | None = None
        self._completion_event = threading.Event()
        self._subscriber_fn: Callable[[], None] | None = None
        self._capacity_fn: Callable[[], None] | None = None
        self._completion_cb_ref = None
        # Contract runner init registers the native → Python completion callback once.
        if self._async_enabled and hasattr(self._lib, "contract_register_completion_callback"):
            completion_fn_type = ctypes.CFUNCTYPE(None, ctypes.c_int64, ctypes.c_int32)
            runtime = self

            @completion_fn_type
            def _native_completion_cb(step_token: int, status: int) -> None:
                runtime._on_native_complete(int(step_token), int(status))

            self._completion_cb_ref = _native_completion_cb
            self._lib.contract_register_completion_callback(self._completion_cb_ref)

    def _bindings_still_valid(self, m: int) -> bool:
        if not self._bindings_ready:
            return False
        cap = getattr(self.model, "_train_batch_cap", 0) or m
        return (
            self._buffers is not None
            and m <= self._buffers.dense_z.shape[0]
            and m <= cap
        )

    
    def _bind_conv_layers(self, X: np.ndarray, m: int) -> tuple[ContractExecCtx, list[tuple[int, Any]]]:
        """Wire conv blocks into ctx (once, or after batch cap bump)."""
        if self._conv_bindings_ready and self._bindings_still_valid(m):
            return self._ctx, self._bound_layers

        from src.spatial_layers import ConvBlock

        arena = self.model.scratch_arena
        ctx = self._ctx
        ctx.lam_l2 = float(self.model.lam_l2)
        ctx.max_norm = float(self.model.max_norm)
        ctx.adam.beta1 = float(self.model.optimizer.beta1)
        ctx.adam.beta2 = float(self.model.optimizer.beta2)
        ctx.adam.eps = float(self.model.optimizer.eps)

        w_logical = self._input_logical_w
        if w_logical is None:
            if X.ndim == 4 and X.shape[3] == 32:
                w_logical = 28
            else:
                w_logical = X.shape[3] if X.ndim == 4 else 28
            self._input_logical_w = w_logical

        opt = self.model.optimizer
        if not opt._setup_done:
            opt.setup(self.model.weights, self.model.biases)

        bound: list[tuple[int, Any]] = []
        max_layer_idx = -1
        cur_c, cur_h, cur_w_log, cur_w_stride = X.shape[1], X.shape[2], w_logical, X.shape[3]

        for li, layer in enumerate(self.model.layers):
            if not isinstance(layer, ConvBlock):
                continue
            if li >= 8:
                raise ValueError("Contract path supports at most 8 ConvBlock layers")

            w_idx = self.model._layer_param_idx[li]
            W = self.model.weights[w_idx]
            b = self.model.biases[w_idx].reshape(-1)

            scratch = arena.ensure_conv_block_train(
                li,
                out_channels=layer.out_channels,
                in_channels=layer.in_channels,
                k_h=layer.k_h,
                k_w=layer.k_w,
                conv_stride=layer.conv_stride,
                conv_pad=layer.conv_pad,
                pool_size=layer.pool_size,
                pool_stride=layer.pool_stride,
                N=m,
                C=cur_c,
                H=cur_h,
                W_stride=cur_w_stride,
                W_logical=cur_w_log,
                dtype=X.dtype,
            )

            if not hasattr(layer, "_contract_dW") or layer._contract_dW.shape != W.shape:
                layer._contract_dW = np.zeros_like(W)
                layer._contract_db = np.zeros_like(self.model.biases[w_idx])
            layer.dW = layer._contract_dW
            layer.db = layer._contract_db
            db_flat = layer.db.reshape(-1)

            lb = ctx.layers[li]
            lb.W = _ptr(W)
            lb.b = _ptr(b)
            lb.dW = _ptr(layer.dW)
            lb.db = _ptr(db_flat)
            lb.out_conv = _ptr(scratch.out_conv_buffer)
            lb.out_pool = _ptr(scratch.out_pool_buffer)
            lb.argmax = _ptr(scratch.argmax_buffer)
            lb.dx = _ptr(scratch.dx_buffer)
            lb.d_conv = _ptr(scratch.d_conv_buffer)
            lb.w_count = int(W.size)
            lb.b_count = int(b.size)
            lb.C_in = layer.in_channels
            lb.C_out = layer.out_channels
            lb.H = cur_h
            lb.W_in = cur_w_log
            lb.W_stride = cur_w_stride
            lb.k_h = layer.k_h
            lb.k_w = layer.k_w
            lb.conv_stride = layer.conv_stride
            lb.conv_pad = layer.conv_pad
            lb.pool_size = layer.pool_size
            lb.pool_stride = layer.pool_stride
            lb.ms_w = _ptr(opt.ms_w[w_idx])
            lb.vs_w = _ptr(opt.vs_w[w_idx])
            lb.ms_b = _ptr(opt.ms_b[w_idx].reshape(-1))
            lb.vs_b = _ptr(opt.vs_b[w_idx].reshape(-1))

            bound.append((w_idx, layer))
            max_layer_idx = li
            cur_c, cur_h, cur_w_log, cur_w_stride = _conv_block_output_geom(
                cur_c, cur_h, cur_w_log, layer
            )

        ctx.num_layers = max_layer_idx + 1 if max_layer_idx >= 0 else 0
        self._bound_layers = bound
        self._conv_bindings_ready = True
        self._bindings_ready = self._conv_bindings_ready and self._dense_bindings_ready
        return self._ctx, self._bound_layers

    
    def _bind_dense(self, ctx: ContractExecCtx, m: int, dtype) -> None:
        """Wire dense head into ctx (once, or after batch cap bump)."""
        if self._dense_bindings_ready and self._bindings_still_valid(m):
            return

        bufs = self._ensure_buffers(m, dtype)
        w_idx = self.model._dense_w_indices[0]
        Wd = self.model.weights[w_idx]
        bd = self.model.biases[w_idx].reshape(-1)
        opt = self.model.optimizer

        d = ctx.dense
        d.W = _ptr(Wd)
        d.b = _ptr(bd)
        d.dW = _ptr(bufs.dense_dW)
        d.db = _ptr(bufs.dense_db)
        d.z = _ptr(bufs.dense_z)
        d.output = _ptr(bufs.dense_output)
        d.delta = _ptr(bufs.dense_delta)
        d.dx_flat = _ptr(bufs.dense_dx_flat)
        d.fan_in = Wd.shape[0]
        d.fan_out = Wd.shape[1]
        d.ms_w = _ptr(opt.ms_w[w_idx])
        d.vs_w = _ptr(opt.vs_w[w_idx])
        d.ms_b = _ptr(opt.ms_b[w_idx].reshape(-1))
        d.vs_b = _ptr(opt.vs_b[w_idx].reshape(-1))

        ctx.loss_out = _ptr(self._loss_scalar)
        self._dense_bindings_ready = True
        self._bindings_ready = self._conv_bindings_ready and self._dense_bindings_ready

    
    def _ensure_static_bindings(self, X: np.ndarray, m: int) -> None:
        """Ensure conv + dense static bindings are warm."""
        if self._bindings_still_valid(m):
            return
        self._conv_bindings_ready = False
        self._dense_bindings_ready = False
        self._bindings_ready = False
        self._bind_conv_layers(X, m)
        self._bind_dense(self._ctx, m, X.dtype)

    
    def _refresh_step_bindings(
        self,
        X: np.ndarray,
        y: np.ndarray,
        m: int,
        lr: float,
        *,
        apply_adam: bool,
    ) -> None:
        """Per-step: batch pointers, hyperparams, zero grad outputs."""
        ctx = self._ctx
        ctx.N = m
        ctx.lr = float(lr)
        ctx.skip_adam = 0 if apply_adam else 1
        ctx.X = _ptr(X)
        ctx.y = _ptr(y)
        ctx.adam.t = int(self.model.optimizer.t)

        if self._buffers is not None:
            self._buffers.dense_dW.fill(0.0)
            self._buffers.dense_db.fill(0.0)
        for _, layer in self._bound_layers:
            layer.dW.fill(0.0)
            layer.db.fill(0.0)

    def _build_op_rows(self) -> list[ContractOpRow]:
        rows: list[ContractOpRow] = []
        for op in self.contract.ops:
            rows.append(
                ContractOpRow(
                    int(op.opcode),
                    op.layer_idx,
                    op.param_idx,
                    op.flags,
                    op.i0,
                    op.i1,
                    op.i2,
                )
            )
        return rows

    
    def _ensure_buffers(self, m: int, dtype) -> ContractBuffers:
        if self._buffers is not None:
            return self._buffers

        w_idx = self.model._dense_w_indices[0]
        W = self.model.weights[w_idx]
        fan_in, fan_out = W.shape
        cap = max(m, getattr(self.model, "_train_batch_cap", 0) or m)

        self._buffers = ContractBuffers(
            dense_z=np.empty((cap, fan_out), dtype=dtype),
            dense_output=np.empty((cap, fan_out), dtype=dtype),
            dense_delta=np.empty((cap, fan_out), dtype=dtype),
            dense_dW=np.zeros((fan_in, fan_out), dtype=dtype),
            dense_db=np.zeros((fan_out,), dtype=dtype),
            dense_dx_flat=np.empty((cap, fan_in), dtype=dtype),
        )
        return self._buffers

    
    def set_engine_driven(self, enabled: bool = True) -> None:
        """When True, submit/reap are driven by TrainingEngine (subscriber)."""
        self._engine_driven = bool(enabled)

    def subscribe_completion(self, on_complete: Callable[[], None]) -> None:
        """Register on-event handler; invoked when native step completes."""
        self._subscriber_fn = on_complete

    def subscribe_capacity(self, on_capacity: Callable[[], None]) -> None:
        """Register capacity handler; invoked when native slot is free again."""
        self._capacity_fn = on_capacity

    def has_completed(self) -> bool:
        return self._completed is not None

    def waiting_on_native_worker(self) -> bool:
        """True while native contract runs and callback has not fired yet."""
        if not self._async_enabled or self._completed is not None:
            return False
        if self._submitted is not None:
            return True
        return bool(self._lib.contract_async_in_flight())

    def wait_for_completion(self, timeout: float | None = None) -> bool:
        """Test helper: block until native callback posts _completed."""
        if self._completed is not None:
            return True
        if timeout is None:
            return self._completion_event.wait()
        if timeout <= 0:
            return self._completion_event.is_set()
        return self._completion_event.wait(timeout)

    def native_in_flight(self) -> bool:
        if not self._async_enabled:
            return False
        if self._submitted is not None or self._completed is not None:
            return True
        return bool(self._lib.contract_async_in_flight())

    def completion_signaled(self) -> bool:
        """True when native callback has posted _completed (non-blocking check)."""
        return self._completed is not None

    def is_busy(self) -> bool:
        """Single in-flight slot occupied — caller should push back BUSY."""
        return self.native_in_flight()

    def _publish_completion(self) -> None:
        self._completion_event.set()
        if self._subscriber_fn is not None:
            self._subscriber_fn()
        if self._capacity_fn is not None:
            self._capacity_fn()

    def try_submit_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        *,
        apply_adam: bool = False,
        step_token: int | None = None,
    ) -> bool:
        """Submit native contract if idle. Returns False if BUSY."""
        if not (self._async_enabled and self._engine_driven):
            raise RuntimeError("try_submit_step is only valid in engine-driven async mode")
        if self.is_busy():
            return False

        X = np.ascontiguousarray(X)
        y = np.ascontiguousarray(y)
        m = int(X.shape[0])

        ctx, bound = self._bind_conv_layers(X, m)
        self._bind_dense(ctx, m, X.dtype)
        self._refresh_step_bindings(X, y, m, lr, apply_adam=apply_adam)

        token = int(step_token if step_token is not None else self.model.optimizer.t + 1)
        self._completion_event.clear()
        self._submit_native(ctx, token)
        self._submitted = _SubmittedStep(m, X.dtype, y, bound, apply_adam)
        return True

    def submit_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        *,
        apply_adam: bool = False,
        step_token: int | None = None,
    ) -> None:
        """Submit native contract; raise if BUSY."""
        if not self.try_submit_step(
            X, y, lr, apply_adam=apply_adam, step_token=step_token
        ):
            raise RuntimeError("submit_step: native BUSY")

    def try_reap_step(
        self,
    ) -> tuple[float, list[np.ndarray], list[np.ndarray], int] | None:
        """Non-blocking: return post-contract result if native callback finished."""
        if self._completed is None:
            if self._submitted is None:
                return None
            if not self._try_reap_native():
                return None
            submitted = self._submitted
            self._submitted = None
            self._completed = self._finish_submitted(submitted)
        result = self._completed
        self._completed = None
        return result

    def _on_native_complete(self, step_token: int, status: int) -> None:
        """Native worker callback: pack result and publish to subscribers."""
        if self._submitted is None:
            raise RuntimeError(
                f"native completion callback token={step_token} with no submitted step"
            )
        if status != 0:
            raise RuntimeError(
                f"async contract step token={step_token} failed with status {status}"
            )
        submitted = self._submitted
        self._submitted = None
        self._pending_token = None
        self._completed = self._finish_submitted(submitted)
        self._publish_completion()

    def _finish_submitted(
        self, submitted: _SubmittedStep
    ) -> tuple[float, list[np.ndarray], list[np.ndarray], int]:
        loss, gw, gb = self._collect_grads(
            submitted.m,
            submitted.dtype,
            submitted.y,
            submitted.bound,
            apply_adam=submitted.apply_adam,
            ctx=self._ctx,
        )
        return loss, gw, gb, submitted.m

    def run_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        *,
        apply_adam: bool = False,
        step_token: int | None = None,
        tick_fn: Callable[[], None] | None = None,
    ) -> tuple[float, list[np.ndarray], list[np.ndarray], int]:
        """Sync one-shot for unit tests (blocking native invoke)."""
        del tick_fn
        X = np.ascontiguousarray(X)
        y = np.ascontiguousarray(y)
        m = int(X.shape[0])

        ctx, bound = self._bind_conv_layers(X, m)
        self._bind_dense(ctx, m, X.dtype)
        self._refresh_step_bindings(X, y, m, lr, apply_adam=apply_adam)

        self._invoke_native_sync(ctx)

        loss, grad_weights, grad_biases = self._collect_grads(
            m, X.dtype, y, bound, apply_adam=apply_adam, ctx=ctx
        )
        return loss, grad_weights, grad_biases, m

    def _invoke_native_sync(self, ctx: ContractExecCtx) -> None:
        status = self._lib.run_contract_training_step(
            ctypes.cast(self._ops, ctypes.POINTER(ContractOpRow)),
            ctypes.c_int32(self.contract.op_count),
            ctypes.byref(ctx),
        )
        if status != 0:
            raise RuntimeError(f"run_contract_training_step failed with status {status}")

    def _submit_native(self, ctx: ContractExecCtx, step_token: int) -> None:
        status = self._lib.submit_contract_training_step(
            ctypes.cast(self._ops, ctypes.POINTER(ContractOpRow)),
            ctypes.c_int32(self.contract.op_count),
            ctypes.byref(ctx),
            ctypes.c_int64(step_token),
        )
        if status == -2:
            raise RuntimeError("submit_contract_training_step: native BUSY")
        if status == -3:
            raise RuntimeError("submit_contract_training_step: reap prior completion first")
        if status != 0:
            raise RuntimeError(f"submit_contract_training_step failed with status {status}")
        self._pending_token = step_token

    def _try_reap_native(self) -> bool:
        if not self._async_enabled:
            return True
        out_token = ctypes.c_int64()
        out_status = ctypes.c_int32()
        rc = self._lib.try_reap_contract_completion(
            ctypes.byref(out_token),
            ctypes.byref(out_status),
        )
        if rc == 0:
            return False
        if rc != 1:
            raise RuntimeError(f"try_reap_contract_completion failed with status {rc}")
        if out_status.value != 0:
            raise RuntimeError(
                f"async contract step token={out_token.value} failed with status {out_status.value}"
            )
        self._pending_token = None
        return True

    
    def _invoke_native(self, ctx: ContractExecCtx) -> None:
        """Sync invoke (legacy); prefer submit + wait for overlap."""
        self._invoke_native_sync(ctx)

    
    def _collect_grads(
        self,
        m: int,
        dtype,
        y: np.ndarray,
        bound: list[tuple[int, Any]],
        *,
        apply_adam: bool,
        ctx: ContractExecCtx,
    ) -> tuple[float, list[np.ndarray | None], list[np.ndarray | None]]:
        if apply_adam:
            self.model.optimizer.t = int(ctx.adam.t)

        bufs = self._ensure_buffers(m, dtype)
        w_idx_dense = self.model._dense_w_indices[0]
        grad_weights: list[np.ndarray | None] = [None] * len(self.model.weights)
        grad_biases: list[np.ndarray | None] = [None] * len(self.model.biases)

        grad_weights[w_idx_dense] = np.copy(bufs.dense_dW)
        grad_biases[w_idx_dense] = np.copy(bufs.dense_db).reshape(1, -1)

        for w_idx, layer in bound:
            grad_weights[w_idx] = np.copy(layer.dW)
            grad_biases[w_idx] = np.copy(layer.db)

        loss = float(self._loss_scalar[0])
        # Native already computed CE in OP_DENSE_FWD; only add L1/L2 reporting terms.
        if self.model.lam_l2 > 0.0:
            l2_sum = sum(float(np.sum(w * w)) for w in self.model.weights)
            loss += (self.model.lam_l2 / (2.0 * m)) * l2_sum
        if self.model.lam_l1 > 0.0:
            l1_sum = sum(float(np.sum(np.abs(w))) for w in self.model.weights)
            loss += (self.model.lam_l1 / m) * l1_sum
        return loss, grad_weights, grad_biases
