# src/contract_runtime.py
"""Phase F: ctypes bindings for native contract-list executor."""
from __future__ import annotations

import ctypes
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from src.contract import ContractList, ContractOp
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
        ("W_next", ctypes.c_void_p),
        ("b_next", ctypes.c_void_p),
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
        ("ms_w_next", ctypes.c_void_p),
        ("vs_w_next", ctypes.c_void_p),
        ("ms_b_next", ctypes.c_void_p),
        ("vs_b_next", ctypes.c_void_p),
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
        ("d_conv_prezeroed", ctypes.c_int64),
        ("dx_prezeroed", ctypes.c_int64),
        ("dw_prezeroed", ctypes.c_int64),
    ]


class DenseBinding(ctypes.Structure):
    _fields_ = [
        ("W", ctypes.c_void_p),
        ("b", ctypes.c_void_p),
        ("W_next", ctypes.c_void_p),
        ("b_next", ctypes.c_void_p),
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
        ("ms_w_next", ctypes.c_void_p),
        ("vs_w_next", ctypes.c_void_p),
        ("ms_b_next", ctypes.c_void_p),
        ("vs_b_next", ctypes.c_void_p),
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


class MhsaLayerBind(ctypes.Structure):
    """Per Pre-LN block — mirrors native MhsaLayerBind."""

    _fields_ = [
        ("W_qkv", ctypes.c_void_p),
        ("b_qkv", ctypes.c_void_p),
        ("W_o", ctypes.c_void_p),
        ("b_o", ctypes.c_void_p),
        ("W_ff1", ctypes.c_void_p),
        ("b_ff1", ctypes.c_void_p),
        ("W_ff2", ctypes.c_void_p),
        ("b_ff2", ctypes.c_void_p),
        ("ln1_gamma", ctypes.c_void_p),
        ("ln1_beta", ctypes.c_void_p),
        ("ln2_gamma", ctypes.c_void_p),
        ("ln2_beta", ctypes.c_void_p),
        ("dW_qkv", ctypes.c_void_p),
        ("db_qkv", ctypes.c_void_p),
        ("dW_o", ctypes.c_void_p),
        ("db_o", ctypes.c_void_p),
        ("dW_ff1", ctypes.c_void_p),
        ("db_ff1", ctypes.c_void_p),
        ("dW_ff2", ctypes.c_void_p),
        ("db_ff2", ctypes.c_void_p),
        ("d_ln1_gamma", ctypes.c_void_p),
        ("d_ln1_beta", ctypes.c_void_p),
        ("d_ln2_gamma", ctypes.c_void_p),
        ("d_ln2_beta", ctypes.c_void_p),
        ("ms_W_qkv", ctypes.c_void_p),
        ("vs_W_qkv", ctypes.c_void_p),
        ("ms_b_qkv", ctypes.c_void_p),
        ("vs_b_qkv", ctypes.c_void_p),
        ("ms_W_o", ctypes.c_void_p),
        ("vs_W_o", ctypes.c_void_p),
        ("ms_b_o", ctypes.c_void_p),
        ("vs_b_o", ctypes.c_void_p),
        ("ms_W_ff1", ctypes.c_void_p),
        ("vs_W_ff1", ctypes.c_void_p),
        ("ms_b_ff1", ctypes.c_void_p),
        ("vs_b_ff1", ctypes.c_void_p),
        ("ms_W_ff2", ctypes.c_void_p),
        ("vs_W_ff2", ctypes.c_void_p),
        ("ms_b_ff2", ctypes.c_void_p),
        ("vs_b_ff2", ctypes.c_void_p),
        ("ms_ln1_g", ctypes.c_void_p),
        ("vs_ln1_g", ctypes.c_void_p),
        ("ms_ln1_b", ctypes.c_void_p),
        ("vs_ln1_b", ctypes.c_void_p),
        ("ms_ln2_g", ctypes.c_void_p),
        ("vs_ln2_g", ctypes.c_void_p),
        ("ms_ln2_b", ctypes.c_void_p),
        ("vs_ln2_b", ctypes.c_void_p),
        ("qkv", ctypes.c_void_p),
        ("scores", ctypes.c_void_p),
        ("attn_out", ctypes.c_void_p),
        ("ln1_out", ctypes.c_void_p),
        ("h1", ctypes.c_void_p),
        ("ln2_out", ctypes.c_void_p),
        ("ffn_pre", ctypes.c_void_p),
        ("ffn_h", ctypes.c_void_p),
        ("O", ctypes.c_void_p),
        ("q_pack", ctypes.c_void_p),
        ("k_pack", ctypes.c_void_p),
        ("v_pack", ctypes.c_void_p),
        ("d_scores", ctypes.c_void_p),
        ("W_qkv_next", ctypes.c_void_p),
        ("b_qkv_next", ctypes.c_void_p),
        ("W_o_next", ctypes.c_void_p),
        ("b_o_next", ctypes.c_void_p),
        ("W_ff1_next", ctypes.c_void_p),
        ("b_ff1_next", ctypes.c_void_p),
        ("W_ff2_next", ctypes.c_void_p),
        ("b_ff2_next", ctypes.c_void_p),
        ("ln1_gamma_next", ctypes.c_void_p),
        ("ln1_beta_next", ctypes.c_void_p),
        ("ln2_gamma_next", ctypes.c_void_p),
        ("ln2_beta_next", ctypes.c_void_p),
        ("ms_W_qkv_next", ctypes.c_void_p),
        ("vs_W_qkv_next", ctypes.c_void_p),
        ("ms_b_qkv_next", ctypes.c_void_p),
        ("vs_b_qkv_next", ctypes.c_void_p),
        ("ms_W_o_next", ctypes.c_void_p),
        ("vs_W_o_next", ctypes.c_void_p),
        ("ms_b_o_next", ctypes.c_void_p),
        ("vs_b_o_next", ctypes.c_void_p),
        ("ms_W_ff1_next", ctypes.c_void_p),
        ("vs_W_ff1_next", ctypes.c_void_p),
        ("ms_b_ff1_next", ctypes.c_void_p),
        ("vs_b_ff1_next", ctypes.c_void_p),
        ("ms_W_ff2_next", ctypes.c_void_p),
        ("vs_W_ff2_next", ctypes.c_void_p),
        ("ms_b_ff2_next", ctypes.c_void_p),
        ("vs_b_ff2_next", ctypes.c_void_p),
        ("ms_ln1_g_next", ctypes.c_void_p),
        ("vs_ln1_g_next", ctypes.c_void_p),
        ("ms_ln1_b_next", ctypes.c_void_p),
        ("vs_ln1_b_next", ctypes.c_void_p),
        ("ms_ln2_g_next", ctypes.c_void_p),
        ("vs_ln2_g_next", ctypes.c_void_p),
        ("ms_ln2_b_next", ctypes.c_void_p),
        ("vs_ln2_b_next", ctypes.c_void_p),
    ]


class MhsaBinding(ctypes.Structure):
    """Composed MHSA bind — mirrors native mhsa_kernels.h."""

    _fields_ = [
        ("B", ctypes.c_int64),
        ("T", ctypes.c_int64),
        ("D", ctypes.c_int64),
        ("H", ctypes.c_int64),
        ("d_head", ctypes.c_int64),
        ("action_dim", ctypes.c_int64),
        ("ffn_hidden", ctypes.c_int64),
        ("num_layers", ctypes.c_int64),
        ("layers", MhsaLayerBind * 8),
        ("W_act", ctypes.c_void_p),
        ("b_act", ctypes.c_void_p),
        ("dW_act", ctypes.c_void_p),
        ("db_act", ctypes.c_void_p),
        ("ms_W_act", ctypes.c_void_p),
        ("vs_W_act", ctypes.c_void_p),
        ("ms_b_act", ctypes.c_void_p),
        ("vs_b_act", ctypes.c_void_p),
        ("pos", ctypes.c_void_p),
        ("d_pos", ctypes.c_void_p),
        ("ms_pos", ctypes.c_void_p),
        ("vs_pos", ctypes.c_void_p),
        ("max_seq_len", ctypes.c_int64),
        ("scratch", ctypes.c_void_p),
        ("actions", ctypes.c_void_p),
        ("dO", ctypes.c_void_p),
        ("dX", ctypes.c_void_p),
        ("d_qkv", ctypes.c_void_p),
        ("d_stream", ctypes.c_void_p),
        ("y", ctypes.c_void_p),
        ("loss_out", ctypes.c_void_p),
        ("O", ctypes.c_void_p),
        ("W_in", ctypes.c_void_p),
        ("b_in", ctypes.c_void_p),
        ("dW_in", ctypes.c_void_p),
        ("db_in", ctypes.c_void_p),
        ("ms_W_in", ctypes.c_void_p),
        ("vs_W_in", ctypes.c_void_p),
        ("ms_b_in", ctypes.c_void_p),
        ("vs_b_in", ctypes.c_void_p),
        ("W_in_next", ctypes.c_void_p),
        ("b_in_next", ctypes.c_void_p),
        ("ms_W_in_next", ctypes.c_void_p),
        ("vs_W_in_next", ctypes.c_void_p),
        ("ms_b_in_next", ctypes.c_void_p),
        ("vs_b_in_next", ctypes.c_void_p),
        ("X_emb", ctypes.c_void_p),
        ("W_act_next", ctypes.c_void_p),
        ("b_act_next", ctypes.c_void_p),
        ("ms_W_act_next", ctypes.c_void_p),
        ("vs_W_act_next", ctypes.c_void_p),
        ("ms_b_act_next", ctypes.c_void_p),
        ("vs_b_act_next", ctypes.c_void_p),
        ("pos_next", ctypes.c_void_p),
        ("ms_pos_next", ctypes.c_void_p),
        ("vs_pos_next", ctypes.c_void_p),
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
        ("num_dense", ctypes.c_int32),
        ("dense", DenseBinding * 8),
        ("adam", AdamBinding),
        ("loss_out", ctypes.c_void_p),
        ("has_mhsa", ctypes.c_int32),
        ("mhsa", MhsaBinding),
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
        if hasattr(lib, "wait_contract_completion"):
            lib.wait_contract_completion.restype = ctypes.c_int32
            lib.wait_contract_completion.argtypes = [
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_int64,
            ]
        if hasattr(lib, "contract_async_debug_snapshot"):
            lib.contract_async_debug_snapshot.restype = ctypes.c_int32
            lib.contract_async_debug_snapshot.argtypes = [
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_int64),
            ]
        lib.contract_async_in_flight.restype = ctypes.c_int32
        lib.contract_async_in_flight.argtypes = []
        lib.contract_async_shutdown.restype = None
        lib.contract_async_shutdown.argtypes = []
        # Optional native API: must stay NULL. Python must never register a
        # worker→Python completion callback (GIL deadlock with the trainer).
        if hasattr(lib, "contract_register_completion_callback"):
            lib.contract_register_completion_callback.restype = None
            lib.contract_register_completion_callback.argtypes = [ctypes.c_void_p]
    if hasattr(lib, "stage_conv_x_pad"):
        # Main-thread setup: build layer 0's padded input for the next step
        # while the worker owns the OMP team for the current one.
        lib.stage_conv_x_pad.restype = ctypes.c_int32
        lib.stage_conv_x_pad.argtypes = [
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        lib.invalidate_conv_x_pad_stage.restype = None
        lib.invalidate_conv_x_pad_stage.argtypes = [ctypes.c_int32]
    if hasattr(lib, "stage_dx_cin_blocked_wt"):
        # Main-thread setup: rebuild the transposed-W buffer for the
        # cin-blocked backward-dX kernel right before submit, once Adam has
        # applied the weights the upcoming job will read.
        lib.stage_dx_cin_blocked_wt.restype = ctypes.c_int32
        lib.stage_dx_cin_blocked_wt.argtypes = [
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        lib.invalidate_dx_cin_blocked_wt_stage.restype = None
        lib.invalidate_dx_cin_blocked_wt_stage.argtypes = [ctypes.c_int32]
    if hasattr(lib, "stage_brgemm_dw_x_pack"):
        # Main-thread: pack BRGEMM dW x panels for layer 0 when Cin%8==0 while
        # the worker owns the OMP team. L1's x is requested after L0 fwd; main
        # drains via service_brgemm_dw_x_pack_requests during wait.
        lib.stage_brgemm_dw_x_pack.restype = ctypes.c_int32
        lib.stage_brgemm_dw_x_pack.argtypes = [
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        lib.invalidate_brgemm_dw_x_pack.restype = None
        lib.invalidate_brgemm_dw_x_pack.argtypes = [ctypes.c_int32]
    if hasattr(lib, "service_brgemm_dw_x_pack_requests"):
        lib.service_brgemm_dw_x_pack_requests.restype = ctypes.c_int32
        lib.service_brgemm_dw_x_pack_requests.argtypes = []
    if hasattr(lib, "set_contract_async_overlap"):
        lib.set_contract_async_overlap.restype = None
        lib.set_contract_async_overlap.argtypes = [ctypes.c_int32]
    if hasattr(lib, "native_tenant_create"):
        lib.native_tenant_create.restype = ctypes.c_int32
        lib.native_tenant_create.argtypes = []
        lib.native_tenant_set_current.restype = None
        lib.native_tenant_set_current.argtypes = [ctypes.c_int32]
        lib.native_tenant_current.restype = ctypes.c_int32
        lib.native_tenant_current.argtypes = []
        lib.native_tenant_invalidate_all.restype = None
        lib.native_tenant_invalidate_all.argtypes = [ctypes.c_int32]
        lib.native_tenant_release.restype = None
        lib.native_tenant_release.argtypes = [ctypes.c_int32]
    if hasattr(lib, "contract_async_shutdown_tenant"):
        lib.contract_async_shutdown_tenant.restype = None
        lib.contract_async_shutdown_tenant.argtypes = [ctypes.c_int32]


@dataclass
class DenseLayerBuffers:
    z: np.ndarray
    output: np.ndarray
    delta: np.ndarray
    dW: np.ndarray
    db: np.ndarray
    dx_flat: np.ndarray


@dataclass
class ContractBuffers:
    dense_layers: list[DenseLayerBuffers]
    batch_cap: int = 0


@dataclass
class _ParameterBank:
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    ms_w: list[np.ndarray]
    vs_w: list[np.ndarray]
    ms_b: list[np.ndarray]
    vs_b: list[np.ndarray]


@dataclass
class _MhsaParameterBank:
    """Dual-bank MHSA params + Adam moments (weights/biases/LN/pos/W_in)."""

    weights: list[np.ndarray]
    biases: list[np.ndarray]
    ln1_gamma: list[np.ndarray]
    ln1_beta: list[np.ndarray]
    ln2_gamma: list[np.ndarray]
    ln2_beta: list[np.ndarray]
    ms_w: list[np.ndarray]
    vs_w: list[np.ndarray]
    ms_b: list[np.ndarray]
    vs_b: list[np.ndarray]
    ms_g: list[np.ndarray]
    vs_g: list[np.ndarray]
    ms_beta: list[np.ndarray]
    vs_beta: list[np.ndarray]
    pos_embed: np.ndarray | None = None
    ms_pos: np.ndarray | None = None
    vs_pos: np.ndarray | None = None
    W_in: np.ndarray | None = None
    b_in: np.ndarray | None = None
    ms_W_in: np.ndarray | None = None
    vs_W_in: np.ndarray | None = None
    ms_b_in: np.ndarray | None = None
    vs_b_in: np.ndarray | None = None


@dataclass
class _ExecutionSlot:
    ctx: ContractExecCtx
    buffers: ContractBuffers
    conv_grads: list[tuple[int, np.ndarray, np.ndarray]]
    loss_scalar: np.ndarray
    owners: list[Any]
    # Zeroed by the main thread during prepare; native skips its own memset.
    d_conv_buffers: list[np.ndarray]
    dx_buffers: list[np.ndarray]


@dataclass
class _EvalSlot:
    ctx: ContractExecCtx
    output: np.ndarray
    batch_cap: int
    dtype: Any
    owners: list[Any]


@dataclass
class _PreparedStep:
    slot_idx: int
    input_bank_idx: int
    output_bank_idx: int
    m: int
    dtype: Any
    X: np.ndarray
    y: np.ndarray
    apply_adam: bool
    step_token: int


@dataclass
class _SubmittedStep:
    slot_idx: int
    output_bank_idx: int
    m: int
    dtype: Any
    X: np.ndarray
    y: np.ndarray
    apply_adam: bool
    step_token: int


def shutdown_contract_async() -> None:
    """Stop all native contract workers and wipe every tenant's stage tables."""
    lib = _load_conv_dll()
    if lib is None:
        return
    if hasattr(lib, "contract_async_shutdown"):
        lib.contract_async_shutdown()
    # Wipe stages for every tenant id (0..7) so process teardown leaves no ABA.
    inv = getattr(lib, "native_tenant_invalidate_all", None)
    set_t = getattr(lib, "native_tenant_set_current", None)
    if inv is not None and set_t is not None:
        for tid in range(8):
            set_t(ctypes.c_int32(tid))
            inv(ctypes.c_int32(tid))
        set_t(ctypes.c_int32(0))


class ContractRuntime:
    """Binds a compiled ContractList to live CNN weights + scratch arena."""

    def __init__(self, model: Any, contract: ContractList, *, native_async_submit: bool = False):
        self.model = model
        self.contract = contract
        self._lib = _load_conv_dll()
        if self._lib is None or not hasattr(self._lib, "run_contract_training_step"):
            raise RuntimeError("conv_kernels.dll missing run_contract_training_step — rebuild native")
        _bind_runner(self._lib)

        self._mhsa_mode = self.contract.graph_id.startswith("mhsa") or any(
            op.opcode
            in (
                ContractOp.MHSA_BLOCK_FWD,
                ContractOp.MHSA_ACTION_FWD,
            )
            for op in self.contract.ops
        )
        if not self._mhsa_mode:
            if len(model._dense_w_indices) < 1:
                raise ValueError("Contract path requires at least one dense head layer")
            if len(model._dense_w_indices) > 8:
                raise ValueError("Contract path supports at most 8 dense layers")

        self._ops = (ContractOpRow * contract.op_count)(*self._build_op_rows())
        self._loss_scalar = np.zeros(1, dtype=np.float32)
        self._buffers: ContractBuffers | None = None
        self._bound_layers: list[tuple[int, Any]] = []
        self._ctx = ContractExecCtx()
        self._bindings_ready = False
        self._conv_bindings_ready = False
        self._dense_bindings_ready = False
        self._input_logical_w: int | None = getattr(model, "input_logical_w", None)
        self._tenant_id = 0
        create_tenant = getattr(self._lib, "native_tenant_create", None)
        if create_tenant is not None:
            tid = int(create_tenant())
            if tid < 0:
                raise RuntimeError("native_tenant_create: no free tenant slots")
            self._tenant_id = tid
        self._activate_tenant()
        self._async_enabled = (
            bool(native_async_submit)
            and hasattr(self._lib, "submit_contract_training_step")
            and hasattr(self._lib, "wait_contract_completion")
        )
        set_overlap = getattr(self._lib, "set_contract_async_overlap", None)
        if set_overlap is not None:
            # async on → serial main packs (OMP busy); async off → OMP packs
            set_overlap(ctypes.c_int32(1 if self._async_enabled else 0))
        self._pending_token: int | None = None
        self._engine_driven = False
        self._submitted: _SubmittedStep | None = None
        self._completed: _SubmittedStep | None = None
        self._prepared: _PreparedStep | None = None
        self._parameter_banks: list[_ParameterBank] = []
        self._published_bank_idx = 0
        self._slots: list[_ExecutionSlot] = []
        self._eval_slot: _EvalSlot | None = None
        self._forward_op_count = next(
            (
                i
                for i, op in enumerate(self.contract.ops)
                if op.opcode
                in (
                    ContractOp.CONV2D_BWD,
                    ContractOp.RELU_BWD,
                    ContractOp.MAXPOOL_BWD,
                    ContractOp.FLATTEN_BWD,
                    ContractOp.DENSE_BWD,
                    ContractOp.ADAM_APPLY,
                    ContractOp.CONV_BLOCK_BWD,
                    ContractOp.MHSA_ACTION_BWD,
                    ContractOp.MHSA_BLOCK_BWD,
                )
            ),
            self.contract.op_count,
        )
        # Tail contract for closed-loop: BLOCK_BWD (+ ADAM) after Python fills dO.
        self._mhsa_do_bwd_ops = None
        self._mhsa_do_bwd_op_count = 0
        if self._mhsa_mode:
            tail_rows = [
                ContractOpRow(
                    int(op.opcode),
                    op.layer_idx,
                    op.param_idx,
                    op.flags,
                    op.i0,
                    op.i1,
                    op.i2,
                )
                for op in self.contract.ops
                if op.opcode
                in (ContractOp.MHSA_BLOCK_BWD, ContractOp.ADAM_APPLY)
            ]
            if tail_rows:
                self._mhsa_do_bwd_op_count = len(tail_rows)
                self._mhsa_do_bwd_ops = (
                    ContractOpRow * self._mhsa_do_bwd_op_count
                )(*tail_rows)
        self._last_mhsa_dX: np.ndarray | None = None
        self._mhsa_ws: dict[str, Any] | None = None
        self._mhsa_workspaces: list[dict[str, Any]] = []
        self._mhsa_banks: list[_MhsaParameterBank] = []
        self._mhsa_w_f32: list[np.ndarray] | None = None
        self._mhsa_b_f32: list[np.ndarray] | None = None
        self._mhsa_ln_f32: list[dict[str, np.ndarray]] | None = None
        self._mhsa_dw_f32: list[np.ndarray] | None = None
        self._mhsa_db_f32: list[np.ndarray] | None = None
        self._mhsa_dln_f32: list[dict[str, np.ndarray]] | None = None
        self._subscriber_fn: Callable[[], None] | None = None
        self._capacity_fn: Callable[[], None] | None = None
        # Native worker posts ASYNC_READY only; this thread reaps + finishes under the GIL.
        if self._async_enabled and hasattr(self._lib, "contract_register_completion_callback"):
            self._lib.contract_register_completion_callback(None)
        self._trace_mailbox("init")

    def _activate_tenant(self) -> None:
        """Bind this runtime's native tenant on the calling thread."""
        set_t = getattr(self._lib, "native_tenant_set_current", None)
        if set_t is not None:
            set_t(ctypes.c_int32(int(self._tenant_id)))

    def close(self) -> None:
        """Release native tenant stages + async worker for this runtime only."""
        self._activate_tenant()
        self._invalidate_input_pad_stage()
        shut = getattr(self._lib, "contract_async_shutdown_tenant", None)
        if shut is not None and self._tenant_id > 0:
            shut(ctypes.c_int32(int(self._tenant_id)))
        release = getattr(self._lib, "native_tenant_release", None)
        if release is not None and self._tenant_id > 0:
            release(ctypes.c_int32(int(self._tenant_id)))
            self._tenant_id = 0

    def __del__(self) -> None:
        try:
            if getattr(self, "_tenant_id", 0) > 0:
                self.close()
        except Exception:
            pass

    def _bindings_still_valid(self, m: int) -> bool:
        if not self._bindings_ready:
            return False
        if self._buffers is None:
            return False
        cap = getattr(self.model, "_train_batch_cap", 0) or m
        return m <= self._buffers.batch_cap and m <= cap

    
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
            # Sync path has no main-thread prepare phase: native still zeroes.
            lb.d_conv_prezeroed = 0
            lb.dx_prezeroed = 0
            lb.dw_prezeroed = 0
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
            # Must match train out_conv buffer (SIMD halo). Native no longer
            # overwrites this — it used to round_up here and that broke dense eval.
            lb.conv_out_w_stride = int(scratch.out_conv_buffer.shape[3])
            conv_out_h = (
                cur_h + 2 * layer.conv_pad - layer.k_h
            ) // layer.conv_stride + 1
            conv_out_w = (
                cur_w_log + 2 * layer.conv_pad - layer.k_w
            ) // layer.conv_stride + 1
            lb.pool_out_h = (conv_out_h - layer.pool_size) // layer.pool_stride + 1
            lb.pool_out_w = (conv_out_w - layer.pool_size) // layer.pool_stride + 1
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
        """Wire dense head layers into ctx (once, or after batch cap bump)."""
        if self._dense_bindings_ready and self._bindings_still_valid(m):
            return

        bufs = self._ensure_buffers(m, dtype)
        opt = self.model.optimizer
        num_dense = len(self.model._dense_w_indices)
        ctx.num_dense = num_dense

        for di, w_idx in enumerate(self.model._dense_w_indices):
            Wd = self.model.weights[w_idx]
            bd = self.model.biases[w_idx].reshape(-1)
            layer_bufs = bufs.dense_layers[di]

            d = ctx.dense[di]
            d.W = _ptr(Wd)
            d.b = _ptr(bd)
            d.dW = _ptr(layer_bufs.dW)
            d.db = _ptr(layer_bufs.db)
            d.z = _ptr(layer_bufs.z)
            d.output = _ptr(layer_bufs.output)
            d.delta = _ptr(layer_bufs.delta)
            d.dx_flat = _ptr(layer_bufs.dx_flat)
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
            for layer_bufs in self._buffers.dense_layers:
                layer_bufs.dW.fill(0.0)
                layer_bufs.db.fill(0.0)
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
        cap = max(m, getattr(self.model, "_train_batch_cap", 0) or m)
        if self._buffers is not None and cap <= self._buffers.batch_cap:
            return self._buffers

        dense_layers: list[DenseLayerBuffers] = []
        for w_idx in self.model._dense_w_indices:
            W = self.model.weights[w_idx]
            fan_in, fan_out = W.shape
            dense_layers.append(
                DenseLayerBuffers(
                    z=np.empty((cap, fan_out), dtype=dtype),
                    output=np.empty((cap, fan_out), dtype=dtype),
                    delta=np.empty((cap, fan_out), dtype=dtype),
                    dW=np.zeros((fan_in, fan_out), dtype=dtype),
                    db=np.zeros((fan_out,), dtype=dtype),
                    dx_flat=np.empty((cap, fan_in), dtype=dtype),
                )
            )

        self._buffers = ContractBuffers(dense_layers=dense_layers, batch_cap=cap)
        return self._buffers

    def _make_async_slot(self, X: np.ndarray, m: int) -> _ExecutionSlot:
        """Allocate one independently owned contract execution/result slot."""
        from src.scratch_arena import ScratchArena
        from src.spatial_layers import ConvBlock

        cap = max(m, getattr(self.model, "_train_batch_cap", 0) or m)
        arena = ScratchArena(self.model.backend)
        arena.set_train_batch_cap(cap)
        ctx = ContractExecCtx()
        ctx.lam_l2 = float(self.model.lam_l2)
        ctx.max_norm = float(self.model.max_norm)
        ctx.adam.beta1 = float(self.model.optimizer.beta1)
        ctx.adam.beta2 = float(self.model.optimizer.beta2)
        ctx.adam.eps = float(self.model.optimizer.eps)

        w_logical = self._input_logical_w
        if w_logical is None:
            w_logical = 28 if X.ndim == 4 and X.shape[3] == 32 else X.shape[3]
            self._input_logical_w = w_logical

        conv_grads: list[tuple[int, np.ndarray, np.ndarray]] = []
        d_conv_buffers: list[np.ndarray] = []
        dx_buffers: list[np.ndarray] = []
        owners: list[Any] = [arena]
        max_layer_idx = -1
        cur_c, cur_h = X.shape[1], X.shape[2]
        cur_w_log, cur_w_stride = w_logical, X.shape[3]

        for li, layer in enumerate(self.model.layers):
            if not isinstance(layer, ConvBlock):
                continue
            if li >= 8:
                raise ValueError("Contract path supports at most 8 ConvBlock layers")
            w_idx = self.model._layer_param_idx[li]
            W = self.model.weights[w_idx]
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
                N=cap,
                C=cur_c,
                H=cur_h,
                W_stride=cur_w_stride,
                W_logical=cur_w_log,
                dtype=X.dtype,
            )
            dW = np.zeros_like(W)
            db = np.zeros_like(self.model.biases[w_idx])
            owners.extend((dW, db))
            conv_grads.append((w_idx, dW, db))

            lb = ctx.layers[li]
            lb.dW = _ptr(dW)
            lb.db = _ptr(db.reshape(-1))
            lb.out_conv = _ptr(scratch.out_conv_buffer)
            lb.out_pool = _ptr(scratch.out_pool_buffer)
            lb.argmax = _ptr(scratch.argmax_buffer)
            lb.dx = _ptr(scratch.dx_buffer)
            lb.d_conv = _ptr(scratch.d_conv_buffer)
            lb.d_conv_prezeroed = 1
            # prepare_step's conv_grads loop (dW.fill(0.0)) already zeroes
            # this every step before submission; skip native's duplicate memset.
            lb.dw_prezeroed = 1
            d_conv_buffers.append(scratch.d_conv_buffer)
            if li != 0:
                # Layer 0's dx has no consumer (input image); backward skips it
                # entirely (nullptr), so nothing to prezero there.
                lb.dx_prezeroed = 1
                dx_buffers.append(scratch.dx_buffer)
            else:
                lb.dx_prezeroed = 0
            lb.w_count = int(W.size)
            lb.b_count = int(db.size)
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

            conv_out_h = (
                cur_h + 2 * layer.conv_pad - layer.k_h
            ) // layer.conv_stride + 1
            conv_out_w = (
                cur_w_log + 2 * layer.conv_pad - layer.k_w
            ) // layer.conv_stride + 1
            lb.conv_out_w_stride = _round_up_simd(conv_out_w)
            lb.pool_out_h = (
                conv_out_h - layer.pool_size
            ) // layer.pool_stride + 1
            lb.pool_out_w = (
                conv_out_w - layer.pool_size
            ) // layer.pool_stride + 1
            max_layer_idx = li
            cur_c, cur_h, cur_w_log, cur_w_stride = _conv_block_output_geom(
                cur_c, cur_h, cur_w_log, layer
            )

        ctx.num_layers = max_layer_idx + 1 if max_layer_idx >= 0 else 0

        dense_layers: list[DenseLayerBuffers] = []
        for di, w_idx in enumerate(self.model._dense_w_indices):
            W = self.model.weights[w_idx]
            fan_in, fan_out = W.shape
            layer_bufs = DenseLayerBuffers(
                z=np.empty((cap, fan_out), dtype=X.dtype),
                output=np.empty((cap, fan_out), dtype=X.dtype),
                delta=np.empty((cap, fan_out), dtype=X.dtype),
                dW=np.zeros((fan_in, fan_out), dtype=X.dtype),
                db=np.zeros((fan_out,), dtype=X.dtype),
                dx_flat=np.empty((cap, fan_in), dtype=X.dtype),
            )
            dense_layers.append(layer_bufs)
            d = ctx.dense[di]
            d.dW = _ptr(layer_bufs.dW)
            d.db = _ptr(layer_bufs.db)
            d.z = _ptr(layer_bufs.z)
            d.output = _ptr(layer_bufs.output)
            d.delta = _ptr(layer_bufs.delta)
            d.dx_flat = _ptr(layer_bufs.dx_flat)
            d.fan_in = fan_in
            d.fan_out = fan_out

        ctx.num_dense = len(dense_layers)
        loss_scalar = np.zeros(1, dtype=np.float32)
        ctx.loss_out = _ptr(loss_scalar)
        return _ExecutionSlot(
            ctx=ctx,
            buffers=ContractBuffers(dense_layers=dense_layers, batch_cap=cap),
            conv_grads=conv_grads,
            loss_scalar=loss_scalar,
            owners=owners,
            d_conv_buffers=d_conv_buffers,
            dx_buffers=dx_buffers,
        )

    def _ensure_async_resources(self, X: np.ndarray, m: int) -> None:
        if self._mhsa_mode:
            self._ensure_mhsa_async_resources(X)
            return
        opt = self.model.optimizer
        if not opt._setup_done:
            opt.setup(self.model.weights, self.model.biases)

        if not self._parameter_banks:
            bank0 = _ParameterBank(
                weights=list(self.model.weights),
                biases=list(self.model.biases),
                ms_w=list(opt.ms_w),
                vs_w=list(opt.vs_w),
                ms_b=list(opt.ms_b),
                vs_b=list(opt.vs_b),
            )
            bank1 = _ParameterBank(
                weights=[np.empty_like(a) for a in bank0.weights],
                biases=[np.empty_like(a) for a in bank0.biases],
                ms_w=[np.empty_like(a) for a in bank0.ms_w],
                vs_w=[np.empty_like(a) for a in bank0.vs_w],
                ms_b=[np.empty_like(a) for a in bank0.ms_b],
                vs_b=[np.empty_like(a) for a in bank0.vs_b],
            )
            self._parameter_banks = [bank0, bank1]

        cap = max(m, getattr(self.model, "_train_batch_cap", 0) or m)
        if self._slots and cap <= self._slots[0].buffers.batch_cap:
            return
        if self._submitted is not None or self._completed is not None:
            raise RuntimeError("cannot resize async slots while a result is owned")
        # Slot buffers are about to be replaced; drop staged pads keyed on the
        # old input pointers so a recycled address cannot produce a false hit.
        self._invalidate_input_pad_stage()
        self._slots = [self._make_async_slot(X, cap), self._make_async_slot(X, cap)]

    def _ensure_mhsa_async_resources(self, X: np.ndarray) -> None:
        """Dual-bank MHSA async: two param banks + two B/T workspaces + two slots."""
        if X.ndim != 3:
            raise ValueError(f"MHSA async expects (B,T,D); got {X.shape}")
        B, T = int(X.shape[0]), int(X.shape[1])
        self._ensure_mhsa_workspace(B, T)
        if (
            self._mhsa_banks
            and len(self._mhsa_workspaces) == 2
            and self._slots
            and getattr(self._slots[0], "mhsa_ready", False)
        ):
            # Refresh workspace geometry if B/T changed.
            for wi in range(2):
                self._ensure_mhsa_workspace_slot(B, T, wi)
            return
        if self._submitted is not None or self._completed is not None:
            raise RuntimeError("cannot resize MHSA async slots while a result is owned")

        m = self.model
        if hasattr(m, "ensure_adam_moments"):
            m.ensure_adam_moments()
        opt = m.optimizer
        bank0 = self._snapshot_mhsa_bank_from_model()
        bank1 = self._clone_mhsa_bank_empty(bank0)
        self._mhsa_banks = [bank0, bank1]
        self._published_bank_idx = 0

        ws0 = self._mhsa_ws
        assert ws0 is not None
        ws1 = self._alloc_mhsa_workspace_buffers(B, T)
        # Move sync-shared grads into ws0; give ws1 private grads.
        ws0["dW"] = self._mhsa_dw_f32
        ws0["db"] = self._mhsa_db_f32
        ws0["dln"] = self._mhsa_dln_f32
        D = int(m.d_model)
        L = int(m.num_layers)
        ws1["dW"] = [np.zeros_like(w) for w in m.weights]
        ws1["db"] = [np.zeros_like(b.reshape(-1)) for b in m.biases]
        ws1["dln"] = [
            {
                "ln1_g": np.zeros(D, dtype=np.float32),
                "ln1_b": np.zeros(D, dtype=np.float32),
                "ln2_g": np.zeros(D, dtype=np.float32),
                "ln2_b": np.zeros(D, dtype=np.float32),
            }
            for _ in range(L)
        ]
        self._mhsa_workspaces = [ws0, ws1]
        self._slots = [
            self._make_mhsa_async_slot(0),
            self._make_mhsa_async_slot(1),
        ]

    def _make_mhsa_async_slot(self, ws_idx: int = 0) -> _ExecutionSlot:
        ws = self._mhsa_workspaces[ws_idx] if self._mhsa_workspaces else self._mhsa_ws
        assert ws is not None
        slot = _ExecutionSlot(
            ctx=ContractExecCtx(),
            buffers=ContractBuffers(dense_layers=[], batch_cap=0),
            conv_grads=[],
            loss_scalar=ws["loss"],
            owners=[],
            d_conv_buffers=[],
            dx_buffers=[],
        )
        slot.mhsa_ready = True  # type: ignore[attr-defined]
        slot.mhsa_ws_idx = ws_idx  # type: ignore[attr-defined]
        return slot

    def _snapshot_mhsa_bank_from_model(self) -> _MhsaParameterBank:
        m = self.model
        opt = m.optimizer
        return _MhsaParameterBank(
            weights=list(m.weights),
            biases=list(m.biases),
            ln1_gamma=list(m.ln1_gamma),
            ln1_beta=list(m.ln1_beta),
            ln2_gamma=list(m.ln2_gamma),
            ln2_beta=list(m.ln2_beta),
            ms_w=list(opt.ms_w),
            vs_w=list(opt.vs_w),
            ms_b=list(opt.ms_b),
            vs_b=list(opt.vs_b),
            ms_g=list(opt.ms_g),
            vs_g=list(opt.vs_g),
            ms_beta=list(opt.ms_beta),
            vs_beta=list(opt.vs_beta),
            pos_embed=m.pos_embed,
            ms_pos=m._ms_pos,
            vs_pos=m._vs_pos,
            W_in=getattr(m, "W_in", None),
            b_in=getattr(m, "b_in", None),
            ms_W_in=getattr(m, "_ms_W_in", None),
            vs_W_in=getattr(m, "_vs_W_in", None),
            ms_b_in=getattr(m, "_ms_b_in", None),
            vs_b_in=getattr(m, "_vs_b_in", None),
        )

    def _clone_mhsa_bank_empty(self, src: _MhsaParameterBank) -> _MhsaParameterBank:
        def _clone_opt(arr: np.ndarray | None) -> np.ndarray | None:
            return None if arr is None else np.empty_like(arr)

        return _MhsaParameterBank(
            weights=[np.empty_like(a) for a in src.weights],
            biases=[np.empty_like(a) for a in src.biases],
            ln1_gamma=[np.empty_like(a) for a in src.ln1_gamma],
            ln1_beta=[np.empty_like(a) for a in src.ln1_beta],
            ln2_gamma=[np.empty_like(a) for a in src.ln2_gamma],
            ln2_beta=[np.empty_like(a) for a in src.ln2_beta],
            ms_w=[np.empty_like(a) for a in src.ms_w],
            vs_w=[np.empty_like(a) for a in src.vs_w],
            ms_b=[np.empty_like(a) for a in src.ms_b],
            vs_b=[np.empty_like(a) for a in src.vs_b],
            ms_g=[np.empty_like(a) for a in src.ms_g],
            vs_g=[np.empty_like(a) for a in src.vs_g],
            ms_beta=[np.empty_like(a) for a in src.ms_beta],
            vs_beta=[np.empty_like(a) for a in src.vs_beta],
            pos_embed=_clone_opt(src.pos_embed),
            ms_pos=_clone_opt(src.ms_pos),
            vs_pos=_clone_opt(src.vs_pos),
            W_in=_clone_opt(src.W_in),
            b_in=_clone_opt(src.b_in),
            ms_W_in=_clone_opt(src.ms_W_in),
            vs_W_in=_clone_opt(src.vs_W_in),
            ms_b_in=_clone_opt(src.ms_b_in),
            vs_b_in=_clone_opt(src.vs_b_in),
        )

    def _ensure_mhsa_workspace_slot(self, B: int, T: int, wi: int) -> None:
        if wi >= len(self._mhsa_workspaces):
            return
        ws = self._mhsa_workspaces[wi]
        m = self.model
        D = int(m.d_model)
        H = int(m.num_heads)
        Hff = int(m.ffn_hidden)
        A = int(m.action_dim)
        L = int(m.num_layers)
        key = (B, T, D, H, Hff, A, L)
        if ws.get("_key") == key:
            return
        fresh = self._alloc_mhsa_workspace_buffers(B, T)
        fresh["dW"] = ws.get("dW") or [np.zeros_like(w) for w in m.weights]
        fresh["db"] = ws.get("db") or [np.zeros_like(b.reshape(-1)) for b in m.biases]
        if "dln" in ws:
            fresh["dln"] = ws["dln"]
        else:
            fresh["dln"] = [
                {
                    "ln1_g": np.zeros(D, dtype=np.float32),
                    "ln1_b": np.zeros(D, dtype=np.float32),
                    "ln2_g": np.zeros(D, dtype=np.float32),
                    "ln2_b": np.zeros(D, dtype=np.float32),
                }
                for _ in range(L)
            ]
        self._mhsa_workspaces[wi] = fresh
        if wi == 0:
            self._mhsa_ws = fresh
            self._mhsa_dw_f32 = fresh["dW"]
            self._mhsa_db_f32 = fresh["db"]
            self._mhsa_dln_f32 = fresh["dln"]
        if self._slots and wi < len(self._slots):
            self._slots[wi].loss_scalar = fresh["loss"]

    def _bind_slot_parameter_banks(
        self, slot: _ExecutionSlot, input_bank_idx: int, output_bank_idx: int
    ) -> None:
        if self._mhsa_mode:
            return
        from src.spatial_layers import ConvBlock

        src = self._parameter_banks[input_bank_idx]
        dst = self._parameter_banks[output_bank_idx]
        ctx = slot.ctx
        for li, layer in enumerate(self.model.layers):
            if not isinstance(layer, ConvBlock):
                continue
            w_idx = self.model._layer_param_idx[li]
            lb = ctx.layers[li]
            lb.W = _ptr(src.weights[w_idx])
            lb.b = _ptr(src.biases[w_idx].reshape(-1))
            lb.W_next = _ptr(dst.weights[w_idx])
            lb.b_next = _ptr(dst.biases[w_idx].reshape(-1))
            lb.ms_w = _ptr(src.ms_w[w_idx])
            lb.vs_w = _ptr(src.vs_w[w_idx])
            lb.ms_b = _ptr(src.ms_b[w_idx].reshape(-1))
            lb.vs_b = _ptr(src.vs_b[w_idx].reshape(-1))
            lb.ms_w_next = _ptr(dst.ms_w[w_idx])
            lb.vs_w_next = _ptr(dst.vs_w[w_idx])
            lb.ms_b_next = _ptr(dst.ms_b[w_idx].reshape(-1))
            lb.vs_b_next = _ptr(dst.vs_b[w_idx].reshape(-1))

        for di, w_idx in enumerate(self.model._dense_w_indices):
            d = ctx.dense[di]
            d.W = _ptr(src.weights[w_idx])
            d.b = _ptr(src.biases[w_idx].reshape(-1))
            d.W_next = _ptr(dst.weights[w_idx])
            d.b_next = _ptr(dst.biases[w_idx].reshape(-1))
            d.ms_w = _ptr(src.ms_w[w_idx])
            d.vs_w = _ptr(src.vs_w[w_idx])
            d.ms_b = _ptr(src.ms_b[w_idx].reshape(-1))
            d.vs_b = _ptr(src.vs_b[w_idx].reshape(-1))
            d.ms_w_next = _ptr(dst.ms_w[w_idx])
            d.vs_w_next = _ptr(dst.vs_w[w_idx])
            d.ms_b_next = _ptr(dst.ms_b[w_idx].reshape(-1))
            d.vs_b_next = _ptr(dst.vs_b[w_idx].reshape(-1))

    def _publish_parameter_bank(self, bank_idx: int, adam_t: int) -> None:
        if self._mhsa_mode:
            if self._mhsa_banks:
                bank = self._mhsa_banks[bank_idx]
                self._published_bank_idx = bank_idx
                m = self.model
                m.weights = bank.weights
                m.biases = bank.biases
                m.ln1_gamma = bank.ln1_gamma
                m.ln1_beta = bank.ln1_beta
                m.ln2_gamma = bank.ln2_gamma
                m.ln2_beta = bank.ln2_beta
                m.pos_embed = bank.pos_embed
                m._ms_pos = bank.ms_pos
                m._vs_pos = bank.vs_pos
                m.W_in = bank.W_in
                m.b_in = bank.b_in
                m._ms_W_in = bank.ms_W_in
                m._vs_W_in = bank.vs_W_in
                m._ms_b_in = bank.ms_b_in
                m._vs_b_in = bank.vs_b_in
                opt = m.optimizer
                opt.ms_w = bank.ms_w
                opt.vs_w = bank.vs_w
                opt.ms_b = bank.ms_b
                opt.vs_b = bank.vs_b
                opt.ms_g = bank.ms_g
                opt.vs_g = bank.vs_g
                opt.ms_beta = bank.ms_beta
                opt.vs_beta = bank.vs_beta
                self._mhsa_w_f32 = bank.weights
                self._mhsa_b_f32 = [b.reshape(-1) for b in bank.biases]
                self._mhsa_ln_f32 = [
                    {
                        "ln1_g": bank.ln1_gamma[li].reshape(-1),
                        "ln1_b": bank.ln1_beta[li].reshape(-1),
                        "ln2_g": bank.ln2_gamma[li].reshape(-1),
                        "ln2_b": bank.ln2_beta[li].reshape(-1),
                    }
                    for li in range(int(m.num_layers))
                ]
            self.model.optimizer.t = int(adam_t)
            return
        bank = self._parameter_banks[bank_idx]
        self._published_bank_idx = bank_idx
        self.model.weights = bank.weights
        self.model.biases = bank.biases
        opt = self.model.optimizer
        opt.ms_w = bank.ms_w
        opt.vs_w = bank.vs_w
        opt.ms_b = bank.ms_b
        opt.vs_b = bank.vs_b
        opt.t = int(adam_t)

    def uses_async_forward(self) -> bool:
        return self._async_enabled

    def _make_eval_slot(self, X: np.ndarray, m: int) -> _EvalSlot:
        """Bind a forward-only context with evaluation-sized private buffers."""
        from src.scratch_arena import ScratchArena
        from src.spatial_layers import ConvBlock

        arena = ScratchArena(self.model.backend)
        ctx = ContractExecCtx()
        owners: list[Any] = [arena]
        w_logical = self._input_logical_w
        if w_logical is None:
            w_logical = 28 if X.ndim == 4 and X.shape[3] == 32 else X.shape[3]
            self._input_logical_w = w_logical

        max_layer_idx = -1
        cur_c, cur_h = X.shape[1], X.shape[2]
        cur_w_log, cur_w_stride = w_logical, X.shape[3]
        for li, layer in enumerate(self.model.layers):
            if not isinstance(layer, ConvBlock):
                continue
            if li >= 8:
                raise ValueError("Contract path supports at most 8 ConvBlock layers")
            scratch = arena.ensure_conv_block_eval(
                li,
                out_channels=layer.out_channels,
                k_h=layer.k_h,
                k_w=layer.k_w,
                conv_stride=layer.conv_stride,
                conv_pad=layer.conv_pad,
                pool_size=layer.pool_size,
                pool_stride=layer.pool_stride,
                N=m,
                C=cur_c,
                H=cur_h,
                W_logical=cur_w_log,
                dtype=X.dtype,
            )
            lb = ctx.layers[li]
            lb.out_conv = _ptr(scratch.eval_out_conv_buffer)
            lb.out_pool = _ptr(scratch.eval_out_pool_buffer)
            lb.argmax = _ptr(scratch.eval_argmax_buffer)
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

            conv_out_h = (
                cur_h + 2 * layer.conv_pad - layer.k_h
            ) // layer.conv_stride + 1
            conv_out_w = (
                cur_w_log + 2 * layer.conv_pad - layer.k_w
            ) // layer.conv_stride + 1
            # Match ensure_conv_block_eval: dense W (no SIMD halo). Rounding here
            # made native write past the eval buffer → heap corruption.
            lb.conv_out_w_stride = conv_out_w
            lb.pool_out_h = (
                conv_out_h - layer.pool_size
            ) // layer.pool_stride + 1
            lb.pool_out_w = (
                conv_out_w - layer.pool_size
            ) // layer.pool_stride + 1
            max_layer_idx = li
            cur_c, cur_h, cur_w_log, cur_w_stride = _conv_block_output_geom(
                cur_c, cur_h, cur_w_log, layer
            )
        ctx.num_layers = max_layer_idx + 1 if max_layer_idx >= 0 else 0

        output: np.ndarray | None = None
        for di, w_idx in enumerate(self.model._dense_w_indices):
            W = self.model.weights[w_idx]
            fan_in, fan_out = W.shape
            z = np.empty((m, fan_out), dtype=X.dtype)
            out = np.empty((m, fan_out), dtype=X.dtype)
            owners.extend((z, out))
            d = ctx.dense[di]
            d.z = _ptr(z)
            d.output = _ptr(out)
            d.fan_in = fan_in
            d.fan_out = fan_out
            output = out
        if output is None:
            raise RuntimeError("forward contract requires at least one dense output")
        ctx.num_dense = len(self.model._dense_w_indices)
        ctx.loss_out = None
        return _EvalSlot(ctx, output, m, X.dtype, owners)

    def _bind_eval_weights(self, slot: _EvalSlot) -> None:
        from src.spatial_layers import ConvBlock

        for li, layer in enumerate(self.model.layers):
            if not isinstance(layer, ConvBlock):
                continue
            w_idx = self.model._layer_param_idx[li]
            slot.ctx.layers[li].W = _ptr(self.model.weights[w_idx])
            slot.ctx.layers[li].b = _ptr(self.model.biases[w_idx].reshape(-1))
        for di, w_idx in enumerate(self.model._dense_w_indices):
            slot.ctx.dense[di].W = _ptr(self.model.weights[w_idx])
            slot.ctx.dense[di].b = _ptr(self.model.biases[w_idx].reshape(-1))

    def run_async_forward(self, X: np.ndarray) -> np.ndarray:
        """Execute the compiled contract's forward prefix on the native worker."""
        if not self._async_enabled:
            raise RuntimeError("async forward requires native_async_submit")
        if (
            self._submitted is not None
            or self._completed is not None
            or self._prepared is not None
            or self._lib.contract_async_in_flight()
        ):
            raise RuntimeError("async forward requires a drained training pipeline")

        X = np.ascontiguousarray(X)
        m = int(X.shape[0])
        if (
            self._eval_slot is None
            or m > self._eval_slot.batch_cap
            or self._eval_slot.dtype != X.dtype
        ):
            self._eval_slot = self._make_eval_slot(X, m)
        slot = self._eval_slot
        self._bind_eval_weights(slot)
        slot.ctx.N = m
        slot.ctx.X = _ptr(X)
        slot.ctx.y = None
        slot.ctx.act = None

        token = -(int(self.model.optimizer.t) + 1)
        self._activate_tenant()
        status = self._lib.submit_contract_training_step(
            ctypes.cast(self._ops, ctypes.POINTER(ContractOpRow)),
            ctypes.c_int32(self._forward_op_count),
            ctypes.byref(slot.ctx),
            ctypes.c_int64(token),
        )
        if status != 0:
            raise RuntimeError(f"submit async forward failed with status {status}")
        self._pending_token = token
        if not self._wait_reap_native(-1):
            raise RuntimeError("async forward wait returned without completion")
        self._invalidate_input_pad_stage()
        return np.copy(slot.output[:m])

    
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

    def _trace_mailbox(self, where: str) -> None:
        """Diagnostic-only cross-check of stable Python/native mailbox state."""
        if not (
            self._async_enabled
            and hasattr(self._lib, "contract_async_debug_snapshot")
        ):
            return

        state = ctypes.c_int32()
        has_job = ctypes.c_int32()
        worker_started = ctypes.c_int32()
        shutdown = ctypes.c_int32()
        submit_token = ctypes.c_int64()
        ready_token = ctypes.c_int64()
        rc = self._lib.contract_async_debug_snapshot(
            ctypes.byref(state),
            ctypes.byref(has_job),
            ctypes.byref(worker_started),
            ctypes.byref(shutdown),
            ctypes.byref(submit_token),
            ctypes.byref(ready_token),
        )
        if rc != 0:
            print(
                f"[MAILBOX_DESYNC][python] where={where} snapshot_rc={rc}",
                file=sys.stderr,
                flush=True,
            )
            return

        submitted = self._submitted is not None
        completed = self._completed is not None
        pending = self._pending_token
        native_state = int(state.value)
        valid = True
        reasons: list[str] = []

        if submitted:
            if native_state not in (1, 2):
                valid = False
                reasons.append("submitted_requires_RUNNING_or_READY")
            native_token = ready_token.value if native_state == 2 else submit_token.value
            if pending is None or native_token != pending:
                valid = False
                reasons.append("submitted_token_mismatch")
        elif completed:
            if native_state != 0 or pending is not None:
                valid = False
                reasons.append("completed_requires_native_IDLE")
        elif native_state != 0 or pending is not None:
            valid = False
            reasons.append("empty_python_requires_native_IDLE")

        if has_job.value and native_state != 1:
            valid = False
            reasons.append("has_job_requires_RUNNING")

        if not valid:
            print(
                "[MAILBOX_DESYNC][python] "
                f"where={where} reasons={'+'.join(reasons)} "
                f"native_state={native_state} has_job={has_job.value} "
                f"worker_started={worker_started.value} shutdown={shutdown.value} "
                f"native_submit_token={submit_token.value} "
                f"native_ready_token={ready_token.value} "
                f"py_submitted={int(submitted)} py_completed={int(completed)} "
                f"py_pending_token={pending}",
                file=sys.stderr,
                flush=True,
            )

    def waiting_on_native_worker(self) -> bool:
        """True while a submitted step is not yet finished on this thread."""
        if not self._async_enabled:
            return False
        if self._submitted is not None:
            return True
        return bool(self._lib.contract_async_in_flight())

    def _poll_complete_on_main(self) -> bool:
        """Non-blocking: reap ASYNC_READY on this thread, finish grads, publish.

        Must run on the Python trainer thread (holds/reacquires GIL around numpy).
        Native worker never calls into Python. No-op when async is off.
        """
        if not self._async_enabled:
            return False
        if self._completed is not None:
            return True
        if self._submitted is None:
            return False
        if not self._try_reap_native():
            self._trace_mailbox("poll_not_ready")
            return False
        submitted = self._submitted
        self._submitted = None
        if submitted.apply_adam:
            self._publish_parameter_bank(
                submitted.output_bank_idx,
                self._slots[submitted.slot_idx].ctx.adam.t,
            )
        self._completed = submitted
        self._publish_completion()
        self._trace_mailbox("poll_completed")
        return True

    def wait_for_completion(self, timeout: float | None = None) -> bool:
        """Block until native READY is reaped and finished on this thread.

        Async only: polls with a short wait so this thread can drain mid-step
        BRGEMM x-pack requests while the OMP team runs L1/dense.
        Sync path has no worker — returns immediately.
        """
        if not self._async_enabled:
            return True
        if self._poll_complete_on_main():
            return True
        if timeout is not None and timeout <= 0:
            return False
        if self._submitted is None:
            return False

        service = getattr(self._lib, "service_brgemm_dw_x_pack_requests", None)

        def _drain_pack() -> None:
            if service is None:
                return
            self._activate_tenant()
            for _ in range(8):
                if int(service()) <= 0:
                    break

        deadline = None if timeout is None else (time.perf_counter() + float(timeout))
        while True:
            _drain_pack()
            if self._poll_complete_on_main():
                return True
            if deadline is not None and time.perf_counter() >= deadline:
                self._trace_mailbox("wait_not_ready")
                return False
            # Short park so we wake often enough to service packs; still avoids
            # a pure spin when the worker has not posted a request yet.
            if not self._wait_reap_native(1):
                continue
            _drain_pack()
            submitted = self._submitted
            self._submitted = None
            if submitted.apply_adam:
                self._publish_parameter_bank(
                    submitted.output_bank_idx,
                    self._slots[submitted.slot_idx].ctx.adam.t,
                )
            self._completed = submitted
            self._publish_completion()
            self._trace_mailbox("wait_completed")
            return True

    def native_in_flight(self) -> bool:
        if not self._async_enabled:
            return False
        if self._submitted is not None:
            return True
        return bool(self._lib.contract_async_in_flight())

    def completion_signaled(self) -> bool:
        """True when this thread has finished a native step into `_completed`."""
        if not self._async_enabled:
            return False
        self._poll_complete_on_main()
        return self._completed is not None

    def is_busy(self) -> bool:
        """Native worker occupied — a prepared step may not commit yet."""
        return self.native_in_flight()

    def _publish_completion(self) -> None:
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
        if self._prepared is None:
            if not self.prepare_step(
                X, y, lr, apply_adam=apply_adam, step_token=step_token
            ):
                return False
        prepared = self._prepared
        if prepared is None:
            return False
        if prepared.X is not X or prepared.y is not y:
            raise RuntimeError("prepared step does not match committed batch")

        slot = self._slots[prepared.slot_idx]
        slot.ctx.adam.t = int(self.model.optimizer.t)
        if not self._mhsa_mode:
            self._stage_dx_wt(prepared.slot_idx, slot)
        self._submit_native(slot.ctx, prepared.step_token)
        self._submitted = _SubmittedStep(
            slot_idx=prepared.slot_idx,
            output_bank_idx=prepared.output_bank_idx,
            m=prepared.m,
            dtype=prepared.dtype,
            X=prepared.X,
            y=prepared.y,
            apply_adam=prepared.apply_adam,
            step_token=prepared.step_token,
        )
        self._prepared = None
        self._trace_mailbox("submitted")
        return True

    def prepare_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        *,
        apply_adam: bool = False,
        step_token: int | None = None,
    ) -> bool:
        """Prepare the inactive execution slot while the current job runs."""
        if not (self._async_enabled and self._engine_driven):
            raise RuntimeError("prepare_step is only valid in engine-driven async mode")
        if self._mhsa_mode:
            return self._prepare_mhsa_step(
                X, y, lr, apply_adam=apply_adam, step_token=step_token
            )
        if self._prepared is not None:
            return self._prepared.X is X and self._prepared.y is y

        X = np.ascontiguousarray(X)
        y = np.ascontiguousarray(y)
        m = int(X.shape[0])
        self._ensure_async_resources(X, m)

        occupied = {
            step.slot_idx
            for step in (self._submitted, self._completed)
            if step is not None
        }
        free_slots = [i for i in range(2) if i not in occupied]
        if not free_slots:
            return False
        slot_idx = free_slots[0]
        input_bank_idx = (
            self._submitted.output_bank_idx
            if self._submitted is not None
            else self._published_bank_idx
        )
        output_bank_idx = 1 - input_bank_idx if apply_adam else input_bank_idx
        slot = self._slots[slot_idx]
        self._bind_slot_parameter_banks(slot, input_bank_idx, output_bank_idx)
        slot.ctx.N = m
        slot.ctx.lr = float(lr)
        slot.ctx.skip_adam = 0 if apply_adam else 1
        slot.ctx.X = _ptr(X)
        slot.ctx.y = _ptr(y)
        slot.ctx.adam.t = int(self.model.optimizer.t)

        for layer_bufs in slot.buffers.dense_layers:
            layer_bufs.dW.fill(0.0)
            layer_bufs.db.fill(0.0)
        for _, dW, db in slot.conv_grads:
            dW.fill(0.0)
            db.fill(0.0)
        # This slot's d_conv is dirty from the step it last ran; the maxpool
        # backward scatters into it, so it must start at zero. The other slot is
        # in flight right now, so this thread is otherwise idle.
        for d_conv in slot.d_conv_buffers:
            d_conv.fill(0.0)
        # Same reasoning as d_conv: dx's shape depends only on (N, C_in, H,
        # W_in_stride), which are fixed for this slot, so it can be zeroed
        # here in the overlap window instead of via memset on the OMP worker.
        for dx in slot.dx_buffers:
            dx.fill(0.0)

        self._stage_input_pad(slot_idx, slot, X)
        self._stage_brgemm_dw_x(slot_idx, slot, X)

        token = int(step_token if step_token is not None else self.model.optimizer.t + 1)
        self._prepared = _PreparedStep(
            slot_idx=slot_idx,
            input_bank_idx=input_bank_idx,
            output_bank_idx=output_bank_idx,
            m=m,
            dtype=X.dtype,
            X=X,
            y=y,
            apply_adam=apply_adam,
            step_token=token,
        )
        return True

    def _prepare_mhsa_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        *,
        apply_adam: bool = False,
        step_token: int | None = None,
    ) -> bool:
        """Prepare inactive MHSA slot/bank while a submitted step may still run."""
        if self._prepared is not None:
            return self._prepared.X is X and self._prepared.y is y

        X = np.ascontiguousarray(X)
        y = np.ascontiguousarray(y)
        m = int(X.shape[0])
        self._ensure_mhsa_async_resources(X)

        occupied = {
            step.slot_idx
            for step in (self._submitted, self._completed)
            if step is not None
        }
        free_slots = [i for i in range(2) if i not in occupied]
        if not free_slots:
            return False
        slot_idx = free_slots[0]

        input_bank_idx = (
            self._submitted.output_bank_idx
            if self._submitted is not None
            else self._published_bank_idx
        )
        output_bank_idx = 1 - input_bank_idx if apply_adam else input_bank_idx

        slot = self._slots[slot_idx]
        ctx = self._bind_mhsa(
            X,
            y,
            apply_adam=apply_adam,
            slot_idx=slot_idx,
            input_bank_idx=input_bank_idx,
            output_bank_idx=output_bank_idx,
        )
        ctx.lr = float(lr)
        ctx.skip_adam = 0 if apply_adam else 1
        ctx.adam.t = int(self.model.optimizer.t)
        self._zero_mhsa_grads(slot_idx=slot_idx)
        slot.ctx = ctx
        ws = self._mhsa_workspaces[slot_idx]
        slot.loss_scalar = ws["loss"]

        token = int(step_token if step_token is not None else self.model.optimizer.t + 1)
        self._prepared = _PreparedStep(
            slot_idx=slot_idx,
            input_bank_idx=input_bank_idx,
            output_bank_idx=output_bank_idx,
            m=m,
            dtype=X.dtype,
            X=X,
            y=y,
            apply_adam=apply_adam,
            step_token=token,
        )
        return True

    def _stage_input_pad(
        self, slot_idx: int, slot: "_ExecutionSlot", X: np.ndarray
    ) -> None:
        """Build conv layer 0's padded input (OMP when async off)."""
        self._stage_input_pad_ctx(slot_idx, slot.ctx, X)

    def _stage_input_pad_ctx(
        self, slot_idx: int, ctx: ContractExecCtx, X: np.ndarray
    ) -> None:
        self._activate_tenant()
        stage = getattr(self._lib, "stage_conv_x_pad", None)
        if stage is None or ctx.num_layers <= 0:
            return
        lb = ctx.layers[0]
        if lb.conv_stride != 1:
            return
        conv_out_w = (
            lb.W_in + 2 * lb.conv_pad - lb.k_w
        ) // lb.conv_stride + 1
        stage(
            ctypes.c_int32(slot_idx),
            ctypes.c_void_p(_ptr(X)),
            ctypes.c_int64(int(ctx.N)),
            ctypes.c_int64(int(lb.C_in)),
            ctypes.c_int64(int(lb.H)),
            ctypes.c_int64(int(lb.W_in)),
            ctypes.c_int64(int(lb.W_stride)),
            ctypes.c_int64(int(lb.k_w)),
            ctypes.c_int64(int(lb.conv_pad)),
            ctypes.c_int64(int(conv_out_w)),
        )

    def _invalidate_input_pad_stage(self, slot_idx: int = -1) -> None:
        self._activate_tenant()
        drop = getattr(self._lib, "invalidate_conv_x_pad_stage", None)
        if drop is not None:
            drop(ctypes.c_int32(slot_idx))
        drop_wt = getattr(self._lib, "invalidate_dx_cin_blocked_wt_stage", None)
        if drop_wt is not None:
            drop_wt(ctypes.c_int32(slot_idx))
        drop_brg = getattr(self._lib, "invalidate_brgemm_dw_x_pack", None)
        if drop_brg is not None:
            drop_brg(ctypes.c_int32(slot_idx))

    def _stage_brgemm_dw_x(
        self, slot_idx: int, slot: "_ExecutionSlot", X: np.ndarray
    ) -> None:
        """Pack layer-0 BRGEMM dW x panels when eligible (OMP when async off)."""
        self._stage_brgemm_dw_x_ctx(slot_idx, slot.ctx, X)

    def _stage_brgemm_dw_x_ctx(
        self, slot_idx: int, ctx: ContractExecCtx, X: np.ndarray
    ) -> None:
        self._activate_tenant()
        stage = getattr(self._lib, "stage_brgemm_dw_x_pack", None)
        if stage is None or ctx.num_layers <= 0:
            return
        lb = ctx.layers[0]
        if (
            lb.conv_stride != 1
            or lb.k_h != lb.k_w
            or lb.k_h < 1
            or lb.k_h > 7
            or (lb.C_in % 8) != 0
            or (lb.C_out % 8) != 0
        ):
            return
        stage(
            ctypes.c_int32(slot_idx),
            ctypes.c_void_p(_ptr(X)),
            ctypes.c_int64(int(ctx.N)),
            ctypes.c_int64(int(lb.C_in)),
            ctypes.c_int64(int(lb.H)),
            ctypes.c_int64(int(lb.W_in)),
            ctypes.c_int64(int(lb.W_stride)),
            ctypes.c_int64(int(lb.k_w)),
            ctypes.c_int64(int(lb.conv_pad)),
        )

    def _stage_sync_useful_work(self, ctx: ContractExecCtx, X: np.ndarray) -> None:
        """When async is off: OMP-parallelize prep packs before the sync step."""
        if self._async_enabled:
            return
        self._stage_input_pad_ctx(0, ctx, X)
        self._stage_brgemm_dw_x_ctx(0, ctx, X)

    def _stage_dx_wt(self, slot_idx: int, slot: "_ExecutionSlot") -> None:
        """Rebuild the cin-blocked backward-dX transposed-W buffer on main.

        Called right before submit, when W for this slot's input bank is
        guaranteed final: the Adam apply that wrote it happened inside the
        native call we just reaped, before this submit could be reached.
        Mirrors native's try_cin_blocked_dx gate exactly; on any mismatch
        native falls back to rebuilding it itself.
        """
        self._activate_tenant()
        stage = getattr(self._lib, "stage_dx_cin_blocked_wt", None)
        if stage is None:
            return
        for li in range(slot.ctx.num_layers):
            if li == 0:
                continue  # layer 0's dx is never computed; nothing to stage
            lb = slot.ctx.layers[li]
            if lb.conv_stride != 1 or lb.k_h != lb.k_w or (lb.C_in % 8) != 0:
                continue
            stage(
                ctypes.c_int32(slot_idx),
                ctypes.c_void_p(lb.W),
                ctypes.c_int64(int(lb.C_out)),
                ctypes.c_int64(int(lb.C_in)),
                ctypes.c_int64(int(lb.k_h)),
            )

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
        """Non-blocking: return post-contract result if native READY was reaped."""
        if not self._async_enabled:
            return None
        self._poll_complete_on_main()
        if self._completed is None:
            return None
        submitted = self._completed
        self._completed = None
        result = self._finish_submitted(submitted)
        self._trace_mailbox("result_consumed")
        return result

    def _finish_submitted(
        self, submitted: _SubmittedStep
    ) -> tuple[float, list[np.ndarray], list[np.ndarray], int]:
        if self._mhsa_mode:
            ws = (
                self._mhsa_workspaces[submitted.slot_idx]
                if self._mhsa_workspaces
                else self._mhsa_ws
            )
            assert ws is not None
            dw = ws.get("dW", self._mhsa_dw_f32)
            db = ws.get("db", self._mhsa_db_f32)
            assert dw is not None and db is not None
            loss = float(ws["loss"][0])
            return loss, dw, db, submitted.m

        slot = self._slots[submitted.slot_idx]
        grad_weights: list[np.ndarray | None] = [None] * len(self.model.weights)
        grad_biases: list[np.ndarray | None] = [None] * len(self.model.biases)
        for di, w_idx in enumerate(self.model._dense_w_indices):
            layer_bufs = slot.buffers.dense_layers[di]
            grad_weights[w_idx] = np.copy(layer_bufs.dW)
            grad_biases[w_idx] = np.copy(layer_bufs.db).reshape(1, -1)
        for w_idx, dW, db in slot.conv_grads:
            grad_weights[w_idx] = np.copy(dW)
            grad_biases[w_idx] = np.copy(db)

        loss = float(slot.loss_scalar[0])
        if self.model.lam_l2 > 0.0:
            l2_sum = sum(float(np.sum(w * w)) for w in self.model.weights)
            loss += (self.model.lam_l2 / (2.0 * submitted.m)) * l2_sum
        if self.model.lam_l1 > 0.0:
            l1_sum = sum(float(np.sum(np.abs(w))) for w in self.model.weights)
            loss += (self.model.lam_l1 / submitted.m) * l1_sum
        gw = grad_weights
        gb = grad_biases
        # X for this step is no longer guaranteed live / unique at this address.
        self._invalidate_input_pad_stage(submitted.slot_idx)
        return loss, gw, gb, submitted.m

    def _alloc_mhsa_workspace_buffers(self, B: int, T: int) -> dict[str, Any]:
        m = self.model
        D = int(m.d_model)
        H = int(m.num_heads)
        Hff = int(m.ffn_hidden)
        A = int(m.action_dim)
        L = int(m.num_layers)
        Dh = D // H
        rows = B * T
        key = (B, T, D, H, Hff, A, L)
        layers_ws = []
        for _ in range(L):
            layers_ws.append(
                {
                    "qkv": np.zeros((rows, 3 * D), dtype=np.float32),
                    "scores": np.zeros((B, H, T, T), dtype=np.float32),
                    "attn_out": np.zeros((rows, D), dtype=np.float32),
                    "ln1_out": np.zeros((rows, D), dtype=np.float32),
                    "h1": np.zeros((rows, D), dtype=np.float32),
                    "ln2_out": np.zeros((rows, D), dtype=np.float32),
                    "ffn_pre": np.zeros((rows, Hff), dtype=np.float32),
                    "ffn_h": np.zeros((rows, Hff), dtype=np.float32),
                    "O": np.zeros((rows, D), dtype=np.float32),
                }
            )
        return {
            "_key": key,
            "layers": layers_ws,
            "scratch": np.zeros((rows, D), dtype=np.float32),
            "actions": np.zeros((B, A), dtype=np.float32),
            "d_qkv": np.zeros((rows, 3 * D), dtype=np.float32),
            "dO": np.zeros((rows, D), dtype=np.float32),
            "dX": np.zeros((rows, D), dtype=np.float32),
            "d_stream": np.zeros((rows, D), dtype=np.float32),
            "X": np.zeros((rows, D), dtype=np.float32),
            "X_emb": np.zeros((rows, D), dtype=np.float32),
            "y": np.zeros((B, A), dtype=np.float32),
            "loss": np.zeros(1, dtype=np.float32),
            "d_pos": np.zeros((int(m.max_seq_len), D), dtype=np.float32),
            "dW_in": np.zeros((D, D), dtype=np.float32),
            "db_in": np.zeros(D, dtype=np.float32),
            "q_pack": np.zeros((T * Dh,), dtype=np.float32),
            "k_pack": np.zeros((T * Dh,), dtype=np.float32),
            "v_pack": np.zeros((T * Dh,), dtype=np.float32),
            "d_scores": np.zeros((T * T,), dtype=np.float32),
        }

    def _ensure_mhsa_workspace(self, B: int, T: int) -> None:
        m = self.model
        D = int(m.d_model)
        H = int(m.num_heads)
        Hff = int(m.ffn_hidden)
        A = int(m.action_dim)
        L = int(m.num_layers)
        key = (B, T, D, H, Hff, A, L)
        if self._mhsa_ws is not None and self._mhsa_ws.get("_key") == key:
            return
        self._mhsa_ws = self._alloc_mhsa_workspace_buffers(B, T)
        # Live f32 parameter banks (native Adam mutates these in place on sync).
        for i, w in enumerate(m.weights):
            if w.dtype != np.float32 or not w.flags["C_CONTIGUOUS"]:
                m.weights[i] = np.ascontiguousarray(w, dtype=np.float32)
        for i, b in enumerate(m.biases):
            flat = np.ascontiguousarray(b.reshape(-1), dtype=np.float32)
            m.biases[i] = flat.reshape(1, -1)
        for li in range(L):
            m.ln1_gamma[li] = np.ascontiguousarray(
                m.ln1_gamma[li].reshape(1, -1), dtype=np.float32
            )
            m.ln1_beta[li] = np.ascontiguousarray(
                m.ln1_beta[li].reshape(1, -1), dtype=np.float32
            )
            m.ln2_gamma[li] = np.ascontiguousarray(
                m.ln2_gamma[li].reshape(1, -1), dtype=np.float32
            )
            m.ln2_beta[li] = np.ascontiguousarray(
                m.ln2_beta[li].reshape(1, -1), dtype=np.float32
            )
        if getattr(m, "W_in", None) is not None:
            m.W_in = np.ascontiguousarray(m.W_in, dtype=np.float32)
            m.b_in = np.ascontiguousarray(m.b_in.reshape(1, -1), dtype=np.float32)
        if getattr(m, "pos_embed", None) is not None:
            m.pos_embed = np.ascontiguousarray(m.pos_embed, dtype=np.float32)

        self._mhsa_w_f32 = m.weights
        self._mhsa_b_f32 = [b.reshape(-1) for b in m.biases]
        self._mhsa_ln_f32 = []
        self._mhsa_dln_f32 = []
        for li in range(L):
            self._mhsa_ln_f32.append(
                {
                    "ln1_g": m.ln1_gamma[li].reshape(-1),
                    "ln1_b": m.ln1_beta[li].reshape(-1),
                    "ln2_g": m.ln2_gamma[li].reshape(-1),
                    "ln2_b": m.ln2_beta[li].reshape(-1),
                }
            )
            self._mhsa_dln_f32.append(
                {
                    "ln1_g": np.zeros(D, dtype=np.float32),
                    "ln1_b": np.zeros(D, dtype=np.float32),
                    "ln2_g": np.zeros(D, dtype=np.float32),
                    "ln2_b": np.zeros(D, dtype=np.float32),
                }
            )
        self._mhsa_dw_f32 = [np.zeros_like(w) for w in self._mhsa_w_f32]
        self._mhsa_db_f32 = [np.zeros_like(b) for b in self._mhsa_b_f32]
        self._mhsa_ws["dW"] = self._mhsa_dw_f32
        self._mhsa_ws["db"] = self._mhsa_db_f32
        self._mhsa_ws["dln"] = self._mhsa_dln_f32
        if hasattr(m, "ensure_adam_moments"):
            m.ensure_adam_moments()

    def _zero_mhsa_grads(self, slot_idx: int = 0) -> None:
        ws = (
            self._mhsa_workspaces[slot_idx]
            if self._mhsa_workspaces
            else self._mhsa_ws
        )
        assert ws is not None
        dw = ws.get("dW", self._mhsa_dw_f32)
        db = ws.get("db", self._mhsa_db_f32)
        dln = ws.get("dln", self._mhsa_dln_f32)
        assert dw is not None and db is not None and dln is not None
        for dW in dw:
            dW.fill(0.0)
        for dbi in db:
            dbi.fill(0.0)
        for dln_i in dln:
            for v in dln_i.values():
                v.fill(0.0)
        ws["dO"].fill(0.0)
        ws["dX"].fill(0.0)
        ws["d_stream"].fill(0.0)
        ws["d_qkv"].fill(0.0)
        ws["loss"].fill(0.0)
        ws["d_pos"].fill(0.0)
        ws["dW_in"].fill(0.0)
        ws["db_in"].fill(0.0)

    def _bind_mhsa(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        *,
        apply_adam: bool = False,
        slot_idx: int = 0,
        input_bank_idx: int | None = None,
        output_bank_idx: int | None = None,
    ) -> ContractExecCtx:
        """Bind MHSA geometry + float32 banks into ContractExecCtx.mhsa."""
        if X.ndim != 3:
            raise ValueError(f"MHSA input must be (B,T,D); got shape {X.shape}")
        B, T, D_in = (int(X.shape[0]), int(X.shape[1]), int(X.shape[2]))
        m = self.model
        if D_in != int(m.d_model):
            raise ValueError(f"MHSA D mismatch: X has {D_in}, model d_model={m.d_model}")
        if T > int(m.max_seq_len):
            raise ValueError(f"MHSA T={T} exceeds max_seq_len={m.max_seq_len}")

        self._ensure_mhsa_workspace(B, T)
        if self._mhsa_workspaces:
            self._ensure_mhsa_workspace_slot(B, T, slot_idx)
            ws = self._mhsa_workspaces[slot_idx]
        else:
            ws = self._mhsa_ws
        assert ws is not None

        use_banks = bool(self._mhsa_banks) and input_bank_idx is not None
        if use_banks:
            assert output_bank_idx is not None
            src = self._mhsa_banks[input_bank_idx]
            dst = self._mhsa_banks[output_bank_idx]
            w = src.weights
            b = [bb.reshape(-1) for bb in src.biases]
            ln_f32 = [
                {
                    "ln1_g": src.ln1_gamma[li].reshape(-1),
                    "ln1_b": src.ln1_beta[li].reshape(-1),
                    "ln2_g": src.ln2_gamma[li].reshape(-1),
                    "ln2_b": src.ln2_beta[li].reshape(-1),
                }
                for li in range(int(m.num_layers))
            ]
            pos_embed = src.pos_embed
            ms_pos, vs_pos = src.ms_pos, src.vs_pos
            W_in, b_in = src.W_in, src.b_in
            ms_W_in, vs_W_in = src.ms_W_in, src.vs_W_in
            ms_b_in, vs_b_in = src.ms_b_in, src.vs_b_in
            ms_w, vs_w = src.ms_w, src.vs_w
            ms_b, vs_b = src.ms_b, src.vs_b
            ms_g, vs_g = src.ms_g, src.vs_g
            ms_beta, vs_beta = src.ms_beta, src.vs_beta
            dw = ws["dW"]
            db = ws["db"]
            dln = ws["dln"]
        else:
            assert self._mhsa_w_f32 is not None
            assert self._mhsa_dw_f32 is not None and self._mhsa_db_f32 is not None
            assert self._mhsa_dln_f32 is not None and self._mhsa_ln_f32 is not None
            w = self._mhsa_w_f32
            b = self._mhsa_b_f32
            ln_f32 = self._mhsa_ln_f32
            dw = self._mhsa_dw_f32
            db = self._mhsa_db_f32
            dln = self._mhsa_dln_f32
            pos_embed = m.pos_embed
            ms_pos, vs_pos = m._ms_pos, m._vs_pos
            W_in = getattr(m, "W_in", None)
            b_in = getattr(m, "b_in", None)
            ms_W_in = getattr(m, "_ms_W_in", None)
            vs_W_in = getattr(m, "_vs_W_in", None)
            ms_b_in = getattr(m, "_ms_b_in", None)
            vs_b_in = getattr(m, "_vs_b_in", None)
            dst = None
            opt = m.optimizer
            ms_w = getattr(opt, "ms_w", None) if apply_adam else None
            vs_w = getattr(opt, "vs_w", None) if apply_adam else None
            ms_b = getattr(opt, "ms_b", None) if apply_adam else None
            vs_b = getattr(opt, "vs_b", None) if apply_adam else None
            ms_g = getattr(opt, "ms_g", None) if apply_adam else None
            vs_g = getattr(opt, "vs_g", None) if apply_adam else None
            ms_beta = getattr(opt, "ms_beta", None) if apply_adam else None
            vs_beta = getattr(opt, "vs_beta", None) if apply_adam else None

        L = int(m.num_layers)
        # Raw tokens only — native applies optional input proj then pos into X_emb.
        Xf = np.ascontiguousarray(X.reshape(B * T, D_in), dtype=np.float32)
        ws["X"] = Xf
        if y is not None:
            yf = np.ascontiguousarray(y, dtype=np.float32)
            if yf.shape != (B, int(m.action_dim)):
                raise ValueError(
                    f"MHSA target shape {yf.shape} != ({B}, {m.action_dim})"
                )
            ws["y"] = yf

        opt = m.optimizer
        if apply_adam and hasattr(m, "ensure_adam_moments"):
            m.ensure_adam_moments()

        ctx = ContractExecCtx()
        ctx.N = B
        ctx.lr = 0.0
        ctx.lam_l2 = float(getattr(m, "lam_l2", 0.0))
        ctx.max_norm = float(getattr(m, "max_norm", 5.0))
        ctx.skip_adam = 0 if apply_adam else 1
        ctx.X = _ptr(Xf)
        ctx.y = _ptr(ws["y"]) if y is not None else None
        ctx.act = None
        ctx.flat_dim = 0
        ctx.num_layers = 0
        ctx.num_dense = 0
        ctx.loss_out = _ptr(ws["loss"])
        ctx.has_mhsa = 1
        ctx.adam.beta1 = float(opt.beta1)
        ctx.adam.beta2 = float(opt.beta2)
        ctx.adam.eps = float(opt.eps)
        ctx.adam.t = int(opt.t)

        mb = ctx.mhsa
        mb.B = B
        mb.T = T
        mb.D = int(m.d_model)
        mb.H = int(m.num_heads)
        mb.d_head = int(m.d_head)
        mb.action_dim = int(m.action_dim)
        mb.ffn_hidden = int(m.ffn_hidden)
        mb.num_layers = L

        use_pos = bool(getattr(m, "use_pos_encoding", False) and pos_embed is not None)
        use_proj = bool(getattr(m, "use_input_proj", False) and W_in is not None)

        for li in range(L):
            base = 4 * li
            lb = mb.layers[li]
            lws = ws["layers"][li]
            ln = ln_f32[li]
            dln_i = dln[li]
            lb.W_qkv, lb.b_qkv = _ptr(w[base + 0]), _ptr(b[base + 0])
            lb.W_o, lb.b_o = _ptr(w[base + 1]), _ptr(b[base + 1])
            lb.W_ff1, lb.b_ff1 = _ptr(w[base + 2]), _ptr(b[base + 2])
            lb.W_ff2, lb.b_ff2 = _ptr(w[base + 3]), _ptr(b[base + 3])
            lb.dW_qkv, lb.db_qkv = _ptr(dw[base + 0]), _ptr(db[base + 0])
            lb.dW_o, lb.db_o = _ptr(dw[base + 1]), _ptr(db[base + 1])
            lb.dW_ff1, lb.db_ff1 = _ptr(dw[base + 2]), _ptr(db[base + 2])
            lb.dW_ff2, lb.db_ff2 = _ptr(dw[base + 3]), _ptr(db[base + 3])
            lb.ln1_gamma, lb.ln1_beta = _ptr(ln["ln1_g"]), _ptr(ln["ln1_b"])
            lb.ln2_gamma, lb.ln2_beta = _ptr(ln["ln2_g"]), _ptr(ln["ln2_b"])
            lb.d_ln1_gamma, lb.d_ln1_beta = _ptr(dln_i["ln1_g"]), _ptr(dln_i["ln1_b"])
            lb.d_ln2_gamma, lb.d_ln2_beta = _ptr(dln_i["ln2_g"]), _ptr(dln_i["ln2_b"])
            if apply_adam and ms_w is not None:
                g0 = 2 * li
                lb.ms_W_qkv, lb.vs_W_qkv = _ptr(ms_w[base + 0]), _ptr(vs_w[base + 0])
                lb.ms_b_qkv, lb.vs_b_qkv = _ptr(ms_b[base + 0]), _ptr(vs_b[base + 0])
                lb.ms_W_o, lb.vs_W_o = _ptr(ms_w[base + 1]), _ptr(vs_w[base + 1])
                lb.ms_b_o, lb.vs_b_o = _ptr(ms_b[base + 1]), _ptr(vs_b[base + 1])
                lb.ms_W_ff1, lb.vs_W_ff1 = _ptr(ms_w[base + 2]), _ptr(vs_w[base + 2])
                lb.ms_b_ff1, lb.vs_b_ff1 = _ptr(ms_b[base + 2]), _ptr(vs_b[base + 2])
                lb.ms_W_ff2, lb.vs_W_ff2 = _ptr(ms_w[base + 3]), _ptr(vs_w[base + 3])
                lb.ms_b_ff2, lb.vs_b_ff2 = _ptr(ms_b[base + 3]), _ptr(vs_b[base + 3])
                lb.ms_ln1_g, lb.vs_ln1_g = _ptr(ms_g[g0]), _ptr(vs_g[g0])
                lb.ms_ln1_b, lb.vs_ln1_b = _ptr(ms_beta[g0]), _ptr(vs_beta[g0])
                lb.ms_ln2_g, lb.vs_ln2_g = _ptr(ms_g[g0 + 1]), _ptr(vs_g[g0 + 1])
                lb.ms_ln2_b, lb.vs_ln2_b = _ptr(ms_beta[g0 + 1]), _ptr(vs_beta[g0 + 1])
                if dst is not None:
                    g0d = 2 * li
                    lb.W_qkv_next = _ptr(dst.weights[base + 0])
                    lb.b_qkv_next = _ptr(dst.biases[base + 0].reshape(-1))
                    lb.W_o_next = _ptr(dst.weights[base + 1])
                    lb.b_o_next = _ptr(dst.biases[base + 1].reshape(-1))
                    lb.W_ff1_next = _ptr(dst.weights[base + 2])
                    lb.b_ff1_next = _ptr(dst.biases[base + 2].reshape(-1))
                    lb.W_ff2_next = _ptr(dst.weights[base + 3])
                    lb.b_ff2_next = _ptr(dst.biases[base + 3].reshape(-1))
                    lb.ln1_gamma_next = _ptr(dst.ln1_gamma[li].reshape(-1))
                    lb.ln1_beta_next = _ptr(dst.ln1_beta[li].reshape(-1))
                    lb.ln2_gamma_next = _ptr(dst.ln2_gamma[li].reshape(-1))
                    lb.ln2_beta_next = _ptr(dst.ln2_beta[li].reshape(-1))
                    lb.ms_W_qkv_next = _ptr(dst.ms_w[base + 0])
                    lb.vs_W_qkv_next = _ptr(dst.vs_w[base + 0])
                    lb.ms_b_qkv_next = _ptr(dst.ms_b[base + 0])
                    lb.vs_b_qkv_next = _ptr(dst.vs_b[base + 0])
                    lb.ms_W_o_next = _ptr(dst.ms_w[base + 1])
                    lb.vs_W_o_next = _ptr(dst.vs_w[base + 1])
                    lb.ms_b_o_next = _ptr(dst.ms_b[base + 1])
                    lb.vs_b_o_next = _ptr(dst.vs_b[base + 1])
                    lb.ms_W_ff1_next = _ptr(dst.ms_w[base + 2])
                    lb.vs_W_ff1_next = _ptr(dst.vs_w[base + 2])
                    lb.ms_b_ff1_next = _ptr(dst.ms_b[base + 2])
                    lb.vs_b_ff1_next = _ptr(dst.vs_b[base + 2])
                    lb.ms_W_ff2_next = _ptr(dst.ms_w[base + 3])
                    lb.vs_W_ff2_next = _ptr(dst.vs_w[base + 3])
                    lb.ms_b_ff2_next = _ptr(dst.ms_b[base + 3])
                    lb.vs_b_ff2_next = _ptr(dst.vs_b[base + 3])
                    lb.ms_ln1_g_next = _ptr(dst.ms_g[g0d])
                    lb.vs_ln1_g_next = _ptr(dst.vs_g[g0d])
                    lb.ms_ln1_b_next = _ptr(dst.ms_beta[g0d])
                    lb.vs_ln1_b_next = _ptr(dst.vs_beta[g0d])
                    lb.ms_ln2_g_next = _ptr(dst.ms_g[g0d + 1])
                    lb.vs_ln2_g_next = _ptr(dst.vs_g[g0d + 1])
                    lb.ms_ln2_b_next = _ptr(dst.ms_beta[g0d + 1])
                    lb.vs_ln2_b_next = _ptr(dst.vs_beta[g0d + 1])
            lb.qkv = _ptr(lws["qkv"])
            lb.scores = _ptr(lws["scores"])
            lb.attn_out = _ptr(lws["attn_out"])
            lb.ln1_out = _ptr(lws["ln1_out"])
            lb.h1 = _ptr(lws["h1"])
            lb.ln2_out = _ptr(lws["ln2_out"])
            lb.ffn_pre = _ptr(lws["ffn_pre"])
            lb.ffn_h = _ptr(lws["ffn_h"])
            lb.O = _ptr(lws["O"])
            lb.q_pack = _ptr(ws["q_pack"])
            lb.k_pack = _ptr(ws["k_pack"])
            lb.v_pack = _ptr(ws["v_pack"])
            lb.d_scores = _ptr(ws["d_scores"])

        act_i = 4 * L
        mb.W_act, mb.b_act = _ptr(w[act_i]), _ptr(b[act_i])
        mb.dW_act, mb.db_act = _ptr(dw[act_i]), _ptr(db[act_i])
        if apply_adam and ms_w is not None:
            mb.ms_W_act, mb.vs_W_act = _ptr(ms_w[act_i]), _ptr(vs_w[act_i])
            mb.ms_b_act, mb.vs_b_act = _ptr(ms_b[act_i]), _ptr(vs_b[act_i])
            if dst is not None:
                mb.W_act_next = _ptr(dst.weights[act_i])
                mb.b_act_next = _ptr(dst.biases[act_i].reshape(-1))
                mb.ms_W_act_next = _ptr(dst.ms_w[act_i])
                mb.vs_W_act_next = _ptr(dst.vs_w[act_i])
                mb.ms_b_act_next = _ptr(dst.ms_b[act_i])
                mb.vs_b_act_next = _ptr(dst.vs_b[act_i])

        if use_pos:
            mb.pos = _ptr(pos_embed)
            mb.d_pos = _ptr(ws["d_pos"])
            mb.max_seq_len = int(m.max_seq_len)
            if apply_adam and ms_pos is not None:
                mb.ms_pos, mb.vs_pos = _ptr(ms_pos), _ptr(vs_pos)
                if dst is not None and dst.pos_embed is not None:
                    mb.pos_next = _ptr(dst.pos_embed)
                    mb.ms_pos_next = _ptr(dst.ms_pos)
                    mb.vs_pos_next = _ptr(dst.vs_pos)
        else:
            mb.pos = None
            mb.d_pos = None
            mb.ms_pos = None
            mb.vs_pos = None
            mb.max_seq_len = 0

        if use_proj:
            mb.W_in = _ptr(W_in)
            mb.b_in = _ptr(b_in.reshape(-1))
            mb.dW_in = _ptr(ws["dW_in"])
            mb.db_in = _ptr(ws["db_in"])
            if apply_adam and ms_W_in is not None:
                mb.ms_W_in, mb.vs_W_in = _ptr(ms_W_in), _ptr(vs_W_in)
                mb.ms_b_in, mb.vs_b_in = _ptr(ms_b_in.reshape(-1)), _ptr(
                    vs_b_in.reshape(-1)
                )
                if dst is not None and dst.W_in is not None:
                    mb.W_in_next = _ptr(dst.W_in)
                    mb.b_in_next = _ptr(dst.b_in.reshape(-1))
                    mb.ms_W_in_next = _ptr(dst.ms_W_in)
                    mb.vs_W_in_next = _ptr(dst.vs_W_in)
                    mb.ms_b_in_next = _ptr(dst.ms_b_in.reshape(-1))
                    mb.vs_b_in_next = _ptr(dst.vs_b_in.reshape(-1))
        else:
            mb.W_in = None
            mb.b_in = None
            mb.dW_in = None
            mb.db_in = None
            mb.ms_W_in = None
            mb.vs_W_in = None
            mb.ms_b_in = None
            mb.vs_b_in = None
            mb.W_in_next = None
            mb.b_in_next = None
            mb.ms_W_in_next = None
            mb.vs_W_in_next = None
            mb.ms_b_in_next = None
            mb.vs_b_in_next = None

        mb.X_emb = _ptr(ws["X_emb"]) if (use_pos or use_proj) else None
        mb.scratch = _ptr(ws["scratch"])
        mb.actions = _ptr(ws["actions"])
        mb.dO = _ptr(ws["dO"])
        mb.dX = _ptr(ws["dX"])
        mb.d_qkv = _ptr(ws["d_qkv"])
        mb.d_stream = _ptr(ws["d_stream"])
        mb.y = _ptr(ws["y"]) if y is not None else None
        mb.loss_out = _ptr(ws["loss"])
        mb.O = _ptr(ws["layers"][L - 1]["O"])
        self._ctx = ctx
        return ctx

    def run_mhsa_forward(self, X: np.ndarray) -> np.ndarray:
        """Forward-only MHSA contract (block + action); returns actions (B, action_dim)."""
        if not self._mhsa_mode:
            raise RuntimeError("run_mhsa_forward requires an MHSA contract")
        X = np.ascontiguousarray(X)
        ctx = self._bind_mhsa(X)
        self._activate_tenant()
        status = self._lib.run_contract_training_step(
            ctypes.cast(self._ops, ctypes.POINTER(ContractOpRow)),
            ctypes.c_int32(self._forward_op_count),
            ctypes.byref(ctx),
        )
        if status != 0:
            raise RuntimeError(f"MHSA forward contract failed status={status}")
        # Prefer the workspace that `_bind_mhsa` just wired (slot 0 if dual-bank).
        ws = (
            self._mhsa_workspaces[0]
            if self._mhsa_workspaces
            else self._mhsa_ws
        )
        assert ws is not None
        return np.copy(ws["actions"])

    def get_last_dX(self) -> np.ndarray | None:
        """Copy of ∂L/∂X from the last MHSA block backward, shape (B, T, D)."""
        if self._last_mhsa_dX is None:
            return None
        return np.copy(self._last_mhsa_dX)

    def _mhsa_ws_active(self, slot_idx: int = 0) -> dict[str, Any]:
        if self._mhsa_workspaces:
            return self._mhsa_workspaces[slot_idx]
        assert self._mhsa_ws is not None
        return self._mhsa_ws

    def _mhsa_action_backward_from_dA(
        self, dA: np.ndarray, *, slot_idx: int = 0
    ) -> None:
        """
        Fill action-head grads + dO from ∂L/∂actions (post-tanh), without MSE(y).

        Requires a prior forward that left ``actions`` and layer O live in workspace.
        """
        ws = self._mhsa_ws_active(slot_idx)
        m = self.model
        B = int(ws["actions"].shape[0])
        A = int(m.action_dim)
        D = int(m.d_model)
        # Infer T from O rows.
        O_flat = ws["layers"][int(m.num_layers) - 1]["O"]
        rows = int(O_flat.shape[0])
        if rows % B != 0:
            raise RuntimeError(f"MHSA O rows {rows} not divisible by B={B}")
        T = rows // B
        dA = np.ascontiguousarray(dA, dtype=np.float32).reshape(B, A)
        actions = ws["actions"].reshape(B, A)
        dz = dA * (1.0 - actions * actions)
        W_act = m.weights[-1]
        if W_act.dtype != np.float32:
            W_act = W_act.astype(np.float32)
        O = O_flat.reshape(B, T, D)
        last = O[:, -1, :]
        dw = ws.get("dW", self._mhsa_dw_f32)
        db = ws.get("db", self._mhsa_db_f32)
        assert dw is not None and db is not None
        dw[-1] += last.T @ dz
        db_act = np.sum(dz, axis=0)
        if db[-1].ndim == 2:
            db[-1] += db_act.reshape(1, -1)
        else:
            db[-1] += db_act.reshape(db[-1].shape)
        d_last = dz @ W_act.T
        ws["dO"].fill(0.0)
        dO = ws["dO"].reshape(B, T, D)
        dO[:, -1, :] = d_last
        # Report mean |dA| proxy in loss slot for debugging (not MSE).
        ws["loss"][0] = float(np.mean(np.abs(dA)))

    def run_mhsa_backward_from_dA(
        self,
        X: np.ndarray,
        dA: np.ndarray,
        *,
        apply_adam: bool = False,
        lr: float = 0.0,
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """
        Re-forward X, inject external ∂L/∂actions, run BLOCK_BWD (Adam optional).

        Returns (dW, db, dX) with dX shaped (B, T, D). Does not require y.
        """
        if not self._mhsa_mode:
            raise RuntimeError("run_mhsa_backward_from_dA requires an MHSA contract")
        if self._mhsa_do_bwd_ops is None or self._mhsa_do_bwd_op_count < 1:
            raise RuntimeError("MHSA contract missing BLOCK_BWD tail ops")
        X = np.ascontiguousarray(X, dtype=np.float32)
        ctx = self._bind_mhsa(X, y=None, apply_adam=apply_adam)
        self._zero_mhsa_grads()
        ctx.lr = float(lr)
        ctx.skip_adam = 0 if apply_adam else 1
        ctx.adam.t = int(self.model.optimizer.t)
        self._activate_tenant()
        status = self._lib.run_contract_training_step(
            ctypes.cast(self._ops, ctypes.POINTER(ContractOpRow)),
            ctypes.c_int32(self._forward_op_count),
            ctypes.byref(ctx),
        )
        if status != 0:
            raise RuntimeError(f"MHSA forward (for dA bwd) failed status={status}")
        self._mhsa_action_backward_from_dA(dA, slot_idx=0)
        status = self._lib.run_contract_training_step(
            ctypes.cast(self._mhsa_do_bwd_ops, ctypes.POINTER(ContractOpRow)),
            ctypes.c_int32(self._mhsa_do_bwd_op_count),
            ctypes.byref(ctx),
        )
        if status != 0:
            raise RuntimeError(f"MHSA BLOCK_BWD from dA failed status={status}")
        if apply_adam:
            self.model.optimizer.t = int(ctx.adam.t)
        ws = self._mhsa_ws_active(0)
        B, T, D = int(X.shape[0]), int(X.shape[1]), int(X.shape[2])
        dX = np.array(ws["dX"], dtype=np.float32, copy=True).reshape(B, T, D)
        self._last_mhsa_dX = dX
        dw = ws.get("dW", self._mhsa_dw_f32)
        db = ws.get("db", self._mhsa_db_f32)
        assert dw is not None and db is not None
        if apply_adam:
            return dw, db, dX
        grad_weights = [np.array(g, dtype=np.float64) for g in dw]
        grad_biases = [
            np.array(g, dtype=np.float64).reshape(1, -1) for g in db
        ]
        return grad_weights, grad_biases, dX

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
        del tick_fn, step_token
        if self._mhsa_mode:
            X = np.ascontiguousarray(X)
            y = np.ascontiguousarray(y)
            ctx = self._bind_mhsa(X, y, apply_adam=apply_adam)
            self._zero_mhsa_grads()
            ctx.lr = float(lr)
            ctx.skip_adam = 0 if apply_adam else 1
            ctx.adam.t = int(self.model.optimizer.t)
            self._activate_tenant()
            status = self._lib.run_contract_training_step(
                ctypes.cast(self._ops, ctypes.POINTER(ContractOpRow)),
                ctypes.c_int32(self.contract.op_count),
                ctypes.byref(ctx),
            )
            if status != 0:
                raise RuntimeError(f"MHSA train contract failed status={status}")
            assert self._mhsa_ws is not None and self._mhsa_dw_f32 is not None
            assert self._mhsa_db_f32 is not None and self._mhsa_dln_f32 is not None
            m = int(X.shape[0])
            loss = float(self._mhsa_ws["loss"][0])
            B, T, D = int(X.shape[0]), int(X.shape[1]), int(X.shape[2])
            self._last_mhsa_dX = np.array(
                self._mhsa_ws["dX"], dtype=np.float32, copy=True
            ).reshape(B, T, D)
            if apply_adam:
                self.model.optimizer.t = int(ctx.adam.t)
                return loss, self._mhsa_dw_f32, self._mhsa_db_f32, m
            grad_weights = [np.array(g, dtype=np.float64) for g in self._mhsa_dw_f32]
            grad_biases = [
                np.array(g, dtype=np.float64).reshape(1, -1) for g in self._mhsa_db_f32
            ]
            return loss, grad_weights, grad_biases, m

        X = np.ascontiguousarray(X)
        y = np.ascontiguousarray(y)
        m = int(X.shape[0])

        ctx, bound = self._bind_conv_layers(X, m)
        self._bind_dense(ctx, m, X.dtype)
        self._refresh_step_bindings(X, y, m, lr, apply_adam=apply_adam)

        self._stage_sync_useful_work(ctx, X)
        self._invoke_native_sync(ctx)

        loss, grad_weights, grad_biases = self._collect_grads(
            m, X.dtype, y, bound, apply_adam=apply_adam, ctx=ctx
        )
        return loss, grad_weights, grad_biases, m

    def _invoke_native_sync(self, ctx: ContractExecCtx) -> None:
        self._activate_tenant()
        status = self._lib.run_contract_training_step(
            ctypes.cast(self._ops, ctypes.POINTER(ContractOpRow)),
            ctypes.c_int32(self.contract.op_count),
            ctypes.byref(ctx),
        )
        # Drop staged x_pad / brgemm packs keyed on this step's X pointer so a
        # later allocation that reuses the address cannot false-hit stale pads
        # (breaks subsequent non-contract sync forwards).
        self._invalidate_input_pad_stage()
        if status != 0:
            raise RuntimeError(f"run_contract_training_step failed with status {status}")

    def _submit_native(self, ctx: ContractExecCtx, step_token: int) -> None:
        self._activate_tenant()
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
        self._activate_tenant()
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

    def _wait_reap_native(self, timeout_ms: int) -> bool:
        self._activate_tenant()
        out_token = ctypes.c_int64()
        out_status = ctypes.c_int32()
        rc = self._lib.wait_contract_completion(
            ctypes.byref(out_token),
            ctypes.byref(out_status),
            ctypes.c_int64(timeout_ms),
        )
        if rc == 0:
            return False
        if rc != 1:
            raise RuntimeError(f"wait_contract_completion failed with status {rc}")
        if self._pending_token is not None and out_token.value != self._pending_token:
            raise RuntimeError(
                f"async completion token mismatch: expected {self._pending_token}, "
                f"got {out_token.value}"
            )
        if out_status.value != 0:
            raise RuntimeError(
                f"async contract step token={out_token.value} failed with status "
                f"{out_status.value}"
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
        grad_weights: list[np.ndarray | None] = [None] * len(self.model.weights)
        grad_biases: list[np.ndarray | None] = [None] * len(self.model.biases)

        for di, w_idx in enumerate(self.model._dense_w_indices):
            layer_bufs = bufs.dense_layers[di]
            grad_weights[w_idx] = np.copy(layer_bufs.dW)
            grad_biases[w_idx] = np.copy(layer_bufs.db).reshape(1, -1)

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
