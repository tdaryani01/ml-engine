# src/contract.py
"""Phase F: compiled contract list (writer side). Native executes; Python compiles once at init."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, List, Optional


class ContractOp(IntEnum):
    """Fixed opcodes — native dispatch table must match these values."""

    CONV2D_FWD = 1
    CONV2D_BWD = 2
    RELU_FWD = 3
    RELU_BWD = 4
    MAXPOOL_FWD = 5
    MAXPOOL_BWD = 6
    FLATTEN_FWD = 7
    FLATTEN_BWD = 8
    DENSE_FWD = 9
    DENSE_BWD = 10
    ADAM_APPLY = 11
    # Fused block (ConvBlock path) — one native entry per fused spatial block
    CONV_BLOCK_FWD = 20
    CONV_BLOCK_BWD = 21


@dataclass
class ContractOpDesc:
    """One row in the contract list — opaque to Python at runtime."""

    opcode: ContractOp
    layer_idx: int
    param_idx: int = -1
    # Geometry / flags packed for native (stride, pad, pool, etc.)
    flags: int = 0
    i0: int = 0
    i1: int = 0
    i2: int = 0


@dataclass
class ContractList:
    """Compiled plan for one training step. Built at init; batch handles bound per submit."""

    graph_id: str
    ops: List[ContractOpDesc] = field(default_factory=list)
    num_params: int = 0

    def to_bytes(self) -> bytes:
        """Serialize for native (fixed layout: u32 opcode, i32 layer, i32 param, u32 flags, i32×3)."""
        import struct

        parts = [struct.pack("<i", len(self.ops))]
        for op in self.ops:
            parts.append(
                struct.pack(
                    "<iiiiiii",
                    int(op.opcode),
                    op.layer_idx,
                    op.param_idx,
                    op.flags,
                    op.i0,
                    op.i1,
                    op.i2,
                )
            )
        return b"".join(parts)

    @property
    def op_count(self) -> int:
        return len(self.ops)


def compile_cnn_training_step(
    layers: list[Any],
    *,
    layer_param_idx: dict[int, int],
    dense_w_indices: list[int],
    graph_id: str = "cnn_v1",
) -> ContractList:
    """
    Build forward + backward op list from a live CNN layer stack.
    Order: all forward ops (input→output), then all backward ops (output→input).
    """
    contract = ContractList(graph_id=graph_id, num_params=len(dense_w_indices) + len(layer_param_idx))

    # Forward pass
    for li, layer in enumerate(layers):
        from src.spatial_layers import Conv2D, ConvBlock, Flatten, MaxPool2D

        if isinstance(layer, ConvBlock):
            pidx = layer_param_idx.get(li, -1)
            contract.ops.append(
                ContractOpDesc(
                    opcode=ContractOp.CONV_BLOCK_FWD,
                    layer_idx=li,
                    param_idx=pidx,
                    flags=1,
                    i0=layer.conv_stride,
                    i1=layer.conv_pad,
                    i2=layer.pool_size,
                )
            )
        elif isinstance(layer, Conv2D):
            pidx = layer_param_idx.get(li, -1)
            contract.ops.append(
                ContractOpDesc(
                    opcode=ContractOp.CONV2D_FWD,
                    layer_idx=li,
                    param_idx=pidx,
                    i0=layer.stride,
                    i1=layer.pad,
                )
            )
        elif isinstance(layer, MaxPool2D):
            contract.ops.append(
                ContractOpDesc(
                    opcode=ContractOp.MAXPOOL_FWD,
                    layer_idx=li,
                    i0=layer.pool_size,
                    i1=layer.stride,
                )
            )
        elif isinstance(layer, Flatten):
            contract.ops.append(ContractOpDesc(opcode=ContractOp.FLATTEN_FWD, layer_idx=li))
        elif layer == "relu":
            contract.ops.append(ContractOpDesc(opcode=ContractOp.RELU_FWD, layer_idx=li))

    for di, w_idx in enumerate(dense_w_indices):
        contract.ops.append(
            ContractOpDesc(opcode=ContractOp.DENSE_FWD, layer_idx=di, param_idx=w_idx)
        )

    # Backward pass (reverse): dense head first, then spatial
    for di in reversed(range(len(dense_w_indices))):
        w_idx = dense_w_indices[di]
        contract.ops.append(
            ContractOpDesc(opcode=ContractOp.DENSE_BWD, layer_idx=di, param_idx=w_idx)
        )

    for li in range(len(layers) - 1, -1, -1):
        layer = layers[li]
        from src.spatial_layers import Conv2D, ConvBlock, Flatten, MaxPool2D

        if isinstance(layer, ConvBlock):
            pidx = layer_param_idx.get(li, -1)
            contract.ops.append(
                ContractOpDesc(
                    opcode=ContractOp.CONV_BLOCK_BWD,
                    layer_idx=li,
                    param_idx=pidx,
                    flags=1,
                )
            )
        elif isinstance(layer, Conv2D):
            pidx = layer_param_idx.get(li, -1)
            contract.ops.append(ContractOpDesc(opcode=ContractOp.CONV2D_BWD, layer_idx=li, param_idx=pidx))
        elif isinstance(layer, MaxPool2D):
            contract.ops.append(ContractOpDesc(opcode=ContractOp.MAXPOOL_BWD, layer_idx=li))
        elif isinstance(layer, Flatten):
            contract.ops.append(ContractOpDesc(opcode=ContractOp.FLATTEN_BWD, layer_idx=li))
        elif layer == "relu":
            contract.ops.append(ContractOpDesc(opcode=ContractOp.RELU_BWD, layer_idx=li))

    contract.ops.append(ContractOpDesc(opcode=ContractOp.ADAM_APPLY, layer_idx=-1))

    return contract
