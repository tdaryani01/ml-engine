# src/training_cache.py
"""Per-batch forward state for training backward (model-agnostic step cache)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.scratch_arena import ConvBlockScratch


@dataclass
class ConvBlockStepCache:
    x: np.ndarray
    conv_act: np.ndarray
    argmax: np.ndarray
    col: np.ndarray | None = None
    scratch: ConvBlockScratch | None = None


@dataclass
class Conv2DStepCache:
    x: np.ndarray
    col: np.ndarray | None = None


@dataclass
class MaxPoolStepCache:
    x_shape: tuple
    pool_cache: np.ndarray


@dataclass
class FlattenStepCache:
    logical_shape: tuple
    padded_shape: tuple


@dataclass
class ForwardCache:
    """
    Activations and layer step state from one training forward pass.

    Used by backward to avoid re-forwarding. Model-agnostic: any architecture
    can populate the lists and typed step dicts by layer index.
    """

    spatial_inputs: list = field(default_factory=list)
    spatial_logical_ws: list = field(default_factory=list)
    dense_inputs: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    activations: list = field(default_factory=list)
    conv_blocks: dict[int, ConvBlockStepCache] = field(default_factory=dict)
    conv2d: dict[int, Conv2DStepCache] = field(default_factory=dict)
    maxpool: dict[int, MaxPoolStepCache] = field(default_factory=dict)
    flatten: dict[int, FlattenStepCache] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        if not self.activations:
            raise ValueError("ForwardCache has no activations")
        return int(self.activations[0].shape[0])

    @property
    def output(self) -> np.ndarray:
        if not self.activations:
            raise ValueError("ForwardCache has no activations")
        return self.activations[-1]


def new_forward_cache(num_layers: int, num_dense: int) -> ForwardCache:
    """Pre-sized step store; spatial slots indexed by pipeline layer index."""
    return ForwardCache(
        spatial_inputs=[None] * num_layers,
        spatial_logical_ws=[None] * num_layers,
        dense_inputs=[None] * num_dense,
        masks=[None] * num_dense,
        activations=[],
    )
