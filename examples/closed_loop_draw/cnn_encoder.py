# examples/closed_loop_draw/cnn_encoder.py
"""CNN-as-UpstreamEncoder wrapper (app layer)."""
from __future__ import annotations

import numpy as np

from src.cnn_network import CNNNetwork
from src.training_cache import ForwardCache, new_forward_cache


class CnnUpstreamEncoder:
    """
    Wraps a CNNNetwork as a closed-loop UpstreamEncoder.

    ``encode`` runs a training forward and returns the network output.
    Multi-step BPTT stashes one ForwardCache per encode (``backward_at``).
    """

    def __init__(self, cnn: CNNNetwork) -> None:
        self.cnn = cnn
        self._caches: list[ForwardCache] = []
        self._out_dim: int | None = None

    @property
    def out_dim(self) -> int:
        if self._out_dim is None:
            raise RuntimeError("encode once before reading out_dim")
        return int(self._out_dim)

    def encode(self, obs: np.ndarray) -> np.ndarray:
        obs = np.ascontiguousarray(obs, dtype=np.float32)
        cache = new_forward_cache(len(self.cnn.layers), len(self.cnn._dense_w_indices))
        out = self.cnn._forward(obs, training=True, cache=cache)
        self._caches.append(cache)
        self._out_dim = int(out.shape[1])
        return np.ascontiguousarray(out, dtype=np.float32)

    def backward(self, dV: np.ndarray) -> None:
        if not self._caches:
            raise RuntimeError("backward without encode")
        self.backward_at(len(self._caches) - 1, dV)

    def backward_at(self, step_idx: int, dV: np.ndarray) -> None:
        cache = self._caches[step_idx]
        self.cnn.accumulate_grads_from_delta(cache, dV)

    def zero_grad(self) -> None:
        self.cnn.zero_pending_grads()
        self._caches.clear()

    def apply_pending(self, lr: float) -> None:
        self.cnn.apply_pending_grads(float(lr))
        self._caches.clear()
