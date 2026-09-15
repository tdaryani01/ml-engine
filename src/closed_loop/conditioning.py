# src/closed_loop/conditioning.py
"""Trainable goal / command embedding bank."""
from __future__ import annotations

import numpy as np


class ConditioningBank:
    """
    Dictionary G ∈ R^{K×D}. Step 0 of every interleaved sequence is G[cmd].

    Adam moments are owned here so the bank can update independently of MHSA.
    """

    def __init__(self, num_commands: int, d_model: int, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.num_commands = int(num_commands)
        self.d_model = int(d_model)
        self.embeddings = (
            rng.standard_normal((self.num_commands, self.d_model)).astype(np.float32)
            * 0.02
        )
        self._d_emb = np.zeros_like(self.embeddings)
        self._m = np.zeros_like(self.embeddings)
        self._v = np.zeros_like(self.embeddings)
        self._t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def embed(self, command_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(command_ids, dtype=np.int64).reshape(-1)
        if ids.min() < 0 or ids.max() >= self.num_commands:
            raise ValueError(
                f"command_ids out of range [0,{self.num_commands}): "
                f"min={ids.min()} max={ids.max()}"
            )
        return np.ascontiguousarray(self.embeddings[ids])

    def backward(self, dG: np.ndarray, command_ids: np.ndarray) -> None:
        ids = np.asarray(command_ids, dtype=np.int64).reshape(-1)
        dG = np.asarray(dG, dtype=np.float32)
        if dG.shape != (ids.shape[0], self.d_model):
            raise ValueError(f"dG shape {dG.shape} != ({ids.shape[0]}, {self.d_model})")
        for i, cid in enumerate(ids):
            self._d_emb[int(cid)] += dG[i]

    def zero_grad(self) -> None:
        self._d_emb.fill(0.0)

    def apply_pending(self, lr: float) -> None:
        self._t += 1
        g = self._d_emb
        self._m = self.beta1 * self._m + (1.0 - self.beta1) * g
        self._v = self.beta2 * self._v + (1.0 - self.beta2) * (g * g)
        m_hat = self._m / (1.0 - self.beta1**self._t)
        v_hat = self._v / (1.0 - self.beta2**self._t)
        self.embeddings -= np.float32(lr) * m_hat / (np.sqrt(v_hat) + self.eps)
        self.zero_grad()
