# src/closed_loop/adapter.py
"""Linear bridge V→D (and optional A→D action token embed)."""
from __future__ import annotations

import numpy as np


class LinearAdapter:
    """
    Trainable affine map: Y = X @ W + b.

    W is (D_in, D_out). Used for CNN→MHSA features and action→token embeds.
    """

    def __init__(self, d_in: int, d_out: int, *, seed: int = 0) -> None:
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        limit = np.sqrt(6.0 / (self.d_in + self.d_out))
        rng = np.random.default_rng(seed)
        self.W = rng.uniform(-limit, limit, (self.d_in, self.d_out)).astype(np.float32)
        self.b = np.zeros((1, self.d_out), dtype=np.float32)
        self._dW = np.zeros_like(self.W)
        self._db = np.zeros_like(self.b)
        self._mW = np.zeros_like(self.W)
        self._vW = np.zeros_like(self.W)
        self._mb = np.zeros_like(self.b)
        self._vb = np.zeros_like(self.b)
        self._t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self._last_x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.d_in:
            raise ValueError(f"expected (B,{self.d_in}), got {x.shape}")
        self._last_x = x
        return x @ self.W + self.b

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """Accumulate dW/db; return dX."""
        if self._last_x is None:
            raise RuntimeError("LinearAdapter.backward without forward")
        dY = np.ascontiguousarray(dY, dtype=np.float32)
        x = self._last_x
        if dY.shape != (x.shape[0], self.d_out):
            raise ValueError(f"dY shape {dY.shape} != ({x.shape[0]}, {self.d_out})")
        self._dW += x.T @ dY
        self._db += np.sum(dY, axis=0, keepdims=True)
        return dY @ self.W.T

    def zero_grad(self) -> None:
        self._dW.fill(0.0)
        self._db.fill(0.0)

    def apply_pending(self, lr: float) -> None:
        self._t += 1
        lr = float(lr)
        for p, g, m, v in (
            (self.W, self._dW, self._mW, self._vW),
            (self.b, self._db, self._mb, self._vb),
        ):
            m[:] = self.beta1 * m + (1.0 - self.beta1) * g
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (g * g)
            m_hat = m / (1.0 - self.beta1**self._t)
            v_hat = v / (1.0 - self.beta2**self._t)
            p -= np.float32(lr) * m_hat / (np.sqrt(v_hat) + self.eps)
        self.zero_grad()
