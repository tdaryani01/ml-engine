# src/closed_loop/interleaver.py
"""Pack goal + state + action tokens into a causal MHSA sequence."""
from __future__ import annotations

import numpy as np


class TokenInterleaver:
    """
    Frozen layout (B, T_seq, D):

        X = [G, S_1, A_1, S_2, A_2, …, S_t]

    At prediction step ``t`` (1-indexed), the sequence ends on ``S_t`` so the
    MHSA last-token action head reads the current visual state while attending
    back to ``G`` and prior (S, A) pairs.

    Sequence length at step t: ``seq_len(t) = 2 * t``.

    Index helpers:
      - goal: 0
      - state S_k  (k=1..t):  1 + 2*(k-1)
      - action A_k (k=1..t-1): 2 + 2*(k-1)
    """

    def __init__(self, d_model: int) -> None:
        self.d_model = int(d_model)

    @staticmethod
    def seq_len(t: int) -> int:
        """Token count when predicting action at 1-indexed step t."""
        if t < 1:
            raise ValueError(f"t must be >= 1, got {t}")
        return 2 * int(t)

    @staticmethod
    def state_index(k: int) -> int:
        """Absolute index of S_k (k 1-indexed)."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        return 1 + 2 * (int(k) - 1)

    @staticmethod
    def action_index(k: int) -> int:
        """Absolute index of A_k (k 1-indexed), only valid once A_k is packed."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        return 2 + 2 * (int(k) - 1)

    def build(
        self,
        goal: np.ndarray,
        states: list[np.ndarray],
        action_embeds: list[np.ndarray],
    ) -> np.ndarray:
        """
        Build X for predicting the next action after ``len(states)`` states.

        ``action_embeds`` must have length ``len(states) - 1`` (prior actions).
        """
        goal = np.ascontiguousarray(goal, dtype=np.float32)
        if goal.ndim != 2 or goal.shape[1] != self.d_model:
            raise ValueError(f"goal must be (B,{self.d_model}), got {goal.shape}")
        B = int(goal.shape[0])
        t = len(states)
        if t < 1:
            raise ValueError("need at least one state token S_1")
        if len(action_embeds) != t - 1:
            raise ValueError(
                f"expected {t - 1} prior action embeds, got {len(action_embeds)}"
            )

        T = self.seq_len(t)
        X = np.zeros((B, T, self.d_model), dtype=np.float32)
        X[:, 0, :] = goal
        for k, S in enumerate(states, start=1):
            S = np.ascontiguousarray(S, dtype=np.float32)
            if S.shape != (B, self.d_model):
                raise ValueError(f"S_{k} shape {S.shape} != ({B},{self.d_model})")
            X[:, self.state_index(k), :] = S
        for k, A in enumerate(action_embeds, start=1):
            A = np.ascontiguousarray(A, dtype=np.float32)
            if A.shape != (B, self.d_model):
                raise ValueError(f"A_{k} shape {A.shape} != ({B},{self.d_model})")
            X[:, self.action_index(k), :] = A
        return X

    def split_dX(
        self, dX: np.ndarray, t: int
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """
        Split ∂L/∂X at prediction step t into (dG, dS_1..dS_t, dA_1..dA_{t-1}).
        """
        dX = np.ascontiguousarray(dX, dtype=np.float32)
        if dX.ndim != 3 or dX.shape[1] != self.seq_len(t) or dX.shape[2] != self.d_model:
            raise ValueError(
                f"dX shape {dX.shape} incompatible with t={t}, D={self.d_model}"
            )
        dG = np.copy(dX[:, 0, :])
        d_states = [np.copy(dX[:, self.state_index(k), :]) for k in range(1, t + 1)]
        d_actions = [
            np.copy(dX[:, self.action_index(k), :]) for k in range(1, t)
        ]
        return dG, d_states, d_actions
