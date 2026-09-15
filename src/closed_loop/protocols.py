# src/closed_loop/protocols.py
"""Plug points for closed-loop trajectory training."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class UpstreamEncoder(Protocol):
    """Maps observations to feature vectors V ∈ R^{B×D_in}."""

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """Forward; may cache activations for a matching ``backward``."""
        ...

    def backward(self, dV: np.ndarray) -> None:
        """Accumulate parameter grads from ∂L/∂V (same batch as last ``encode``)."""
        ...

    def apply_pending(self, lr: float) -> None:
        """Apply accumulated grads (Adam/SGD) once per trajectory."""
        ...

    def zero_grad(self) -> None:
        """Clear accumulated parameter grads."""
        ...


@runtime_checkable
class Environment(Protocol):
    """Differentiable (or soft) world step; owns obs state."""

    def reset(self, batch_size: int) -> np.ndarray:
        """Return initial observation (B, …)."""
        ...

    def step(self, action: np.ndarray) -> np.ndarray:
        """Apply action → next obs; stash info needed for ``action_grad``."""
        ...

    def action_grad(self, d_obs: np.ndarray) -> np.ndarray:
        """
        ∂L/∂action from ∂L/∂obs_after_step (and any deferred canvas chain).

        Must match the last ``step`` call. May also accumulate ∂L/∂obs_before
        into an internal buffer for BPTT through the observation chain.
        """
        ...

    def pop_obs_grad(self) -> np.ndarray | None:
        """Optional ∂L/∂obs_before from the last ``action_grad`` (for encoder path)."""
        ...


@runtime_checkable
class TrajectoryLoss(Protocol):
    """Pluggable per-step (or terminal) checker."""

    def step_loss(self, obs: np.ndarray, target: np.ndarray, t: int) -> float:
        """Scalar loss contribution at step t (after env.step)."""
        ...

    def step_obs_grad(self, obs: np.ndarray, target: np.ndarray, t: int) -> np.ndarray:
        """∂(step_loss)/∂obs at step t."""
        ...


@runtime_checkable
class ConditioningBankProto(Protocol):
    """Trainable goal / command embeddings G ∈ R^{K×D}."""

    def embed(self, command_ids: np.ndarray) -> np.ndarray:
        """command_ids (B,) int → G (B, D)."""
        ...

    def backward(self, dG: np.ndarray, command_ids: np.ndarray) -> None:
        """Accumulate embedding table grads."""
        ...

    def apply_pending(self, lr: float) -> None:
        ...

    def zero_grad(self) -> None:
        ...


# Marker for optional stash bags during rollout (app-defined).
RolloutAux = dict[str, Any]
