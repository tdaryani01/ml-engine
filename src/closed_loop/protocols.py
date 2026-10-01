# src/closed_loop/protocols.py
"""Generic plug points for closed-loop trajectory training.

Phase 1: the trainer is NOT yet wired to these protocols. They exist so the
boundaries (Environment / Actor / LossEvaluator) are explicit and the
app-layer adapters can be validated against them.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

# Opaque per-batch goal payload. Interpreted ONLY by the Actor / Environment /
# LossEvaluator. For drawing this is ``command_ids`` (policy condition); the
# loss evaluator receives whatever payload it was configured for.
Goal = Any
# Opaque observation tensor with a leading batch dimension.
Obs = np.ndarray


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

    def obs_spec(self) -> tuple[int, ...]:
        """Shape of a single observation, excluding the batch dimension."""
        ...

    def reset(self, batch_size: int, goal: Goal) -> Obs:
        """Return initial observation (B, …) for the batch."""
        ...

    def step(self, action: np.ndarray) -> Obs:
        """Apply action → next obs; stash info needed for the reverse pass."""
        ...


@runtime_checkable
class DifferentiableEnvironment(Environment, Protocol):
    """Environment that exposes observation history + transition gradients."""

    def obs_at(self, t: int) -> Obs:
        """Observation produced at step ``t`` (0-indexed)."""
        ...

    def step_backward(
        self, t: int, d_obs_after: Obs
    ) -> tuple[np.ndarray, Obs | None]:
        """∂L/∂action_t and carried ∂L/∂prior_obs_t (or None)."""
        ...

    def aux_loss(self) -> tuple[float, list[np.ndarray]]:
        """Trajectory prior: ``(loss, [∂loss/∂action_t])``."""
        ...


@runtime_checkable
class Actor(Protocol):
    """Policy plugin: obs history → action.

    Owns goal conditioning and the token grammar. For the draw stack this
    absorbs ConditioningBank, the V→D and A→D LinearAdapters, MHSANetwork, and
    TokenInterleaver.
    """

    action_dim: int

    def reset(self, batch_size: int, goal: Goal, max_steps: int) -> None:
        """Clear caches and bind the goal + trajectory length for a new rollout."""
        ...

    def act(self, obs: Obs) -> np.ndarray:
        """Forward one step → action (B, action_dim); caches internals."""
        ...

    def accumulate_grads(self, t: int, d_action: np.ndarray) -> None:
        """Reverse-pass step: accumulate ∂L/∂action_t (no optimizer step)."""
        ...

    def backward_done(self) -> None:
        """Flush deferred grads after the reverse sweep."""
        ...

    def apply_updates(self, lr: float) -> None:
        """Single optimizer step over all actor params."""
        ...

    def zero_grad(self) -> None:
        """Clear accumulated parameter grads."""
        ...

    def checkpoint(self, version: int, val_loss: float) -> bytes:
        """Serialize actor weights + config for a checkpoint blob."""
        ...

    def seq_len(self, t: int) -> int:
        """Policy token-grammar length when predicting the action at step ``t``."""
        ...


@runtime_checkable
class LossEvaluator(Protocol):
    """Per-step (or terminal) checker scored against an opaque goal payload."""

    def step_loss(self, obs: Obs, goal: Goal, t: int) -> float:
        """Scalar loss contribution at step t (after env.step)."""
        ...

    def step_obs_grad(self, obs: Obs, goal: Goal, t: int) -> np.ndarray:
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


# Back-compat: the trainer (until Phase 2) imports the old name.
TrajectoryLoss = LossEvaluator

# Marker for optional stash bags during rollout (app-defined).
RolloutAux = dict[str, Any]


__all__ = [
    "Actor",
    "ConditioningBankProto",
    "DifferentiableEnvironment",
    "Environment",
    "Goal",
    "LossEvaluator",
    "Obs",
    "RolloutAux",
    "TrajectoryLoss",
    "UpstreamEncoder",
]
