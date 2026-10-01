# src/closed_loop/trainer.py
"""Python-orchestrated closed-loop rollout over pluggable plugins.

The loop is task-agnostic. It owns only:

  * the trajectory schedule (``max_steps``),
  * the forward sweep (``actor.act`` ⇄ ``env.step``),
  * the reverse sweep (``loss_fn`` → ``env.step_backward`` → ``actor``),
  * loss aggregation and the checkpoint surface.

Domain mechanics live behind :class:`Actor`, :class:`Environment` and
:class:`LossEvaluator` — see ``src/closed_loop/protocols.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.closed_loop.protocols import (
    Actor,
    DifferentiableEnvironment,
    Environment,
    Goal,
    LossEvaluator,
)


@dataclass
class RolloutResult:
    total_loss: float
    actions: list[np.ndarray]
    seq_lens: list[int]
    extras: dict[str, Any] = field(default_factory=dict)


def _goal_batch_size(goal: Goal) -> int:
    """Batch-dimension convention: a goal exposes ``batch_size`` or is an array."""
    explicit = getattr(goal, "batch_size", None)
    if explicit is not None:
        return int(explicit)
    return int(np.asarray(goal).reshape(-1).shape[0])


class ClosedLoopTrainer:
    """Generic closed-loop rollout: actor ⇄ environment, scored by an evaluator.

    Actor grads are deferred: the reverse sweep accumulates, then one
    ``actor.apply_updates(lr)`` performs the optimizer step.
    """

    def __init__(
        self,
        *,
        actor: Actor,
        env: Environment,
        loss_fn: LossEvaluator,
        max_steps: int,
    ) -> None:
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.actor = actor
        self.env = env
        self.loss_fn = loss_fn

    # -- internals ---------------------------------------------------------

    def _seq_lens(self) -> list[int]:
        return [int(self.actor.seq_len(t)) for t in range(1, self.max_steps + 1)]

    def _forward(
        self, goal: Goal, *, score: bool
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[float], float]:
        """Run the trajectory. Returns (frames, actions, step_losses, total)."""
        batch = _goal_batch_size(goal)
        self.actor.reset(batch, goal, self.max_steps)
        obs = self.env.reset(batch, goal)
        frames: list[np.ndarray] = [np.array(obs, copy=True)]
        actions: list[np.ndarray] = []
        step_losses: list[float] = []
        total = 0.0
        for t in range(1, self.max_steps + 1):
            action = np.ascontiguousarray(self.actor.act(obs), dtype=np.float32)
            actions.append(action)
            obs = self.env.step(action)
            frames.append(np.array(obs, copy=True))
            if score:
                step_loss = float(self.loss_fn.step_loss(obs, goal, t - 1))
                step_losses.append(step_loss)
                total += step_loss
        return frames, actions, step_losses, total

    # -- public ------------------------------------------------------------

    def rollout_eval(self, *, goal: Goal) -> RolloutResult:
        """Inference-only rollout. No loss, no gradients; frames in ``extras``."""
        self.actor.zero_grad()
        frames, actions, _unused, _zero = self._forward(goal, score=False)
        self.actor.zero_grad()  # drop forward caches
        return RolloutResult(
            total_loss=0.0,
            actions=actions,
            seq_lens=self._seq_lens(),
            extras={"frames": frames},
        )

    def rollout_train(
        self, *, goal: Goal, lr: float, apply_updates: bool = True
    ) -> RolloutResult:
        """One trajectory: forward, reverse BPTT, optional optimizer step."""
        if not isinstance(self.env, DifferentiableEnvironment):
            raise RuntimeError(
                "rollout_train requires a DifferentiableEnvironment "
                "(obs_at / step_backward / aux_loss)"
            )
        self.actor.zero_grad()
        frames, actions, step_losses, total_loss = self._forward(goal, score=True)

        aux_loss, d_aux = self.env.aux_loss()
        total_loss += float(aux_loss)

        d_carry: np.ndarray | None = np.zeros_like(
            frames[self.max_steps], dtype=np.float32
        )
        for t in range(self.max_steps, 0, -1):
            ti = t - 1
            d_loss = self.loss_fn.step_obs_grad(frames[t], goal, ti)
            d_after = d_loss if d_carry is None else (d_loss + d_carry)
            dA, d_carry = self.env.step_backward(ti, d_after)
            if d_aux is not None:
                dA = dA + d_aux[ti]
            self.actor.accumulate_grads(t, dA)
        self.actor.backward_done()

        if apply_updates:
            self.actor.apply_updates(lr)

        extras: dict[str, Any] = {"step_losses": step_losses, "frames": frames}
        if aux_loss:
            extras["aux_loss"] = float(aux_loss)

        return RolloutResult(
            total_loss=float(total_loss),
            actions=actions,
            seq_lens=self._seq_lens(),
            extras=extras,
        )


__all__ = ["ClosedLoopTrainer", "RolloutResult"]
