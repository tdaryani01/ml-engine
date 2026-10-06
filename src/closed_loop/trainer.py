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
        feedback: Any = None,
    ) -> None:
        self.feedback = feedback if feedback is not None else BackpropFeedback()
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
        """One trajectory scored and fed back by the trainer's feedback strategy (backprop through the environment by default)."""
        return self.feedback.run(self, goal=goal, lr=lr, apply_updates=apply_updates)


class BackpropFeedback:
    """Feedback by reverse-mode differentiation through a differentiable environment (the default)."""

    def run(self, trainer: "ClosedLoopTrainer", *, goal: Goal, lr: float, apply_updates: bool = True) -> RolloutResult:
        """One trajectory: forward, reverse BPTT, optional optimizer step."""
        if not isinstance(trainer.env, DifferentiableEnvironment):
            raise RuntimeError(
                "rollout_train requires a DifferentiableEnvironment "
                "(obs_at / step_backward / aux_loss)"
            )
        trainer.actor.zero_grad()
        frames, actions, step_losses, total_loss = trainer._forward(goal, score=True)

        aux_loss, d_aux = trainer.env.aux_loss()
        total_loss += float(aux_loss)

        d_carry: np.ndarray | None = np.zeros_like(
            frames[trainer.max_steps], dtype=np.float32
        )
        for t in range(trainer.max_steps, 0, -1):
            ti = t - 1
            d_loss = trainer.loss_fn.step_obs_grad(frames[t], goal, ti)
            d_after = d_loss if d_carry is None else (d_loss + d_carry)
            dA, d_carry = trainer.env.step_backward(ti, d_after)
            if d_aux is not None:
                dA = dA + d_aux[ti]
            trainer.actor.accumulate_grads(t, dA)
        trainer.actor.backward_done()

        if apply_updates:
            trainer.actor.apply_updates(lr)

        extras: dict[str, Any] = {"step_losses": step_losses, "frames": frames}
        if aux_loss:
            extras["aux_loss"] = float(aux_loss)

        return RolloutResult(
            total_loss=float(total_loss),
            actions=actions,
            seq_lens=trainer._seq_lens(),
            extras=extras,
        )


class PolicyGradientFeedback:
    """Feedback by policy gradient (REINFORCE with a batch-mean baseline) for environments that cannot pass gradients back.

    The actor's output is the mean of a Gaussian policy; the executed action is that mean plus noise. The environment needs only
    ``reset``/``step``; the reward comes from ``loss_fn.step_reward(obs, goal, t) -> (B,)`` (higher is better). The gradient on the
    actor's action output is ``-advantage * noise / noise_std`` per sample, fed through the actor's own ``accumulate_grads``.
    Validation (``apply_updates=False``) runs without noise and without gradients.
    """

    def __init__(self, *, noise_std: float = 0.1, gamma: float = 1.0, normalize_advantage: bool = False, seed: int = 0) -> None:
        if noise_std <= 0.0:
            raise ValueError("noise_std must be > 0")
        self.noise_std = float(noise_std)
        self.gamma = float(gamma)
        self.normalize_advantage = bool(normalize_advantage)
        self._rng = np.random.default_rng(int(seed))

    def run(self, trainer: "ClosedLoopTrainer", *, goal: Goal, lr: float, apply_updates: bool = True) -> RolloutResult:
        reward_fn = getattr(trainer.loss_fn, "step_reward", None)
        if not callable(reward_fn):
            raise RuntimeError("PolicyGradientFeedback needs a loss evaluator with step_reward(obs, goal, t) -> (B,)")
        actor, env, T = trainer.actor, trainer.env, trainer.max_steps
        explore = bool(apply_updates)
        batch = _goal_batch_size(goal)
        actor.zero_grad()
        actor.reset(batch, goal, T)
        obs = env.reset(batch, goal)
        actions: list[np.ndarray] = []
        noises: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        for t in range(1, T + 1):
            mean = np.ascontiguousarray(actor.act(obs), dtype=np.float32)
            eps = self._rng.standard_normal(mean.shape).astype(np.float32) if explore else np.zeros_like(mean)
            action = mean + np.float32(self.noise_std) * eps
            actions.append(action)
            noises.append(eps)
            obs = env.step(action)
            rewards.append(np.asarray(reward_fn(obs, goal, t - 1), dtype=np.float32).reshape(batch))
        step_losses = [float(-r.mean()) for r in rewards]
        if explore:
            returns = np.zeros((T, batch), dtype=np.float32)
            running = np.zeros(batch, dtype=np.float32)
            for t in range(T - 1, -1, -1):
                running = rewards[t] + np.float32(self.gamma) * running
                returns[t] = running
            adv = returns - returns.mean(axis=1, keepdims=True)  # the batch average is the baseline
            if self.normalize_advantage:
                adv = adv / (adv.std(axis=1, keepdims=True) + 1e-8)
            for t in range(T, 0, -1):
                d_action = -(adv[t - 1][:, None] * noises[t - 1]) / np.float32(self.noise_std)
                actor.accumulate_grads(t, d_action.astype(np.float32))
            actor.backward_done()
            actor.apply_updates(lr)
        else:
            actor.zero_grad()  # drop forward caches
        return RolloutResult(
            total_loss=float(sum(step_losses)),
            actions=actions,
            seq_lens=trainer._seq_lens(),
            extras={"step_losses": step_losses},
        )


__all__ = ["BackpropFeedback", "ClosedLoopTrainer", "PolicyGradientFeedback", "RolloutResult"]
