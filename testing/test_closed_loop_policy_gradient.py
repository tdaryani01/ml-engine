"""Policy-gradient feedback: an environment that cannot pass gradients back is trained through the same trainer."""
from __future__ import annotations

import numpy as np
import pytest

from src.closed_loop.trainer import BackpropFeedback, ClosedLoopTrainer, PolicyGradientFeedback


class _ReachEnv:
    """1-D reach: the observation is (target - position); only reset/step exist (no step_backward)."""

    def reset(self, batch_size, goal):
        self.target = np.asarray(goal, dtype=np.float32).reshape(-1, 1)
        self.pos = np.zeros_like(self.target)
        return self.target - self.pos

    def step(self, action):
        self.pos = self.pos + np.asarray(action, dtype=np.float32)
        return self.target - self.pos


class _ReachReward:
    def step_loss(self, obs, goal, t):
        return float(np.abs(obs).mean())

    def step_reward(self, obs, goal, t):
        return -np.abs(obs[:, 0])


class _LinearActor:
    """mean action = observation @ W (a Gaussian policy's mean); implements the Actor surface the trainer uses."""

    action_dim = 1

    def __init__(self):
        self.W = np.zeros((1, 1), dtype=np.float32)
        self._dW = np.zeros_like(self.W)
        self._obs: list[np.ndarray] = []

    def reset(self, batch_size, goal, max_steps):
        self._obs, self._B = [], int(batch_size)

    def act(self, obs):
        self._obs.append(np.asarray(obs, dtype=np.float32))
        return self._obs[-1] @ self.W

    def accumulate_grads(self, t, d_action):
        self._dW += self._obs[t - 1].T @ np.asarray(d_action, dtype=np.float32)

    def backward_done(self):
        pass

    def apply_updates(self, lr):
        self.W -= np.float32(lr) * self._dW / self._B
        self._dW[:] = 0.0

    def zero_grad(self):
        self._dW[:] = 0.0

    def checkpoint(self, version, val_loss):
        return b""

    def seq_len(self, t):
        return t


def _trainer(feedback=None, steps=3):
    return ClosedLoopTrainer(actor=_LinearActor(), env=_ReachEnv(), loss_fn=_ReachReward(), max_steps=steps, feedback=feedback)


def _goals(rng, n=64):
    return rng.uniform(0.5, 1.5, size=n).astype(np.float32)


def test_the_default_feedback_is_the_backward_pass() -> None:
    assert isinstance(_trainer().feedback, BackpropFeedback)


def test_a_policy_gradient_trainer_learns_an_environment_without_a_backward_pass() -> None:
    rng = np.random.default_rng(0)
    tr = _trainer(PolicyGradientFeedback(noise_std=0.1, seed=1))
    before = tr.rollout_train(goal=_goals(rng), lr=0.0, apply_updates=False).extras["step_losses"][-1]
    for _ in range(300):
        tr.rollout_train(goal=_goals(rng), lr=0.05)
    after = tr.rollout_train(goal=_goals(rng), lr=0.0, apply_updates=False)
    assert before > 0.5 and after.extras["step_losses"][-1] < 0.1  # the final distance to the target fell from about 1 to about 0
    assert float(tr.actor.W[0, 0]) > 0.1  # it learned to move toward the target


def test_validation_adds_no_noise_and_changes_no_weights() -> None:
    rng = np.random.default_rng(0)
    tr = _trainer(PolicyGradientFeedback(noise_std=0.5, seed=1))
    tr.rollout_train(goal=_goals(rng), lr=0.05)
    w = tr.actor.W.copy()
    g = _goals(rng)
    a = tr.rollout_train(goal=g, lr=0.05, apply_updates=False)
    b = tr.rollout_train(goal=g, lr=0.05, apply_updates=False)
    assert np.array_equal(tr.actor.W, w) and a.total_loss == b.total_loss


def test_a_loss_without_a_reward_is_refused() -> None:
    class _NoReward:
        def step_loss(self, obs, goal, t):
            return 0.0

    tr = ClosedLoopTrainer(actor=_LinearActor(), env=_ReachEnv(), loss_fn=_NoReward(), max_steps=2, feedback=PolicyGradientFeedback())
    with pytest.raises(RuntimeError, match="step_reward"):
        tr.rollout_train(goal=np.ones(4, dtype=np.float32), lr=0.01)
