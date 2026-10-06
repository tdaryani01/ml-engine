"""Options for learning from demonstrations (imitation): recorded observations in, the recorded actions as the target.

A demonstration file is an ``.npz`` with ``observations`` (episodes, steps, obs_dim) and ``actions`` (episodes, steps, action_dim).
Whole episodes are held out for validation. Run with ``closed_loop.feedback: "teacher_forcing"``.

Seats: encoder ``identity``, policy ``mlp`` (continuous actions, one step of observation), env ``demo_replay``, loss ``action_mse``
(schema ``loss_type`` ``action_mse``), data ``demonstrations``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.closed_loop import needs
from src.closed_loop.registry import register


@dataclass
class DemoGoal:
    """One batch of demonstrations: what the policy sees and what it is scored against."""

    observations: np.ndarray  # (B, T, obs_dim)
    actions: np.ndarray  # (B, T, action_dim)

    @property
    def batch_size(self) -> int:
        return int(self.observations.shape[0])


class IdentityEncoder:
    """Vector observations need no encoder: the features are the observation."""

    def encode(self, obs: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(obs, dtype=np.float32).reshape(len(obs), -1)

    def backward(self, dV: np.ndarray) -> None:
        pass

    def apply_pending(self, lr: float) -> None:
        pass

    def zero_grad(self) -> None:
        pass


class MlpActor:
    """obs -> tanh hidden layer -> action (continuous). Adam. Implements the actor surface the trainer uses."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.action_dim = int(action_dim)
        self.params = {
            "W1": (rng.standard_normal((obs_dim, hidden)) * np.sqrt(1.0 / obs_dim)).astype(np.float32), "b1": np.zeros((1, hidden), np.float32),
            "W2": (rng.standard_normal((hidden, action_dim)) * np.sqrt(1.0 / hidden)).astype(np.float32), "b2": np.zeros((1, action_dim), np.float32),
        }
        self._g = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0
        self._cache: list[tuple[np.ndarray, np.ndarray]] = []

    def reset(self, batch_size: int, goal: Any, max_steps: int) -> None:
        self._cache = []

    def _features(self, obs: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(obs, dtype=np.float32).reshape(len(obs), -1)

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = self._features(obs)
        h = np.tanh(x @ self.params["W1"] + self.params["b1"])
        self._cache.append((x, h))
        return (h @ self.params["W2"] + self.params["b2"]).astype(np.float32)

    def accumulate_grads(self, t: int, d_action: np.ndarray) -> None:
        x, h = self._cache[t - 1]
        dA = np.asarray(d_action, dtype=np.float32)
        dh = (dA @ self.params["W2"].T) * (1.0 - h * h)
        self._g["W2"] += h.T @ dA
        self._g["b2"] += dA.sum(axis=0, keepdims=True)
        self._g["W1"] += x.T @ dh
        self._g["b1"] += dh.sum(axis=0, keepdims=True)

    def backward_done(self) -> None:
        pass

    def apply_updates(self, lr: float) -> None:
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, p in self.params.items():
            g = self._g[k]
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * g * g
            p -= np.float32(lr) * (self._m[k] / (1 - b1**self._t)) / (np.sqrt(self._v[k] / (1 - b2**self._t)) + eps)
        self.zero_grad()

    def zero_grad(self) -> None:
        for g in self._g.values():
            g.fill(0.0)

    def checkpoint(self, version: int, val_loss: float) -> bytes:
        return b""

    def seq_len(self, t: int) -> int:
        return 1

    # the checkpoint state: weights and optimizer moments as named arrays
    def state_arrays(self) -> dict[str, np.ndarray]:
        out = {k: v for k, v in self.params.items()}
        out.update({f"m_{k}": v for k, v in self._m.items()})
        out.update({f"v_{k}": v for k, v in self._v.items()})
        out["opt_t"] = np.array([self._t], dtype=np.int64)
        return out

    def load_state_arrays(self, flat: dict[str, np.ndarray]) -> None:
        for k in self.params:
            if k not in flat or flat[k].shape != self.params[k].shape:
                raise ValueError(f"checkpoint state does not match this policy ({k})")
            self.params[k][...] = flat[k]
            self._m[k][...] = flat.get(f"m_{k}", np.zeros_like(self._m[k]))
            self._v[k][...] = flat.get(f"v_{k}", np.zeros_like(self._v[k]))
        self._t = int(np.asarray(flat.get("opt_t", [0])).reshape(-1)[0])


class MlpHistoryActor(MlpActor):
    """The MLP policy with a short memory: it sees its last ``history`` observations, newest first, zero-padded at the start of an episode.

    Memory is what lets a policy tell where it has just been (a loop, a dead end) or what just changed; the observation alone cannot."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int, history: int, *, seed: int = 0) -> None:
        if history < 1:
            raise ValueError("closed_loop.history must be at least 1")
        super().__init__(obs_dim * history, action_dim, hidden, seed=seed)
        self.obs_dim, self.history = int(obs_dim), int(history)
        self._past: list[np.ndarray] = []

    def reset(self, batch_size: int, goal: Any, max_steps: int) -> None:
        super().reset(batch_size, goal, max_steps)
        self._past = []

    def _features(self, obs: np.ndarray) -> np.ndarray:
        now = np.ascontiguousarray(obs, dtype=np.float32).reshape(len(obs), -1)
        self._past.append(now)
        window = self._past[-self.history:][::-1]  # newest first
        window += [np.zeros_like(now)] * (self.history - len(window))
        return np.concatenate(window, axis=1)


class ReplayEnv:
    """Replays the recorded observations. The action does not change the world; there is no backward pass."""

    def reset(self, batch_size: int, goal: DemoGoal) -> np.ndarray:
        self._obs, self._t = goal.observations, 0
        return self._obs[:, 0]

    def step(self, action: np.ndarray) -> np.ndarray:
        self._t = min(self._t + 1, self._obs.shape[1] - 1)
        return self._obs[:, self._t]


class ActionMse:
    """Mean squared error between the policy's action and the recorded one."""

    def step_action_loss(self, action: np.ndarray, goal: DemoGoal, t: int) -> float:
        return float(np.mean((action - goal.actions[:, t]) ** 2))

    def step_action_grad(self, action: np.ndarray, goal: DemoGoal, t: int) -> np.ndarray:
        return (2.0 * (action - goal.actions[:, t]) / action.size).astype(np.float32)


class Demonstrations:
    """Training batches from the training episodes; one fixed validation batch from whole held-out episodes.

    A subclass names its file key (``PATH_KEY``), the arrays it reads (``ARRAYS``: the first is the observations, the second the
    target actions, the rest ride along) and the goal it builds (``_goal``)."""

    PATH_KEY = "demonstrations_path"
    ARRAYS: tuple[str, ...] = ("observations", "actions")

    def __init__(self, cfg: dict[str, Any]) -> None:
        cl = cfg["closed_loop"]
        path = str(cl.get(self.PATH_KEY) or "").strip()
        if not path:
            raise ValueError(f"closed_loop.{self.PATH_KEY} is required (the .npz with {', '.join(self.ARRAYS)})")
        with np.load(path, allow_pickle=False) as z:
            missing = [k for k in self.ARRAYS if k not in z]
            if missing:
                raise ValueError(f"{path}: needs {', '.join(repr(k) for k in self.ARRAYS)} (episodes, steps, features); missing {missing}")
            arrays = [np.asarray(z[k], np.float32) for k in self.ARRAYS]
        steps = int(cl["max_steps"])
        if any(a.ndim != 3 or a.shape[:2] != arrays[0].shape[:2] for a in arrays):
            raise ValueError(f"{path}: {', '.join(self.ARRAYS)} must be (episodes, steps, features) with the same episodes and steps")
        if arrays[0].shape[1] < steps:
            raise ValueError(f"{path}: episodes have {arrays[0].shape[1]} steps but closed_loop.max_steps is {steps}")
        self.arrays = [a[:, :steps] for a in arrays]
        self.obs, self.act = self.arrays[0], self.arrays[1]
        n = len(self.obs)
        if n < 2:
            raise ValueError(f"{path}: need at least 2 episodes (one is held out for validation)")
        self.batch_size = int(cl["batch_size"])
        self.seed = int(cl.get("seed") or (cfg.get("optimization") or {}).get("seed") or 0)
        order = np.random.default_rng(self.seed).permutation(n)
        n_val = max(1, int(round(n * float(cl.get("val_fraction", 0.2)))))
        self.val_idx, self.train_idx = order[:n_val], order[n_val:]
        if len(self.train_idx) == 0:
            raise ValueError(f"{path}: no training episodes are left after holding out {n_val}")
        self._plate = 0
        # Each fit of a chain (Autopilot) is a stretch: it must draw NEW training batches, not replay the first fit's.
        self._stretch = max(0, int(((cfg.get("fit") or {}).get("stretch_index")) or 1) - 1)

    def _batch(self, idx: np.ndarray) -> DemoGoal:
        return self._goal(*[a[idx] for a in self.arrays])

    def _goal(self, observations: np.ndarray, actions: np.ndarray) -> DemoGoal:
        return DemoGoal(observations, actions)

    def train_goal(self, step: int) -> DemoGoal:
        rng = np.random.default_rng([self.seed, self._stretch, self._plate, int(step)])
        take = rng.choice(self.train_idx, size=self.batch_size, replace=len(self.train_idx) < self.batch_size)
        return self._batch(take)

    def val_goal(self, step: int) -> DemoGoal:
        return self._batch(self.val_idx[: self.batch_size])

    def next_plate(self) -> None:
        self._plate += 1


@dataclass
class PreferenceGoal(DemoGoal):
    """Observations, the action that was chosen and the one that was rejected at each step."""

    rejected: np.ndarray = None  # (B, T, action_dim)


class BradleyTerryMargin:
    """Pairwise preference loss on the policy's action ``a``: ``softplus(-beta * (|a - rejected|^2 - |a - chosen|^2))``.

    The policy is rewarded for being closer to the chosen action than to the rejected one; ``beta`` sets how sharp the
    preference is. (Bradley-Terry: the probability that chosen beats rejected is ``sigmoid(beta * margin)``.)"""

    def __init__(self, beta: float) -> None:
        if beta <= 0:
            raise ValueError("closed_loop.preference_beta must be > 0")
        self.beta = float(beta)

    def _margin(self, action: np.ndarray, goal: PreferenceGoal, t: int) -> np.ndarray:
        c, r = goal.actions[:, t], goal.rejected[:, t]
        return np.sum((action - r) ** 2, axis=1) - np.sum((action - c) ** 2, axis=1)

    def step_action_loss(self, action: np.ndarray, goal: PreferenceGoal, t: int) -> float:
        return float(np.mean(np.logaddexp(0.0, -self.beta * self._margin(action, goal, t))))

    def step_action_grad(self, action: np.ndarray, goal: PreferenceGoal, t: int) -> np.ndarray:
        m = self._margin(action, goal, t)
        d_loss_d_margin = -self.beta / (1.0 + np.exp(np.clip(self.beta * m, -50.0, 50.0)))  # -beta * sigmoid(-beta * m)
        d_margin_d_action = 2.0 * (goal.actions[:, t] - goal.rejected[:, t])
        return (d_loss_d_margin[:, None] * d_margin_d_action / len(action)).astype(np.float32)

    @staticmethod
    def accuracy(action: np.ndarray, goal: PreferenceGoal, t: int) -> float:
        """How often the policy's action is closer to the chosen one than to the rejected one (for checking, not training)."""
        return float(np.mean(np.sum((action - goal.actions[:, t]) ** 2, axis=1) < np.sum((action - goal.rejected[:, t]) ** 2, axis=1)))


class Preferences(Demonstrations):
    """Like demonstrations, with a rejected action beside each chosen one."""

    PATH_KEY = "preferences_path"
    ARRAYS = ("observations", "chosen_actions", "rejected_actions")

    def _goal(self, observations: np.ndarray, chosen: np.ndarray, rejected: np.ndarray) -> PreferenceGoal:
        return PreferenceGoal(observations, chosen, rejected)


@register("encoder", "identity")
def identity_encoder(cfg: dict[str, Any], seed: int):
    return IdentityEncoder()


@register("policy", "mlp")
def mlp_policy(cfg: dict[str, Any], encoder: Any, seed: int):
    cl = cfg["closed_loop"]
    return MlpActor(int(cl["obs_dim"]), int(cl["action_dim"]), int(cl.get("hidden", 32)), seed=seed)


@register("policy", "mlp_history")
def mlp_history_policy(cfg: dict[str, Any], encoder: Any, seed: int):
    cl = cfg["closed_loop"]
    return MlpHistoryActor(int(cl["obs_dim"]), int(cl["action_dim"]), int(cl.get("hidden", 32)), int(cl.get("history", 4)), seed=seed)


@register("env", "demo_replay")
def demo_replay(cfg: dict[str, Any]):
    return ReplayEnv()


@register("loss", "action_mse", loss_types=("action_mse",))
def action_mse(cfg: dict[str, Any]):
    return ActionMse()


@register("data", "demonstrations")
def demonstrations(cfg: dict[str, Any]):
    return Demonstrations(cfg)


needs.declare_needs("data", "demonstrations", [
    {"name": "demonstrations_path", "kind": "file", "label": "Demonstrations (.npz)", "required": True, "config_key": "closed_loop.demonstrations_path",
     "hint": "Recorded observations (episodes, steps, obs_dim) and actions (episodes, steps, action_dim). It is read on this computer and never sent to TM."},
])


@register("loss", "preference_margin", loss_types=("preference",))
def preference_margin(cfg: dict[str, Any]):
    return BradleyTerryMargin(float(cfg["closed_loop"].get("preference_beta", 4.0)))


@register("data", "preferences")
def preferences(cfg: dict[str, Any]):
    return Preferences(cfg)


needs.declare_needs("data", "preferences", [
    {"name": "preferences_path", "kind": "file", "label": "Preferences (.npz)", "required": True, "config_key": "closed_loop.preferences_path",
     "hint": "Observations (episodes, steps, obs_dim) with the chosen and the rejected action at each step (episodes, steps, action_dim). "
             "It is read on this computer and never sent to TM."},
])
