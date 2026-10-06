"""Reach: a serial robot arm moves its end effector to a target. A real instance (the UR5), in exact kinematics.

The world is the arm's kinematic chain from its Denavit-Hartenberg parameters. The policy commands joint velocities (a smooth
``tanh`` of its output, scaled to a per-step limit); the end effector follows by forward kinematics, so the whole rollout is
differentiable (the geometric Jacobian is the backward pass) and trains by backprop through the environment. Joint limits, a
table plane the end effector must stay above, and control effort are penalised.

Observation per step: ``[joint angles (n), end-effector position (3), target (3), target - position (3)]``.
Seats: env ``arm_reach``, loss ``reach_error`` (schema ``loss_type`` ``reach_error``), data ``reach_targets``; encoder
``identity`` and policy ``mlp`` are shared with the other vector families.

Config (``closed_loop``): ``arm`` (``"ur5"`` or ``{"dh": [[a, alpha, d, theta_offset], ...], "joint_limits": [[lo, hi], ...],
"home": [...]}``), ``max_steps``, ``batch_size``, ``max_joint_step`` (rad per step, default 0.15), ``effort_weight``,
``limit_weight``, ``table_weight``, ``table_z``, ``reach_dense_weight``, ``target_spread``, optional ``targets_path`` (an ``.npz``
with ``targets`` (N, 3), metres, in the arm's base frame).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.closed_loop import needs
from src.closed_loop.registry import register

# The UR5 (Universal Robots), standard DH: alpha, a, d (metres). Zero pose end effector: (-0.81725, -0.19145, -0.005491).
_UR5 = {
    "dh": [[0.0, np.pi / 2, 0.089159, 0.0], [-0.425, 0.0, 0.0, 0.0], [-0.39225, 0.0, 0.0, 0.0],
           [0.0, np.pi / 2, 0.10915, 0.0], [0.0, -np.pi / 2, 0.09465, 0.0], [0.0, 0.0, 0.0823, 0.0]],  # rows: a, alpha, d, theta_offset
    "joint_limits": [[-2 * np.pi, 2 * np.pi]] * 6,
    "home": [0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0],
}


def arm_from_config(cl: dict[str, Any]) -> dict[str, np.ndarray]:
    raw = cl.get("arm", "ur5")
    spec = _UR5 if raw == "ur5" else raw
    if not isinstance(spec, dict) or "dh" not in spec:
        raise ValueError("closed_loop.arm must be 'ur5' or {'dh': [[a, alpha, d, theta_offset], ...], 'joint_limits': ..., 'home': ...}")
    dh = np.asarray(spec["dh"], dtype=np.float64)
    if dh.ndim != 2 or dh.shape[1] != 4:
        raise ValueError("arm.dh must be rows of [a, alpha, d, theta_offset]")
    n = len(dh)
    limits = np.asarray(spec.get("joint_limits") or [[-np.pi, np.pi]] * n, dtype=np.float64)
    home = np.asarray(spec.get("home") or np.zeros(n), dtype=np.float64)
    if limits.shape != (n, 2) or home.shape != (n,):
        raise ValueError("arm.joint_limits must be (joints, 2) and arm.home (joints,)")
    return {"dh": dh, "limits": limits, "home": home}


def forward_kinematics(dh: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """End-effector position (B, 3) and the geometric position Jacobian (B, 3, n) of the chain at joint angles ``q`` (B, n)."""
    B, n = q.shape
    R = np.tile(np.eye(3), (B, 1, 1))
    p = np.zeros((B, 3))
    axes, origins = [], []
    for i in range(n):
        a, alpha, d, off = dh[i]
        axes.append(R[:, :, 2].copy())
        origins.append(p.copy())
        th = q[:, i] + off
        c, s = np.cos(th), np.sin(th)
        ca, sa = np.cos(alpha), np.sin(alpha)
        p = p + np.einsum("bij,bj->bi", R, np.stack([a * c, a * s, np.full(B, d)], axis=1))
        Rz = np.zeros((B, 3, 3))
        Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 0, 2] = c, -s * ca, s * sa
        Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 1, 2] = s, c * ca, -c * sa
        Rz[:, 2, 1], Rz[:, 2, 2] = sa, ca
        R = R @ Rz
    J = np.stack([np.cross(axes[i], p - origins[i]) for i in range(n)], axis=2)
    return p, J


@dataclass
class ReachGoal:
    targets: np.ndarray  # (B, 3) metres, base frame
    q_noise: np.ndarray  # (B, n) offset from the home pose at the start

    @property
    def batch_size(self) -> int:
        return int(self.targets.shape[0])


class ArmReachEnv:
    """A differentiable robot-arm world: joint velocities in, end-effector pose out."""

    def __init__(self, cl: dict[str, Any]) -> None:
        arm = arm_from_config(cl)
        self.dh, self.limits, self.home = arm["dh"], arm["limits"], arm["home"]
        self.n = len(self.dh)
        self.scale = float(cl.get("max_joint_step", 0.15))
        self.effort_w = float(cl.get("effort_weight", 0.01))
        self.limit_w = float(cl.get("limit_weight", 1.0))
        self.table_w = float(cl.get("table_weight", 10.0))
        self.table_z = float(cl.get("table_z", -0.1))  # the end effector must stay above this height (m) in the base frame
        want = self.n + 9
        if "obs_dim" in cl and int(cl["obs_dim"]) != want:
            raise ValueError(f"closed_loop.obs_dim is {cl['obs_dim']} but this arm's observation has {want} values ({self.n} joints + 9)")
        if "action_dim" in cl and int(cl["action_dim"]) != self.n:
            raise ValueError(f"closed_loop.action_dim is {cl['action_dim']} but this arm has {self.n} joints")

    def obs_spec(self) -> tuple[int, ...]:
        return (self.n + 9,)

    def _obs(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.concatenate([q, p, self.g, self.g - p], axis=1).astype(np.float32)

    def reset(self, batch_size: int, goal: ReachGoal) -> np.ndarray:
        self.g = np.asarray(goal.targets, dtype=np.float64)
        self.q = self.home[None, :] + np.asarray(goal.q_noise, dtype=np.float64)
        p, J = forward_kinematics(self.dh, self.q)
        self._qs, self._ps, self._Js, self._us, self._obs_hist = [self.q.copy()], [p], [J], [], [self._obs(self.q, p)]
        return self._obs_hist[0]

    def step(self, action: np.ndarray) -> np.ndarray:
        u = np.tanh(np.asarray(action, dtype=np.float64))
        self.q = self.q + self.scale * u
        p, J = forward_kinematics(self.dh, self.q)
        self._us.append(u)
        self._qs.append(self.q.copy())
        self._ps.append(p)
        self._Js.append(J)
        self._obs_hist.append(self._obs(self.q, p))
        return self._obs_hist[-1]

    def obs_at(self, t: int) -> np.ndarray:
        return self._obs_hist[t + 1]

    def step_backward(self, t: int, d_after: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gradient on the action of step ``t`` and on the previous observation, from the gradient on this step's observation."""
        n = self.n
        d = np.asarray(d_after, dtype=np.float64)
        J = self._Js[t + 1]
        dp = d[:, n:n + 3] - d[:, n + 6:n + 9]  # position enters the observation directly and through target - position
        dq = d[:, :n] + np.einsum("bin,bi->bn", J, dp)
        u = self._us[t]
        d_action = dq * self.scale * (1.0 - u * u)
        carry = np.zeros_like(d)
        carry[:, :n] = dq  # q after the step is q before it plus the action: the gradient reaches the previous joint angles unchanged
        return d_action.astype(np.float32), carry.astype(np.float32)

    def aux_loss(self) -> tuple[float, list[np.ndarray]]:
        """Effort, joint-limit and table penalties over the whole trajectory, with their gradient on each step's action."""
        T = len(self._us)
        if T == 0:
            return 0.0, []
        B = len(self.g)
        lo, hi = self.limits[:, 0], self.limits[:, 1]
        total = 0.0
        d_q = [np.zeros((B, self.n)) for _ in range(T)]  # gradient on the joint angles after each step
        for k in range(T):
            q, p, J = self._qs[k + 1], self._ps[k + 1], self._Js[k + 1]
            below, above = np.maximum(lo - q, 0.0), np.maximum(q - hi, 0.0)
            total += self.limit_w * float(np.sum(below**2 + above**2) / B)
            d_q[k] += self.limit_w * 2.0 * (above - below) / B
            drop = np.maximum(self.table_z - p[:, 2], 0.0)
            total += self.table_w * float(np.sum(drop**2) / B)
            d_p = np.zeros((B, 3))
            d_p[:, 2] = -self.table_w * 2.0 * drop / B
            d_q[k] += np.einsum("bin,bi->bn", J, d_p)
        d_action: list[np.ndarray] = []
        running = np.zeros((B, self.n))
        for j in range(T - 1, -1, -1):  # the angle after step k depends on every action up to k
            running = running + d_q[j]
            u = self._us[j]
            total += self.effort_w * float(np.sum(u * u) / B)
            d_action.append(((running * self.scale + self.effort_w * 2.0 * u / B) * (1.0 - u * u)).astype(np.float32))
        return total, d_action[::-1]


class ReachError:
    """Squared distance from the end effector to the target (m^2): all of it at the last step, a small share at the others."""

    def __init__(self, max_steps: int, dense_weight: float) -> None:
        self.T, self.dense = int(max_steps), float(dense_weight)

    def _w(self, t: int) -> float:
        return 1.0 if t == self.T - 1 else self.dense

    def step_loss(self, obs: np.ndarray, goal: ReachGoal, t: int) -> float:
        delta = np.asarray(obs, dtype=np.float64)[:, -3:]
        return self._w(t) * float(np.mean(np.sum(delta**2, axis=1)))

    def step_obs_grad(self, obs: np.ndarray, goal: ReachGoal, t: int) -> np.ndarray:
        grad = np.zeros_like(obs, dtype=np.float32)
        grad[:, -3:] = (self._w(t) * 2.0 * np.asarray(obs, dtype=np.float64)[:, -3:] / len(obs)).astype(np.float32)
        return grad


class ReachTargets:
    """Targets the arm should reach. Without a file they are drawn from the arm's own workspace (the end-effector position of
    random joint angles around the home pose, so every target is reachable); with ``targets_path`` they are the user's points."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        cl = cfg["closed_loop"]
        arm = arm_from_config(cl)
        self.dh, self.home, self.n = arm["dh"], arm["home"], len(arm["dh"])
        self.batch_size = int(cl["batch_size"])
        self.seed = int(cl.get("seed") or (cfg.get("optimization") or {}).get("seed") or 0)
        self.spread = float(cl.get("target_spread", 0.6))
        self.start_noise = float(cl.get("start_noise", 0.05))
        path = str(cl.get("targets_path") or "").strip()
        self._plate = 0
        if path:
            with np.load(path, allow_pickle=False) as z:
                if "targets" not in z:
                    raise ValueError(f"{path}: needs 'targets' (points, 3), metres, in the arm's base frame")
                pts = np.asarray(z["targets"], dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
                raise ValueError(f"{path}: 'targets' must be (points, 3) with at least 2 points")
            order = np.random.default_rng(self.seed).permutation(len(pts))
            n_val = max(1, int(round(len(pts) * float(cl.get("val_fraction", 0.2)))))
            self.val_pts, self.train_pts = pts[order[:n_val]], pts[order[n_val:]]
            if len(self.train_pts) == 0:
                raise ValueError(f"{path}: no training points are left after holding out {n_val}")
        else:
            self.train_pts = None
            rng = np.random.default_rng([self.seed, 99])
            self.val_pts = self._workspace(rng, max(self.batch_size, 8))

    def _workspace(self, rng: np.random.Generator, count: int) -> np.ndarray:
        q = self.home[None, :] + rng.uniform(-self.spread, self.spread, size=(count, self.n))
        return forward_kinematics(self.dh, q)[0]

    def _goal(self, rng: np.random.Generator, pts: np.ndarray) -> ReachGoal:
        return ReachGoal(pts, rng.uniform(-self.start_noise, self.start_noise, size=(len(pts), self.n)))

    def train_goal(self, step: int) -> ReachGoal:
        rng = np.random.default_rng([self.seed, self._plate, int(step)])
        if self.train_pts is None:
            pts = self._workspace(rng, self.batch_size)
        else:
            pts = self.train_pts[rng.choice(len(self.train_pts), size=self.batch_size, replace=len(self.train_pts) < self.batch_size)]
        return self._goal(rng, pts)

    def val_goal(self, step: int) -> ReachGoal:
        rng = np.random.default_rng([self.seed, 7])
        return self._goal(rng, self.val_pts[: self.batch_size])

    def next_plate(self) -> None:
        self._plate += 1


@register("env", "arm_reach")
def arm_reach(cfg: dict[str, Any]):
    return ArmReachEnv(cfg["closed_loop"])


@register("loss", "reach_error", loss_types=("reach_error",))
def reach_error(cfg: dict[str, Any]):
    cl = cfg["closed_loop"]
    return ReachError(int(cl["max_steps"]), float(cl.get("reach_dense_weight", 0.05)))


@register("data", "reach_targets")
def reach_targets(cfg: dict[str, Any]):
    return ReachTargets(cfg)


needs.declare_needs("data", "reach_targets", [
    {"name": "targets_path", "kind": "file", "label": "Target points (.npz, optional)", "required": False, "config_key": "closed_loop.targets_path",
     "hint": "Points to reach: 'targets' (points, 3), metres, in the arm's base frame. Leave empty to train on targets drawn from the arm's workspace."},
])
