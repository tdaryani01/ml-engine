"""Reach with a real arm (the UR5): exact kinematics, a differentiable world, trained by backprop through it."""
from __future__ import annotations

import numpy as np
import pytest

from src.closed_loop.assembler import assemble_closed_loop, modules_from_config
from src.closed_loop.fit import fit_closed_loop
from src.closed_loop.reach_options import ArmReachEnv, ReachError, ReachGoal, arm_from_config, forward_kinematics
from testing.test_closed_loop_demonstrations import _Ledger

MODS = {"encoder": "identity", "policy": "mlp", "env": "arm_reach", "loss": "reach_error", "data": "reach_targets"}


def _cfg(**extra):
    cl = {"arm": "ur5", "max_steps": 12, "batch_size": 32, "obs_dim": 15, "action_dim": 6, "hidden": 64, "seed": 0, "max_joint_step": 0.15, **extra}
    return {"assembly": {"modules": dict(MODS)}, "closed_loop": cl, "optimization": {"learning_rate": 0.003}}


def test_the_ur5_forward_kinematics_match_the_published_zero_pose() -> None:
    arm = arm_from_config({"arm": "ur5"})
    p, _ = forward_kinematics(arm["dh"], np.zeros((1, 6)))
    np.testing.assert_allclose(p[0], [-0.81725, -0.19145, -0.005491], atol=1e-6)  # from the manufacturer's DH table


def test_the_jacobian_matches_finite_differences() -> None:
    arm = arm_from_config({"arm": "ur5"})
    q = np.random.default_rng(0).uniform(-1.5, 1.5, size=(4, 6))
    _, J = forward_kinematics(arm["dh"], q)
    for j in range(6):
        up, dn = q.copy(), q.copy()
        up[:, j] += 1e-6
        dn[:, j] -= 1e-6
        fd = (forward_kinematics(arm["dh"], up)[0] - forward_kinematics(arm["dh"], dn)[0]) / 2e-6
        np.testing.assert_allclose(J[:, :, j], fd, atol=1e-5)


def test_the_gradient_through_the_whole_rollout_matches_finite_differences() -> None:
    """The reverse sweep (loss, step_backward carry, aux penalties) against numeric differentiation of the same total loss."""
    cl = _cfg(max_steps=5, batch_size=3, table_z=0.15, limit_weight=2.0, effort_weight=0.05)["closed_loop"]  # a high table so that penalty is active
    env, T, B = ArmReachEnv(cl), 5, 3
    loss = ReachError(T, 0.1)
    rng = np.random.default_rng(3)
    goal = ReachGoal(rng.uniform(-0.3, 0.3, size=(B, 3)) + np.array([-0.5, -0.2, 0.2]), rng.uniform(-0.1, 0.1, size=(B, 6)))
    A = rng.standard_normal((T, B, 6)) * 0.5

    def total(actions):
        obs = env.reset(B, goal)
        s = 0.0
        for t in range(T):
            obs = env.step(actions[t])
            s += loss.step_loss(obs, goal, t)
        return s + env.aux_loss()[0]

    total(A)  # analytic: the reverse sweep, as BackpropFeedback runs it
    frames = [env._obs_hist[i] for i in range(T + 1)]
    _, d_aux = env.aux_loss()
    d_carry = np.zeros_like(frames[T])
    analytic = [None] * T
    for t in range(T, 0, -1):
        d_after = loss.step_obs_grad(frames[t], goal, t - 1) + d_carry
        dA, d_carry = env.step_backward(t - 1, d_after)
        analytic[t - 1] = dA + d_aux[t - 1]
    eps = 1e-5
    for t, b, j in [(0, 0, 0), (0, 1, 3), (2, 2, 5), (4, 0, 1), (3, 1, 2)]:
        up, dn = A.copy(), A.copy()
        up[t, b, j] += eps
        dn[t, b, j] -= eps
        fd = (total(up) - total(dn)) / (2 * eps)
        assert float(analytic[t][b, j]) == pytest.approx(fd, rel=2e-2, abs=1e-5)


def test_a_dimension_that_does_not_match_the_arm_is_refused() -> None:
    with pytest.raises(ValueError, match="observation has 15 values"):
        assemble_closed_loop(_cfg(obs_dim=12), seed=0)
    with pytest.raises(ValueError, match="this arm has 6 joints"):
        assemble_closed_loop(_cfg(action_dim=3), seed=0)


def test_the_policy_learns_to_reach_targets_it_has_not_seen(tmp_path) -> None:
    run = assemble_closed_loop(_cfg(), seed=0)
    led = _Ledger()
    fit_closed_loop(run, led, lr=0.003, steps=400, patience=0, checkpoint_every=200)
    val = [d[3] for d in led.docs if d[0] == "step"]
    first, last = np.mean(val[:10]), np.mean(val[-10:])
    assert last < 0.25 * first  # the held-out squared distance (m^2) fell to under a quarter
    assert np.sqrt(last) < 0.1  # and the end effector lands within 10 cm of targets it was not trained on


def test_the_schema_loss_type_picks_the_reach_loss_and_the_targets_file_is_optional() -> None:
    from src.closed_loop.needs import assembly_needs

    mods = {k: v for k, v in MODS.items() if k != "loss"}
    assert modules_from_config({"assembly": {"modules": mods}, "schema_template": {"loss_type": "reach_error"}})["loss"] == "reach_error"
    need = assembly_needs(MODS)["needs"]
    assert [n["name"] for n in need] == ["targets_path"] and need[0]["required"] is False


def test_a_users_target_points_are_used_and_held_out(tmp_path) -> None:
    pts = np.random.default_rng(0).uniform(-0.2, 0.2, size=(40, 3)) + np.array([-0.5, -0.2, 0.2])
    np.savez(tmp_path / "t.npz", targets=pts)
    run = assemble_closed_loop(_cfg(targets_path=str(tmp_path / "t.npz")), seed=0)
    d = run.data
    assert len(d.val_pts) == 8 and len(d.train_pts) == 32
    assert not any(np.allclose(v, t) for v in d.val_pts for t in d.train_pts)
    np.savez(tmp_path / "bad.npz", targets=np.zeros((5, 2)))
    with pytest.raises(ValueError, match="must be \\(points, 3\\)"):
        assemble_closed_loop(_cfg(targets_path=str(tmp_path / "bad.npz")), seed=0)
