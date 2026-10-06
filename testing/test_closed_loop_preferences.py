"""Learning from preferences: the chosen action over the rejected one, through the same loop as demonstrations."""
from __future__ import annotations

import numpy as np
import pytest

from src.closed_loop.assembler import assemble_closed_loop, modules_from_config
from src.closed_loop.demonstration_options import BradleyTerryMargin, PreferenceGoal
from src.closed_loop.fit import fit_closed_loop
from testing.test_closed_loop_demonstrations import _Ledger

MODS = {"encoder": "identity", "policy": "mlp", "env": "demo_replay", "loss": "preference_margin", "data": "preferences"}


def _prefs(path, episodes=60, steps=4, obs_dim=3, act_dim=2, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((episodes, steps, obs_dim)).astype(np.float32)
    good = rng.standard_normal((obs_dim, act_dim)).astype(np.float32)
    chosen = np.tanh(obs @ good).astype(np.float32)
    rejected = (chosen + 1.5 * rng.choice([-1.0, 1.0], size=chosen.shape)).astype(np.float32)  # clearly worse
    np.savez(path, observations=obs, chosen_actions=chosen, rejected_actions=rejected)
    return path


def _cfg(path, **extra):
    cl = {"max_steps": 4, "batch_size": 16, "obs_dim": 3, "action_dim": 2, "hidden": 16, "preferences_path": str(path),
          "feedback": "teacher_forcing", "seed": 0, "preference_beta": 2.0, **extra}
    return {"assembly": {"modules": dict(MODS)}, "closed_loop": cl, "optimization": {"learning_rate": 0.02}}


def test_the_loss_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(1)
    goal = PreferenceGoal(rng.standard_normal((5, 3, 2)).astype(np.float32), rng.standard_normal((5, 3, 2)).astype(np.float32),
                          rng.standard_normal((5, 3, 2)).astype(np.float32))
    loss = BradleyTerryMargin(1.7)
    a = rng.standard_normal((5, 2)).astype(np.float32)
    g = loss.step_action_grad(a, goal, 1)
    eps = 1e-3
    for i in range(5):
        for j in range(2):
            up, dn = a.copy(), a.copy()
            up[i, j] += eps
            dn[i, j] -= eps
            fd = (loss.step_action_loss(up, goal, 1) - loss.step_action_loss(dn, goal, 1)) / (2 * eps)
            assert g[i, j] == pytest.approx(fd, abs=2e-3)


def test_a_policy_learns_to_prefer_the_chosen_action_on_held_out_episodes(tmp_path) -> None:
    run = assemble_closed_loop(_cfg(_prefs(tmp_path / "p.npz")), seed=0)
    led = _Ledger()
    val = run.data.val_goal(1)

    def accuracy():
        run.actor.reset(val.batch_size, val, 4)
        obs = run.env.reset(val.batch_size, val)
        acc = []
        for t in range(4):
            a = run.actor.act(obs)
            acc.append(BradleyTerryMargin.accuracy(a, val, t))
            obs = run.env.step(a)
        run.actor.zero_grad()
        return float(np.mean(acc))

    before = accuracy()
    fit_closed_loop(run, led, lr=0.02, steps=300, patience=0, checkpoint_every=100)
    steps = [d for d in led.docs if d[0] == "step"]
    assert steps[-1][3] < 0.5 * steps[0][3]  # the held-out preference loss fell
    assert accuracy() >= 0.95 and accuracy() > before  # and the policy's action is nearer the chosen one on episodes it never saw


def test_a_preference_file_needs_both_actions(tmp_path) -> None:
    np.savez(tmp_path / "x.npz", observations=np.zeros((4, 4, 3)), chosen_actions=np.zeros((4, 4, 2)))
    with pytest.raises(ValueError, match="missing \\['rejected_actions'\\]"):
        assemble_closed_loop(_cfg(tmp_path / "x.npz"), seed=0)
    with pytest.raises(ValueError, match="preferences_path is required"):
        assemble_closed_loop(_cfg(""), seed=0)


def test_the_schema_loss_type_picks_the_preference_loss_and_the_file_is_a_need() -> None:
    from src.closed_loop.needs import assembly_needs

    mods = {k: v for k, v in MODS.items() if k != "loss"}
    assert modules_from_config({"assembly": {"modules": mods}, "schema_template": {"loss_type": "preference"}})["loss"] == "preference_margin"
    out = assembly_needs(MODS)
    assert [n["name"] for n in out["needs"]] == ["preferences_path"] and out["needs"][0]["config_key"] == "closed_loop.preferences_path"


def test_the_fit_reports_the_held_out_accuracy_beside_the_loss(tmp_path) -> None:
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((60, 4, 3)).astype(np.float32)
    chosen = np.tanh(obs @ rng.standard_normal((3, 2))).astype(np.float32)
    other = np.tanh(obs @ rng.standard_normal((3, 2))).astype(np.float32)  # an equally plausible action that was ranked lower
    np.savez(tmp_path / "p.npz", observations=obs, chosen_actions=chosen, rejected_actions=other)
    run = assemble_closed_loop(_cfg(tmp_path / "p.npz"), seed=0)
    led = _Ledger()
    fit_closed_loop(run, led, lr=0.02, steps=300, patience=0, checkpoint_every=300)
    acc = [e["val_accuracy"] for e in led.extra if e]
    assert len(acc) == 300 and all(0.0 <= a <= 1.0 for a in acc)
    assert np.mean(acc[-10:]) >= 0.95 and np.mean(acc[-10:]) > acc[0] + 0.2  # the number a person judging it wants, rising to nearly always right


def test_the_runner_carries_the_extra_metrics_out_with_each_snapshot(tmp_path) -> None:
    from src.launcher import FamilyFitSpec, run_me_fit

    rng = np.random.default_rng(0)
    obs = rng.standard_normal((40, 4, 3)).astype(np.float32)
    chosen = np.tanh(obs @ rng.standard_normal((3, 2))).astype(np.float32)
    np.savez(tmp_path / "p.npz", observations=obs, chosen_actions=chosen, rejected_actions=np.tanh(obs @ rng.standard_normal((3, 2))).astype(np.float32))
    cfg = _cfg(tmp_path / "p.npz", max_steps=4)
    cfg["assembly"]["family_id"] = "closed_loop"  # the runner picks the pipeline from the family
    spec = FamilyFitSpec(model_id="m", config=cfg, fit={"run_budget": 40, "patience": 0, "checkpoint_every": 20, "lr": 0.02}, num_threads=1)
    snaps = list(run_me_fit(spec, work_dir=tmp_path / "w", should_stop=lambda: False))
    carried = [s["extra_metrics"] for s in snaps if s.get("extra_metrics")]
    assert carried and all("val_accuracy" in c for c in carried)  # what the Desktop turns into the thin row's extra metrics
