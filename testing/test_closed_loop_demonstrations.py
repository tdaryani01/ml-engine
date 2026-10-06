"""Learning from demonstrations through the closed-loop assembly: replayed observations, recorded actions as targets."""
from __future__ import annotations

import numpy as np
import pytest

from src.closed_loop.assembler import assemble_closed_loop, modules_from_config
from src.closed_loop.fit import fit_closed_loop
from src.closed_loop.state import capture_state, restore_state

MODS = {"encoder": "identity", "policy": "mlp", "env": "demo_replay", "loss": "action_mse", "data": "demonstrations"}


def _demos(path, episodes=40, steps=4, obs_dim=3, act_dim=2, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((episodes, steps, obs_dim)).astype(np.float32)
    expert = rng.standard_normal((obs_dim, act_dim)).astype(np.float32)
    np.savez(path, observations=obs, actions=(obs @ expert).astype(np.float32))
    return path


def _cfg(path, **extra):
    cl = {"max_steps": 4, "batch_size": 16, "obs_dim": 3, "action_dim": 2, "hidden": 16, "demonstrations_path": str(path),
          "feedback": "teacher_forcing", "seed": 0, **extra}
    return {"assembly": {"modules": dict(MODS)}, "closed_loop": cl, "optimization": {"learning_rate": 0.02}}


class _Ledger:
    def __init__(self):
        self.docs, self.version = [], 0

        class _S:
            def flush(self):
                pass

        self.store = _S()

    def push_checkpoint_state(self, state, version, **kw):
        self.docs.append(("checkpoint", version, kw))

    def push_step_metrics(self, step, version, train, val, **kw):
        self.docs.append(("step", step, train, val))

    def push_run_end(self, end):
        self.docs.append(("end", end))


def test_a_policy_learns_the_experts_actions_from_demonstrations(tmp_path) -> None:
    run = assemble_closed_loop(_cfg(_demos(tmp_path / "d.npz")), seed=0)
    led = _Ledger()
    end = fit_closed_loop(run, led, lr=0.02, steps=300, patience=0, checkpoint_every=100)
    steps = [d for d in led.docs if d[0] == "step"]
    first_val, last_val = steps[0][3], steps[-1][3]
    assert first_val > 0.3 and last_val < 0.05 * first_val  # held-out episodes: the error fell by far more than 20x
    assert end["reason"] == "success" and any(d[0] == "checkpoint" for d in led.docs)


def test_the_checkpoint_state_restores_exactly(tmp_path) -> None:
    cfg = _cfg(_demos(tmp_path / "d.npz"))
    a = assemble_closed_loop(cfg, seed=0)
    fit_closed_loop(a, _Ledger(), lr=0.02, steps=20)
    b = assemble_closed_loop(cfg, seed=9)  # a different init: restore must overwrite it completely
    restore_state(b.actor, capture_state(a.actor))
    goal = a.data.val_goal(1)
    ra = a.trainer.rollout_train(goal=goal, lr=0.02, apply_updates=False)
    rb = b.trainer.rollout_train(goal=goal, lr=0.02, apply_updates=False)
    assert ra.total_loss == rb.total_loss
    assert int(capture_state(b.actor)["opt_t"][0]) == int(capture_state(a.actor)["opt_t"][0]) == 20


def test_whole_episodes_are_held_out_and_training_batches_never_contain_them(tmp_path) -> None:
    run = assemble_closed_loop(_cfg(_demos(tmp_path / "d.npz")), seed=0)
    d = run.data
    assert set(d.val_idx).isdisjoint(set(d.train_idx)) and len(d.val_idx) == 8 and len(d.train_idx) == 32
    held = {tuple(d.obs[i].reshape(-1)) for i in d.val_idx}
    for step in range(1, 6):
        assert all(tuple(row.reshape(-1)) not in held for row in d.train_goal(step).observations)


def test_a_demonstration_file_that_does_not_fit_the_run_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="episodes have 4 steps but closed_loop.max_steps is 9"):
        assemble_closed_loop(_cfg(_demos(tmp_path / "d.npz"), max_steps=9), seed=0)
    np.savez(tmp_path / "bad.npz", observations=np.zeros((3, 4)))
    with pytest.raises(ValueError, match="needs 'observations'"):
        assemble_closed_loop(_cfg(tmp_path / "bad.npz"), seed=0)
    with pytest.raises(ValueError, match="demonstrations_path is required"):
        assemble_closed_loop(_cfg(""), seed=0)


def test_the_schema_loss_type_picks_the_action_loss() -> None:
    mods = {k: v for k, v in MODS.items() if k != "loss"}
    assert modules_from_config({"assembly": {"modules": mods}, "schema_template": {"loss_type": "action_mse"}})["loss"] == "action_mse"


def test_the_demonstrations_option_asks_the_user_for_the_file() -> None:
    from src.closed_loop.needs import assembly_needs

    out = assembly_needs(MODS)
    assert [n["name"] for n in out["needs"]] == ["demonstrations_path"] and out["needs"][0]["kind"] == "file" and out["missing"] == []
