# testing/test_closed_loop_assembler.py
"""The registry-built closed-loop run is the same run the hard-wired assembler builds, for the same config and seed."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.closed_loop.assembler import assemble_closed_loop, modules_from_config
from src.closed_loop.registry import options, resolve

SMOKE = os.path.join(os.path.dirname(__file__), "..", "examples", "closed_loop_draw", "config_draw_smoke.yaml")
MODULES = {"encoder": "cnn_upstream", "policy": "mhsa", "env": "soft_canvas", "loss": "canvas_reconstruction"}


def _cfg():
    cfg = yaml.safe_load(open(SMOKE))
    cfg["assembly"] = {"family_id": "closed_loop", "modules": dict(MODULES)}
    return cfg


def _goal(cfg):
    from examples.closed_loop_draw.assemble import make_target
    from examples.closed_loop_draw.goal import DrawGoal

    B = int(cfg["closed_loop"]["batch_size"])
    return DrawGoal(command_ids=np.zeros(B, dtype=np.int64), target=make_target(cfg, B))


def test_the_registry_knows_the_built_in_options_by_the_names_tm_uses() -> None:
    # the names TM's catalog uses are there (other families register more options beside them)
    assert "cnn_upstream" in options("encoder") and "mhsa" in options("policy")
    assert "soft_canvas" in options("env") and "canvas_reconstruction" in options("loss")


def test_an_unknown_option_is_refused_and_names_what_is_available() -> None:
    with pytest.raises(ValueError, match="no encoder option named 'nope'.*cnn_upstream"):
        resolve("encoder", "nope")


def test_every_seat_must_be_named() -> None:
    with pytest.raises(ValueError, match="missing: env, loss"):
        modules_from_config({"assembly": {"modules": {"encoder": "cnn_upstream", "policy": "mhsa"}}})


def test_the_assembled_run_trains_exactly_like_the_hard_wired_one() -> None:
    from examples.closed_loop_draw.assemble import assemble as hard_wired

    cfg = _cfg()
    old = hard_wired(cfg, seed=0)
    new = assemble_closed_loop(cfg, seed=0)
    goal = _goal(cfg)
    lr = float(cfg["optimization"]["learning_rate"])
    a = old.trainer.rollout_train(goal=goal, lr=lr)
    b = new.trainer.rollout_train(goal=goal, lr=lr)
    assert a.total_loss == pytest.approx(b.total_loss, rel=0, abs=0)  # identical, not just close
    for x, y in zip(a.actions, b.actions):
        np.testing.assert_array_equal(x, y)
    c = old.trainer.rollout_train(goal=goal, lr=lr)  # after one optimizer step the weights must still agree
    d = new.trainer.rollout_train(goal=goal, lr=lr)
    assert c.total_loss == pytest.approx(d.total_loss, rel=0, abs=0)
    old.close()
    new.close()


def test_a_closed_loop_payload_may_author_the_policy_dims_in_its_mhsa_block_beside_a_schema_template() -> None:
    from config.config_loader import TMConfigError, parse_tm_production_config

    base = _cfg()
    base["schema_template"] = {"inputs": [{"name": "observation", "shape": ["__OBSERVATION_DIM__"], "type": "float32"}],
                               "outputs": [{"name": "reconstruction", "shape": ["__TARGET_DIM__"], "type": "float32"}],
                               "loss_type": "mse", "schema_version": "1.0", "task_class": "match_reconstruct"}
    assert parse_tm_production_config(base, profile="closed_loop")["mhsa"]["d_model"] == 32  # dims come from the mhsa block
    no_dims = dict(base)
    no_dims.pop("mhsa")
    with pytest.raises(TMConfigError, match="d_model"):
        parse_tm_production_config(no_dims, profile="closed_loop")


def test_state_survives_the_checkpoint_bytes_and_restores_exactly() -> None:
    """Capture -> checkpoint document bytes -> read back -> restore. Scalars (the optimizer step) come back as arrays."""
    import numpy as np

    from src.closed_loop.state import capture_state, restore_state
    from src.ledger import CHECKPOINT, LedgerDocument, document_from_bytes, document_to_bytes

    cfg = _cfg()
    run = assemble_closed_loop(cfg, seed=0)
    goal = _goal(cfg)
    for _ in range(2):
        run.trainer.rollout_train(goal=goal, lr=0.002)  # moves weights and the optimizer step off their start
    state = capture_state(run.actor)
    body = {"version": 2, "val_loss": 0.1, "is_local_best": True, "weights": [], "biases": [], "gammas": None, "betas": None,
            "optimizer": {"type": "none", "t": 0, "beta1": 0.0, "beta2": 0.0, "eps": 0.0,
                          **{k: None for k in ("ms_w", "vs_w", "ms_b", "vs_b", "ms_g", "vs_g", "ms_beta", "vs_beta")}}, "state": state}
    doc = LedgerDocument(doc_type=CHECKPOINT, branch_id="main", model_instance_id="t", architecture_id="a", version=2, step_id=2, body=body)
    back = document_from_bytes(document_to_bytes(doc)).body["state"]
    fresh = assemble_closed_loop(cfg, seed=7)  # a different init: restore must overwrite it completely
    restore_state(fresh.actor, back)  # (this raised on a newer NumPy: int() of a one-element array)
    again = capture_state(fresh.actor)
    assert set(again) == set(state)
    for k in state:
        # the checkpoint wire stores float32, so a float64 moment comes back rounded (~6e-8 relative)
        np.testing.assert_allclose(np.asarray(again[k]).reshape(-1), np.asarray(state[k]).reshape(-1), rtol=1e-6, atol=1e-9)
    for r in (run, fresh):
        r.close()


def test_autopilot_starts_each_stretch_on_the_plate_after_the_one_the_last_early_stop_left() -> None:
    from src.closed_loop.run import rotate_plates

    cfg = _cfg()
    run = assemble_closed_loop(cfg, seed=0)
    ids = run.data.stock_ids
    first = run.data.command_id
    assert rotate_plates(run, {"rotate_on_es": True, "stretch_index": 1}) == 0 and run.data.command_id == first  # first stretch: no rotation
    assert rotate_plates(run, {"rotate_on_es": False, "stretch_index": 4}) == 0 and run.data.command_id == first  # policy off: stay put
    assert rotate_plates(run, {"rotate_on_es": True, "stretch_index": 3}) == 2
    assert run.data.command_id == ids[(ids.index(first) + 2) % len(ids)]
    assert run.data.val_goal(1).command_ids[0] != run.data.command_id  # validation stays on a different plate than training
    run.close()


def _mods(loss=None):
    m = {"encoder": "cnn_upstream", "policy": "mhsa", "env": "soft_canvas"}
    if loss:
        m["loss"] = loss
    return m


def test_a_loss_the_assembly_leaves_out_is_the_one_the_schema_loss_type_names() -> None:
    cfg = {"assembly": {"modules": _mods()}, "schema_template": {"loss_type": "MSE"}}
    assert modules_from_config(cfg)["loss"] == "canvas_reconstruction"


def test_a_loss_that_does_not_implement_the_schema_loss_type_is_refused() -> None:
    cfg = {"assembly": {"modules": _mods("canvas_reconstruction")}, "schema_template": {"loss_type": "pairwise"}}
    with pytest.raises(ValueError, match="implements \\['mse'\\].*loss_type 'pairwise'"):
        modules_from_config(cfg)


def test_a_schema_loss_type_nothing_implements_is_refused_when_the_assembly_names_no_loss() -> None:
    with pytest.raises(ValueError, match="no loss option implements the schema loss_type 'pairwise'"):
        modules_from_config({"assembly": {"modules": _mods()}, "schema_template": {"loss_type": "pairwise"}})


def test_two_losses_for_one_loss_type_need_the_assembly_to_choose(monkeypatch) -> None:
    from src.closed_loop import registry

    monkeypatch.setitem(registry._LOSS_TYPES, "another_mse", ("mse",))
    with pytest.raises(ValueError, match="several loss options.*another_mse.*canvas_reconstruction"):
        modules_from_config({"assembly": {"modules": _mods()}, "schema_template": {"loss_type": "mse"}})
    assert modules_from_config({"assembly": {"modules": _mods("canvas_reconstruction")}, "schema_template": {"loss_type": "mse"}})["loss"] == "canvas_reconstruction"


def test_without_a_schema_loss_type_the_named_loss_is_used_as_before() -> None:
    assert modules_from_config({"assembly": {"modules": _mods("canvas_reconstruction")}})["loss"] == "canvas_reconstruction"


def test_the_feedback_comes_from_the_closed_loop_config_and_defaults_to_backprop() -> None:
    from src.closed_loop.assembler import feedback_from_config
    from src.closed_loop.trainer import BackpropFeedback, PolicyGradientFeedback

    assert feedback_from_config({"closed_loop": {}}) is None  # the trainer's own default
    assert isinstance(feedback_from_config({"closed_loop": {"feedback": "backprop"}}), BackpropFeedback)
    pg = feedback_from_config({"closed_loop": {"feedback": {"kind": "policy_gradient", "noise_std": 0.3, "gamma": 0.9}}}, seed=5)
    assert isinstance(pg, PolicyGradientFeedback) and pg.noise_std == 0.3 and pg.gamma == 0.9
    assert isinstance(feedback_from_config({"closed_loop": {"feedback": "policy_gradient"}}), PolicyGradientFeedback)


def test_an_unknown_feedback_or_parameter_is_refused() -> None:
    from src.closed_loop.assembler import feedback_from_config

    with pytest.raises(ValueError, match="unknown feedback 'ppo'.*backprop.*policy_gradient"):
        feedback_from_config({"closed_loop": {"feedback": "ppo"}})
    with pytest.raises(ValueError, match="takes \\['noise_std', 'gamma', 'normalize_advantage'\\], not \\['lr'\\]"):
        feedback_from_config({"closed_loop": {"feedback": {"kind": "policy_gradient", "lr": 1}}})
    with pytest.raises(ValueError, match="takes no parameters"):
        feedback_from_config({"closed_loop": {"feedback": {"kind": "backprop", "noise_std": 0.1}}})


def test_policy_gradient_is_refused_at_assembly_when_the_loss_has_no_reward() -> None:
    cfg = _cfg()
    cfg["closed_loop"]["feedback"] = "policy_gradient"
    with pytest.raises(ValueError, match="needs a loss option with step_reward"):
        assemble_closed_loop(cfg, seed=0)
