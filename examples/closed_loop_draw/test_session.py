# examples/closed_loop_draw/test_session.py
"""DrawSession: a fork restored gives the same trajectories; val is deterministic and on another shape. Needs the native kernels (skipped where they are not built)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CFG = Path(__file__).resolve().parent / "config_draw_smoke.yaml"


@pytest.fixture()
def session():
    try:
        from utils.conv_dispatch import bootstrap_im2col_gemm_runtime

        bootstrap_im2col_gemm_runtime()
        from examples.closed_loop_draw.session import DrawSession

        s = DrawSession(CFG, seed=0, train_command=0, val_command=1)
    except FileNotFoundError as exc:      # conv_kernels.so is not built here
        pytest.skip(f"native kernels not available: {exc}")
    yield s
    s.close()


def test_restore_gives_the_same_trajectories(session):
    session.train(2)
    f = session.fork()
    a = session.train(3)
    session.restore(f)
    b = session.train(3)
    session.restore(f)
    c = session.train(3)
    assert a == b == c


def test_val_is_deterministic_and_k_buys_nothing(session):
    session.train(2)
    assert session.val(1) == session.val(1) == session.val(8)


def test_val_does_not_move_the_weights_and_training_does(session):
    v0 = session.val()
    assert session.val() == v0
    session.train(3)
    assert session.val() != v0


def test_the_validation_shape_must_differ_from_the_training_shape():
    from examples.closed_loop_draw.session import DrawSession

    with pytest.raises(ValueError):
        DrawSession(CFG, seed=0, train_command=0, val_command=0)


def test_tick_runs_after_each_trajectory_and_the_canvas_does_not_move_the_weights(session):
    seen = []
    session.tick = seen.append
    session.train(3)
    v = session.val()
    assert seen == ["train"] * 3 + ["val"]
    f = session.fork()
    c = session.canvas()
    assert c.shape[0] == 1 and session.target().shape[0] == 1
    session.restore(f)
    assert session.val() == v



def test_the_canvas_is_the_drawing_fit_scores_when_the_model_sees_its_target():
    """With ``target_conditioning`` the model's goal includes the target image. The canvas must be drawn with that same goal: it is then the last frame of the rollout ``fit`` scores, not a drawing made without the target (the window showed the latter: its picture drifted from the scored one as the model learned to use the target)."""
    import numpy as np

    try:
        from utils.conv_dispatch import bootstrap_im2col_gemm_runtime

        bootstrap_im2col_gemm_runtime()
        from examples.closed_loop_draw.assemble import load_config
        from examples.closed_loop_draw.goal import DrawGoal
        from examples.closed_loop_draw.session import DrawSession

        cfg = load_config(CFG)
        cfg["closed_loop"]["target_conditioning"] = True
        s = DrawSession(cfg, seed=0, train_command=0, val_command=1)
    except FileNotFoundError as exc:      # conv_kernels.so is not built here
        pytest.skip(f"native kernels not available: {exc}")
    try:
        s.train(6)
        scored = s.app.trainer.rollout_train(goal=s._goal(s.train_command), lr=s.lr, apply_updates=False).extras["frames"][-1][:1]
        blind = s.app.trainer.rollout_eval(goal=DrawGoal.render(s._ids[s.train_command][:1])).extras["frames"][-1]
        c = s.canvas()
        assert np.allclose(c, scored, atol=1e-6)
        assert not np.allclose(blind, scored, atol=1e-6)      # the goal without the target draws something else: the test would pass trivially otherwise
    finally:
        s.close()
