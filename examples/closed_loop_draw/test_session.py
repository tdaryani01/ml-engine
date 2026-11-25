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


# --- the guards of fork/restore: stand-in parts, no native kernels and no example assets, so they run in the ml CI ---
import types

import numpy as np


def _fake_session(monkeypatch, cnn_opt):
    from examples.closed_loop_draw import session as S

    monkeypatch.setattr(S, "capture_state", lambda actor: {})
    monkeypatch.setattr(S, "restore_state", lambda actor, st: None)
    mk = lambda: types.SimpleNamespace(W=np.zeros(2), b=np.zeros(2), _mW=np.zeros(2), _vW=np.zeros(2), _mb=np.zeros(2), _vb=np.zeros(2), _t=0)
    app = types.SimpleNamespace(actor=None, adapter=mk(), action_embed=mk(), conditioning=types.SimpleNamespace(embeddings=np.zeros(2), _m=np.zeros(2), _v=np.zeros(2), _t=0), cnn=types.SimpleNamespace(optimizer=cnn_opt))
    s = S.DrawSession.__new__(S.DrawSession)
    s.app = app
    return s


def test_fork_restore_roundtrip_with_a_list_of_moments(monkeypatch):
    opt = types.SimpleNamespace(m=[np.ones(2), None], v=[np.ones(2), None], t=3)
    s = _fake_session(monkeypatch, opt)
    f = s.fork()
    opt.m[0][...] = 9.0
    opt.t = 7
    s.restore(f)
    assert opt.m[0].tolist() == [1.0, 1.0] and opt.t == 3


def test_fork_refuses_a_moment_store_it_would_silently_skip(monkeypatch):
    opt = types.SimpleNamespace(state={"w": {"m": np.zeros(2)}}, t=0)
    s = _fake_session(monkeypatch, opt)
    with pytest.raises(TypeError, match="holds arrays"):
        s.fork()


def test_restore_refuses_a_moment_list_that_changed_shape(monkeypatch):
    opt = types.SimpleNamespace(m=[np.ones(2), None], t=0)
    s = _fake_session(monkeypatch, opt)
    f = s.fork()
    opt.m = [np.ones(2)]
    with pytest.raises(ValueError, match="changed shape"):
        s.restore(f)
    opt.m = [np.ones(2), np.ones(2)]
    with pytest.raises(ValueError, match="changed shape"):
        s.restore(f)
