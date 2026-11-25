# testing/test_closed_loop_draw_session_guards.py
"""Guards of ``examples.closed_loop_draw.session.DrawSession.fork/restore`` on stand-in parts: no native kernels, no example assets, so they run in the ml CI suite (testing/). The exact-trajectory tests stay in examples/closed_loop_draw/test_session.py (they need the kernels and are skipped where those are not built).
A moment store of torch tensors in a dict would not be recognised by ``_holds_arrays`` (numpy arrays only): this engine has none."""
from __future__ import annotations

import types

import numpy as np
import pytest


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
