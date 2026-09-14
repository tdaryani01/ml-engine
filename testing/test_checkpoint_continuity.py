# BL-009: checkpoint continuity — save full live config; restore applies it.
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from examples.closed_loop_draw.agent import DrawStudentAgent
from examples.closed_loop_draw.draw_checkpoint import (
    job_overlay_only,
    knobs_from_cfg,
    public_run_config,
)
from src.manager_heartbeat import decode_checkpoint_blob, encode_checkpoint_blob


def test_regression_checkpoint_blob_embeds_and_returns_config():
    cfg = {
        "closed_loop": {"sigma": 0.11, "train_patience": 55, "max_steps": 10},
        "optimization": {"learning_rate": 0.001, "seed": 0},
        "mhsa": {"d_model": 64},
        "source": "strip-me",
        "resume_checkpoint": {"version": 1},
    }
    pub = public_run_config(cfg, lr=0.0007)
    assert pub["optimization"]["learning_rate"] == 0.0007
    assert "source" not in pub
    assert "resume_checkpoint" not in pub
    assert pub["closed_loop"]["sigma"] == 0.11

    body = {
        "kind": "draw_student_v1",
        "version": 25,
        "weights": {"cnn_w": [np.zeros((2, 2), dtype=np.float32)]},
        "config": pub,
        "knobs": knobs_from_cfg(cfg, lr=0.0007),
    }
    blob = encode_checkpoint_blob(body)
    out = decode_checkpoint_blob(blob)
    assert out["config"]["closed_loop"]["train_patience"] == 55
    assert out["knobs"]["learning_rate"] == 0.0007


def test_regression_job_overlay_only_strips_hot_knobs():
    ov = job_overlay_only(
        {
            "source": "user",
            "feed": {"on_es": "noop"},
            "closed_loop": {"sigma": 0.99},
            "optimization": {"learning_rate": 9.0},
        }
    )
    assert ov == {"source": "user", "feed": {"on_es": "noop"}}
    assert "closed_loop" not in ov


def test_regression_restore_applies_checkpoint_config(monkeypatch):
    """Explicit restore must re-apply live config from the checkpoint body."""
    agent = object.__new__(DrawStudentAgent)
    agent.cfg = {
        "closed_loop": {"sigma": 0.06, "train_patience": 32},
        "optimization": {"learning_rate": 0.0005},
    }
    agent.lr = 0.0005
    agent._sigma = 0.06
    agent._train_patience = 32
    agent._traj = 3
    agent._stale = 9
    agent._es_tripped = True
    agent._remix_data_on_es = False
    agent._remix_after_restore = False
    agent._last_loss = 0.1
    agent._checkpoint_version = None
    agent._config_version = None
    agent.app = object()

    applied: list[dict] = []

    def fake_apply(self, config, *, rebuild=False):
        del rebuild
        applied.append(dict(config))
        self.cfg = dict(config)
        opt = config.get("optimization") or {}
        cl = config.get("closed_loop") or {}
        if "learning_rate" in opt:
            self.lr = float(opt["learning_rate"])
        if "sigma" in cl:
            self._sigma = float(cl["sigma"])
        if "train_patience" in cl:
            self._train_patience = int(cl["train_patience"])
        return {"learning_rate": self.lr}

    monkeypatch.setattr(DrawStudentAgent, "apply_run_config", fake_apply)
    monkeypatch.setattr(
        "examples.closed_loop_draw.agent.apply_checkpoint_blob",
        lambda _app, _body: {"knobs": {}, "config": {}},
    )

    ckpt_cfg = {
        "closed_loop": {"sigma": 0.19, "train_patience": 77},
        "optimization": {"learning_rate": 0.0025},
    }

    class HB:
        def fetch_blob(self, _key):
            return b"blob"

        def set_active_checkpoint(self, _ckpt):
            return None

        def queue_ack(self, *_a, **_k):
            return None

    agent.hb = HB()
    monkeypatch.setattr(
        "examples.closed_loop_draw.agent.load_checkpoint_blob",
        lambda _data: {
            "weights": {"x": 1},
            "config": ckpt_cfg,
            "knobs": {"learning_rate": 0.0025},
        },
    )

    cmd = SimpleNamespace(
        id="c-restore",
        payload={"blob_key": "ckpts/m/v25.pkl", "version": 25},
    )
    DrawStudentAgent._restore_from_command(agent, cmd)

    assert applied and applied[0]["optimization"]["learning_rate"] == 0.0025
    assert agent.lr == 0.0025
    assert agent._sigma == 0.19
    assert agent._train_patience == 77
    assert agent._checkpoint_version == 25
    assert agent._traj == 25
    assert agent._stale == 0


def test_regression_claim_resume_checkpoint_restores_before_overlay(monkeypatch):
    """Any-worker claim with resume_checkpoint pin restores ckpt; overlay only."""
    agent = object.__new__(DrawStudentAgent)
    agent.cfg = {"closed_loop": {"sigma": 0.06}, "optimization": {"learning_rate": 1e-3}}
    agent.lr = 1e-3
    agent._sigma = 0.06
    agent._traj = 0
    agent._es_park_hold = True
    agent._user_pause_hold = True
    agent._last_loss = None
    agent.app = object()

    restored = []

    def fake_restore(self, *, version, blob_key, command_id):
        restored.append((version, blob_key, command_id))
        self.lr = 0.004
        self._sigma = 0.22
        self.cfg = {
            "closed_loop": {"sigma": 0.22},
            "optimization": {"learning_rate": 0.004},
        }
        self._traj = int(version)
        return True

    monkeypatch.setattr(DrawStudentAgent, "_restore_checkpoint", fake_restore)

    job = {
        "job_id": "j1",
        "model_id": "draw-1",
        "config": {
            "source": "user",
            "feed": {"on_es": "noop"},
            "closed_loop": {"sigma": 0.99},
            "optimization": {"learning_rate": 9.0},
        },
        "data": {
            "resume_checkpoint": {
                "version": 40,
                "blob_key": "ckpts/draw-1/v40.pkl",
            }
        },
    }
    assert DrawStudentAgent.on_claim_config(agent, job) is True
    assert restored == [(40, "ckpts/draw-1/v40.pkl", None)]
    # Hot knobs from restore, not from stale Start stamp.
    assert agent.lr == 0.004
    assert agent._sigma == 0.22
    assert agent.cfg["source"] == "user"
    assert agent.cfg["feed"]["on_es"] == "noop"
    assert agent.cfg["closed_loop"]["sigma"] == 0.22
    assert agent._es_park_hold is False
    assert agent._user_pause_hold is False
