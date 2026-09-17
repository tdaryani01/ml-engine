"""BL-028f: restore rewind must ignore future-timeline onsets (Flash Fix B)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from examples.closed_loop_draw.agent import DrawStudentAgent


def _fake_make_target(cfg, batch_size=None):
    cid = int(cfg["closed_loop"].get("command_id", 0))
    B = int(batch_size or cfg["closed_loop"].get("batch_size", 1))
    return np.full((B, 1, 4, 4), float(cid), dtype=np.float32)


def _bare_agent():
    cfg = {
        "closed_loop": {
            "batch_size": 2,
            "canvas": [1, 28, 28],
            "command_id": 0,
            "remix_data_on_es": False,
            "train_patience": 32,
            "target": {"kind": "stock"},
        }
    }
    agent = object.__new__(DrawStudentAgent)
    agent.cfg = cfg
    agent.B = 2
    agent.command_id = 0
    agent.command_ids = np.full(2, 0, dtype=np.int64)
    agent.target = _fake_make_target(cfg, batch_size=2)
    agent._probe_target = _fake_make_target(cfg, batch_size=2)
    agent._remix_data_on_es = False
    agent._remix_count = 0
    agent._remix_after_restore = False
    agent._traj = 425
    agent._stale = 0
    agent._train_patience = 32
    agent._handled_onset_version = None
    agent._es_tripped = False
    agent._checkpoint_version = None
    agent._config_version = None
    agent._last_loss = 0.0
    return agent


def _health_hb(health: dict):
    class HB:
        cfg = SimpleNamespace(instance_id="draw-test", timeout_s=0.5)
        job_bound = True
        bound_model_id = "tm-brain"

        def get_json(self, path, timeout_s=3.0):
            assert "session-health" in path
            return dict(health)

        def fetch_blob(self, _key):
            return None

        def set_active_checkpoint(self, *_a, **_k):
            return None

        def queue_ack(self, *_a, **_k):
            return None

    return HB()


def test_regression_onset_future_timeline_not_within_patience(monkeypatch):
    """traj=425 vs onset_version=1136 must not trip (old age=max(0,…) → 0 bug)."""
    monkeypatch.setattr(
        "examples.closed_loop_draw.agent.make_target", _fake_make_target
    )
    agent = _bare_agent()
    agent._traj = 425
    agent._handled_onset_version = None
    agent.hb = _health_hb({"onset_version": 1136, "onsets": [{"version": 1136}]})

    within, onset = DrawStudentAgent._onset_within_patience(agent)
    assert within is False
    assert onset is None


def test_regression_restore_advances_handled_past_future_onsets(monkeypatch):
    """After restore to 425 with future onsets on tape, mark them handled."""
    monkeypatch.setattr(
        "examples.closed_loop_draw.agent.make_target", _fake_make_target
    )
    agent = _bare_agent()
    agent._traj = 2000  # pre-restore
    agent._handled_onset_version = None
    agent.hb = _health_hb(
        {
            "onset_version": 1136,
            "onsets": [{"version": 800}, {"version": 1136}],
        }
    )

    # Simulate successful restore rewind without a real blob.
    agent._traj = 425
    agent._stale = 0
    agent._es_tripped = False
    DrawStudentAgent._advance_handled_past_future_onsets(agent)

    assert agent._handled_onset_version is not None
    assert int(agent._handled_onset_version) >= 1136

    within, onset = DrawStudentAgent._onset_within_patience(agent)
    assert within is False
    assert onset is None


def test_regression_past_onset_still_within_patience(monkeypatch):
    """Onsets on the current timeline still trip when age <= patience."""
    monkeypatch.setattr(
        "examples.closed_loop_draw.agent.make_target", _fake_make_target
    )
    agent = _bare_agent()
    agent._traj = 450
    agent._handled_onset_version = None
    agent.hb = _health_hb({"onset_version": 440, "onsets": [{"version": 440}]})

    within, onset = DrawStudentAgent._onset_within_patience(agent)
    assert within is True
    assert onset == 440
