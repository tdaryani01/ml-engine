# Restore-then-remix: ES calls brain first; terrain remix after restore lands.
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from examples.closed_loop_draw.agent import DrawStudentAgent
from examples.closed_loop_draw.commands import STOCK_COMMAND_IDS


def _fake_make_target(cfg, batch_size=None):
    cid = int(cfg["closed_loop"].get("command_id", 0))
    B = int(batch_size or cfg["closed_loop"].get("batch_size", 1))
    return np.full((B, 1, 4, 4), float(cid), dtype=np.float32)


def _bare_agent(*, remix=True, command_id=0):
    cfg = {
        "closed_loop": {
            "batch_size": 2,
            "canvas": [1, 28, 28],
            "command_id": command_id,
            "remix_data_on_es": remix,
            "target": {"kind": "stock"},
        }
    }
    agent = object.__new__(DrawStudentAgent)
    agent.cfg = cfg
    agent.B = 2
    agent.command_id = command_id
    agent.command_ids = np.full(2, command_id, dtype=np.int64)
    agent.target = _fake_make_target(cfg, batch_size=2)
    agent._probe_target = _fake_make_target(cfg, batch_size=2)
    agent._remix_data_on_es = remix
    agent._remix_count = 0
    agent._remix_after_restore = False
    agent._traj = 40
    agent._stale = 99
    agent._best_ink_miss = 0.5
    agent._best_probe = 0.4
    agent._last_ink_miss = 0.5
    agent._last_probe = 0.4
    agent._last_loss = 0.3
    agent.lr = 1e-3
    agent._es_park_hold = True
    agent._es_tripped = True
    agent._checkpoint_version = None
    return agent


def test_regression_remix_helper_changes_stock_target(monkeypatch):
    monkeypatch.setattr(
        "examples.closed_loop_draw.agent.make_target", _fake_make_target
    )
    agent = _bare_agent()
    before = int(agent.command_id)
    before_target = np.array(agent.target, copy=True)
    DrawStudentAgent._remix_training_data(agent)

    assert int(agent.command_id) != before
    assert int(agent.command_id) in set(STOCK_COMMAND_IDS)
    assert not np.allclose(agent.target, before_target)
    assert agent._stale == 0
    assert agent._best_ink_miss is None
    assert agent._remix_count == 1


def test_regression_es_shadow_does_not_remix_before_restore(monkeypatch):
    """Onset/ES posts tm-brain/act and arms remix — does not remix yet."""
    monkeypatch.setattr(
        "examples.closed_loop_draw.agent.make_target", _fake_make_target
    )
    agent = _bare_agent()
    agent._es_shadow = True
    agent._es_apply = True
    agent._es_min_traj = 1
    agent._train_patience = 1
    agent._es_tripped = False
    agent._handled_onset_version = None
    agent._outcome_horizon = 10
    agent._pending_outcome_episode = None
    agent._outcome_due_traj = None
    agent._last_status = None

    remixed = []

    def boom(*_a, **_k):
        remixed.append(True)
        raise AssertionError("remix must not run inside _maybe_es_shadow")

    monkeypatch.setattr(DrawStudentAgent, "_remix_training_data", boom)
    monkeypatch.setattr(
        DrawStudentAgent,
        "_onset_within_patience",
        lambda self: (True, 30),
    )
    monkeypatch.setattr(DrawStudentAgent, "_set_status", lambda self, *a, **k: None)

    posts = []

    class HB:
        cfg = SimpleNamespace(instance_id="draw-test")

        def post_json(self, path, body, timeout_s=15.0):
            posts.append((path, body))
            return {
                "decision": {
                    "action": "restore_best",
                    "authority": "rules",
                    "reason": "within patience of unhealthy onset",
                    "episode_id": "ep-1",
                    "model_action": None,
                },
                "actuation": {"applied": True, "sequence": ["pause", "restore", "resume"]},
            }

    agent.hb = HB()
    before_cid = int(agent.command_id)
    DrawStudentAgent._maybe_es_shadow(agent)

    assert remixed == []
    assert agent._remix_after_restore is True
    assert int(agent.command_id) == before_cid
    assert agent._es_park_hold is False  # restore_best clears sticky park
    assert any("/tm-brain/act" in p[0] for p in posts)


def test_regression_es_act_uses_bound_model_not_pool_session(monkeypatch):
    """Pool session id must not be the tm-brain/act path (was HTTP 404)."""
    monkeypatch.setattr(
        "examples.closed_loop_draw.agent.make_target", _fake_make_target
    )
    agent = _bare_agent()
    agent._es_shadow = True
    agent._es_apply = True
    agent._es_min_traj = 1
    agent._train_patience = 1
    agent._es_tripped = False
    agent._handled_onset_version = None
    agent._outcome_horizon = 10
    agent._pending_outcome_episode = None
    agent._outcome_due_traj = None

    monkeypatch.setattr(
        DrawStudentAgent,
        "_onset_within_patience",
        lambda self: (False, None),
    )
    monkeypatch.setattr(DrawStudentAgent, "_set_status", lambda self, *a, **k: None)

    posts: list[str] = []

    class HB:
        cfg = SimpleNamespace(instance_id="22409324")  # pool session

        @property
        def bound_model_id(self) -> str:
            return "tm-brain"

        def post_json(self, path, body, timeout_s=15.0):
            posts.append(path)
            return {
                "decision": {
                    "action": "noop",
                    "authority": "rules",
                    "reason": "stable / continue",
                    "episode_id": "ep-pool",
                    "model_action": None,
                },
                "actuation": {"applied": False, "reason": "no actuator for action"},
            }

    agent.hb = HB()
    DrawStudentAgent._maybe_es_shadow(agent)

    assert posts, "expected tm-brain/act POST"
    assert posts[0] == "/api/instances/tm-brain/tm-brain/act"
    assert all("22409324" not in p for p in posts)


def test_regression_restore_ack_remixes_terrain_once(monkeypatch):
    monkeypatch.setattr(
        "examples.closed_loop_draw.agent.make_target", _fake_make_target
    )
    agent = _bare_agent()
    agent._remix_after_restore = True

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
        lambda _data: {"weights": {}},
    )
    monkeypatch.setattr(
        "examples.closed_loop_draw.agent.apply_checkpoint_blob",
        lambda _app, _body: None,
    )
    agent.app = object()

    before = int(agent.command_id)
    cmd = SimpleNamespace(
        id="c1",
        payload={"blob_key": "k", "version": 7},
    )
    DrawStudentAgent._restore_from_command(agent, cmd)

    assert int(agent.command_id) != before
    assert int(agent.command_id) in set(STOCK_COMMAND_IDS)
    assert agent._stale == 0
    assert agent._remix_after_restore is False
    assert agent._es_park_hold is False
    assert agent._remix_count == 1

    # after_restore resume must not double-remix
    before2 = int(agent.command_id)
    count2 = agent._remix_count
    resume = SimpleNamespace(
        id="c2",
        action="resume",
        payload={"source": "tm_brain", "phase": "after_restore"},
    )
    agent._user_pause_hold = False
    agent._run_authorized = False
    agent._paused = True
    agent.hb.set_desired_state = lambda *_a, **_k: None
    agent.hb.mark_command_seen = lambda *_a, **_k: None
    agent._apply_config_payload = lambda _c: {}
    agent._live_config = lambda: {}
    agent._print_config = lambda *_a, **_k: None
    DrawStudentAgent._handle_start_resume(agent, resume)
    assert int(agent.command_id) == before2
    assert agent._remix_count == count2


def test_regression_after_restore_remixes_if_restore_deferred(monkeypatch):
    """If remix was only armed (not consumed on restore), after_restore consumes it."""
    monkeypatch.setattr(
        "examples.closed_loop_draw.agent.make_target", _fake_make_target
    )
    agent = _bare_agent()
    agent._remix_after_restore = True
    agent._user_pause_hold = False
    agent.hb = SimpleNamespace(
        set_desired_state=lambda *_a, **_k: None,
        mark_command_seen=lambda *_a, **_k: None,
        queue_ack=lambda *_a, **_k: None,
    )
    agent._apply_config_payload = lambda _c: {}
    agent._live_config = lambda: {}
    agent._print_config = lambda *_a, **_k: None

    before = int(agent.command_id)
    resume = SimpleNamespace(
        id="c3",
        action="resume",
        payload={"source": "tm_brain", "phase": "after_restore"},
    )
    DrawStudentAgent._handle_start_resume(agent, resume)
    assert int(agent.command_id) != before
    assert agent._remix_after_restore is False
    assert agent._es_park_hold is False
    assert agent._remix_count == 1
