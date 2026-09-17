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


def test_regression_es_autopilot_trip_awaits_tm_no_act(monkeypatch):
    """BL-024: Autopilot ES publishes trip + soft hold; no /act or /work/pause."""
    monkeypatch.setattr(
        "examples.closed_loop_draw.agent.make_target", _fake_make_target
    )
    agent = _bare_agent()
    agent._es_shadow = True
    agent._es_apply = True
    agent._autopilot = True
    agent._es_min_traj = 1
    agent._train_patience = 1
    agent._es_tripped = False
    agent._handled_onset_version = None
    agent._user_pause_hold = False
    agent._paused = False
    agent._traj = 2
    agent._remix_data_on_es = True
    agent._remix_after_restore = False

    monkeypatch.setattr(
        DrawStudentAgent,
        "_onset_within_patience",
        lambda self: (True, 30),
    )
    statuses: list[str] = []

    def _capture_status(self, state, **_k):
        statuses.append(str(state))

    monkeypatch.setattr(DrawStudentAgent, "_set_status", _capture_status)

    posts: list[tuple[str, dict]] = []

    class HB:
        cfg = SimpleNamespace(instance_id="draw-test", timeout_s=0.5)

        @property
        def bound_model_id(self) -> str:
            return "tm-brain"

        def post_json(self, path, body, timeout_s=15.0):
            posts.append((path, dict(body)))
            return {}

        def set_desired_state(self, *_a, **_k):
            return None

        def set_metrics(self, *_a, **_k):
            return None

    agent.hb = HB()
    DrawStudentAgent._maybe_es_shadow(agent)

    assert agent._paused is True
    assert agent._es_park_hold is True
    assert agent._user_pause_hold is False
    assert agent._remix_after_restore is True
    assert "es-onset" in statuses or any(s.startswith("es-") for s in statuses)
    assert not any("/tm-brain/act" in p[0] for p in posts)
    assert not any(p[0] == "/api/work/pause" for p in posts)


def test_regression_es_shadow_does_not_remix_before_restore(monkeypatch):
    """Onset/ES soft-holds for TM; does not remix yet."""
    monkeypatch.setattr(
        "examples.closed_loop_draw.agent.make_target", _fake_make_target
    )
    agent = _bare_agent()
    agent._es_shadow = True
    agent._es_apply = True
    agent._autopilot = True
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

        def set_desired_state(self, *_a, **_k):
            return None

    agent.hb = HB()
    before_cid = int(agent.command_id)
    DrawStudentAgent._maybe_es_shadow(agent)

    assert remixed == []
    assert agent._remix_after_restore is True
    assert int(agent.command_id) == before_cid
    assert agent._es_park_hold is True  # soft-hold awaiting TM restore/resume
    assert not any("/tm-brain/act" in p[0] for p in posts)


def test_regression_es_autopilot_uses_bound_model_not_pool_session(monkeypatch):
    """BL-024: trip uses bound agent id for metrics; no /act to pool session."""
    monkeypatch.setattr(
        "examples.closed_loop_draw.agent.make_target", _fake_make_target
    )
    agent = _bare_agent()
    agent._es_shadow = True
    agent._es_apply = True
    agent._autopilot = True
    agent._es_min_traj = 1
    agent._train_patience = 1
    agent._es_tripped = False
    agent._handled_onset_version = None
    agent._traj = 2
    agent._stale = 99
    agent._last_status = None
    agent._last_loss = 0.0
    agent._sigma = 0.0
    agent._max_steps = 64
    agent._continuity_weight = 0.0
    agent._ledger_on = False
    agent._paused = False
    agent._user_pause_hold = False
    agent._es_run_done = False
    agent._remix_data_on_es = False
    agent._es_apply = True

    monkeypatch.setattr(
        DrawStudentAgent,
        "_onset_within_patience",
        lambda self: (False, None),
    )
    published: list[dict] = []

    class HB:
        cfg = SimpleNamespace(instance_id="22409324")  # pool session

        @property
        def bound_model_id(self) -> str:
            return "tm-brain"

        def post_json(self, path, body, timeout_s=15.0):
            raise AssertionError(f"no HTTP on Autopilot ES trip: {path}")

        def set_desired_state(self, *_a, **_k):
            return None

        def set_metrics(self, metrics):
            published.append(dict(metrics))

        def maybe_ping(self, force=False):
            return None

    agent.hb = HB()
    DrawStudentAgent._maybe_es_shadow(agent)
    assert agent._tm_agent_id() == "tm-brain"
    assert agent._es_park_hold is True
    assert published
    assert str(published[-1].get("state") or "").startswith("es-")


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


def test_regression_manual_es_does_not_apply_restore(monkeypatch):
    """Start without Autopilot: ES ends the job — never restore / park claimed."""
    monkeypatch.setattr(
        "examples.closed_loop_draw.agent.make_target", _fake_make_target
    )
    agent = _bare_agent()
    agent._es_shadow = True
    agent._es_apply = True  # setup default — must still not apply without Autopilot
    agent._autopilot = False
    agent._es_min_traj = 1
    agent._train_patience = 1
    agent._es_tripped = False
    agent._handled_onset_version = None
    agent._outcome_horizon = 10
    agent._pending_outcome_episode = None
    agent._outcome_due_traj = None
    agent._paused = False
    agent._last_status = None
    agent._traj = 2

    statuses: list[str] = []
    stops: list[str] = []
    acks: list[dict] = []

    monkeypatch.setattr(
        DrawStudentAgent,
        "_onset_within_patience",
        lambda self: (True, 30),
    )

    def _capture_status(self, state, *, loss=None):
        statuses.append(state)
        self._last_status = state

    monkeypatch.setattr(DrawStudentAgent, "_set_status", _capture_status)

    posts = []

    class HB:
        cfg = SimpleNamespace(instance_id="draw-test")
        job_bound = True
        job_id = "job-manual-es"

        def post_json(self, path, body, timeout_s=15.0):
            posts.append((path, body))
            return {
                "decision": {
                    "action": "restore_best",
                    "authority": "rules",
                    "reason": "within patience of unhealthy onset",
                    "episode_id": "ep-manual",
                    "model_action": None,
                },
                "actuation": {"applied": False},
            }

        def set_desired_state(self, state):
            posts.append(("desired", state))

        def ack_work(self, *, result=None):
            acks.append(dict(result or {}))
            self.job_bound = False
            return {"job_id": "job-manual-es", "state": "done"}

    agent.hb = HB()
    agent.bind_engine_stop(lambda: stops.append("stop"))
    DrawStudentAgent._maybe_es_shadow(agent)

    # Manual: no tm-brain/act (durable episodes look like live steering in UI).
    assert not any(
        isinstance(p, str) and "/tm-brain/act" in p for p, _ in posts
    )
    assert agent._es_park_hold is False
    assert agent._es_run_done is True
    assert agent._paused is False
    assert agent._remix_after_restore is False
    assert agent._pending_outcome_episode is None
    assert ("desired", "idle") in posts
    assert ("desired", "paused") not in posts
    assert not any(
        isinstance(p, str) and p.endswith("/control/pause") for p, _ in posts
    )
    assert "es-stop:manual" in statuses
    assert not any(s.startswith("es-act:") for s in statuses)
    assert stops == ["stop"]
    assert len(acks) == 1
    assert acks[0].get("reason") == "es_manual_stop"
    assert acks[0].get("ok") is True
    assert agent.hb.job_bound is False


def test_regression_after_restore_prefers_feed_stock_over_local_remix(monkeypatch):
    """BL-005i: resume after_restore with feed.command_id skips local remix."""
    monkeypatch.setattr(
        "examples.closed_loop_draw.agent.make_target", _fake_make_target
    )
    agent = _bare_agent(command_id=0)
    agent._remix_after_restore = True
    agent._es_park_hold = True
    agent._user_pause_hold = False
    agent._es_run_done = False
    agent.hb = SimpleNamespace(
        set_desired_state=lambda *_a, **_k: None,
        mark_command_seen=lambda *_a, **_k: None,
        queue_ack=lambda *_a, **_k: None,
    )
    agent._apply_config_payload = lambda _c: {}
    agent._live_config = lambda: {}
    agent._print_config = lambda *_a, **_k: None
    remixed = []

    def boom(*_a, **_k):
        remixed.append(True)
        raise AssertionError("local remix must not run when feed stamps command_id")

    monkeypatch.setattr(DrawStudentAgent, "_remix_training_data", boom)

    resume = SimpleNamespace(
        id="c-feed",
        action="resume",
        payload={
            "source": "tm_brain",
            "phase": "after_restore",
            "config": {
                "feed": {
                    "plate_id": "plate-x",
                    "recipe_id": "stock_targets_default",
                    "command_id": 3,
                    "plate_kind": "stock_targets",
                }
            },
        },
    )
    ok = DrawStudentAgent.on_engine_start_resume(agent, resume)
    assert ok is True
    assert remixed == []
    assert int(agent.command_id) == 3
    assert agent._remix_after_restore is False
    assert agent._es_park_hold is False
    assert agent._remix_count == 1
