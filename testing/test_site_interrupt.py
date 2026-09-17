# BL-023: pool HB fail streak → local pause; reconnect → /api/work/pause.
from __future__ import annotations

from src.manager_heartbeat import ManagerHeartbeat, ManagerHeartbeatConfig


def test_regression_site_interrupt_hb_streak_pauses_and_work_pause():
    hb = ManagerHeartbeat(
        ManagerHeartbeatConfig(
            enabled=True,
            uri="http://tm.test",
            instance_id="25476195",
            kind="engine",
            interval_s=0.01,
            timeout_s=0.2,
            site_interrupt_fail_threshold=3,
        )
    )
    trips: list[str] = []
    hb.set_on_site_interrupt(lambda: trips.append("trip"))
    hb.bind_job("job-1", model_id="tm-brain")
    calls: list[str] = []

    def fake_post(url, body, *, expect_commands, timeout_s=None):
        calls.append(url)
        if "/api/workers/" in url and url.rstrip("/").endswith("/heartbeat"):
            return None  # TM down
        return None

    hb._post_json = fake_post  # type: ignore[method-assign]

    for _ in range(3):
        hb._sync_pool_session(
            "http://tm.test",
            need_pool=False,
            metrics={"run_state": "training", "state": "training"},
        )

    assert hb.site_interrupt_hold is True
    assert trips == ["trip"]
    assert hb.desired_state == "paused"

    # TM returns — next successful pool HB should POST /api/work/pause.
    pause_bodies: list[dict] = []

    def fake_post_up(url, body, *, expect_commands, timeout_s=None):
        calls.append(url)
        if "/api/work/pause" in url:
            pause_bodies.append(dict(body))
            return {"job_id": "job-1", "state": "paused"}
        if "/api/workers/" in url and url.rstrip("/").endswith("/heartbeat"):
            return {
                "session_id": "25476195",
                "should_exit": False,
                "release_job": True,
            }
        if expect_commands:
            return {}
        return True

    hb._post_json = fake_post_up  # type: ignore[method-assign]
    # post_json for work/pause uses public post_json → _post_json path via url build.
    # _note_pool_hb_success calls self.post_json which hits network — stub post_json.
    def fake_post_json(path, body, *, timeout_s=None):
        if path == "/api/work/pause":
            pause_bodies.append(dict(body))
            return {"state": "paused"}
        return None

    hb.post_json = fake_post_json  # type: ignore[method-assign]
    hb._sync_pool_session(
        "http://tm.test",
        need_pool=False,
        metrics={"run_state": "paused", "state": "paused"},
    )
    assert pause_bodies and pause_bodies[0].get("model_id") == "tm-brain"
    assert hb.job_bound is False
    assert hb.site_interrupt_hold is True
    mets = hb._metrics
    assert mets.get("state") == "paused"


def test_regression_site_interrupt_no_trip_when_idle():
    hb = ManagerHeartbeat(
        ManagerHeartbeatConfig(
            enabled=True,
            uri="http://tm.test",
            instance_id="11111111",
            site_interrupt_fail_threshold=2,
        )
    )
    trips: list[str] = []
    hb.set_on_site_interrupt(lambda: trips.append("trip"))

    def fake_post(url, body, *, expect_commands, timeout_s=None):
        return None

    hb._post_json = fake_post  # type: ignore[method-assign]
    hb._sync_pool_session("http://tm.test", need_pool=False, metrics={"run_state": "idle"})
    hb._sync_pool_session("http://tm.test", need_pool=False, metrics={"run_state": "idle"})
    assert trips == []
    assert hb.site_interrupt_hold is False
