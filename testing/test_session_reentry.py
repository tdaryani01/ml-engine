# BL-023b: session-identity 404 → bounded re-register (Rules 1–4).
from __future__ import annotations

import time

from src.manager_heartbeat import (
    HttpResult,
    ManagerHeartbeat,
    ManagerHeartbeatConfig,
    _REENTRY_COOLDOWN_S,
)


def _hb(uri: str = "http://tm.test") -> ManagerHeartbeat:
    return ManagerHeartbeat(
        ManagerHeartbeatConfig(
            enabled=True,
            uri=uri,
            instance_id="86426584",
            kind="engine",
            interval_s=0.01,
            timeout_s=0.2,
            site_interrupt_fail_threshold=6,
        )
    )


def test_bl023b_hb_session_404_reregisters_same_id_no_site_streak():
    hb = _hb()
    hb.bind_job("job-1", model_id="draw-student")
    calls: list[str] = []
    register_n = {"n": 0}

    def fake_post(url, body, *, expect_commands, timeout_s=None):
        calls.append(url)
        if url.rstrip("/").endswith("/register"):
            register_n["n"] += 1
            return HttpResult(ok=True, status=200, payload=True)
        if "/heartbeat" in url:
            # First HB in a sync: 404; after register, success.
            if register_n["n"] == 0:
                return HttpResult(
                    ok=False,
                    status=404,
                    detail='{"detail":"session not found"}',
                )
            return HttpResult(
                ok=True,
                status=200,
                payload={"session_id": "86426584", "should_exit": False},
            )
        return HttpResult(ok=False, status=500, detail="unexpected")

    hb._post_result = fake_post  # type: ignore[method-assign]
    sleeps: list[float] = []
    hb_mod_sleep = time.sleep

    def fake_sleep(s):
        sleeps.append(s)

    import src.manager_heartbeat as mh

    mh.time.sleep = fake_sleep  # type: ignore[attr-defined]
    try:
        streak0 = hb._hb_fail_streak
        hb._sync_pool_session(
            "http://tm.test",
            need_pool=False,
            metrics={"run_state": "training", "state": "training"},
        )
    finally:
        mh.time.sleep = hb_mod_sleep  # type: ignore[attr-defined]

    assert register_n["n"] == 1
    assert any(u.rstrip("/").endswith("/register") for u in calls)
    assert hb._pool_registered is True
    assert hb._hb_fail_streak == streak0
    assert hb.site_interrupt_hold is False
    assert hb.claim_suspect is False
    assert hb.job_bound is True
    assert sleeps and sleeps[0] <= 1.0


def test_bl023b_cooldown_blocks_second_register_storm():
    hb = _hb()
    hb.bind_job("job-1", model_id="draw-student")
    register_n = {"n": 0}

    def always_404(url, body, *, expect_commands, timeout_s=None):
        if url.rstrip("/").endswith("/register"):
            register_n["n"] += 1
            return HttpResult(ok=True, status=200, payload=True)
        if "/heartbeat" in url:
            return HttpResult(ok=False, status=404, detail="session not found")
        return HttpResult(ok=False, status=500, detail="x")

    hb._post_result = always_404  # type: ignore[method-assign]
    import src.manager_heartbeat as mh

    real_sleep = mh.time.sleep
    mh.time.sleep = lambda s: None  # type: ignore[attr-defined]
    try:
        hb._sync_pool_session(
            "http://tm.test",
            need_pool=False,
            metrics={"run_state": "training", "state": "training"},
        )
        # First cycle: one register, retry HB 404 → cooldown + park
        assert register_n["n"] == 1
        assert hb.job_bound is False
        assert hb.site_interrupt_hold is True

        hb.bind_job("job-2", model_id="draw-student")
        hb._sync_pool_session(
            "http://tm.test",
            need_pool=False,
            metrics={"run_state": "training", "state": "training"},
        )
        # Cooling: no second register
        assert register_n["n"] == 1
    finally:
        mh.time.sleep = real_sleep  # type: ignore[attr-defined]


def test_bl023b_generic_404_does_not_reregister():
    hb = _hb()
    calls: list[str] = []

    def fake_post(url, body, *, expect_commands, timeout_s=None):
        calls.append(url)
        if "/heartbeat" in url:
            return HttpResult(ok=False, status=404, detail="checkpoint not found")
        return HttpResult(ok=True, status=200, payload=True)

    hb._post_result = fake_post  # type: ignore[method-assign]
    hb._sync_pool_session(
        "http://tm.test",
        need_pool=False,
        metrics={"run_state": "idle", "state": "idle"},
    )
    assert not any(u.rstrip("/").endswith("/register") for u in calls)
    assert hb._pool_registered is False


def test_bl023b_malformed_404_body_no_raise():
    hb = _hb()

    def fake_post(url, body, *, expect_commands, timeout_s=None):
        if url.rstrip("/").endswith("/register"):
            return HttpResult(ok=True, status=200, payload=True)
        if "/heartbeat" in url:
            return HttpResult(
                ok=False,
                status=404,
                detail="<html>session not found</html>",
            )
        return HttpResult(ok=False, status=500, detail="")

    hb._post_result = fake_post  # type: ignore[method-assign]
    import src.manager_heartbeat as mh

    real_sleep = mh.time.sleep
    mh.time.sleep = lambda s: None  # type: ignore[attr-defined]
    try:
        hb._sync_pool_session(
            "http://tm.test",
            need_pool=False,
            metrics={"run_state": "idle", "state": "idle"},
        )
    finally:
        mh.time.sleep = real_sleep  # type: ignore[attr-defined]
    # Idle + failed retry → cooldown, no crash
    assert hb._reentry_cooling() or not hb._pool_registered


def test_bl023b_cooldown_constant():
    assert _REENTRY_COOLDOWN_S == 3.0
