# BL-006a / BL-011: Engine heartbeats register as TM pool workers only (capacity).
# Agent /api/instances HB happens only after bind_job with job.model_id.
from __future__ import annotations

from src.manager_heartbeat import ManagerHeartbeat, ManagerHeartbeatConfig


def test_regression_engine_heartbeat_pool_only_on_boot():
    """Cold-start worker: pool register + pool HB; no agent InstanceRecord."""
    hb = ManagerHeartbeat(
        ManagerHeartbeatConfig(
            enabled=True,
            uri="http://tm.test",
            instance_id="55443322",
            kind="engine",
            interval_s=0.01,
            timeout_s=0.2,
        )
    )
    calls: list[tuple[str, dict]] = []

    def fake_post(url, body, *, expect_commands):
        calls.append((url, dict(body)))
        if "/api/workers/" in url and url.rstrip("/").endswith("/heartbeat"):
            return {"session_id": "55443322", "should_exit": False}
        if expect_commands:
            return {"desired_state": "idle", "commands": []}
        return True

    hb._post_json = fake_post  # type: ignore[method-assign]
    hb.set_metrics({"run_state": "idle"})
    hb._worker(
        need_register=False,
        need_pool=True,
        metrics={"run_state": "idle"},
        job_bound=False,
    )

    urls = [u for u, _ in calls]
    assert not any("/api/instances" in u for u in urls)
    assert any(u.endswith("/api/workers/register") for u in urls)
    assert any("/api/workers/55443322/heartbeat" in u for u in urls)

    reg = next(b for u, b in calls if u.endswith("/api/workers/register"))
    assert reg["session_id"] == "55443322"
    assert "pool" in reg["caps"]
    assert "engine" in reg["caps"]

    pool_hb = next(b for u, b in calls if "/api/workers/55443322/heartbeat" in u)
    assert pool_hb["status"] == "idle"
    assert hb._pool_registered is True
    assert hb.job_bound is False


def test_regression_empty_instance_id_gets_random_pool_id():
    hb = ManagerHeartbeat(
        ManagerHeartbeatConfig(enabled=True, uri="http://tm.test", instance_id="")
    )
    assert hb.pool_session_id.isdigit()
    assert len(hb.pool_session_id) == 8


def test_regression_bind_job_heartbeats_agent_model_id():
    """Only after bind_job(model_id=…) does the worker touch /api/instances/{model}."""
    hb = ManagerHeartbeat(
        ManagerHeartbeatConfig(
            enabled=True,
            uri="http://tm.test",
            instance_id="55443322",
            kind="engine",
        )
    )
    calls: list[tuple[str, dict]] = []

    def fake_post(url, body, *, expect_commands):
        calls.append((url, dict(body)))
        if "/api/workers/" in url and url.rstrip("/").endswith("/heartbeat"):
            return {"session_id": "55443322", "should_exit": False}
        if expect_commands:
            return {"desired_state": "running", "commands": []}
        return True

    hb._post_json = fake_post  # type: ignore[method-assign]
    hb.bind_job("job-1", model_id="tm-brain-1")
    assert hb.job_bound is True
    assert hb.job_id == "job-1"
    assert hb.bound_model_id == "tm-brain-1"
    hb._worker(
        need_register=True,
        need_pool=True,
        metrics={"run_state": "training"},
        job_bound=True,
    )
    urls = [u for u, _ in calls]
    assert not any(u.endswith("/api/instances") and not u.rstrip("/").endswith("heartbeat") for u in urls)
    assert any(u.endswith("/api/instances/tm-brain-1/heartbeat") for u in urls)
    assert any(u.endswith("/api/workers/register") for u in urls)

    hb.unbind_job()
    assert hb.job_bound is False
    assert hb.bound_model_id is None
    calls.clear()
    hb._worker(
        need_register=False,
        need_pool=False,
        metrics={"run_state": "idle"},
        job_bound=False,
    )
    urls2 = [u for u, _ in calls]
    assert not any("/api/instances" in u for u in urls2)
    assert any("/api/workers/55443322/heartbeat" in u for u in urls2)


def test_regression_pool_heartbeat_miss_clears_registration():
    hb = ManagerHeartbeat(
        ManagerHeartbeatConfig(
            enabled=True,
            uri="http://tm.test",
            instance_id="55443322",
        )
    )
    hb._pool_registered = True

    def fake_post(url, body, *, expect_commands):
        del body, expect_commands
        if "/api/workers/" in url and url.endswith("/heartbeat"):
            return None
        return True

    hb._post_json = fake_post  # type: ignore[method-assign]
    hb._worker(
        need_register=False,
        need_pool=False,
        metrics={},
        job_bound=False,
    )
    assert hb._pool_registered is False


def test_regression_claim_work_binds_job_then_ack_unbinds():
    """BL-006c/011: claim → bind model_id; ack → unbind (agent HB only while leased)."""
    hb = ManagerHeartbeat(
        ManagerHeartbeatConfig(
            enabled=True,
            uri="http://tm.test",
            instance_id="55443322",
        )
    )
    calls: list[tuple[str, dict]] = []

    def fake_post(url, body, *, expect_commands):
        calls.append((url, dict(body)))
        if url.endswith("/api/work/claim"):
            return {
                "job": {
                    "job_id": "job-42",
                    "model_id": "agent-x",
                    "kind": "train",
                    "claim_token": "tok-1",
                    "config": {},
                }
            }
        if "/ack" in url:
            return {"job_id": "job-42", "state": "done"}
        if expect_commands:
            return {"desired_state": "running", "commands": []}
        return True

    hb._post_json = fake_post  # type: ignore[method-assign]
    job = hb.try_claim_work()
    assert job is not None
    assert job["job_id"] == "job-42"
    assert hb.job_bound is True
    assert hb.job_id == "job-42"
    assert hb.bound_model_id == "agent-x"
    claim_call = next(b for u, b in calls if u.endswith("/api/work/claim"))
    assert claim_call["session_id"] == "55443322"
    assert "model_id" not in claim_call

    calls.clear()
    hb._worker(
        need_register=True,
        need_pool=False,
        metrics={"run_state": "training"},
        job_bound=True,
    )
    assert any("/api/instances/agent-x/heartbeat" in u for u, _ in calls)

    out = hb.ack_work(result={"ok": True})
    assert out is not None
    assert hb.job_bound is False
    assert hb.job_id is None
    assert hb.bound_model_id is None


def test_regression_append_ledger_uses_bound_model_id():
    """HTTP ledger POST must stamp job.model_id, not pool worker id."""
    from src.manager_heartbeat import ManagerHeartbeat, ManagerHeartbeatConfig

    hb = ManagerHeartbeat(
        ManagerHeartbeatConfig(
            enabled=True,
            uri="http://tm.test",
            instance_id="55443322",
            kind="engine",
        )
    )
    posts: list[dict] = []

    def fake_urlopen(req, timeout=0):
        import json

        body = json.loads(req.data.decode("utf-8"))
        posts.append(body)

        class _Resp:
            status = 200

            def read(self):
                return b"{}"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Resp()

    import urllib.request

    hb.bind_job("job-1", model_id="tm-brain")
    orig = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen  # type: ignore[assignment]
    try:
        ok = hb.append_ledger_doc(doc_type="step_complete", body={"version": 1})
    finally:
        urllib.request.urlopen = orig  # type: ignore[assignment]
    assert ok is True
    assert posts[0]["instance_id"] == "tm-brain"


def test_regression_claim_without_model_id_refuses_bind():
    hb = ManagerHeartbeat(
        ManagerHeartbeatConfig(
            enabled=True,
            uri="http://tm.test",
            instance_id="55443322",
        )
    )

    def fake_post(url, body, *, expect_commands):
        del body, expect_commands
        if url.endswith("/api/work/claim"):
            return {"job": {"job_id": "j1", "claim_token": "t", "kind": "train"}}
        return True

    hb._post_json = fake_post  # type: ignore[method-assign]
    assert hb.try_claim_work() is None
    assert hb.job_bound is False


def test_regression_pool_hb_release_job_unbinds():
    """BL-006d: pool HB release_job drops agent bind after TM pause/cancel."""
    hb = ManagerHeartbeat(
        ManagerHeartbeatConfig(
            enabled=True,
            uri="http://tm.test",
            instance_id="55443322",
        )
    )
    hb.bind_job("job-99", model_id="agent-y")
    hb._pool_registered = True

    def fake_post(url, body, *, expect_commands):
        if "/api/workers/" in url and url.endswith("/heartbeat"):
            assert body.get("meta", {}).get("job_id") == "job-99"
            return {"session_id": "55443322", "release_job": True}
        return True

    hb._post_json = fake_post  # type: ignore[method-assign]
    hb._sync_pool_session("http://tm.test", need_pool=False, metrics={"run_state": "training"})
    assert hb.job_bound is False
    assert hb.bound_model_id is None
