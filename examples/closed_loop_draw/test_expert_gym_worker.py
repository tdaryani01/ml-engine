# Smoke + unit: gym worker policy, stub loop, auth, golden curation.
# Run: PYTHONPATH=. .venv/bin/python -m pytest examples/closed_loop_draw/test_expert_gym_worker.py
from __future__ import annotations

import pytest

from examples.closed_loop_draw.expert_gym_worker import (
    decide_gym_action,
    flag_episode_golden,
    is_golden_run,
    read_gym_from_metrics,
    run_worker_round,
    worker_loop,
)


def test_decide_wait_when_disarmed():
    assert decide_gym_action({"state": "training", "tm_gym_armed": False}) == "wait"


def test_decide_wait_when_paused():
    assert (
        decide_gym_action(
            {"state": "paused", "tm_gym_armed": True, "tm_user_paused": True}
        )
        == "wait"
    )


def test_decide_run_when_armed_even_if_idle():
    """suggest/outcome used to pulse idle mid-round — armed must still run."""
    assert (
        decide_gym_action({"state": "idle", "tm_gym_armed": True, "tm_gym_batch": 2})
        == "run"
    )


def test_decide_run_when_armed_training():
    assert (
        decide_gym_action({"state": "training", "tm_gym_armed": True, "tm_gym_batch": 2})
        == "run"
    )


def test_decide_stop_done_on_val_target():
    assert (
        decide_gym_action(
            {
                "state": "training",
                "tm_gym_armed": True,
                "tm_gym_val_target": 0.7,
                "tm_gym_min_n_train": 10,
                "val_accuracy": 0.8,
                "n_train": 12,
            }
        )
        == "stop_done"
    )


def test_worker_loop_respects_pause_and_stop():
    """Stub episode/train — Pause between rounds; stop_done disarms."""
    calls = {"episodes": 0, "trains": 0, "disarm": 0, "ticks": 0}
    # fetch order: run → (post-episode still run for train) → pause waits → stop_done
    seq = [
        {
            "state": "training",
            "tm_gym_armed": True,
            "tm_gym_batch": 2,
            "tm_gym_pre_traj": 1,
            "tm_gym_post_traj": 1,
        },
        {
            "state": "training",
            "tm_gym_armed": True,
            "tm_gym_batch": 2,
            "tm_gym_pre_traj": 1,
            "tm_gym_post_traj": 1,
        },
        {
            "state": "paused",
            "tm_gym_armed": True,
            "tm_user_paused": True,
            "tm_gym_batch": 2,
        },
        {
            "state": "paused",
            "tm_gym_armed": True,
            "tm_user_paused": True,
            "tm_gym_batch": 2,
        },
        {
            "state": "training",
            "tm_gym_armed": True,
            "tm_gym_batch": 2,
            "tm_gym_val_target": 0.5,
            "tm_gym_min_n_train": 4,
            "val_accuracy": 0.9,
            "n_train": 8,
        },
    ]

    def fetch(_uri: str) -> dict:
        i = min(calls["ticks"], len(seq) - 1)
        calls["ticks"] += 1
        return dict(seq[i])

    def episode(**_kwargs):
        calls["episodes"] += 1
        return {"ok": True}

    def train(_uri: str) -> dict:
        calls["trains"] += 1
        return {
            "ok": True,
            "version": calls["trains"],
            "val_accuracy": 0.9,
            "n_train": 8,
        }

    def pulse(*_a, **_k):
        return None

    def disarm(_uri: str):
        calls["disarm"] += 1

    sleeps: list[float] = []

    out = worker_loop(
        tm_uri="http://test",
        cfg={},
        poll_s=0.0,
        seed0=0,
        max_rounds=10,
        episode_fn=episode,
        fetch_fn=fetch,
        train_fn=train,
        pulse_fn=pulse,
        disarm_fn=disarm,
        sleep_fn=lambda s: sleeps.append(s),
    )
    assert out["reason"] == "val_target"
    assert calls["episodes"] >= 2
    assert calls["trains"] >= 1
    assert calls["disarm"] >= 1
    assert read_gym_from_metrics(seq[0])["batch"] == 2


def test_is_golden_run_heuristic():
    assert is_golden_run({"outcome": 0.95}) is True
    assert is_golden_run({"outcome": 0.9}) is True
    assert is_golden_run({"outcome": 0.5}) is False
    assert is_golden_run({"outcome": None}) is False
    assert is_golden_run({}) is False
    assert is_golden_run({"outcome": "bad"}) is False
    # Exact zero post-run error counts even below the outcome bar.
    assert is_golden_run({"outcome": 0.1, "ink_post": 0.0}) is True
    # Explicit threshold override.
    assert is_golden_run({"outcome": 0.4}, min_outcome=0.3) is True
    assert is_golden_run({"outcome": 0.4}, min_outcome=0.5) is False


def test_run_worker_round_flags_only_perfect_episodes():
    flagged: list[tuple[str, str]] = []

    def episode(**kwargs):
        seed = int(kwargs["seed"])
        return {
            "episode_id": f"ep-{seed}",
            "outcome": 0.99 if seed % 2 == 0 else 0.1,
            "ink_post": 0.5,
        }

    def golden_fn(tm_uri, episode_id):
        flagged.append((tm_uri, episode_id))
        return True

    rows = run_worker_round(
        cfg={},
        tm_uri="http://test",
        gym={"batch": 4, "pre_traj": 1, "post_traj": 1, "complexity": "mixed"},
        seed0=0,
        episode_fn=episode,
        golden_fn=golden_fn,
    )
    assert len(rows) == 4
    assert flagged == [("http://test", "ep-0"), ("http://test", "ep-2")]


def test_run_worker_round_curation_failure_does_not_crash():
    """A failed curation PATCH must not crash the round or halt data gen."""

    def episode(**_kwargs):
        return {"episode_id": "ep-boom", "outcome": 1.0}

    def golden_fn(_tm_uri, _episode_id):
        raise RuntimeError("tm down")

    rows = run_worker_round(
        cfg={},
        tm_uri="http://test",
        gym={"batch": 2, "pre_traj": 1, "post_traj": 1, "complexity": "mixed"},
        seed0=0,
        episode_fn=episode,
        golden_fn=golden_fn,
    )
    assert len(rows) == 2


def test_flag_episode_golden_posts_expected_patch(monkeypatch):
    import examples.closed_loop_draw.expert_gym_worker as gw

    seen: dict = {}

    def ok(url, body, timeout_s=30.0):
        seen["url"] = url
        seen["body"] = body
        return {"status": "ok", "is_golden": True}

    monkeypatch.setattr(gw, "_patch_json", ok)
    assert flag_episode_golden("http://tm:8000/", "abc") is True
    assert seen["url"] == "http://tm:8000/api/tm-brain/episodes/abc/golden"
    assert seen["body"] == {"is_golden": True}


def test_flag_episode_golden_swallows_failure(monkeypatch):
    import examples.closed_loop_draw.expert_gym_worker as gw

    def boom(url, body, timeout_s=30.0):
        raise RuntimeError("404 not found")

    monkeypatch.setattr(gw, "_patch_json", boom)
    # Network/HTTP failure -> False, never raises.
    assert flag_episode_golden("http://test", "ep-1") is False
    # Blank id -> no request attempted, no raise.
    assert flag_episode_golden("http://test", "") is False


def test_configure_worker_auth_requires_key(monkeypatch):
    import examples.closed_loop_draw.expert_gym_worker as gw

    monkeypatch.delenv("TM_API_KEY", raising=False)
    monkeypatch.setattr(gw, "_api_key", None)
    with pytest.raises(gw.WorkerConfigError):
        gw.configure_worker_auth()
    # Whitespace-only is treated as missing, too.
    with pytest.raises(gw.WorkerConfigError):
        gw.configure_worker_auth("   ")


def test_configure_worker_auth_reads_env(monkeypatch):
    import examples.closed_loop_draw.expert_gym_worker as gw

    monkeypatch.setenv("TM_API_KEY", "sekret")
    monkeypatch.setattr(gw, "_api_key", None)
    assert gw.configure_worker_auth() == "sekret"
    assert gw._auth_headers() == {"Authorization": "Bearer sekret"}


def _capture_urlopen(monkeypatch, gw, payload: bytes = b'{"ok": true}'):
    captured: dict = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return payload

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(gw.urllib.request, "urlopen", fake_urlopen)
    return captured


def test_get_json_injects_bearer_header(monkeypatch):
    import examples.closed_loop_draw.expert_gym_worker as gw

    monkeypatch.setenv("TM_API_KEY", "tok-123")
    monkeypatch.setattr(gw, "_api_key", None)
    gw.configure_worker_auth()
    captured = _capture_urlopen(monkeypatch, gw)
    assert gw._get_json("http://x/api/instances") == {"ok": True}
    req = captured["req"]
    assert req.get_header("Authorization") == "Bearer tok-123"
    assert req.get_method() == "GET"


def test_patch_json_injects_bearer_header(monkeypatch):
    import examples.closed_loop_draw.expert_gym_worker as gw

    monkeypatch.setenv("TM_API_KEY", "tok-456")
    monkeypatch.setattr(gw, "_api_key", None)
    gw.configure_worker_auth()
    captured = _capture_urlopen(
        monkeypatch, gw, payload=b'{"status": "ok", "is_golden": true}'
    )
    out = gw._patch_json(
        "http://x/api/tm-brain/episodes/e1/golden", {"is_golden": True}
    )
    assert out["is_golden"] is True
    req = captured["req"]
    assert req.get_header("Authorization") == "Bearer tok-456"
    assert req.get_method() == "PATCH"


def test_main_fails_fast_without_api_key(monkeypatch):
    """Startup must abort with a config error, not 401 in the poll loop."""
    import examples.closed_loop_draw.expert_gym_worker as gw

    monkeypatch.delenv("TM_API_KEY", raising=False)
    monkeypatch.setattr(gw, "_api_key", None)
    with pytest.raises(gw.WorkerConfigError):
        gw.main(["--max-rounds", "0"])



if __name__ == "__main__":
    test_decide_wait_when_disarmed()
    test_decide_wait_when_paused()
    test_decide_run_when_armed_training()
    test_decide_stop_done_on_val_target()
    test_worker_loop_respects_pause_and_stop()
    test_is_golden_run_heuristic()
    test_run_worker_round_flags_only_perfect_episodes()
    test_run_worker_round_curation_failure_does_not_crash()
    print("expert_gym_worker ok")
