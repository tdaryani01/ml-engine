# Smoke + unit: gym worker policy and stub loop (no heavy draw).
# Run: PYTHONPATH=. .venv/bin/python examples/closed_loop_draw/test_expert_gym_worker.py
from __future__ import annotations

from examples.closed_loop_draw.expert_gym_worker import (
    decide_gym_action,
    read_gym_from_metrics,
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


if __name__ == "__main__":
    test_decide_wait_when_disarmed()
    test_decide_wait_when_paused()
    test_decide_run_when_armed_training()
    test_decide_stop_done_on_val_target()
    test_worker_loop_respects_pause_and_stop()
    print("expert_gym_worker ok")
