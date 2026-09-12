# Unit + light integration: drawing_expert gym domain adapter (Slice D).
# Run: PYTHONPATH=. .venv/bin/python examples/closed_loop_draw/test_gym_domain.py
from __future__ import annotations

from examples.closed_loop_draw.commands import STOCK_COMMAND_IDS
from examples.closed_loop_draw.gym_domain import (
    DrawingExpertDomain,
    get_gym_domain,
    list_gym_domains,
)
from examples.closed_loop_draw.expert_gym_worker import (
    read_gym_from_metrics,
    run_worker_round,
)


def test_registry_has_drawing_expert():
    assert "drawing_expert" in list_gym_domains()
    d = get_gym_domain("drawing_expert")
    assert d.name == "drawing_expert"
    assert isinstance(d, DrawingExpertDomain)


def test_sample_returns_stock_task():
    d = DrawingExpertDomain()
    task = d.sample(seed=7, complexity="easy")
    assert task.domain == "drawing_expert"
    assert task.complexity == "easy"
    assert task.command_id in STOCK_COMMAND_IDS
    assert task.seed == 7


def test_score_is_ink_improvement():
    d = DrawingExpertDomain()
    assert d.score({"ink_pre": 0.5, "ink_post": 0.3}) == 0.2
    assert d.score({"ink_pre": 0.2, "ink_post": 0.4}) == -0.2


def test_unknown_domain_raises():
    try:
        get_gym_domain("nope")
        assert False, "expected KeyError"
    except KeyError as exc:
        assert "nope" in str(exc)


def test_integration_worker_round_uses_domain_episode_fn():
    """One batch via injectible episode_fn (no native draw)."""
    calls: list[int] = []

    def ep(**kwargs):
        calls.append(int(kwargs["seed"]))
        return {
            "domain": "drawing_expert",
            "outcome": 0.01,
            "complexity": kwargs.get("complexity") or "easy",
            "command": "stub",
            "ink_pre": 0.4,
            "ink_post": 0.39,
        }

    gym = read_gym_from_metrics(
        {
            "tm_gym_armed": True,
            "tm_gym_batch": 3,
            "tm_gym_pre_traj": 1,
            "tm_gym_post_traj": 1,
            "tm_gym_domain": "drawing_expert",
            "state": "training",
        }
    )
    rows = run_worker_round(
        cfg={},
        tm_uri="http://test",
        gym=gym,
        seed0=100,
        episode_fn=ep,
    )
    assert len(rows) == 3
    assert calls == [100, 101, 102]
    assert all(r["domain"] == "drawing_expert" for r in rows)


if __name__ == "__main__":
    test_registry_has_drawing_expert()
    test_sample_returns_stock_task()
    test_score_is_ink_improvement()
    test_unknown_domain_raises()
    test_integration_worker_round_uses_domain_episode_fn()
    print("gym_domain ok")
