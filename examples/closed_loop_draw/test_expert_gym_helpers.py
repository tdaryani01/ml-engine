# Smoke: curriculum gym helpers (complexity pools + config jitter).
# Run: PYTHONPATH=. .venv/bin/python examples/closed_loop_draw/test_expert_gym_helpers.py
from __future__ import annotations

import numpy as np

from examples.closed_loop_draw.assemble import load_config
from examples.closed_loop_draw.expert_gym import (
    COMPLEXITY_POOLS,
    _jitter_config,
    _pick_complexity,
)
from examples.closed_loop_draw.commands import STOCK_COMMAND_IDS


def test_complexity_pools_are_stock():
    for name, pool in COMPLEXITY_POOLS.items():
        assert name in ("easy", "medium", "hard")
        assert len(pool) >= 1
        for cid in pool:
            assert cid in STOCK_COMMAND_IDS, (name, cid)


def test_pick_and_jitter_stable_bounds():
    cfg = load_config("examples/closed_loop_draw/config_draw_interactive.yaml")
    rng = np.random.default_rng(0)
    assert _pick_complexity(rng, "easy") == "easy"
    assert _pick_complexity(rng, None) in COMPLEXITY_POOLS
    for tier in COMPLEXITY_POOLS:
        j = _jitter_config(cfg, rng, complexity=tier)
        cl = j["closed_loop"]
        assert 6 <= int(cl["max_steps"]) <= 20
        assert float(cl["sigma"]) > 0
        assert float(cl["continuity_weight"]) >= 0
        assert float(j["optimization"]["learning_rate"]) > 0


if __name__ == "__main__":
    test_complexity_pools_are_stock()
    test_pick_and_jitter_stable_bounds()
    print("expert_gym helpers ok")
