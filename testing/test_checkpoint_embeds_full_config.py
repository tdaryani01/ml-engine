# BL-008a: checkpoint ledger body embeds full live run config.
from __future__ import annotations

from examples.closed_loop_draw.draw_checkpoint import knobs_from_cfg, public_run_config


def test_regression_checkpoint_embeds_full_config():
    cfg = {
        "closed_loop": {"sigma": 0.07, "train_patience": 40, "max_steps": 10},
        "optimization": {"learning_rate": 0.001, "seed": 0},
        "mhsa": {"d_model": 64},
        "training_manager": {"uri": "http://secret", "enabled": True},
        "source": "should-strip",
    }
    pub = public_run_config(cfg, lr=0.002)
    assert "training_manager" not in pub
    assert "source" not in pub
    assert pub["closed_loop"]["sigma"] == 0.07
    assert pub["optimization"]["learning_rate"] == 0.002
    assert pub["mhsa"]["d_model"] == 64

    # Ledger body shape (agent _maybe_checkpoint).
    version = 25
    body = {
        "version": version,
        "knobs": knobs_from_cfg(cfg, lr=0.002),
        "config": public_run_config(cfg, lr=0.002),
        "config_snapshot": True,
    }
    assert body["version"] == version
    assert body["config"]["closed_loop"]["train_patience"] == 40
    assert body["knobs"]["learning_rate"] == 0.002
