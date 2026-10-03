"""The small hyperparameters stored beside a checkpoint's model state and config."""

from __future__ import annotations

from typing import Any


def knobs_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    opt = dict(cfg.get("optimization") or {})
    ledger = dict(cfg.get("ledger") or {})
    return {
        "learning_rate": opt.get("learning_rate"),
        "seed": opt.get("seed"),
        "batch_size": opt.get("batch_size"),
        "epochs_full_dataset": opt.get("epochs_full_dataset"),
        "optimizer": opt.get("optimizer"),
        "early_stopping_enabled": opt.get("early_stopping_enabled"),
        "patience": opt.get("patience"),
        "min_delta": opt.get("min_delta"),
        "checkpoint_every": ledger.get("checkpoint_every_steps", ledger.get("checkpoint_every")),
    }
