"""This run's early-stop choice from Start, applied to a copy of a supervised config (the saved model config is not changed)."""

from __future__ import annotations

from typing import Any


class FitSpecError(ValueError):
    """The fit spec TM staged is not something an engine can run."""


def apply_early_stop(config: dict[str, Any], early_stop: Any) -> dict[str, Any]:
    """``early_stop`` is ``{"enabled": bool, "patience": int}``. Absent leaves the config alone; enabled without a patience keeps the
    model's own."""
    if early_stop is None:
        return config
    if not isinstance(early_stop, dict) or not isinstance(early_stop.get("enabled"), bool):
        raise FitSpecError("early_stop needs enabled true or false")
    out = dict(config)
    opt = dict(out.get("optimization") or {})
    opt["early_stopping_enabled"] = early_stop["enabled"]
    if early_stop["enabled"] and early_stop.get("patience") is not None:
        try:
            patience = int(early_stop.get("patience"))
        except (TypeError, ValueError):
            raise FitSpecError("early_stop patience must be a whole number of epochs") from None
        if not 1 <= patience <= 1000:
            raise FitSpecError("early_stop patience must be between 1 and 1000 epochs")
        opt["patience"] = patience
    out["optimization"] = opt
    return out
