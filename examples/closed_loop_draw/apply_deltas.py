# Apply DrawingExpert knob deltas onto a live DrawApp.
from __future__ import annotations

from typing import Any

from examples.closed_loop_draw.assemble import DrawApp


def apply_drawing_deltas(
    app: DrawApp,
    deltas: list[dict[str, Any]],
    *,
    train_lr: float | None = None,
) -> tuple[list[str], float | None]:
    """Apply catalog keys. Returns (applied_keys, maybe_new_train_lr)."""
    applied: list[str] = []
    lr_out = train_lr
    cl = app.cfg.setdefault("closed_loop", {})
    opt = app.cfg.setdefault("optimization", {})

    for d in deltas:
        if not isinstance(d, dict):
            continue
        key = str(d.get("key") or "")
        val = d.get("value")
        if not key:
            continue
        try:
            if key == "closed_loop.sigma":
                app.env.sigma = float(val)
                cl["sigma"] = float(val)
                applied.append(key)
            elif key == "closed_loop.continuity_weight":
                app.env.continuity_weight = float(val)
                cl["continuity_weight"] = float(val)
                applied.append(key)
            elif key == "closed_loop.max_steps":
                steps = int(val)
                cl["max_steps"] = steps
                app.loss_fn.max_steps = steps
                applied.append(key)
            elif key == "closed_loop.loss_blur_sigmas":
                sigmas = tuple(float(x) for x in list(val))
                app.loss_fn.blur_sigmas = sigmas
                # Keep weights aligned if lengths match start weights.
                if len(app.loss_fn.blur_weights_start) != len(sigmas):
                    app.loss_fn.blur_weights_start = tuple(1.0 for _ in sigmas)
                    app.loss_fn.blur_weights_end = app.loss_fn.blur_weights_start
                app.loss_fn._apply_anneal_weights(app.loss_fn.anneal_progress)
                cl["loss_blur_sigmas"] = list(sigmas)
                applied.append(key)
            elif key == "closed_loop.loss_blur_weights":
                weights = tuple(float(x) for x in list(val))
                app.loss_fn.blur_weights_start = weights
                if len(app.loss_fn.blur_weights_end) != len(weights):
                    app.loss_fn.blur_weights_end = weights
                app.loss_fn._apply_anneal_weights(app.loss_fn.anneal_progress)
                cl["loss_blur_weights"] = list(weights)
                applied.append(key)
            elif key == "closed_loop.loss_edt_weight":
                app.loss_fn.edt_weight = float(val)
                cl["loss_edt_weight"] = float(val)
                applied.append(key)
            elif key == "closed_loop.loss_edt_sym_weight":
                app.loss_fn.edt_sym_weight = float(val)
                cl["loss_edt_sym_weight"] = float(val)
                applied.append(key)
            elif key == "closed_loop.loss_edt_soft_tau":
                app.loss_fn.edt_soft_tau = float(val)
                cl["loss_edt_soft_tau"] = float(val)
                applied.append(key)
            elif key in (
                "optimization.learning_rate",
                "interactive.train_lr",
            ):
                lr_out = float(val)
                opt["learning_rate"] = float(val)
                applied.append(key)
        except (TypeError, ValueError):
            continue
    return applied, lr_out
