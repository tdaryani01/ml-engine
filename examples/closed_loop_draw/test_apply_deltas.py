# Unit test apply_drawing_deltas with a lightweight fake DrawApp.
from __future__ import annotations

from types import SimpleNamespace

from examples.closed_loop_draw.apply_deltas import apply_drawing_deltas


def test_apply_drawing_deltas_sigma_and_lr():
    loss_fn = SimpleNamespace(
        max_steps=10,
        blur_sigmas=(3.0, 1.0, 0.0),
        blur_weights_start=(1.0, 1.0, 1.0),
        blur_weights_end=(1.0, 1.0, 1.0),
        anneal_progress=0.0,
        edt_weight=0.5,
        edt_sym_weight=0.5,
        edt_soft_tau=2.0,
        _apply_anneal_weights=lambda self_or_u, u=None: None,
    )

    def _apply(u):
        loss_fn.anneal_progress = float(u)

    loss_fn._apply_anneal_weights = _apply

    app = SimpleNamespace(
        env=SimpleNamespace(sigma=0.06, continuity_weight=0.01),
        loss_fn=loss_fn,
        cfg={"closed_loop": {"max_steps": 10}, "optimization": {"learning_rate": 5e-4}},
        max_steps=10,
    )
    applied, lr = apply_drawing_deltas(
        app,  # type: ignore[arg-type]
        [
            {"key": "closed_loop.sigma", "value": 0.08},
            {"key": "closed_loop.continuity_weight", "value": 0.0},
            {"key": "optimization.learning_rate", "value": 0.0003},
            {"key": "closed_loop.loss_edt_weight", "value": 1.0},
            {"key": "closed_loop.max_steps", "value": 12},
        ],
        train_lr=0.0005,
    )
    assert "closed_loop.sigma" in applied
    assert abs(app.env.sigma - 0.08) < 1e-12
    assert abs(app.env.continuity_weight - 0.0) < 1e-12
    assert lr is not None and abs(lr - 0.0003) < 1e-12
    assert abs(app.loss_fn.edt_weight - 1.0) < 1e-12
    assert app.loss_fn.max_steps == 12
    assert app.cfg["closed_loop"]["max_steps"] == 12


if __name__ == "__main__":
    test_apply_drawing_deltas_sigma_and_lr()
    print("apply_deltas ok")
