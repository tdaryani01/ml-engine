"""The built-in closed-loop options (the names TM's catalog uses). Each factory builds one seat from the run config.

Seat signatures:
  encoder(cfg, seed)                -> UpstreamEncoder
  policy(cfg, encoder, seed)        -> Actor   (owns goal conditioning and the token grammar)
  env(cfg)                          -> Environment
  loss(cfg)                         -> LossEvaluator
The concrete classes still live in ``examples.closed_loop_draw``; they are imported lazily so ``src`` stays importable
without them.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.closed_loop.registry import register


@register("encoder", "cnn_upstream")
def cnn_upstream(cfg: dict[str, Any], seed: int):
    from examples.closed_loop_draw.assemble import make_cnn
    from examples.closed_loop_draw.cnn_encoder import CnnUpstreamEncoder

    return CnnUpstreamEncoder(make_cnn(cfg, seed=seed))


@register("policy", "mhsa")
def mhsa_policy(cfg: dict[str, Any], encoder: Any, seed: int):
    from examples.closed_loop_draw.actor import DrawActor
    from examples.closed_loop_draw.assemble import make_mhsa
    from src.closed_loop import ConditioningBank, LinearAdapter

    cl = cfg["closed_loop"]
    C, H, W = (int(x) for x in cl["canvas"])
    mhsa = make_mhsa(cfg, seed=seed + 1)
    # Keep action-head logits small so tanh starts near 0 (avoids +-1 saturation).
    mhsa.weights[-1] *= float(cl.get("action_head_scale", 0.1))
    d_in = int(encoder.encode(np.zeros((1, C, H, W), dtype=np.float32)).shape[1])
    encoder.zero_grad()
    return DrawActor(
        mhsa=mhsa,
        encoder=encoder,
        adapter=LinearAdapter(d_in, mhsa.d_model, seed=seed + 2),
        action_embed=LinearAdapter(mhsa.action_dim, mhsa.d_model, seed=seed + 3),
        conditioning=ConditioningBank(num_commands=int(cl.get("num_commands", 2)), d_model=mhsa.d_model, seed=seed + 4),
    )


@register("env", "soft_canvas")
def soft_canvas(cfg: dict[str, Any]):
    from examples.closed_loop_draw.env import SoftCanvasEnv

    cl = cfg["closed_loop"]
    C, H, W = (int(x) for x in cl["canvas"])
    return SoftCanvasEnv(
        height=H, width=W, channels=C, sigma=float(cl.get("sigma", 0.1)),
        action_scale=float(cl.get("action_scale", 0.85)), continuity_weight=float(cl.get("continuity_weight", 0.0)),
    )


@register("loss", "canvas_reconstruction")
def canvas_reconstruction(cfg: dict[str, Any]):
    from examples.closed_loop_draw.env import CanvasReconstructionLoss

    cl = cfg["closed_loop"]
    return CanvasReconstructionLoss(
        terminal_only=bool(cl.get("terminal_loss", True)), max_steps=int(cl["max_steps"]), kind=str(cl.get("loss_kind", "balanced")),
        fg_alpha=float(cl.get("fg_alpha", 15.0)), fg_thresh=float(cl.get("fg_thresh", 0.05)),
        blur_sigmas=cl.get("loss_blur_sigmas", None), blur_weights=cl.get("loss_blur_weights", None),
        blur_weights_end=cl.get("loss_blur_weights_end", None), edt_weight=float(cl.get("loss_edt_weight", 0.0)),
        edt_sym_weight=float(cl.get("loss_edt_sym_weight", 0.0)), edt_soft_tau=float(cl.get("loss_edt_soft_tau", 2.0)),
    )
