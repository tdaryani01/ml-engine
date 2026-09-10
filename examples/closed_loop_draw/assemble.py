# examples/closed_loop_draw/assemble.py
"""Build closed-loop draw stack from a config dict (app layer)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from config.constants import EngineBackend
from examples.closed_loop_draw.cnn_encoder import CnnUpstreamEncoder
from examples.closed_loop_draw.env import CanvasReconstructionLoss, SoftCanvasEnv
from examples.closed_loop_draw.targets import target_circle
from src.closed_loop import (
    ClosedLoopTrainer,
    ConditioningBank,
    LinearAdapter,
    TokenInterleaver,
)
from src.model_factory import ModelFactory


@dataclass
class DrawApp:
    trainer: ClosedLoopTrainer
    cnn: Any
    mhsa: Any
    encoder: CnnUpstreamEncoder
    adapter: LinearAdapter
    action_embed: LinearAdapter
    conditioning: ConditioningBank
    env: SoftCanvasEnv
    loss_fn: CanvasReconstructionLoss
    cfg: dict[str, Any]

    @property
    def max_steps(self) -> int:
        return int(self.cfg["closed_loop"]["max_steps"])

    @property
    def batch_size(self) -> int:
        return int(self.cfg["closed_loop"]["batch_size"])

    @property
    def lr(self) -> float:
        return float(self.cfg["optimization"]["learning_rate"])

    def close(self) -> None:
        rt = getattr(self.mhsa, "_contract_runtime", None)
        if rt is not None:
            rt.close()
            self.mhsa._contract_runtime = None


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return cfg


def _default_spatial_pipeline(in_channels: int) -> list[dict[str, Any]]:
    return [
        {
            "type": "conv",
            "in_channels": in_channels,
            "out_channels": 4,
            "kernel_size": 5,
            "stride": 1,
            "pad": 2,
        },
        {"type": "relu"},
        {"type": "pool", "pool_size": 2, "stride": 2},
        {
            "type": "conv",
            "in_channels": 4,
            "out_channels": 8,
            "kernel_size": 5,
            "stride": 1,
            "pad": 2,
        },
        {"type": "relu"},
        {"type": "pool", "pool_size": 2, "stride": 2},
        {"type": "flatten"},
    ]


def make_cnn(cfg: dict[str, Any], *, seed: int = 0):
    ce = cfg.get("cnn_encoder", {})
    shape = list(ce.get("input_shape", cfg["closed_loop"]["canvas"]))
    feature_dim = int(ce.get("feature_dim", 16))
    pipeline = ce.get("spatial_pipeline") or _default_spatial_pipeline(int(shape[0]))
    dense_head = list(ce.get("dense_head", []))
    np.random.seed(seed)
    return ModelFactory.create_model(
        "cnn",
        layer_sizes=[feature_dim],
        backend=EngineBackend.NATIVE,
        optimizer="adam",
        task_type="regression",
        lam_l1=0.0,
        lam_l2=0.0,
        p_dropout=0.0,
        contract_list_enabled=False,
        cnn_config={
            "input_shape": shape,
            "spatial_pipeline": pipeline,
            "dense_head": dense_head,
        },
    )


def make_mhsa(cfg: dict[str, Any], *, seed: int = 1):
    cl = cfg["closed_loop"]
    mh = cfg.get("mhsa", {})
    max_steps = int(cl["max_steps"])
    np.random.seed(seed)
    return ModelFactory.create_model(
        "mhsa",
        layer_sizes=[int(mh.get("action_dim", 4))],
        backend=EngineBackend.NATIVE,
        optimizer="adam",
        mhsa_config={
            "d_model": int(mh.get("d_model", 32)),
            "num_heads": int(mh.get("num_heads", 4)),
            "max_seq_len": TokenInterleaver.seq_len(max_steps),
            "action_dim": int(mh.get("action_dim", 4)),
            "ffn_mult": int(mh.get("ffn_mult", 2)),
            "num_layers": int(mh.get("num_layers", 1)),
            "use_pos_encoding": bool(mh.get("use_pos_encoding", False)),
            "use_input_proj": bool(mh.get("use_input_proj", False)),
        },
        contract_list_enabled=True,
        lam_l2=0.0,
        lam_l1=0.0,
    )


def make_target(cfg: dict[str, Any], batch_size: int | None = None) -> np.ndarray:
    from examples.closed_loop_draw.commands import ALIASES, COMMANDS, load_command_target

    cl = cfg["closed_loop"]
    C, H, W = (int(x) for x in cl["canvas"])
    B = int(batch_size if batch_size is not None else cl["batch_size"])
    tgt = cl.get("target", {})
    kind = str(tgt.get("kind", "stock")).lower()

    if kind == "procedural_circle":
        return target_circle(
            batch_size=B,
            height=H,
            width=W,
            channels=C,
            radius=float(tgt.get("radius", 0.55)),
            edge_soft=float(tgt.get("edge_soft", tgt.get("sigma", 0.08))),
        )

    if kind in COMMANDS.values():
        cid = next(i for i, n in COMMANDS.items() if n == kind)
    elif kind in ALIASES:
        cid = ALIASES[kind]
    else:
        cid = int(cl.get("command_id", 0))
    return load_command_target(cid, batch_size=B, height=H, width=W, channels=C)



def assemble(cfg: dict[str, Any], *, seed: int = 0) -> DrawApp:
    """Wire CNN encoder + adapters + MHSA + canvas env into a ClosedLoopTrainer."""
    cl = cfg["closed_loop"]
    C, H, W = (int(x) for x in cl["canvas"])
    cnn = make_cnn(cfg, seed=seed)
    mhsa = make_mhsa(cfg, seed=seed + 1)
    # Keep action-head logits small so tanh starts near 0 (avoids ±1 saturation).
    mhsa.weights[-1] *= float(cfg.get("closed_loop", {}).get("action_head_scale", 0.1))
    encoder = CnnUpstreamEncoder(cnn)

    probe = np.zeros((1, C, H, W), dtype=np.float32)
    V0 = encoder.encode(probe)
    d_in = int(V0.shape[1])
    encoder.zero_grad()

    adapter = LinearAdapter(d_in, mhsa.d_model, seed=seed + 2)
    action_embed = LinearAdapter(mhsa.action_dim, mhsa.d_model, seed=seed + 3)
    conditioning = ConditioningBank(
        num_commands=int(cl.get("num_commands", 2)),
        d_model=mhsa.d_model,
        seed=seed + 4,
    )
    env = SoftCanvasEnv(
        height=H,
        width=W,
        channels=C,
        sigma=float(cl.get("sigma", 0.1)),
        action_scale=float(cl.get("action_scale", 0.85)),
    )
    loss_fn = CanvasReconstructionLoss(
        terminal_only=bool(cl.get("terminal_loss", True)),
        max_steps=int(cl["max_steps"]),
        kind=str(cl.get("loss_kind", "balanced")),
        fg_alpha=float(cl.get("fg_alpha", 15.0)),
        fg_thresh=float(cl.get("fg_thresh", 0.05)),
        blur_sigmas=cl.get("loss_blur_sigmas", None),
        blur_weights=cl.get("loss_blur_weights", None),
        blur_weights_end=cl.get("loss_blur_weights_end", None),
        edt_weight=float(cl.get("loss_edt_weight", 0.0)),
        edt_sym_weight=float(cl.get("loss_edt_sym_weight", 0.0)),
        edt_soft_tau=float(cl.get("loss_edt_soft_tau", 2.0)),
    )
    trainer = ClosedLoopTrainer(
        mhsa=mhsa,
        encoder=encoder,
        adapter=adapter,
        action_embed=action_embed,
        conditioning=conditioning,
        env=env,
        loss_fn=loss_fn,
    )
    return DrawApp(
        trainer=trainer,
        cnn=cnn,
        mhsa=mhsa,
        encoder=encoder,
        adapter=adapter,
        action_embed=action_embed,
        conditioning=conditioning,
        env=env,
        loss_fn=loss_fn,
        cfg=cfg,
    )


def rollout_forward(app: DrawApp, *, command_ids: np.ndarray) -> np.ndarray:
    """
    Inference-only trajectory; returns final canvas (NCHW).

    Does not update weights. Uses the same interleave layout as training.
    """
    frames = list(rollout_forward_frames(app, command_ids=command_ids))
    return frames[-1] if frames else app.env.reset(int(np.asarray(command_ids).reshape(-1).shape[0]))


def rollout_forward_frames(app: DrawApp, *, command_ids: np.ndarray):
    """
    Yield canvas after each stroke (inference only).

    First yield is the blank canvas after reset; then one frame per stroke.
    """
    B = int(np.asarray(command_ids).reshape(-1).shape[0])
    goal = app.conditioning.embed(command_ids)
    obs = app.env.reset(B)
    yield np.array(obs, copy=True)
    states_S: list[np.ndarray] = []
    action_embs: list[np.ndarray] = []
    for t in range(1, app.max_steps + 1):
        V = app.encoder.encode(obs)
        S = app.adapter.forward(V)
        states_S.append(np.array(S, copy=True))
        X = app.trainer.interleaver.build(goal, states_S, action_embs)
        A = np.ascontiguousarray(app.mhsa.predict(X), dtype=np.float32)
        obs = app.env.step(A)
        yield np.array(obs, copy=True)
        if t < app.max_steps:
            action_embs.append(app.action_embed.forward(A))
    app.encoder.zero_grad()
