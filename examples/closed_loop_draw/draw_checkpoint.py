# Draw student checkpoint capture / restore (weights + training config).
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from examples.closed_loop_draw.assemble import DrawApp
from src.manager_heartbeat import decode_checkpoint_blob, encode_checkpoint_blob


def snapshot_trainable(app: DrawApp) -> dict[str, Any]:
    """Deep-copy trainable tensors + Adam optimizer moments (m, v, t)."""
    mhsa = app.mhsa
    cnn = app.cnn
    snap: dict[str, Any] = {
        "mhsa_w": [np.array(w, copy=True) for w in mhsa.weights],
        "mhsa_b": [np.array(b, copy=True) for b in mhsa.biases],
        "ln1_g": [np.array(g, copy=True) for g in mhsa.ln1_gamma],
        "ln1_b": [np.array(b, copy=True) for b in mhsa.ln1_beta],
        "ln2_g": [np.array(g, copy=True) for g in mhsa.ln2_gamma],
        "ln2_b": [np.array(b, copy=True) for b in mhsa.ln2_beta],
        "cnn_w": [np.array(w, copy=True) for w in cnn.weights],
        "cnn_b": [np.array(b, copy=True) for b in cnn.biases],
        "adapter_W": np.array(app.adapter.W, copy=True),
        "adapter_b": np.array(app.adapter.b, copy=True),
        "act_W": np.array(app.action_embed.W, copy=True),
        "act_b": np.array(app.action_embed.b, copy=True),
        "cond": np.array(app.conditioning.embeddings, copy=True),
    }
    if mhsa.pos_embed is not None:
        snap["pos"] = np.array(mhsa.pos_embed, copy=True)
    if mhsa.W_in is not None:
        snap["W_in"] = np.array(mhsa.W_in, copy=True)
        snap["b_in"] = np.array(mhsa.b_in, copy=True)

    # --- Adam optimizer moments for physical continuity ---
    mhsa.ensure_adam_moments()
    opt = mhsa.optimizer
    snap["opt_ms_w"] = [np.array(m, copy=True) for m in opt.ms_w]
    snap["opt_vs_w"] = [np.array(v, copy=True) for v in opt.vs_w]
    snap["opt_ms_b"] = [np.array(m, copy=True) for m in opt.ms_b]
    snap["opt_vs_b"] = [np.array(v, copy=True) for v in opt.vs_b]
    if getattr(opt, "ms_g", None) is not None:
        snap["opt_ms_g"] = [np.array(m, copy=True) for m in opt.ms_g]
    if getattr(opt, "vs_g", None) is not None:
        snap["opt_vs_g"] = [np.array(v, copy=True) for v in opt.vs_g]
    if getattr(opt, "ms_beta", None) is not None:
        snap["opt_ms_beta"] = [np.array(m, copy=True) for m in opt.ms_beta]
    if getattr(opt, "vs_beta", None) is not None:
        snap["opt_vs_beta"] = [np.array(v, copy=True) for v in opt.vs_beta]
    snap["opt_t"] = int(getattr(opt, "t", 0))

    # Extra MHSA-side moments (pos_embed, input_proj)
    if mhsa.pos_embed is not None and mhsa._ms_pos is not None:
        snap["opt_ms_pos"] = np.array(mhsa._ms_pos, copy=True)
        snap["opt_vs_pos"] = np.array(mhsa._vs_pos, copy=True)
    if mhsa.W_in is not None and mhsa._ms_W_in is not None:
        snap["opt_ms_W_in"] = np.array(mhsa._ms_W_in, copy=True)
        snap["opt_vs_W_in"] = np.array(mhsa._vs_W_in, copy=True)
    if mhsa.b_in is not None and mhsa._ms_b_in is not None:
        snap["opt_ms_b_in"] = np.array(mhsa._ms_b_in, copy=True)
        snap["opt_vs_b_in"] = np.array(mhsa._vs_b_in, copy=True)

    return snap


def _zero_like_list(bufs: Any) -> None:
    if not bufs:
        return
    for x in bufs:
        if x is not None:
            np.asarray(x).fill(0.0)


def restore_trainable(app: DrawApp, snap: dict[str, Any]) -> None:
    mhsa = app.mhsa
    cnn = app.cnn
    for dst, src in zip(mhsa.weights, snap["mhsa_w"]):
        dst[...] = src
    for dst, src in zip(mhsa.biases, snap["mhsa_b"]):
        dst[...] = src
    for dst, src in zip(mhsa.ln1_gamma, snap["ln1_g"]):
        dst[...] = src
    for dst, src in zip(mhsa.ln1_beta, snap["ln1_b"]):
        dst[...] = src
    for dst, src in zip(mhsa.ln2_gamma, snap["ln2_g"]):
        dst[...] = src
    for dst, src in zip(mhsa.ln2_beta, snap["ln2_b"]):
        dst[...] = src
    for dst, src in zip(cnn.weights, snap["cnn_w"]):
        dst[...] = src
    for dst, src in zip(cnn.biases, snap["cnn_b"]):
        dst[...] = src
    app.adapter.W[...] = snap["adapter_W"]
    app.adapter.b[...] = snap["adapter_b"]
    app.action_embed.W[...] = snap["act_W"]
    app.action_embed.b[...] = snap["act_b"]
    app.conditioning.embeddings[...] = snap["cond"]
    if "pos" in snap and mhsa.pos_embed is not None:
        mhsa.pos_embed[...] = snap["pos"]
    if "W_in" in snap and mhsa.W_in is not None:
        mhsa.W_in[...] = snap["W_in"]
        mhsa.b_in[...] = snap["b_in"]

    mhsa.ensure_adam_moments()
    opt = mhsa.optimizer

    def _load_moments(key: str, dst_list: list | None) -> None:
        """Load moments from snap if present; otherwise zero (legacy compat)."""
        if dst_list is None:
            return
        src = snap.get(key)
        if src is not None and len(src) == len(dst_list):
            for d, s in zip(dst_list, src):
                d[...] = s
        else:
            _zero_like_list(dst_list)

    _load_moments("opt_ms_w", getattr(opt, "ms_w", None))
    _load_moments("opt_vs_w", getattr(opt, "vs_w", None))
    _load_moments("opt_ms_b", getattr(opt, "ms_b", None))
    _load_moments("opt_vs_b", getattr(opt, "vs_b", None))
    _load_moments("opt_ms_g", getattr(opt, "ms_g", None))
    _load_moments("opt_vs_g", getattr(opt, "vs_g", None))
    _load_moments("opt_ms_beta", getattr(opt, "ms_beta", None))
    _load_moments("opt_vs_beta", getattr(opt, "vs_beta", None))

    opt.t = int(snap.get("opt_t", 0))

    # Extra MHSA-side moments (pos_embed, input_proj)
    if mhsa.pos_embed is not None and mhsa._ms_pos is not None:
        src = snap.get("opt_ms_pos")
        if src is not None:
            mhsa._ms_pos[...] = src
            mhsa._vs_pos[...] = snap.get("opt_vs_pos")
        else:
            mhsa._ms_pos.fill(0.0)
            mhsa._vs_pos.fill(0.0)
    if mhsa.W_in is not None and mhsa._ms_W_in is not None:
        src = snap.get("opt_ms_W_in")
        if src is not None:
            mhsa._ms_W_in[...] = src
            mhsa._vs_W_in[...] = snap.get("opt_vs_W_in")
        else:
            mhsa._ms_W_in.fill(0.0)
            mhsa._vs_W_in.fill(0.0)
    if mhsa.b_in is not None and mhsa._ms_b_in is not None:
        src = snap.get("opt_ms_b_in")
        if src is not None:
            mhsa._ms_b_in[...] = src
            mhsa._vs_b_in[...] = snap.get("opt_vs_b_in")
        else:
            mhsa._ms_b_in.fill(0.0)
            mhsa._vs_b_in.fill(0.0)


def knobs_from_cfg(cfg: dict[str, Any], *, lr: float) -> dict[str, Any]:
    """Compact training knobs to persist beside weights."""
    cl = dict(cfg.get("closed_loop") or {})
    opt = dict(cfg.get("optimization") or {})
    return {
        "learning_rate": float(lr),
        "seed": opt.get("seed"),
        "max_steps": cl.get("max_steps"),
        "batch_size": cl.get("batch_size"),
        "sigma": cl.get("sigma"),
        "command_id": cl.get("command_id"),
        "continuity_weight": cl.get("continuity_weight", 0.0),
        "loss_kind": cl.get("loss_kind"),
        "fg_alpha": cl.get("fg_alpha"),
        "loss_blur_sigmas": cl.get("loss_blur_sigmas"),
        "loss_blur_weights": cl.get("loss_blur_weights"),
        "loss_edt_weight": cl.get("loss_edt_weight"),
        "train_patience": cl.get("train_patience"),
    }


def public_run_config(cfg: dict[str, Any], *, lr: float | None = None) -> dict[str, Any]:
    """Full YAML-shaped run config for ledger (strip worker-local TM connect)."""
    out = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
    out.pop("training_manager", None)
    # Drop Start/job overlays if present on cfg.
    for k in (
        "source",
        "autopilot",
        "feed",
        "gym",
        "config_version",
        "config_source",
        "config_preset_id",
        "preset_id",
        "resume_checkpoint",
    ):
        out.pop(k, None)
    if lr is not None:
        opt = dict(out.get("optimization") or {})
        opt["learning_rate"] = float(lr)
        out["optimization"] = opt
    return out


def build_checkpoint_blob(
    app: DrawApp,
    *,
    version: int,
    cfg: dict[str, Any],
    lr: float,
    val_loss: float | None,
) -> bytes:
    body = {
        "kind": "draw_student_v1",
        "version": int(version),
        "val_loss": val_loss,
        "weights": snapshot_trainable(app),
        "config": public_run_config(cfg, lr=lr),
        "knobs": knobs_from_cfg(cfg, lr=lr),
    }
    return encode_checkpoint_blob(body)


def load_checkpoint_blob(data: bytes) -> dict[str, Any]:
    body = decode_checkpoint_blob(data)
    if not isinstance(body, dict):
        raise TypeError("draw checkpoint blob must be a dict")
    return body


def apply_checkpoint_blob(app: DrawApp, body: dict[str, Any]) -> dict[str, Any]:
    """Restore weights; return ``{config, knobs}`` for the agent to re-apply."""
    weights = body.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("checkpoint missing weights")
    restore_trainable(app, weights)
    knobs = body.get("knobs") if isinstance(body.get("knobs"), dict) else {}
    cfg = body.get("config") if isinstance(body.get("config"), dict) else {}
    return {"knobs": dict(knobs), "config": dict(cfg)}


def job_overlay_only(job_config: dict[str, Any] | None) -> dict[str, Any]:
    """Start/job overlay keys only — never replace checkpoint hot knobs."""
    if not isinstance(job_config, dict):
        return {}
    out: dict[str, Any] = {}
    for k in ("source", "autopilot", "feed", "gym"):
        if k in job_config:
            out[k] = copy.deepcopy(job_config[k])
    return out
