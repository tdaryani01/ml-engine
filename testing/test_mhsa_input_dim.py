# testing/test_mhsa_input_dim.py
"""Decoupled MHSA input width (D_in != d_model): exact gradient + Adam checks.

The projection X[rows, D_in] @ W_in[D_in, D] + b_in feeds the (already
Torch-verified) no-projection block. So for a projected model A, a twin model B
with the same block weights, *no* projection, fed X_emb = X @ W_in + b_in,
yields the exact upstream gradient delta = dL/dX_emb as its passthrough dX.
The new kernels must then satisfy, to f32 rounding:

    dW_in = X^T @ delta      db_in = sum_rows(delta)      dX = delta @ W_in^T

Covers D_in < D (the case where a D*D-sized update overran every W_in buffer)
and D_in > D (the case where rows >= D of W_in were never trained).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import EngineBackend
from src.model_factory import ModelFactory
from utils.conv_dispatch import bootstrap_im2col_gemm_runtime

D_MODEL = 8
HEADS = 2
T = 3
B = 2
ACTIONS = 3


def _make(input_dim: int | None, *, seed: int):
    bootstrap_im2col_gemm_runtime()
    np.random.seed(seed)
    return ModelFactory.create_model(
        "mhsa",
        layer_sizes=[ACTIONS],
        backend=EngineBackend.NATIVE,
        optimizer="adam",
        mhsa_config={
            "d_model": D_MODEL,
            "num_heads": HEADS,
            "max_seq_len": 4,
            "action_dim": ACTIONS,
            "ffn_mult": 2,
            "num_layers": 2,
            "use_pos_encoding": False,
            "use_input_proj": False,
            "input_dim": input_dim,
        },
        contract_list_enabled=True,
        lam_l2=0.0,
        lam_l1=0.0,
    )


def _close(model) -> None:
    rt = getattr(model, "_contract_runtime", None)
    if rt is not None:
        rt.close()
        model._contract_runtime = None


def _copy_block(src, dst) -> None:
    """Copy every non-projection parameter src -> dst (same D, H, L, A)."""
    for i, w in enumerate(src.weights):
        dst.weights[i][...] = w
    for i, b in enumerate(src.biases):
        dst.biases[i][...] = b
    for name in ("ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta"):
        for a, b in zip(getattr(src, name), getattr(dst, name)):
            b[...] = a
    if hasattr(dst, "_sync_restored_weights"):
        dst._sync_restored_weights()


def _grads(model, X: np.ndarray, y: np.ndarray):
    loss, _, _, _ = model.run_contract_train_step(X, y, lr=0.0, apply_adam=False)
    ws = model._contract_runtime._mhsa_ws
    dX = np.array(ws["dX"], dtype=np.float64).reshape(X.shape)
    dW_in = np.array(ws["dW_in"], dtype=np.float64)
    db_in = np.array(ws["db_in"], dtype=np.float64).reshape(-1)
    return float(loss), dX, dW_in, db_in


def _check_projection_grads(d_in: int, seed: int) -> None:
    proj = _make(d_in, seed=seed)
    twin = _make(None, seed=seed + 1000)
    try:
        assert proj.W_in is not None and proj.W_in.shape == (d_in, D_MODEL)
        assert twin.W_in is None
        _copy_block(proj, twin)

        rng = np.random.default_rng(seed)
        X = (rng.standard_normal((B, T, d_in)) * 0.5).astype(np.float32)
        y = (rng.standard_normal((B, ACTIONS)) * 0.5).astype(np.float32)

        W_in = proj.W_in.astype(np.float64)
        b_in = proj.b_in.astype(np.float64).reshape(-1)
        X2 = X.reshape(B * T, d_in).astype(np.float64)
        X_emb = (X2 @ W_in + b_in).astype(np.float32).reshape(B, T, D_MODEL)

        loss_p, dX_p, dW_in, db_in = _grads(proj, X, y)
        loss_t, delta, _, _ = _grads(twin, X_emb, y)
        delta2 = delta.reshape(B * T, D_MODEL)

        assert abs(loss_p - loss_t) <= 1e-5 * max(1.0, abs(loss_t)), (loss_p, loss_t)
        ref_dW = X2.T @ delta2
        ref_db = delta2.sum(axis=0)
        ref_dX = (delta2 @ W_in.T).reshape(B, T, d_in)
        for name, got, ref in (
            ("dW_in", dW_in, ref_dW),
            ("db_in", db_in, ref_db),
            ("dX", dX_p, ref_dX),
        ):
            assert got.shape == ref.shape, (name, got.shape, ref.shape)
            scale = max(1e-6, float(np.max(np.abs(ref))))
            err = float(np.max(np.abs(got - ref))) / scale
            assert err < 1e-4, f"{name} D_in={d_in}: rel err {err:.2e}"
            print(f"  D_in={d_in:>2} {name:<6} shape={got.shape} rel_err={err:.1e}")
    finally:
        _close(proj)
        _close(twin)


def test_input_dim_narrower_than_d_model_grads():
    _check_projection_grads(5, seed=3)


def test_input_dim_wider_than_d_model_grads():
    _check_projection_grads(13, seed=4)


def test_input_dim_adam_updates_every_w_in_row():
    """Adam must touch all D_in*D entries — rows >= D were skipped under D*D."""
    for d_in in (5, 13):
        model = _make(d_in, seed=21 + d_in)
        try:
            w0 = model.W_in.copy()
            m0 = model._ms_W_in.copy()
            rng = np.random.default_rng(d_in)
            X = (rng.standard_normal((B, T, d_in)) * 0.5).astype(np.float32)
            y = (rng.standard_normal((B, ACTIONS)) * 0.5).astype(np.float32)
            model.run_contract_train_step(X, y, lr=1e-2, apply_adam=True)
            moved = np.any(model.W_in != w0, axis=1)
            assert moved.all(), f"D_in={d_in}: W_in rows never updated: {np.where(~moved)[0]}"
            assert np.any(model._ms_W_in != m0, axis=1).all()
            print(f"  D_in={d_in:>2} adam: all {d_in} W_in rows updated")
        finally:
            _close(model)


def test_input_dim_rejects_wrong_width():
    model = _make(5, seed=31)
    try:
        X = np.zeros((B, T, D_MODEL), dtype=np.float32)
        y = np.zeros((B, ACTIONS), dtype=np.float32)
        try:
            model.run_contract_train_step(X, y, lr=0.0, apply_adam=False)
        except ValueError as exc:
            assert "input width mismatch" in str(exc)
        else:
            raise AssertionError("expected ValueError for X width != input_dim")
    finally:
        _close(model)


if __name__ == "__main__":
    test_input_dim_narrower_than_d_model_grads()
    test_input_dim_wider_than_d_model_grads()
    test_input_dim_adam_updates_every_w_in_row()
    test_input_dim_rejects_wrong_width()
    print("[PASSED] mhsa input_dim: exact grads + adam coverage + width guard")
