# testing/test_closed_loop.py
"""Closed-loop: interleaver layout + MHSA external dA → dX handoff."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import EngineBackend
from src.closed_loop import ConditioningBank, LinearAdapter, TokenInterleaver
from src.model_factory import ModelFactory
from utils.conv_dispatch import bootstrap_im2col_gemm_runtime


def _close(model) -> None:
    rt = getattr(model, "_contract_runtime", None)
    if rt is not None:
        rt.close()
        model._contract_runtime = None


def test_interleaver_layout():
    D = 8
    B = 2
    il = TokenInterleaver(D)
    assert il.seq_len(1) == 2
    assert il.seq_len(3) == 6
    assert il.state_index(1) == 1
    assert il.action_index(1) == 2
    assert il.state_index(2) == 3

    G = np.ones((B, D), dtype=np.float32)
    S1 = np.full((B, D), 2.0, dtype=np.float32)
    S2 = np.full((B, D), 3.0, dtype=np.float32)
    A1 = np.full((B, D), 4.0, dtype=np.float32)
    X = il.build(G, [S1, S2], [A1])
    assert X.shape == (B, 4, D)
    assert np.allclose(X[:, 0], G)
    assert np.allclose(X[:, 1], S1)
    assert np.allclose(X[:, 2], A1)
    assert np.allclose(X[:, 3], S2)

    dX = np.zeros_like(X)
    dX[:, 0] = 1.0
    dX[:, 1] = 2.0
    dX[:, 2] = 3.0
    dX[:, 3] = 4.0
    dG, dS, dA = il.split_dX(dX, t=2)
    assert np.allclose(dG, 1.0)
    assert len(dS) == 2 and len(dA) == 1
    assert np.allclose(dS[0], 2.0) and np.allclose(dS[1], 4.0)
    assert np.allclose(dA[0], 3.0)


def test_mhsa_backward_from_dA_surfaces_dX():
    bootstrap_im2col_gemm_runtime()
    np.random.seed(0)
    model = ModelFactory.create_model(
        "mhsa",
        layer_sizes=[4],
        backend=EngineBackend.NATIVE,
        optimizer="adam",
        mhsa_config={
            "d_model": 16,
            "num_heads": 4,
            "max_seq_len": 8,
            "action_dim": 4,
            "ffn_mult": 2,
            "num_layers": 1,
            "use_pos_encoding": False,
            "use_input_proj": False,
        },
        contract_list_enabled=True,
        lam_l2=0.0,
        lam_l1=0.0,
    )
    try:
        B, T, D = 2, 4, 16
        X = np.random.randn(B, T, D).astype(np.float32) * 0.1
        actions = model.predict(X)
        assert actions.shape == (B, 4)
        dA = np.random.randn(B, 4).astype(np.float32) * 0.01
        dw, db, dX = model.backward_from_dA(X, dA, apply_adam=False)
        assert dX.shape == (B, T, D)
        assert np.isfinite(dX).all()
        assert float(np.abs(dX).max()) > 0.0
        assert model.get_last_dX() is not None
        assert len(dw) == len(model.weights)
        # Last-token structure: most energy on t=T-1 for shallow net (not required strict,
        # but dX should respond).
        assert float(np.abs(dX[:, -1, :]).sum()) > 0.0
    finally:
        _close(model)


def test_adapter_and_conditioning_grads():
    ad = LinearAdapter(5, 7, seed=0)
    x = np.random.randn(3, 5).astype(np.float32)
    y = ad.forward(x)
    dY = np.ones_like(y)
    dX = ad.backward(dY)
    assert dX.shape == x.shape
    assert float(np.abs(ad._dW).sum()) > 0.0
    w0 = ad.W.copy()
    ad.apply_pending(1e-2)
    assert float(np.abs(ad.W - w0).max()) > 0.0

    bank = ConditioningBank(4, 7, seed=1)
    ids = np.array([0, 2, 0], dtype=np.int64)
    G = bank.embed(ids)
    assert G.shape == (3, 7)
    bank.backward(np.ones_like(G), ids)
    assert float(np.abs(bank._d_emb[0]).sum()) > 0.0
    e0 = bank.embeddings.copy()
    bank.apply_pending(1e-2)
    assert float(np.abs(bank.embeddings - e0).max()) > 0.0


if __name__ == "__main__":
    test_interleaver_layout()
    test_adapter_and_conditioning_grads()
    test_mhsa_backward_from_dA_surfaces_dX()
    print("test_closed_loop OK")
