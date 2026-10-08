# testing/test_mhsa_five_actions.py
"""The TM brain's five-action head on the NATIVE MHSA path (token schema 4).

The brain's action width is a runtime parameter of the model, not part of the native struct ABI (``MhsaBinding`` is checked by offsets in ``contract_runtime``). This proves it by running the
brain's real geometry (token width 112 = d_model, 4 heads, 9 positions = history_k 8 + the live token) with FIVE actions: forward, a discrete cross-entropy training step that moves the
weights and lowers the loss, and a fit that separates five classes.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from testing.test_mhsa import _close, _make_mhsa  # noqa: E402


def _model(action_dim: int):
    return _make_mhsa(d_model=112, num_heads=4, max_seq_len=9, action_dim=action_dim, ffn_mult=2, num_layers=2, seed=3, action_mode="discrete")


def test_five_action_head_forward_on_the_brain_geometry():
    model = _model(5)
    try:
        assert model.action_mode == "discrete" and model.action_dim == 5
        X = np.random.default_rng(0).standard_normal((6, 9, 112))
        logits = model.predict(X)
        assert logits.shape == (6, 5) and np.all(np.isfinite(logits))
        assert np.allclose(logits, model.predict(X), atol=1e-6)
        print("[PASSED] mhsa: five-action discrete head, 112/4 heads/9 positions, forward")
    finally:
        _close(model)


def test_five_action_head_learns_five_classes_natively():
    model = _model(5)
    try:
        rng = np.random.default_rng(1)
        y = np.repeat(np.arange(5), 16)
        onehot = np.zeros((len(y), 5)); onehot[np.arange(len(y)), y] = 1.0
        X = rng.standard_normal((len(y), 9, 112)) * 0.1
        X[np.arange(len(y)), -1, y] += 2.0                                  # the live token carries the class in one coordinate
        before = [w.copy() for w in model.weights]
        losses, accs = [], []
        for _ in range(80):
            loss, _gw, _gb, _ = model.run_contract_train_step(X, onehot, lr=3e-3, apply_adam=True)
            losses.append(float(loss))
            accs.append(float(np.mean(np.argmax(model.predict(X), axis=1) == y)))
        assert any(not np.array_equal(a, b) for a, b in zip(before, model.weights)), "the native step did not move any weight"
        assert losses[-1] < 0.5 * losses[0], f"loss {losses[0]:.3f} -> {losses[-1]:.3f}"
        assert accs[-1] >= 0.9 > accs[0], f"accuracy {accs[0]:.2f} -> {accs[-1]:.2f}: five classes were not learned"
        print(f"[PASSED] mhsa: five-action head trains natively, loss {losses[0]:.3f} -> {losses[-1]:.3f}, accuracy {accs[0]:.2f} -> {accs[-1]:.2f}")
    finally:
        _close(model)


if __name__ == "__main__":
    test_five_action_head_forward_on_the_brain_geometry()
    test_five_action_head_learns_five_classes_natively()
