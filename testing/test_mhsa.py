# testing/test_mhsa.py
"""MHSA native contract: smoke, causal/softmax structure, finite-diff grads."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import EngineBackend
from src.contract import compile_mhsa_training_step, ContractOp
from src.model_factory import ModelFactory
from utils.conv_dispatch import bootstrap_im2col_gemm_runtime


def _rel_err(analytic: float, numeric: float) -> float:
    abs_diff = abs(analytic - numeric)
    # float32 MHSA path: near-zero analytic vs FD noise
    if abs(analytic) < 1e-5 and abs(numeric) < 5e-4:
        return abs_diff if abs_diff > 5e-4 else 0.0
    if abs(analytic) < 1e-7 and abs(numeric) < 1e-7:
        return abs_diff
    return abs_diff / max(abs(analytic) + abs(numeric), 1e-12)


def _make_mhsa(
    *,
    d_model: int = 8,
    num_heads: int = 2,
    max_seq_len: int = 8,
    action_dim: int = 2,
    ffn_mult: int = 2,
    seed: int = 0,
):
    bootstrap_im2col_gemm_runtime()
    np.random.seed(seed)
    return ModelFactory.create_model(
        "mhsa",
        layer_sizes=[action_dim],
        backend=EngineBackend.NATIVE,
        optimizer="adam",
        mhsa_config={
            "d_model": d_model,
            "num_heads": num_heads,
            "max_seq_len": max_seq_len,
            "action_dim": action_dim,
            "ffn_mult": ffn_mult,
        },
        contract_list_enabled=True,
        lam_l2=0.0,
        lam_l1=0.0,
    )


def _mse(actions: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((actions.astype(np.float64) - y.astype(np.float64)) ** 2))


def _sample_coords(shape: tuple[int, ...], rng: np.random.Generator, cap: int) -> list[tuple]:
    flat = int(np.prod(shape))
    if flat == 0:
        return []
    n = min(cap, flat)
    # Always include first + last; fill rest randomly without replacement.
    idxs = {0, flat - 1}
    while len(idxs) < n:
        idxs.add(int(rng.integers(0, flat)))
    out = []
    for i in sorted(idxs):
        out.append(tuple(int(x) for x in np.unravel_index(i, shape)))
    return out


def test_mhsa_compile_ops():
    c = compile_mhsa_training_step()
    assert [op.opcode for op in c.ops] == [
        ContractOp.MHSA_BLOCK_FWD,
        ContractOp.MHSA_ACTION_FWD,
        ContractOp.MHSA_ACTION_BWD,
        ContractOp.MHSA_BLOCK_BWD,
        ContractOp.ADAM_APPLY,
    ]
    print("[PASSED] mhsa: contract op list")


def test_mhsa_naive_forward_actions():
    model = _make_mhsa(d_model=32, num_heads=4, action_dim=4, seed=0)
    X = np.random.randn(2, 5, 32).astype(np.float64)
    actions = model.predict(X)
    assert actions.shape == (2, 4)
    assert np.all(np.isfinite(actions))
    assert np.all(actions >= -1.0 - 1e-5) and np.all(actions <= 1.0 + 1e-5)
    a2 = model.predict(X)
    assert np.allclose(actions, a2, atol=1e-6)
    print("[PASSED] mhsa: naive native forward → tanh actions (deterministic)")


def test_mhsa_train_step_smoke():
    model = _make_mhsa(d_model=16, num_heads=2, action_dim=3, seed=1)
    X = np.random.randn(2, 4, 16).astype(np.float64)
    y = np.random.randn(2, 3).astype(np.float64)
    w0 = [w.copy() for w in model.weights]
    loss, gw, gb, m = model.run_contract_train_step(X, y, lr=1e-3, apply_adam=True)
    assert m == 2
    assert np.isfinite(loss)
    assert len(gw) == 5 and len(gb) == 5
    assert all(g is not None and np.all(np.isfinite(g)) for g in gw)
    assert all(g is not None and np.all(np.isfinite(g)) for g in gb)
    assert any(not np.allclose(a, b) for a, b in zip(w0, model.weights))
    print("[PASSED] mhsa: train step smoke (loss+grads+adam)")


def test_mhsa_rejects_bad_geometry():
    try:
        _make_mhsa(d_model=8, num_heads=3)
        raise AssertionError("expected d_model % heads ValueError")
    except ValueError:
        pass
    model = _make_mhsa(max_seq_len=4, seed=2)
    X = np.random.randn(1, 5, 8).astype(np.float64)
    try:
        model.predict(X)
        raise AssertionError("expected T > max_seq_len ValueError")
    except ValueError:
        pass
    print("[PASSED] mhsa: rejects bad heads / T>max")


def test_mhsa_t1_and_causal_scores():
    model = _make_mhsa(d_model=8, num_heads=2, action_dim=2, seed=3)
    X1 = np.random.randn(2, 1, 8).astype(np.float64)
    a1 = model.predict(X1)
    assert a1.shape == (2, 2) and np.all(np.isfinite(a1))

    T = 4
    X = np.random.randn(2, T, 8).astype(np.float64)
    model.predict(X)
    scores = model._contract_runtime._mhsa_ws["scores"]
    # scores layout [B, H, T, T]; causal + row-stochastic on allowed prefix
    for i in range(T):
        future = scores[:, :, i, i + 1 :]
        assert np.allclose(future, 0.0, atol=1e-6), f"causal leak at i={i}"
        prefix = scores[:, :, i, : i + 1]
        assert np.all(prefix >= -1e-6)
        sums = prefix.sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-4), f"softmax row sum i={i}: {sums}"
    print("[PASSED] mhsa: T=1 + causal softmax structure")


def test_mhsa_loss_matches_predict_mse():
    model = _make_mhsa(seed=4)
    X = np.random.randn(2, 3, 8).astype(np.float64)
    y = np.random.randn(2, 2).astype(np.float64)
    actions = model.predict(X)
    loss_py = _mse(actions, y)
    loss_native, _, _, _ = model.run_contract_train_step(
        X, y, lr=0.0, apply_adam=False
    )
    assert abs(loss_native - loss_py) < 1e-5, f"native={loss_native} py={loss_py}"
    print("[PASSED] mhsa: native loss matches predict MSE")


def test_mhsa_adam_loss_decreases():
    model = _make_mhsa(seed=5)
    rng = np.random.default_rng(5)
    X = rng.standard_normal((4, 3, 8)).astype(np.float64) * 0.5
    y = np.zeros((4, 2), dtype=np.float64)
    losses = []
    for _ in range(20):
        loss, _, _, _ = model.run_contract_train_step(X, y, lr=1e-2, apply_adam=True)
        losses.append(loss)
    assert losses[-1] < losses[0] * 0.9, f"losses[0]={losses[0]} losses[-1]={losses[-1]}"
    print(f"[PASSED] mhsa: adam loss decreases ({losses[0]:.4f} → {losses[-1]:.4f})")


def _analytic_pack(model, X: np.ndarray, y: np.ndarray):
    loss, gw, gb, _ = model.run_contract_train_step(X, y, lr=0.0, apply_adam=False)
    rt = model._contract_runtime
    dln = rt._mhsa_dln_f32
    assert dln is not None
    ln_g = [
        np.array(dln["ln1_g"], dtype=np.float64).reshape(1, -1),
        np.array(dln["ln2_g"], dtype=np.float64).reshape(1, -1),
    ]
    ln_b = [
        np.array(dln["ln1_b"], dtype=np.float64).reshape(1, -1),
        np.array(dln["ln2_b"], dtype=np.float64).reshape(1, -1),
    ]
    dX = np.array(rt._mhsa_ws["dX"], dtype=np.float64).reshape(X.shape)
    return loss, gw, gb, ln_g, ln_b, dX


def _fd_check_tensor(
    name: str,
    tensor: np.ndarray,
    grad: np.ndarray,
    *,
    fwd_loss,
    eps: float,
    tol: float,
    rng: np.random.Generator,
    cap: int,
) -> tuple[bool, float]:
    assert tensor.shape == grad.shape, f"{name} shape mismatch {tensor.shape} vs {grad.shape}"
    max_err = 0.0
    worst = None
    for coord in _sample_coords(tensor.shape, rng, cap):
        orig = float(tensor[coord])
        tensor[coord] = orig + eps
        lp = fwd_loss()
        tensor[coord] = orig - eps
        lm = fwd_loss()
        tensor[coord] = orig
        g_num = (lp - lm) / (2.0 * eps)
        g_ana = float(grad[coord])
        err = _rel_err(g_ana, g_num)
        if err > max_err:
            max_err = err
            worst = (coord, g_ana, g_num)
    ok = max_err <= tol
    tag = "PASSED" if ok else "FAILED"
    print(f"[{tag}] mhsa FD {name:<12} max_rel={max_err:.2e} (tol={tol:.0e})")
    if not ok and worst is not None:
        c, ga, gn = worst
        print(f"  └── worst {c}: ana={ga:+.6e} num={gn:+.6e} |Δ|={abs(ga - gn):.6e}")
    return ok, max_err


def test_mhsa_finite_diff_grads():
    """Central FD vs native analytic grads (float32 kernels → looser tol than CNN f64)."""
    model = _make_mhsa(d_model=8, num_heads=2, action_dim=2, ffn_mult=2, seed=7)
    rng = np.random.default_rng(7)
    X = (rng.standard_normal((2, 3, 8)) * 0.3).astype(np.float64)
    y = (rng.standard_normal((2, 2)) * 0.3).astype(np.float64)

    _, gw, gb, ln_g, ln_b, _ = _analytic_pack(model, X, y)

    def fwd_loss() -> float:
        return _mse(model.predict(X), y)

    eps = 1e-3
    tol = 5e-2
    cap = 6
    ok = True

    for i, (W, dW) in enumerate(zip(model.weights, gw)):
        name = model._param_names[i]
        p, _ = _fd_check_tensor(
            f"W[{name}]", W, dW, fwd_loss=fwd_loss, eps=eps, tol=tol, rng=rng, cap=cap
        )
        ok = ok and p

    for i, (b, db) in enumerate(zip(model.biases, gb)):
        name = model._param_names[i]
        p, _ = _fd_check_tensor(
            f"b[{name}]", b, db, fwd_loss=fwd_loss, eps=eps, tol=tol, rng=rng, cap=cap
        )
        ok = ok and p

    for i, (g, dg) in enumerate(
        zip([model.ln1_gamma, model.ln2_gamma], ln_g)
    ):
        p, _ = _fd_check_tensor(
            f"ln{i+1}_g", g, dg, fwd_loss=fwd_loss, eps=eps, tol=tol, rng=rng, cap=cap
        )
        ok = ok and p
    for i, (b, db) in enumerate(zip([model.ln1_beta, model.ln2_beta], ln_b)):
        p, _ = _fd_check_tensor(
            f"ln{i+1}_b", b, db, fwd_loss=fwd_loss, eps=eps, tol=tol, rng=rng, cap=cap
        )
        ok = ok and p

    # dX finite-diff (fresh analytic after weight checks)
    _, _, _, _, _, dX2 = _analytic_pack(model, X, y)
    max_err = 0.0
    worst = None
    for coord in _sample_coords(X.shape, rng, cap=8):
        orig = float(X[coord])
        X[coord] = orig + eps
        lp = _mse(model.predict(X), y)
        X[coord] = orig - eps
        lm = _mse(model.predict(X), y)
        X[coord] = orig
        g_num = (lp - lm) / (2.0 * eps)
        g_ana = float(dX2[coord])
        err = _rel_err(g_ana, g_num)
        if err > max_err:
            max_err = err
            worst = (coord, g_ana, g_num)
    p = max_err <= tol
    tag = "PASSED" if p else "FAILED"
    print(f"[{tag}] mhsa FD dX           max_rel={max_err:.2e} (tol={tol:.0e})")
    if not p and worst is not None:
        c, ga, gn = worst
        print(f"  └── worst {c}: ana={ga:+.6e} num={gn:+.6e}")
    ok = ok and p

    assert ok, "MHSA finite-difference gradient check failed"
    print("[PASSED] mhsa: finite-diff grads (W/b/LN/dX)")


if __name__ == "__main__":
    test_mhsa_compile_ops()
    test_mhsa_naive_forward_actions()
    test_mhsa_train_step_smoke()
    test_mhsa_rejects_bad_geometry()
    test_mhsa_t1_and_causal_scores()
    test_mhsa_loss_matches_predict_mse()
    test_mhsa_adam_loss_decreases()
    test_mhsa_finite_diff_grads()
    print("[SUCCESS] MHSA tests passed")
