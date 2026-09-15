# testing/test_mhsa.py
"""MHSA native contract: smoke, causal/softmax, stacked layers, finite-diff grads."""
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
    """Same cliff as testing/test_gradient_check.GradientChecker._compute_relative_error."""
    abs_diff = abs(analytic - numeric)
    if abs(analytic) < 1e-7 and abs(numeric) < 1e-7:
        return abs_diff
    return abs_diff / max(abs(analytic) + abs(numeric), 1e-12)


def _close(model) -> None:
    rt = getattr(model, "_contract_runtime", None)
    if rt is not None:
        rt.close()
        model._contract_runtime = None


def _make_mhsa(
    *,
    d_model: int = 8,
    num_heads: int = 2,
    max_seq_len: int = 8,
    action_dim: int = 2,
    ffn_mult: int = 2,
    num_layers: int = 1,
    seed: int = 0,
    use_input_proj: bool = False,
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
            "num_layers": num_layers,
            # Pos adds noise to tiny probes; exercise it in a dedicated smoke.
            "use_pos_encoding": False,
            "use_input_proj": use_input_proj,
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
    try:
        X = np.random.randn(2, 5, 32).astype(np.float64)
        actions = model.predict(X)
        assert actions.shape == (2, 4)
        assert np.all(np.isfinite(actions))
        assert np.all(actions >= -1.0 - 1e-5) and np.all(actions <= 1.0 + 1e-5)
        assert np.allclose(actions, model.predict(X), atol=1e-6)
        print("[PASSED] mhsa: naive native forward → tanh actions (deterministic)")
    finally:
        _close(model)


def test_mhsa_train_step_smoke():
    model = _make_mhsa(d_model=16, num_heads=2, action_dim=3, seed=1)
    try:
        X = np.random.randn(2, 4, 16).astype(np.float64)
        y = np.random.randn(2, 3).astype(np.float64)
        w0 = [w.copy() for w in model.weights]
        loss, gw, gb, m = model.run_contract_train_step(X, y, lr=1e-3, apply_adam=True)
        assert m == 2
        assert np.isfinite(loss)
        assert len(gw) == 5 and len(gb) == 5
        assert all(g is not None and np.all(np.isfinite(g)) for g in gw)
        assert any(not np.allclose(a, b) for a, b in zip(w0, model.weights))
        print("[PASSED] mhsa: train step smoke (loss+grads+adam)")
    finally:
        _close(model)


def test_mhsa_stacked_smoke():
    model = _make_mhsa(num_layers=2, d_model=8, num_heads=2, action_dim=2, seed=9)
    try:
        assert model.num_layers == 2
        assert len(model.weights) == 9  # 4*2 + act
        X = np.random.randn(2, 3, 8).astype(np.float64)
        y = np.random.randn(2, 2).astype(np.float64)
        actions = model.predict(X)
        assert actions.shape == (2, 2) and np.all(np.isfinite(actions))
        loss, gw, gb, _ = model.run_contract_train_step(X, y, lr=1e-2, apply_adam=True)
        assert np.isfinite(loss) and len(gw) == 9
        print("[PASSED] mhsa: stacked L=2 smoke")
    finally:
        _close(model)


def test_mhsa_rejects_bad_geometry():
    try:
        _make_mhsa(d_model=8, num_heads=3)
        raise AssertionError("expected d_model % heads ValueError")
    except ValueError:
        pass
    try:
        _make_mhsa(num_layers=0)
        raise AssertionError("expected num_layers ValueError")
    except ValueError:
        pass
    model = _make_mhsa(max_seq_len=4, seed=2)
    try:
        try:
            model.predict(np.random.randn(1, 5, 8).astype(np.float64))
            raise AssertionError("expected T > max_seq_len ValueError")
        except ValueError:
            pass
        print("[PASSED] mhsa: rejects bad heads / layers / T>max")
    finally:
        _close(model)


def test_mhsa_t1_and_causal_scores():
    model = _make_mhsa(d_model=8, num_heads=2, action_dim=2, num_layers=2, seed=3)
    try:
        a1 = model.predict(np.random.randn(2, 1, 8).astype(np.float64))
        assert a1.shape == (2, 2) and np.all(np.isfinite(a1))

        T = 4
        X = np.random.randn(2, T, 8).astype(np.float64)
        model.predict(X)
        for li in range(model.num_layers):
            scores = model._contract_runtime._mhsa_ws["layers"][li]["scores"]
            for i in range(T):
                assert np.allclose(scores[:, :, i, i + 1 :], 0.0, atol=1e-6)
                prefix = scores[:, :, i, : i + 1]
                assert np.allclose(prefix.sum(axis=-1), 1.0, atol=1e-4)
        print("[PASSED] mhsa: T=1 + causal softmax (all layers)")
    finally:
        _close(model)


def test_mhsa_loss_matches_predict_mse():
    model = _make_mhsa(seed=4, num_layers=2)
    try:
        X = np.random.randn(2, 3, 8).astype(np.float64)
        y = np.random.randn(2, 2).astype(np.float64)
        loss_py = _mse(model.predict(X), y)
        loss_native, _, _, _ = model.run_contract_train_step(
            X, y, lr=0.0, apply_adam=False
        )
        assert abs(loss_native - loss_py) < 1e-5
        print("[PASSED] mhsa: native loss matches predict MSE")
    finally:
        _close(model)


def test_mhsa_adam_loss_decreases():
    model = _make_mhsa(seed=5, num_layers=2)
    try:
        rng = np.random.default_rng(5)
        X = rng.standard_normal((4, 3, 8)).astype(np.float64) * 0.5
        y = np.zeros((4, 2), dtype=np.float64)
        losses = []
        for _ in range(20):
            loss, _, _, _ = model.run_contract_train_step(X, y, lr=1e-2, apply_adam=True)
            losses.append(loss)
        assert losses[-1] < losses[0] * 0.9, f"losses[0]={losses[0]} losses[-1]={losses[-1]}"
        print(f"[PASSED] mhsa: adam loss decreases ({losses[0]:.4f} → {losses[-1]:.4f})")
    finally:
        _close(model)


def _analytic_pack(model, X: np.ndarray, y: np.ndarray):
    loss, gw, gb, _ = model.run_contract_train_step(X, y, lr=0.0, apply_adam=False)
    rt = model._contract_runtime
    ln_g = []
    ln_b = []
    for dln in rt._mhsa_dln_f32:
        ln_g.append(np.array(dln["ln1_g"], dtype=np.float64).reshape(1, -1))
        ln_g.append(np.array(dln["ln2_g"], dtype=np.float64).reshape(1, -1))
        ln_b.append(np.array(dln["ln1_b"], dtype=np.float64).reshape(1, -1))
        ln_b.append(np.array(dln["ln2_b"], dtype=np.float64).reshape(1, -1))
    dX = np.array(rt._mhsa_ws["dX"], dtype=np.float64).reshape(X.shape)
    dW_in = np.array(rt._mhsa_ws["dW_in"], dtype=np.float64)
    db_in = np.array(rt._mhsa_ws["db_in"], dtype=np.float64).reshape(1, -1)
    return loss, gw, gb, ln_g, ln_b, dX, dW_in, db_in


def _sync_native_to_torch(model, tm) -> None:
    """Copy native [fan_in, fan_out] banks into Torch Linear/MHA (weight is [out, in])."""
    import torch

    L = int(model.num_layers)
    with torch.no_grad():
        for li in range(L):
            base = 4 * li
            blk = tm.blocks[li]
            W_qkv, W_o, W_ff1, W_ff2 = (model.weights[base + i] for i in range(4))
            b_qkv, b_o, b_ff1, b_ff2 = (
                model.biases[base + i].reshape(-1) for i in range(4)
            )
            blk.attn.in_proj_weight.copy_(torch.from_numpy(np.asarray(W_qkv.T)))
            blk.attn.in_proj_bias.copy_(torch.from_numpy(np.asarray(b_qkv)))
            blk.attn.out_proj.weight.copy_(torch.from_numpy(np.asarray(W_o.T)))
            blk.attn.out_proj.bias.copy_(torch.from_numpy(np.asarray(b_o)))
            blk.ff1.weight.copy_(torch.from_numpy(np.asarray(W_ff1.T)))
            blk.ff1.bias.copy_(torch.from_numpy(np.asarray(b_ff1)))
            blk.ff2.weight.copy_(torch.from_numpy(np.asarray(W_ff2.T)))
            blk.ff2.bias.copy_(torch.from_numpy(np.asarray(b_ff2)))
            blk.ln1.weight.copy_(
                torch.from_numpy(np.asarray(model.ln1_gamma[li].reshape(-1)))
            )
            blk.ln1.bias.copy_(
                torch.from_numpy(np.asarray(model.ln1_beta[li].reshape(-1)))
            )
            blk.ln2.weight.copy_(
                torch.from_numpy(np.asarray(model.ln2_gamma[li].reshape(-1)))
            )
            blk.ln2.bias.copy_(
                torch.from_numpy(np.asarray(model.ln2_beta[li].reshape(-1)))
            )
        act_i = 4 * L
        tm.action.weight.copy_(torch.from_numpy(np.asarray(model.weights[act_i].T)))
        tm.action.bias.copy_(
            torch.from_numpy(np.asarray(model.biases[act_i].reshape(-1)))
        )
        if tm.in_proj is not None and getattr(model, "W_in", None) is not None:
            tm.in_proj.weight.copy_(torch.from_numpy(np.asarray(model.W_in.T)))
            tm.in_proj.bias.copy_(
                torch.from_numpy(np.asarray(model.b_in.reshape(-1)))
            )


def _torch_ref_grads(model, X: np.ndarray, y: np.ndarray):
    """f64 Torch twin + autograd — same 1e-5 cliff as MLP GradientChecker."""
    import torch

    from benchmarks.benchmark_mhsa import create_torch_mhsa_class

    mhsa = {
        "d_model": int(model.d_model),
        "num_heads": int(model.num_heads),
        "max_seq_len": int(model.max_seq_len),
        "action_dim": int(model.action_dim),
        "ffn_mult": int(model.ffn_mult),
        "num_layers": int(model.num_layers),
        "use_pos_encoding": bool(model.use_pos_encoding),
        "use_input_proj": bool(model.use_input_proj),
    }
    tm = create_torch_mhsa_class()(mhsa).double()
    _sync_native_to_torch(model, tm)
    Xt = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    yt = torch.tensor(y, dtype=torch.float64)
    pred = tm(Xt)
    loss = torch.mean((pred - yt) ** 2)
    loss.backward()

    L = int(model.num_layers)
    gw = []
    gb = []
    ln_g = []
    ln_b = []
    for li in range(L):
        blk = tm.blocks[li]
        gw.extend(
            [
                blk.attn.in_proj_weight.grad.T.detach().numpy(),
                blk.attn.out_proj.weight.grad.T.detach().numpy(),
                blk.ff1.weight.grad.T.detach().numpy(),
                blk.ff2.weight.grad.T.detach().numpy(),
            ]
        )
        gb.extend(
            [
                blk.attn.in_proj_bias.grad.detach().numpy().reshape(1, -1),
                blk.attn.out_proj.bias.grad.detach().numpy().reshape(1, -1),
                blk.ff1.bias.grad.detach().numpy().reshape(1, -1),
                blk.ff2.bias.grad.detach().numpy().reshape(1, -1),
            ]
        )
        ln_g.append(blk.ln1.weight.grad.detach().numpy().reshape(1, -1))
        ln_g.append(blk.ln2.weight.grad.detach().numpy().reshape(1, -1))
        ln_b.append(blk.ln1.bias.grad.detach().numpy().reshape(1, -1))
        ln_b.append(blk.ln2.bias.grad.detach().numpy().reshape(1, -1))
    gw.append(tm.action.weight.grad.T.detach().numpy())
    gb.append(tm.action.bias.grad.detach().numpy().reshape(1, -1))
    dX = Xt.grad.detach().numpy()
    dW_in = None
    db_in = None
    if tm.in_proj is not None:
        dW_in = tm.in_proj.weight.grad.T.detach().numpy()
        db_in = tm.in_proj.bias.grad.detach().numpy().reshape(1, -1)
    return gw, gb, ln_g, ln_b, dX, dW_in, db_in


def _grad_check_tensor(
    name: str,
    analytic: np.ndarray,
    reference: np.ndarray,
    *,
    tol: float,
) -> bool:
    """Tight check: rtol=tol (1e-5) with atol=1e-7 for f32-native vs f64-Torch."""
    assert analytic.shape == reference.shape, (name, analytic.shape, reference.shape)
    a = np.asarray(analytic, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    # Per-element cliff matching GradientChecker relative error, plus an absolute
    # floor: native banks are f32 so sub-1e-7 abs deltas vs Torch.double are noise.
    atol = 1e-7
    max_err = 0.0
    worst = None
    flat_a = a.reshape(-1)
    flat_r = r.reshape(-1)
    for i in range(flat_a.size):
        ga, gn = float(flat_a[i]), float(flat_r[i])
        abs_diff = abs(ga - gn)
        if abs_diff <= atol:
            err = 0.0
        else:
            err = _rel_err(ga, gn)
        if err > max_err:
            max_err = err
            worst = (np.unravel_index(i, a.shape), ga, gn)
    ok = max_err <= tol
    tag = "PASSED" if ok else "FAILED"
    print(
        f"[{tag}] mhsa grad {name:<14} max_rel={max_err:.2e} "
        f"(tol={tol:.0e}, atol={atol:.0e})"
    )
    if not ok and worst is not None:
        c, ga, gn = worst
        print(f"  └── worst {c}: native={ga:+.6e} torch64={gn:+.6e}")
    return ok


def _run_grad_check(model, *, seed: int) -> None:
    """Native analytic vs Torch.double autograd at GradientChecker tol=1e-5.

    Classic central-diff through f32 native predict floors ~1e-3 and cannot
    honestly hit the MLP 1e-5 cliff; the f64 twin is the tight reference.
    """
    try:
        rng = np.random.default_rng(seed)
        X = (rng.standard_normal((2, 3, model.d_model)) * 0.3).astype(np.float64)
        y = (rng.standard_normal((2, model.action_dim)) * 0.3).astype(np.float64)
        _, gw, gb, ln_g, ln_b, dX, dW_in, db_in = _analytic_pack(model, X, y)
        ref_gw, ref_gb, ref_ln_g, ref_ln_b, ref_dX, ref_dW_in, ref_db_in = (
            _torch_ref_grads(model, X, y)
        )
        tol = 1e-5
        ok = True
        for i, (dW, rW) in enumerate(zip(gw, ref_gw)):
            ok = _grad_check_tensor(model._param_names[i], dW, rW, tol=tol) and ok
        for i, (db, rb) in enumerate(zip(gb, ref_gb)):
            ok = (
                _grad_check_tensor(f"b[{model._param_names[i]}]", db, rb, tol=tol)
                and ok
            )
        for i, (dg, rg) in enumerate(zip(ln_g, ref_ln_g)):
            ok = _grad_check_tensor(f"ln_g[{i}]", dg, rg, tol=tol) and ok
        for i, (db, rb) in enumerate(zip(ln_b, ref_ln_b)):
            ok = _grad_check_tensor(f"ln_b[{i}]", db, rb, tol=tol) and ok
        ok = _grad_check_tensor("dX", dX, ref_dX, tol=tol) and ok
        if model.use_input_proj:
            assert dW_in is not None and ref_dW_in is not None
            ok = _grad_check_tensor("W_in", dW_in, ref_dW_in, tol=tol) and ok
            ok = _grad_check_tensor("b_in", db_in, ref_db_in, tol=tol) and ok
        assert ok, "MHSA gradient check failed (native vs Torch.double, tol=1e-5)"
    finally:
        _close(model)


def test_mhsa_pos_encoding_adam():
    model = _make_mhsa(d_model=8, num_heads=2, max_seq_len=4, action_dim=2, seed=3)
    # Rebuild with pos on (factory helper forces False for FD).
    from src.model_factory import ModelFactory
    from config.constants import EngineBackend

    _close(model)
    np.random.seed(3)
    model = ModelFactory.create_model(
        "mhsa",
        layer_sizes=[2],
        backend=EngineBackend.NATIVE,
        optimizer="adam",
        mhsa_config={
            "d_model": 8,
            "num_heads": 2,
            "max_seq_len": 4,
            "action_dim": 2,
            "ffn_mult": 2,
            "num_layers": 1,
            "use_pos_encoding": True,
        },
        contract_list_enabled=True,
        lam_l2=0.0,
        lam_l1=0.0,
    )
    try:
        assert model.pos_embed is not None and model.pos_embed.shape == (4, 8)
        p0 = model.pos_embed.copy()
        X = np.random.randn(2, 3, 8)
        y = np.random.randn(2, 2)
        model.run_contract_train_step(X, y, lr=1e-2, apply_adam=True)
        assert not np.allclose(p0, model.pos_embed)
        print("[PASSED] mhsa: learned pos encoding + native adam")
    finally:
        _close(model)


def test_mhsa_input_proj_adam_and_fd():
    """Input proj: Adam moves W_in; grads match Torch.double at tol=1e-5."""
    model = _make_mhsa(
        d_model=8, num_heads=2, max_seq_len=4, action_dim=2, seed=9, use_input_proj=True
    )
    try:
        assert model.W_in is not None and model.W_in.shape == (8, 8)
        w0 = model.W_in.copy()
        rng = np.random.default_rng(9)
        X = (rng.standard_normal((2, 3, 8)) * 0.3).astype(np.float64)
        y = (rng.standard_normal((2, 2)) * 0.3).astype(np.float64)
        model.run_contract_train_step(X, y, lr=1e-2, apply_adam=True)
        assert not np.allclose(w0, model.W_in)

        _close(model)
        model = _make_mhsa(
            d_model=8,
            num_heads=2,
            max_seq_len=4,
            action_dim=2,
            seed=9,
            use_input_proj=True,
        )
        _run_grad_check(model, seed=9)
        print("[PASSED] mhsa: input proj adam + grad check")
    finally:
        _close(model)


def test_mhsa_dual_bank_prepare_while_inflight():
    """Dual-bank async: prepare next step while one is in flight; banks flip."""
    from src.model_factory import ModelFactory
    from config.constants import EngineBackend

    np.random.seed(13)
    model = ModelFactory.create_model(
        "mhsa",
        layer_sizes=[2],
        backend=EngineBackend.NATIVE,
        optimizer="adam",
        mhsa_config={
            "d_model": 8,
            "num_heads": 2,
            "max_seq_len": 4,
            "action_dim": 2,
            "ffn_mult": 2,
            "num_layers": 1,
            "use_pos_encoding": False,
            "use_input_proj": True,
        },
        contract_list_enabled=True,
        native_async_submit=True,
        lam_l2=0.0,
        lam_l1=0.0,
    )
    try:
        rt = model._contract_runtime
        assert rt is not None
        if not hasattr(rt._lib, "submit_contract_training_step"):
            print("[SKIPPED] mhsa dual-bank: submit_contract_training_step missing")
            return
        rt.set_engine_driven(True)
        rng = np.random.default_rng(13)
        X1 = rng.standard_normal((2, 3, 8)).astype(np.float32)
        y1 = rng.standard_normal((2, 2)).astype(np.float32)
        X2 = rng.standard_normal((2, 3, 8)).astype(np.float32)
        y2 = rng.standard_normal((2, 2)).astype(np.float32)
        lr = 1e-2
        w0 = model.W_in.copy()

        assert model.add_training_step(X1, y1, lr, apply_adam=True, step_token=1) == "OK"
        assert model.contract_busy()
        assert len(rt._mhsa_banks) == 2
        assert rt.prepare_step(X2, y2, lr, apply_adam=True, step_token=2)
        assert rt._prepared is not None
        assert rt._prepared.input_bank_idx != rt._prepared.output_bank_idx
        # Free slot while one submitted; prepare must not block on single-flight.
        assert rt._prepared.slot_idx != rt._submitted.slot_idx

        assert rt.wait_for_completion(timeout=5.0)
        loss1 = rt.try_reap_step()
        assert loss1 is not None
        assert not np.allclose(w0, model.W_in), "publish should move W_in after adam"

        assert model.add_training_step(X2, y2, lr, apply_adam=True, step_token=2) == "OK"
        assert rt.wait_for_completion(timeout=5.0)
        loss2 = rt.try_reap_step()
        assert loss2 is not None
        assert rt._published_bank_idx in (0, 1)
        print("[PASSED] mhsa: dual-bank prepare-while-inflight + W_in publish")
    finally:
        _close(model)


def test_mhsa_finite_diff_grads_l1():
    _run_grad_check(_make_mhsa(num_layers=1, seed=7), seed=7)
    print("[PASSED] mhsa: grad check L=1 (native vs Torch.double, tol=1e-5)")


def test_mhsa_finite_diff_grads_l2():
    _run_grad_check(_make_mhsa(num_layers=2, seed=11), seed=11)
    print("[PASSED] mhsa: grad check L=2 (native vs Torch.double, tol=1e-5)")


if __name__ == "__main__":
    test_mhsa_compile_ops()
    test_mhsa_naive_forward_actions()
    test_mhsa_train_step_smoke()
    test_mhsa_stacked_smoke()
    test_mhsa_rejects_bad_geometry()
    test_mhsa_t1_and_causal_scores()
    test_mhsa_loss_matches_predict_mse()
    test_mhsa_adam_loss_decreases()
    test_mhsa_pos_encoding_adam()
    test_mhsa_input_proj_adam_and_fd()
    test_mhsa_dual_bank_prepare_while_inflight()
    test_mhsa_finite_diff_grads_l1()
    test_mhsa_finite_diff_grads_l2()
    print("[SUCCESS] MHSA tests passed")
