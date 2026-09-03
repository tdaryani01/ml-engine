# testing/test_contract.py
"""Phase F: contract list compile + native grad parity."""
import copy
import os
import sys
import tempfile
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import EngineBackend
from src.contract import ContractOp, compile_cnn_training_step
from src.ledger import BatchRef, LedgerConfig
from src.model_factory import ModelFactory
from src.training_engine import StepInput, create_training_engine
from src.training_session import TrainingSession
from utils.conv_dispatch import _load_conv_dll


def _tiny_cnn_config():
    return {
        "input_shape": [1, 28, 28],
        "spatial_pipeline": [
            {"type": "conv", "in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 1, "pad": 1},
            {"type": "relu"},
            {"type": "pool", "pool_size": 2, "stride": 2},
            {"type": "flatten"},
        ],
        "dense_head": [],
    }


def _make_model(seed: int = 42, cnn_config: dict | None = None):
    np.random.seed(seed)
    return ModelFactory.create_model(
        model_type="cnn",
        layer_sizes=[4],
        backend=EngineBackend.NATIVE,
        optimizer="adam",
        cnn_config=cnn_config or _tiny_cnn_config(),
        lam_l1=0.0,
        lam_l2=0.0,
        max_norm=1e9,
    )


def test_compile_synthetic_cnn_contract():
    model = _make_model()
    contract = compile_cnn_training_step(
        model.layers,
        layer_param_idx=model._layer_param_idx,
        dense_w_indices=model._dense_w_indices,
    )
    opcodes = [op.opcode for op in contract.ops]
    assert ContractOp.CONV_BLOCK_FWD in opcodes
    assert opcodes.index(ContractOp.DENSE_FWD) > opcodes.index(ContractOp.CONV_BLOCK_FWD)
    assert opcodes.index(ContractOp.DENSE_BWD) < opcodes.index(ContractOp.CONV_BLOCK_BWD)
    assert contract.op_count >= 6
    blob = contract.to_bytes()
    assert len(blob) > 4
    print(f"[PASSED] compile contract: {contract.op_count} ops, {len(blob)} bytes")


def test_contract_grad_parity_hidden_dense():
    lib = _load_conv_dll()
    if lib is None or not hasattr(lib, "run_contract_training_step"):
        print("[SKIPPED] contract hidden dense parity: rebuild native (run_contract_training_step missing)")
        return

    cfg = _tiny_cnn_config()
    cfg["dense_head"] = [8]

    rng = np.random.default_rng(11)
    X = rng.standard_normal((8, 1, 28, 32), dtype=np.float32)
    y = np.zeros((8, 4), dtype=np.float32)
    y[np.arange(8), rng.integers(0, 4, size=8)] = 1.0
    lr = 0.01

    sync = _make_model(seed=11, cnn_config=cfg)
    contract = _make_model(seed=11, cnn_config=cfg)
    contract.enable_contract_list()

    sess_sync = TrainingSession(model=sync, data_provider=None, initial_lr=lr)
    sess_contract = TrainingSession(model=contract, data_provider=None, initial_lr=lr)

    res_sync = sess_sync.train_step(X, y, lr=lr)
    sess_sync.apply_step(res_sync, lr)
    sess_contract.train_step(X, y, lr=lr)

    for i, (w_a, w_b) in enumerate(zip(sync.weights, contract.weights)):
        np.testing.assert_allclose(w_a, w_b, rtol=1e-4, atol=1e-4, err_msg=f"weight[{i}]")
    for i, (b_a, b_b) in enumerate(zip(sync.biases, contract.biases)):
        np.testing.assert_allclose(b_a, b_b, rtol=1e-4, atol=1e-4, err_msg=f"bias[{i}]")

    print("[PASSED] contract grad parity: hidden dense head matches sync path")


def test_contract_grad_parity_one_step():
    lib = _load_conv_dll()
    if lib is None or not hasattr(lib, "run_contract_training_step"):
        print("[SKIPPED] contract grad parity: rebuild native (run_contract_training_step missing)")
        return

    rng = np.random.default_rng(7)
    X = rng.standard_normal((8, 1, 28, 32), dtype=np.float32)
    y = np.zeros((8, 4), dtype=np.float32)
    y[np.arange(8), rng.integers(0, 4, size=8)] = 1.0
    lr = 0.01

    sync = _make_model(seed=123)
    contract = _make_model(seed=123)
    contract.enable_contract_list()

    sess_sync = TrainingSession(model=sync, data_provider=None, initial_lr=lr)
    sess_contract = TrainingSession(model=contract, data_provider=None, initial_lr=lr)

    res_sync = sess_sync.train_step(X, y, lr=lr)
    sess_sync.apply_step(res_sync, lr)

    res_contract = sess_contract.train_step(X, y, lr=lr)

    for i, (w_a, w_b) in enumerate(zip(sync.weights, contract.weights)):
        np.testing.assert_allclose(w_a, w_b, rtol=1e-4, atol=1e-4, err_msg=f"weight[{i}]")
    for i, (b_a, b_b) in enumerate(zip(sync.biases, contract.biases)):
        np.testing.assert_allclose(b_a, b_b, rtol=1e-4, atol=1e-4, err_msg=f"bias[{i}]")

    print("[PASSED] contract grad parity: one step weights match sync path")


def test_contract_train_step_weights_applied_in_native():
    lib = _load_conv_dll()
    if lib is None or not hasattr(lib, "run_contract_training_step"):
        print("[SKIPPED] contract weights_applied: rebuild native (run_contract_training_step missing)")
        return

    rng = np.random.default_rng(9)
    X = rng.standard_normal((8, 1, 28, 32), dtype=np.float32)
    y = np.zeros((8, 4), dtype=np.float32)
    y[np.arange(8), rng.integers(0, 4, size=8)] = 1.0
    lr = 0.01

    model = _make_model(seed=55)
    model.enable_contract_list()
    session = TrainingSession(model=model, data_provider=None, initial_lr=lr)

    w_before = [w.copy() for w in model.weights]
    result = session.train_step(X, y, lr=lr)

    assert result.weights_applied is True
    assert any(not np.allclose(w, wb) for w, wb in zip(model.weights, w_before))
    print("[PASSED] contract train_step: native Adam applied, weights_applied set")


def test_contract_async_submit_reaps():
    lib = _load_conv_dll()
    if lib is None or not hasattr(lib, "submit_contract_training_step"):
        print("[SKIPPED] contract async: rebuild native (submit_contract_training_step missing)")
        return

    rng = np.random.default_rng(21)
    X = rng.standard_normal((8, 1, 28, 32), dtype=np.float32)
    y = np.zeros((8, 4), dtype=np.float32)
    y[np.arange(8), rng.integers(0, 4, size=8)] = 1.0
    lr = 0.01

    sync = _make_model(seed=31)
    async_model = _make_model(seed=31)
    async_model.enable_contract_list()

    sess_sync = TrainingSession(model=sync, data_provider=None, initial_lr=lr)
    sess_async = TrainingSession(model=async_model, data_provider=None, initial_lr=lr)

    res_sync = sess_sync.train_step(X, y, lr=lr)
    sess_sync.apply_step(res_sync, lr)

    tick_calls = {"n": 0}

    def _tick():
        tick_calls["n"] += 1

    res_async = sess_async.train_step(X, y, lr=lr, tick_fn=_tick)
    assert res_async.weights_applied is True
    assert tick_calls["n"] >= 0

    for w_a, w_b in zip(sync.weights, async_model.weights):
        np.testing.assert_allclose(w_a, w_b, rtol=1e-4, atol=1e-4)
    print("[PASSED] contract async: submit/reap matches sync path")


def test_contract_busy_pushback():
    lib = _load_conv_dll()
    if lib is None or not hasattr(lib, "submit_contract_training_step"):
        print("[SKIPPED] contract busy: rebuild native (submit_contract_training_step missing)")
        return

    rng = np.random.default_rng(33)
    X = rng.standard_normal((8, 1, 28, 32), dtype=np.float32)
    y = np.zeros((8, 4), dtype=np.float32)
    y[np.arange(8), rng.integers(0, 4, size=8)] = 1.0
    lr = 0.01

    model = _make_model(seed=41)
    model.enable_contract_list()
    model._contract_runtime.set_engine_driven(True)

    assert model.add_training_step(X, y, lr, step_token=1) == "OK"
    assert model.contract_busy()
    assert model.add_training_step(X, y, lr, step_token=2) == "BUSY"
    # Drain the in-flight step so the native worker is idle for later tests.
    rt = model._contract_runtime
    assert rt.wait_for_completion(timeout=5.0)
    assert rt.try_reap_step() is not None
    print("[PASSED] contract busy: CNN pushback while native in flight")


def test_contract_engine_finalize_skips_python_apply():
    lib = _load_conv_dll()
    if lib is None or not hasattr(lib, "run_contract_training_step"):
        print("[SKIPPED] contract finalize apply skip: rebuild native (run_contract_training_step missing)")
        return

    rng = np.random.default_rng(13)
    X = rng.standard_normal((8, 1, 28, 32), dtype=np.float32)
    y = np.zeros((8, 4), dtype=np.float32)
    y[np.arange(8), rng.integers(0, 4, size=8)] = 1.0
    lr = 0.01

    with tempfile.TemporaryDirectory() as tmp:
        model = _make_model(seed=77)
        model.enable_contract_list()
        session = TrainingSession(model=model, data_provider=None, initial_lr=lr)
        engine = create_training_engine(
            tmp,
            session=session,
            config=LedgerConfig(
                checkpoint_every_steps=100,
                checkpoint_on_local_best=False,
                contract_list_enabled=True,
                store_backend="file_streaming",
            ),
        )

        with mock.patch.object(session, "apply_step", wraps=session.apply_step) as apply_mock:
            engine.on_fit_start()
            step = StepInput(X=X, y=y, batch_ref=BatchRef.new(), lr=lr)
            assert engine.try_submit(step)
            losses = engine.drain_pending()
            assert losses
            loss = losses[0]
            assert loss is not None
            apply_mock.assert_not_called()

        engine.close()

    print("[PASSED] contract engine: finalize skips Python apply_step")


if __name__ == "__main__":
    try:
        test_compile_synthetic_cnn_contract()
        test_contract_grad_parity_hidden_dense()
        test_contract_grad_parity_one_step()
        test_contract_train_step_weights_applied_in_native()
        test_contract_async_submit_reaps()
        test_contract_busy_pushback()
        test_contract_engine_finalize_skips_python_apply()
    finally:
        from src.contract_runtime import shutdown_contract_async

        shutdown_contract_async()
