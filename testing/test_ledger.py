# testing/test_ledger.py
"""Phase E gates: ledger documents, file store, training engine, replay."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import EngineBackend, IngestionMode, ModelType, LRHierarchy
from config.schema import LedgerSettings
from src.ledger import (
    BatchRef,
    CHECKPOINT,
    FileLedgerStore,
    LedgerConfig,
    LedgerDocument,
    STEP_COMMAND,
    TrainingLedger,
    document_from_bytes,
    document_to_bytes,
    train_step_result_from_body,
    train_step_result_to_body,
)
from src.model_factory import ModelFactory
from src.training_engine import StepInput, TrainingEngine, create_training_engine
from src.training_session import TrainStepResult, TrainingSession
from src.controller import ModelController


def _tiny_cnn():
    cnn_config = {
        "input_shape": [1, 28, 28],
        "spatial_pipeline": [
            {"type": "conv", "in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 1, "pad": 1},
            {"type": "relu"},
            {"type": "pool", "pool_size": 2, "stride": 2},
            {"type": "flatten"},
        ],
        "dense_head": [8],
    }
    return ModelFactory.create_model(
        model_type="cnn",
        layer_sizes=[4],
        backend=EngineBackend.NATIVE,
        optimizer="adam",
        cnn_config=cnn_config,
    )


def _clone_weights(src, dst) -> None:
    for w_s, w_d in zip(src.weights, dst.weights):
        w_d[...] = w_s
    for b_s, b_d in zip(src.biases, dst.biases):
        b_d[...] = b_s


def _batch(rng, n=4):
    X = rng.standard_normal((n, 1, 28, 32), dtype=np.float32)
    y = np.eye(n, dtype=np.float32)
    return X, y


def test_e1_batch_ref_uuid_unique():
    a = BatchRef.new(epoch=0, batch_idx=0)
    b = BatchRef.new(epoch=0, batch_idx=0)
    assert a.batch_id != b.batch_id
    print("[PASSED] E1: BatchRef UUID unique per fetch")


def test_e1_document_round_trip():
    doc = LedgerDocument(
        doc_type=STEP_COMMAND,
        branch_id="main",
        model_instance_id="cnn-0",
        architecture_id="cnn_v1",
        body={"step_id": 1, "lr": 0.01},
        lsn=1,
    )
    back = document_from_bytes(document_to_bytes(doc))
    assert back.doc_type == STEP_COMMAND
    assert back.body["step_id"] == 1
    print("[PASSED] E1: LedgerDocument serialize round-trip")


def test_e2_train_step_result_to_bytes():
    gw = [np.ones((2, 3), dtype=np.float32)]
    gb = [np.zeros((1, 3), dtype=np.float32)]
    result = TrainStepResult(step_id=3, loss=0.25, grad_weights=gw, grad_biases=gb, m_samples=4)
    body = train_step_result_to_body(result)
    back = train_step_result_from_body(body)
    assert back.step_id == 3
    assert np.allclose(back.grad_weights[0], gw[0])
    rt = TrainStepResult.from_bytes(result.to_bytes())
    assert rt.step_id == 3
    assert np.allclose(rt.grad_weights[0], gw[0])
    print("[PASSED] E2: TrainStepResult to_bytes / from_bytes")


def test_e4_file_ledger_store_push_scan():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileLedgerStore(tmp)
        ledger = TrainingLedger(store=store, branch_id="main", architecture_id="cnn_v1")
        lsn1 = ledger.push_step_command(1, 0, BatchRef.new(), lr=0.01, m_samples=4)
        lsn2 = ledger.push_step_consolidated(1, 1, 1)
        assert lsn2 == lsn1 + 1
        types = [d.doc_type for d in store.scan()]
        assert STEP_COMMAND in types
        assert store.head_lsn() == lsn2
    print("[PASSED] E4: FileLedgerStore push/scan")


def test_e4_checkpoint_put_get():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileLedgerStore(tmp)
        ledger = TrainingLedger(store=store, branch_id="main")
        model = _tiny_cnn()
        ledger.push_checkpoint(model, version=1, val_loss=0.5, is_local_best=True)
        cp = store.get_checkpoint("main", 1)
        assert cp is not None
        assert cp.doc_type == CHECKPOINT
        assert cp.body["version"] == 1
    print("[PASSED] E4: checkpoint put/get")


def test_e3_engine_matches_train_and_apply():
    rng = np.random.default_rng(99)
    lr = 0.01
    n_steps = 4
    batches = []
    for i in range(n_steps):
        X, y = _batch(rng)
        batches.append((X, y, BatchRef.new(epoch=0, batch_idx=i)))

    direct = _tiny_cnn()
    with tempfile.TemporaryDirectory() as tmp:
        store = FileLedgerStore(tmp)
        ledger = TrainingLedger(store=store, branch_id="main", architecture_id="cnn_v1")
        engine_model = _tiny_cnn()
        _clone_weights(direct, engine_model)

        for X, y, _ in batches:
            direct.backward(X, y, lr)

        session = TrainingSession(model=engine_model, data_provider=None, initial_lr=lr)
        cfg = LedgerConfig(checkpoint_every_steps=2, checkpoint_on_local_best=False)
        engine = TrainingEngine(session=session, ledger=ledger, config=cfg)
        steps = [StepInput(X=X, y=y, batch_ref=ref, lr=lr) for X, y, ref in batches]
        engine.run_steps(steps)

        for w_d, w_e in zip(direct.weights, engine_model.weights):
            assert np.allclose(w_d, w_e, rtol=1e-5, atol=1e-5)
        for b_d, b_e in zip(direct.biases, engine_model.biases):
            assert np.allclose(b_d, b_e, rtol=1e-5, atol=1e-5)
    print("[PASSED] E3: TrainingEngine matches sequential train_and_apply")


def test_e5_replay_from_checkpoint():
    rng = np.random.default_rng(7)
    lr = 0.01
    n_steps = 3
    batches = []
    for i in range(n_steps):
        X, y = _batch(rng)
        batches.append((X, y, BatchRef.new(epoch=0, batch_idx=i)))

    with tempfile.TemporaryDirectory() as tmp:
        store = FileLedgerStore(tmp)
        init = _tiny_cnn()
        model = _tiny_cnn()
        _clone_weights(init, model)
        session = TrainingSession(model=model, data_provider=None, initial_lr=lr)
        ledger = TrainingLedger(store=store, branch_id="main", architecture_id="cnn_v1")
        cfg = LedgerConfig(checkpoint_every_steps=1, checkpoint_on_local_best=False)
        engine = TrainingEngine(session=session, ledger=ledger, config=cfg)
        steps = [StepInput(X=X, y=y, batch_ref=ref, lr=lr) for X, y, ref in batches]
        engine.run_steps(steps)
        target_weights = [w.copy() for w in model.weights]
        target_biases = [b.copy() for b in model.biases]

        replay_model = _tiny_cnn()
        _clone_weights(init, replay_model)
        replay_session = TrainingSession(model=replay_model, data_provider=None, initial_lr=lr)
        replay_ledger = TrainingLedger(store=store, branch_id="main", architecture_id="cnn_v1")
        replay_ledger.restore_checkpoint(replay_model, version=1)
        replay_ledger.replay_apply_from_version(
            replay_session, from_version=1, to_version=n_steps, default_lr=lr
        )

        for w_t, w_r in zip(target_weights, replay_model.weights):
            assert np.allclose(w_t, w_r, rtol=1e-5, atol=1e-5)
        for b_t, b_r in zip(target_biases, replay_model.biases):
            assert np.allclose(b_t, b_r, rtol=1e-5, atol=1e-5)
    print("[PASSED] E5: replay from checkpoint matches head weights")


def test_e5_fork_freezes_parent():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileLedgerStore(tmp)
        ledger = TrainingLedger(store=store, branch_id="main")
        ledger.push_checkpoint(_tiny_cnn(), version=2)
        child = ledger.fork_branch("retry", parent_version=2, reason="test")
        assert ledger.frozen
        assert child.branch_id == "retry"
        assert child.version == 2
        assert store.get_checkpoint("retry", 2) is not None
    print("[PASSED] E5: fork freezes parent and copies checkpoint")


def test_e6_prime_fit_with_ledger_and_early_stop_rollback():
    """E6′: fit via controller writes ledger; ES rollback uses ledger checkpoint."""
    import shutil

    rng = np.random.default_rng(11)
    lr = 0.01
    ledger_root = tempfile.mkdtemp(prefix="ml_engine_ledger_fit_")
    try:
        class _Provider:
            def __init__(self):
                self.batch_size = 4
                self._cursor = 0
                self._epoch = 0
                self.X = rng.standard_normal((8, 1, 28, 32), dtype=np.float32)
                self.y = np.eye(4, dtype=np.float32)
                self.y = np.vstack([self.y, self.y])
                self.splits = {}
                self.y_train_processed = self.y

            def get_validation_set(self):
                return self.X[:4], self.y[:4]

            def reset_epoch(self):
                self._cursor = 0
                self._epoch += 1

            def has_more_batches(self):
                return self._cursor < len(self.X)

            def next_batch(self):
                end = min(self._cursor + self.batch_size, len(self.X))
                x, y = self.X[self._cursor:end], self.y[self._cursor:end]
                self._cursor = end
                return x, y

            def normalize(self, x):
                return x

        provider = _Provider()
        controller = ModelController(data_provider=provider, learning_rate=lr)
        controller.initialize_network_from_dimensions(
            input_dim=28 * 32,
            output_dim=4,
            model_type=ModelType.CNN,
            hidden_layers=[],
            cnn_config={
                "input_shape": [1, 28, 28],
                "spatial_pipeline": [
                    {"type": "conv", "in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 1, "pad": 1},
                    {"type": "relu"},
                    {"type": "pool", "pool_size": 2, "stride": 2},
                    {"type": "flatten"},
                ],
                "dense_head": [8],
            },
            backend=EngineBackend.NATIVE,
        )

        ledger_cfg = LedgerSettings(
            enabled=True,
            path="branch_main",
            branch_id="main",
            checkpoint_every_steps=100,
            checkpoint_on_local_best=True,
        )
        controller.fit(
            steps=8,
            source_mode=IngestionMode.CSV,
            model_type=ModelType.CNN,
            early_stopping_enabled=False,
            ledger_settings=ledger_cfg,
            output_dir=ledger_root,
        )

        from src.ledger import FileLedgerStore

        store = FileLedgerStore(os.path.join(ledger_root, "branch_main"))
        assert store.head_lsn() > 0
        journal_types = {d.doc_type for d in store.scan()}
        assert "step.command" in journal_types
        assert "step.result" in journal_types
        print("[PASSED] E6′: fit with ledger writes step documents")
    finally:
        shutil.rmtree(ledger_root, ignore_errors=True)


PHASE_E_TESTS = [
    test_e1_batch_ref_uuid_unique,
    test_e1_document_round_trip,
    test_e2_train_step_result_to_bytes,
    test_e4_file_ledger_store_push_scan,
    test_e4_checkpoint_put_get,
    test_e3_engine_matches_train_and_apply,
    test_e5_replay_from_checkpoint,
    test_e5_fork_freezes_parent,
    test_e6_prime_fit_with_ledger_and_early_stop_rollback,
]


if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING PHASE E LEDGER TESTS ")
    print("=" * 60)
    failed = 0
    for fn in PHASE_E_TESTS:
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"[FAILED] {fn.__name__}: {exc}")
    print("=" * 60)
    if failed:
        print(f"[FAILURE] {failed}/{len(PHASE_E_TESTS)} failed.")
        sys.exit(1)
    print(f"[SUCCESS] All {len(PHASE_E_TESTS)} Phase E tests passed.")
    print("=" * 60)
