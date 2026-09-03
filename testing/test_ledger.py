# testing/test_ledger.py
"""Phase E gates: ledger documents, file store, training engine, replay."""
import copy
import os
import sys
import tempfile
from unittest import mock

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
    STEP_COMPLETE,
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


def _close_store(store) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        close()


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
        body={
            "step_id": 1,
            "base_version": 0,
            "batch_ref": BatchRef.new(epoch=0, batch_idx=0).to_dict(),
            "lr": 0.01,
            "m_samples": 4,
            "scheduler_epoch": 0,
        },
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
        try:
            ledger = TrainingLedger(store=store, branch_id="main", architecture_id="cnn_v1")
            rng = np.random.default_rng(0)
            X, y = _batch(rng)
            ref = BatchRef.new()
            result = TrainStepResult(
                step_id=1,
                loss=0.5,
                grad_weights=[np.zeros((4, 1, 3, 3), dtype=np.float32)],
                grad_biases=[np.zeros(4, dtype=np.float32)],
                m_samples=4,
            )
            lsn = ledger.push_step_complete(
                step_id=1,
                base_version=0,
                batch_ref=ref,
                lr=0.01,
                m_samples=4,
                result=result,
                version=1,
                optimizer_t=1,
                train_loss=0.5,
            )
            assert lsn == 1
            store.flush()
            types = [d.doc_type for d in store.scan()]
            assert STEP_COMPLETE in types
            assert store.head_lsn() == lsn
        finally:
            _close_store(store)
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
        _close_store(store)
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
        _close_store(store)
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
        _close_store(store)
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
        _close_store(store)
    print("[PASSED] E5: fork freezes parent and copies checkpoint")


def _assert_weights_equal(model_a, model_b) -> None:
    for w_a, w_b in zip(model_a.weights, model_b.weights):
        assert np.allclose(w_a, w_b), "weight mismatch after restore"
    for b_a, b_b in zip(model_a.biases, model_b.biases):
        assert np.allclose(b_a, b_b), "bias mismatch after restore"


def test_e6_es_ledger_restore_matches_best_snapshot():
    """E6′: ES rollback via ledger matches weights captured at best epoch."""
    rng = np.random.default_rng(17)
    lr = 0.01
    with tempfile.TemporaryDirectory() as tmp:
        model = _tiny_cnn()
        session = TrainingSession(model=model, data_provider=None, initial_lr=lr)
        engine = create_training_engine(
            tmp,
            session=session,
            config=LedgerConfig(checkpoint_every_steps=100, checkpoint_on_local_best=True),
        )
        session.engine = engine

        X, y = _batch(rng)
        ref = BatchRef.new(epoch=0, batch_idx=0)
        engine.run_step(StepInput(X=X, y=y, batch_ref=ref, lr=lr))

        es_state: dict = {
            "best_val_loss": float("inf"),
            "best_epoch": 0,
            "patience_counter": 0,
            "weights": None,
            "biases": None,
            "best_version": None,
        }
        session._handle_early_stopping(
            epoch=0,
            current_val_raw_cost=0.2,
            min_delta=1e-4,
            patience=1,
            es_state=es_state,
            current_val_loss=0.2,
        )
        assert es_state["best_version"] is not None

        best_snapshot = _tiny_cnn()
        _clone_weights(model, best_snapshot)
        best_opt_t = model.optimizer.t

        for i in range(3):
            X, y = _batch(rng)
            engine.run_step(StepInput(X=X, y=y, batch_ref=BatchRef.new(epoch=1, batch_idx=i), lr=lr))

        stopped = session._handle_early_stopping(
            epoch=2,
            current_val_raw_cost=0.9,
            min_delta=1e-4,
            patience=1,
            es_state=es_state,
            current_val_loss=0.9,
        )
        assert stopped is True
        _assert_weights_equal(model, best_snapshot)
        assert model.optimizer.t == best_opt_t
        _close_store(engine.ledger.store)
    print("[PASSED] E6: ES ledger restore matches best-epoch snapshot")


def test_e7_sync_ledger_step_id_order():
    """Sync path: step_id and version monotonic on ledger tape."""
    rng = np.random.default_rng(31)
    lr = 0.01
    n_steps = 4
    batches = [_batch(rng) for _ in range(n_steps)]

    with tempfile.TemporaryDirectory() as tmp:
        model = _tiny_cnn()
        session = TrainingSession(model=model, data_provider=None, initial_lr=lr)
        store = FileLedgerStore(tmp)
        ledger = TrainingLedger(store=store, branch_id="main", architecture_id="cnn_v1")
        engine = TrainingEngine(
            session=session,
            ledger=ledger,
            config=LedgerConfig(checkpoint_every_steps=100, checkpoint_on_local_best=False),
        )

        class _Provider:
            def __init__(self, items):
                self._items = items
                self._i = 0
                self.batch_size = 4

            def reset_epoch(self):
                self._i = 0

            def has_more_batches(self):
                return self._i < len(self._items)

            def next_batch(self):
                X, y = self._items[self._i]
                self._i += 1
                return X, y

            def normalize(self, x):
                return x

        session.engine = engine
        session.data_provider = _Provider(batches)
        session._run_epoch_training_pass(active_lr=lr, steps=999, epoch=0, is_classification=True)

        cmds = [d for d in store.scan() if d.doc_type == STEP_COMPLETE]
        versions = [d.body["consolidated"]["version"] for d in cmds]
        step_ids = [d.body["command"]["step_id"] for d in cmds]

        assert step_ids == list(range(1, n_steps + 1))
        assert versions == list(range(1, n_steps + 1))
        assert ledger.version == n_steps
        _close_store(store)
    print("[PASSED] E7: sync ledger step_id/version order")


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
            max_epochs=1,
        )

        from src.ledger import FileLedgerStore

        store = FileLedgerStore(os.path.join(ledger_root, "branch_main"))
        assert store.head_lsn() > 0
        journal_types = {d.doc_type for d in store.scan()}
        assert STEP_COMPLETE in journal_types
        _close_store(store)
        print("[PASSED] E6-prime: fit with ledger writes step documents")
    finally:
        shutil.rmtree(ledger_root, ignore_errors=True)


def test_streaming_store_push_defers_encode_until_flush():
    """push() queues docs; encode runs on begin_flush(), not push()."""
    from src.ledger_store import StreamingFileLedgerStore

    doc = LedgerDocument(
        doc_type=STEP_COMMAND,
        branch_id="main",
        model_instance_id="cnn-0",
        architecture_id="cnn_v1",
        body={
            "step_id": 1,
            "base_version": 0,
            "batch_ref": BatchRef.new().to_dict(),
            "lr": 0.01,
            "m_samples": 4,
            "scheduler_epoch": 0,
        },
    )
    with tempfile.TemporaryDirectory() as tmp:
        store = StreamingFileLedgerStore(tmp)
        try:
            with mock.patch("src.ledger_store.document_to_bytes", wraps=document_to_bytes) as encode:
                lsn = store.push(doc)
                assert lsn == 1
                assert len(store._queue) == 1
                assert isinstance(store._queue[0][1], LedgerDocument)
                encode.assert_not_called()

                assert store.begin_flush() is True
                assert len(store._queue) == 0
                encode.assert_called_once()
                store.flush()
                assert store.journal_path.stat().st_size > 0
        finally:
            _close_store(store)
    print("[PASSED] streaming store: encode deferred to begin_flush")


class _FlushTrackingStore:
    """Wraps StreamingFileLedgerStore; records flush tick invocations."""

    def __init__(self, root: str):
        from src.ledger_store import StreamingFileLedgerStore

        self._inner = StreamingFileLedgerStore(root)
        self.tick_calls = 0
        self.begin_flush_calls = 0

    def push(self, doc):
        return self._inner.push(doc)

    def try_reap_flush(self) -> bool:
        self.tick_calls += 1
        return self._inner.try_reap_flush()

    def begin_flush(self) -> bool:
        self.begin_flush_calls += 1
        return self._inner.begin_flush()

    def has_flush_pending(self) -> bool:
        return self._inner.has_flush_pending()

    def queue_pending(self) -> bool:
        return self._inner.queue_pending()

    def flush(self) -> None:
        self._inner.flush()

    def close(self) -> None:
        self._inner.close()

    def put_checkpoint(self, doc) -> None:
        return self._inner.put_checkpoint(doc)

    def get_checkpoint(self, branch_id: str, version: int):
        return self._inner.get_checkpoint(branch_id, version)

    def scan(self, from_lsn: int = 1, to_lsn=None):
        return self._inner.scan(from_lsn, to_lsn)

    def head_lsn(self) -> int:
        return self._inner.head_lsn()


def test_engine_flush_tick_in_run_step():
    """Hot path: reap prior flush at step start; begin flush at step end."""
    rng = np.random.default_rng(44)
    lr = 0.01
    X, y = _batch(rng)

    with tempfile.TemporaryDirectory() as tmp:
        store = _FlushTrackingStore(tmp)
        model = _tiny_cnn()
        session = TrainingSession(model=model, data_provider=None, initial_lr=lr)
        ledger = TrainingLedger(
            store=store,
            branch_id="main",
            architecture_id="cnn_v1",
        )
        engine = TrainingEngine(
            session=session,
            ledger=ledger,
            config=LedgerConfig(checkpoint_every_steps=100, checkpoint_on_local_best=False),
        )
        engine.run_step(StepInput(X=X, y=y, batch_ref=BatchRef.new(), lr=lr))
        assert store.tick_calls >= 1
        assert store.begin_flush_calls >= 1
        _close_store(store)
    print("[PASSED] engine: flush tick at step start, begin_flush at step end")


PHASE_E_TESTS = [
    test_e1_batch_ref_uuid_unique,
    test_e1_document_round_trip,
    test_e2_train_step_result_to_bytes,
    test_e4_file_ledger_store_push_scan,
    test_e4_checkpoint_put_get,
    test_e3_engine_matches_train_and_apply,
    test_e5_replay_from_checkpoint,
    test_e5_fork_freezes_parent,
    test_e6_es_ledger_restore_matches_best_snapshot,
    test_e7_sync_ledger_step_id_order,
    test_e6_prime_fit_with_ledger_and_early_stop_rollback,
    test_streaming_store_push_defers_encode_until_flush,
    test_engine_flush_tick_in_run_step,
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
