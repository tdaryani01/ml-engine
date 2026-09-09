# testing/test_session_isolation.py
"""Multi-tenant isolation: one session per model instance; no stage cross-talk."""
from __future__ import annotations

import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import EngineBackend
from src.model_factory import ModelFactory
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


def _make_model(seed: int = 42):
    np.random.seed(seed)
    return ModelFactory.create_model(
        model_type="cnn",
        layer_sizes=[4],
        backend=EngineBackend.NATIVE,
        optimizer="adam",
        cnn_config=_tiny_cnn_config(),
        lam_l1=0.0,
        lam_l2=0.0,
        max_norm=1e9,
    )


def _batch(seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((8, 1, 28, 32), dtype=np.float32)
    y = np.zeros((8, 4), dtype=np.float32)
    y[np.arange(8), rng.integers(0, 4, size=8)] = 1.0
    return X, y


def test_reject_same_model_dual_session():
    model = _make_model(seed=1)
    _ = TrainingSession(model=model, data_provider=None, initial_lr=0.01)
    try:
        TrainingSession(model=model, data_provider=None, initial_lr=0.01)
        raise AssertionError("expected RuntimeError for shared model object")
    except RuntimeError as e:
        assert "already owned" in str(e)
    print("[PASSED] isolation: same model object rejected by second session")


def test_sequential_contract_then_other_model_sync():
    """Contract on A must not poison sync train on independent model B."""
    lib = _load_conv_dll()
    if lib is None or not hasattr(lib, "run_contract_training_step"):
        print("[SKIPPED] sequential isolation: rebuild native")
        return

    X, y = _batch(11)
    lr = 0.01

    # Baseline: fresh model B' sync loss
    base = _make_model(seed=99)
    base_loss = TrainingSession(model=base, data_provider=None, initial_lr=lr).train_step(
        X, y, lr=lr
    ).loss

    # Pollute path: contract step on A, then same seed model B
    a = _make_model(seed=3)
    a.enable_contract_list()
    TrainingSession(model=a, data_provider=None, initial_lr=lr).train_step(X, y, lr=lr)
    if hasattr(a, "_contract_runtime") and a._contract_runtime is not None:
        a._contract_runtime.close()

    b = _make_model(seed=99)
    b_loss = TrainingSession(model=b, data_provider=None, initial_lr=lr).train_step(
        X, y, lr=lr
    ).loss
    assert abs(b_loss - base_loss) < 1e-5, (b_loss, base_loss)
    print("[PASSED] isolation: contract A then sync B matches clean baseline")


def test_parallel_two_model_contract_steps():
    """Two independent contract models can train concurrently without cross-talk."""
    lib = _load_conv_dll()
    if lib is None or not hasattr(lib, "run_contract_training_step"):
        print("[SKIPPED] parallel isolation: rebuild native")
        return
    if not hasattr(lib, "native_tenant_create"):
        print("[SKIPPED] parallel isolation: native_tenant_create missing")
        return

    X, y = _batch(21)
    lr = 0.01

    # Serial baselines
    a0 = _make_model(seed=11)
    a0.enable_contract_list()
    loss_a_ref = TrainingSession(model=a0, data_provider=None, initial_lr=lr).train_step(
        X, y, lr=lr
    ).loss
    a0._contract_runtime.close()

    b0 = _make_model(seed=22)
    b0.enable_contract_list()
    loss_b_ref = TrainingSession(model=b0, data_provider=None, initial_lr=lr).train_step(
        X, y, lr=lr
    ).loss
    b0._contract_runtime.close()

    a = _make_model(seed=11)
    a.enable_contract_list()
    b = _make_model(seed=22)
    b.enable_contract_list()
    sess_a = TrainingSession(model=a, data_provider=None, initial_lr=lr)
    sess_b = TrainingSession(model=b, data_provider=None, initial_lr=lr)
    out: dict[str, float] = {}
    err: list[BaseException] = []

    def _run(name, sess):
        try:
            out[name] = float(sess.train_step(X, y, lr=lr).loss)
        except BaseException as e:
            err.append(e)

    t1 = threading.Thread(target=_run, args=("a", sess_a))
    t2 = threading.Thread(target=_run, args=("b", sess_b))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    if err:
        raise err[0]
    assert abs(out["a"] - loss_a_ref) < 1e-4, (out["a"], loss_a_ref)
    assert abs(out["b"] - loss_b_ref) < 1e-4, (out["b"], loss_b_ref)
    a._contract_runtime.close()
    b._contract_runtime.close()
    print("[PASSED] isolation: parallel contract A+B match serial baselines")


def test_engine_two_sessions_round_robin():
    """Engine keeps two sessions alive; run() advances both; ledger tags session_id."""
    import tempfile

    from config.constants import IngestionMode, ModelType
    from src.ledger import LedgerConfig, TrainingLedger
    from src.ledger_store import FileLedgerStore
    from src.optimizers import AdamOptimizer
    from src.models import MultiClassNetwork
    from src.training_engine import TrainingEngine
    from src.training_session import SessionStatus

    class _Prov:
        def __init__(self, X, y, Xv, yv, batches=2):
            self.X, self.y = X, y
            self.Xv, self.yv = Xv, yv
            self.batches = batches
            self._i = 0
            self.batch_size = X.shape[0]

        def reset_epoch(self):
            self._i = 0

        def has_more_batches(self):
            return self._i < self.batches

        def next_batch(self):
            self._i += 1
            return self.X, self.y

        def normalize(self, X):
            return X

        def get_validation_set(self):
            return self.Xv, self.yv

    rng = np.random.default_rng(3)
    X = rng.standard_normal((8, 4)).astype(np.float64)
    y = np.eye(3)[rng.integers(0, 3, size=8)]
    Xv = rng.standard_normal((4, 4)).astype(np.float64)
    yv = np.eye(3)[rng.integers(0, 3, size=4)]

    def _mlp():
        return MultiClassNetwork(
            layer_sizes=[4, 8, 3],
            optimizer_instance=AdamOptimizer(lr=0.05),
            use_batch_norm=False,
            lam_l1=0.0,
            lam_l2=0.0,
        )

    with tempfile.TemporaryDirectory() as tmp:
        store = FileLedgerStore(tmp)
        ledger = TrainingLedger(store=store, branch_id="main", architecture_id="mlp")
        engine = TrainingEngine(
            ledger=ledger,
            config=LedgerConfig(checkpoint_every_steps=100, checkpoint_on_local_best=False),
        )
        s1 = engine.start_session(
            model=_mlp(),
            data_provider=_Prov(X, y, Xv, yv),
            initial_lr=0.05,
            session_id="sess-a",
            activate=True,
            steps=2,
            source_mode=IngestionMode.CSV,
            model_type=ModelType.MULTI_CLASS,
            early_stopping_enabled=False,
            max_epochs=2,
        )
        s2 = engine.start_session(
            model=_mlp(),
            data_provider=_Prov(X, y, Xv, yv),
            initial_lr=0.05,
            session_id="sess-b",
            activate=False,
            steps=2,
            source_mode=IngestionMode.CSV,
            model_type=ModelType.MULTI_CLASS,
            early_stopping_enabled=False,
            max_epochs=2,
        )
        assert s1.status == SessionStatus.ACTIVE
        assert s2.status == SessionStatus.PENDING
        assert len(engine.sessions) == 2

        results = engine.run(activate_pending=True)
        assert set(results.keys()) == {"sess-a", "sess-b"}
        assert len(results["sess-a"][0]) == 2
        assert len(results["sess-b"][0]) == 2
        assert engine.sessions == []

        tagged_a = list(ledger.scan_session("sess-a"))
        tagged_b = list(ledger.scan_session("sess-b"))
        assert any(d.doc_type == "step.complete" for d in tagged_a)
        assert any(d.doc_type == "step.complete" for d in tagged_b)
        engine.close()
    print("[PASSED] engine: two sessions round-robin + ledger session tags")


def _mlp_prov_bundle(seed: int = 3):
    from src.optimizers import AdamOptimizer
    from src.models import MultiClassNetwork

    class _Prov:
        def __init__(self, X, y, Xv, yv, batches=2):
            self.X, self.y = X, y
            self.Xv, self.yv = Xv, yv
            self.batches = batches
            self._i = 0
            self.batch_size = X.shape[0]

        def reset_epoch(self):
            self._i = 0

        def has_more_batches(self):
            return self._i < self.batches

        def next_batch(self):
            self._i += 1
            return self.X, self.y

        def normalize(self, X):
            return X

        def get_validation_set(self):
            return self.Xv, self.yv

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((8, 4)).astype(np.float64)
    y = np.eye(3)[rng.integers(0, 3, size=8)]
    Xv = rng.standard_normal((4, 4)).astype(np.float64)
    yv = np.eye(3)[rng.integers(0, 3, size=4)]

    def _mlp():
        return MultiClassNetwork(
            layer_sizes=[4, 8, 3],
            optimizer_instance=AdamOptimizer(lr=0.05),
            use_batch_norm=False,
            lam_l1=0.0,
            lam_l2=0.0,
        )

    return _mlp, _Prov(X, y, Xv, yv), X, y


def test_concurrent_engine_sessions_run_step():
    """Two threads each run_step on a distinct session; ledger stays consistent."""
    import tempfile
    import threading

    from src.ledger import LedgerConfig, STEP_COMPLETE, TrainingLedger
    from src.ledger_store import FileLedgerStore
    from src.training_engine import StepInput, TrainingEngine
    from src.ledger import BatchRef

    _mlp, prov, X, y = _mlp_prov_bundle(11)
    with tempfile.TemporaryDirectory() as tmp:
        store = FileLedgerStore(tmp)
        ledger = TrainingLedger(store=store, branch_id="main", architecture_id="mlp")
        engine = TrainingEngine(
            ledger=ledger,
            config=LedgerConfig(checkpoint_every_steps=1, checkpoint_on_local_best=False),
        )
        sa = engine.start_session(
            model=_mlp(), data_provider=prov, initial_lr=0.05, session_id="conc-a"
        )
        sb = engine.start_session(
            model=_mlp(), data_provider=prov, initial_lr=0.05, session_id="conc-b"
        )
        err: list[BaseException] = []
        losses: dict[str, list[float]] = {"conc-a": [], "conc-b": []}

        def _worker(sess, name: str, n: int):
            try:
                for i in range(n):
                    step = StepInput(
                        X=X,
                        y=y,
                        batch_ref=BatchRef.new(epoch=0, batch_idx=i),
                        lr=0.05,
                    )
                    losses[name].append(engine.run_step(step, session=sess))
            except BaseException as e:
                err.append(e)

        t1 = threading.Thread(target=_worker, args=(sa, "conc-a", 4))
        t2 = threading.Thread(target=_worker, args=(sb, "conc-b", 4))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        if err:
            raise err[0]

        assert len(losses["conc-a"]) == 4
        assert len(losses["conc-b"]) == 4
        assert ledger.version == 8

        docs_a = list(ledger.scan_session("conc-a", STEP_COMPLETE))
        docs_b = list(ledger.scan_session("conc-b", STEP_COMPLETE))
        assert len(docs_a) == 4
        assert len(docs_b) == 4
        ids_a = {id(d) for d in docs_a}
        ids_b = {id(d) for d in docs_b}
        assert ids_a.isdisjoint(ids_b)

        versions = sorted(
            int(d.version) for d in list(docs_a) + list(docs_b) if d.version is not None
        )
        assert versions == list(range(1, 9))

        engine.end_session("conc-a", finalize=False)
        engine.finish_session("conc-b", finalize=False)
        assert engine.sessions == []
        engine.close()
    print("[PASSED] concurrent: two sessions run_step + ledger partition")


def test_ledger_multi_session_restore_isolation():
    """Checkpoints/steps tagged per session; restore A does not load B's weights."""
    import tempfile

    from src.ledger import BatchRef, LedgerConfig, TrainingLedger
    from src.ledger_store import FileLedgerStore
    from src.training_engine import StepInput, TrainingEngine

    _mlp, prov, X, y = _mlp_prov_bundle(21)
    with tempfile.TemporaryDirectory() as tmp:
        store = FileLedgerStore(tmp)
        ledger = TrainingLedger(store=store, branch_id="main", architecture_id="mlp")
        engine = TrainingEngine(
            ledger=ledger,
            config=LedgerConfig(checkpoint_every_steps=1, checkpoint_on_local_best=False),
        )
        model_a = _mlp()
        model_b = _mlp()
        sa = engine.start_session(
            model=model_a, data_provider=prov, initial_lr=0.05, session_id="led-a"
        )
        sb = engine.start_session(
            model=model_b, data_provider=prov, initial_lr=0.05, session_id="led-b"
        )

        for i in range(3):
            engine.run_step(
                StepInput(X=X, y=y, batch_ref=BatchRef.new(0, i), lr=0.05),
                session=sa,
            )
            engine.run_step(
                StepInput(X=X, y=y, batch_ref=BatchRef.new(0, i), lr=0.05),
                session=sb,
            )

        snap_a = [w.copy() for w in model_a.weights]
        snap_b = [w.copy() for w in model_b.weights]
        assert not all(np.allclose(a, b) for a, b in zip(snap_a, snap_b))

        head_a = ledger.session_head_version("led-a")
        head_b = ledger.session_head_version("led-b")
        assert head_a > 0 and head_b > 0
        assert ledger.latest_checkpoint_for_session("led-a") is not None
        assert ledger.latest_checkpoint_for_session("led-b") is not None

        # Mutate A, then restore A's checkpoint — must match snap from ledger, not B.
        for w in model_a.weights:
            w += 1.0
        assert not all(np.allclose(a, b) for a, b in zip(model_a.weights, snap_a))
        ledger.restore_session_checkpoint(model_a, "led-a")
        for a, b in zip(model_a.weights, snap_a):
            assert np.allclose(a, b), "restore led-a pulled wrong weights"

        # Restoring A must not change B.
        for a, b in zip(model_b.weights, snap_b):
            assert np.allclose(a, b)

        engine.end_session("led-a", finalize=False)
        engine.end_session("led-b", finalize=False)
        engine.close()
    print("[PASSED] ledger: multi-session restore isolation")


def test_end_and_resume_session():
    """end_session drops; resume_session restores checkpoint and re-registers."""
    import tempfile

    from config.constants import IngestionMode, ModelType
    from src.ledger import BatchRef, LedgerConfig, TrainingLedger
    from src.ledger_store import FileLedgerStore
    from src.training_engine import StepInput, TrainingEngine
    from src.training_session import SessionStatus

    _mlp, prov, X, y = _mlp_prov_bundle(31)
    with tempfile.TemporaryDirectory() as tmp:
        store = FileLedgerStore(tmp)
        ledger = TrainingLedger(store=store, branch_id="main", architecture_id="mlp")
        engine = TrainingEngine(
            ledger=ledger,
            config=LedgerConfig(checkpoint_every_steps=1, checkpoint_on_local_best=False),
        )
        model = _mlp()
        s = engine.start_session(
            model=model,
            data_provider=prov,
            initial_lr=0.05,
            session_id="resume-me",
            activate=True,
            steps=2,
            source_mode=IngestionMode.CSV,
            model_type=ModelType.MULTI_CLASS,
            early_stopping_enabled=False,
            max_epochs=1,
        )
        engine.run_step(
            StepInput(X=X, y=y, batch_ref=BatchRef.new(0, 0), lr=0.05), session=s
        )
        engine.run_step(
            StepInput(X=X, y=y, batch_ref=BatchRef.new(0, 1), lr=0.05), session=s
        )
        snap = [w.copy() for w in model.weights]
        cp_ver = int(ledger.latest_checkpoint_for_session("resume-me").version)

        hist = engine.end_session("resume-me", finalize=False)
        assert hist[0] == [] or isinstance(hist[0], list)
        assert engine.get_session("resume-me") is None
        assert s.status == SessionStatus.FINISHED

        for w in model.weights:
            w *= 0.0
        assert not all(np.allclose(a, b) for a, b in zip(model.weights, snap))

        resumed = engine.resume_session(
            session_id="resume-me",
            model=model,
            data_provider=prov,
            initial_lr=0.05,
            activate=True,
            steps=2,
            source_mode=IngestionMode.CSV,
            model_type=ModelType.MULTI_CLASS,
            early_stopping_enabled=False,
            max_epochs=1,
        )
        assert resumed.session_id == "resume-me"
        assert resumed.status == SessionStatus.ACTIVE
        assert resumed._last_healthy_version == cp_ver
        for a, b in zip(model.weights, snap):
            assert np.allclose(a, b), "resume did not restore checkpoint weights"

        # Continue training after resume without colliding with live registry.
        engine.run_step(
            StepInput(X=X, y=y, batch_ref=BatchRef.new(1, 0), lr=0.05), session=resumed
        )
        engine.finish_session("resume-me", finalize=False)
        engine.close()
    print("[PASSED] end_session + resume_session restore")


if __name__ == "__main__":
    try:
        test_reject_same_model_dual_session()
        test_sequential_contract_then_other_model_sync()
        test_parallel_two_model_contract_steps()
        test_engine_two_sessions_round_robin()
        test_concurrent_engine_sessions_run_step()
        test_ledger_multi_session_restore_isolation()
        test_end_and_resume_session()
    finally:
        from src.contract_runtime import shutdown_contract_async

        shutdown_contract_async()
    print("[SUCCESS] session isolation tests passed")
