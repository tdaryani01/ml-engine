# testing/test_run_end_document.py
"""Fit contract: ME writes an explicit ``run.end`` ledger document when a fit ends.

EE (and anything else that tails the ledger) needs to know HOW a fit ended without
parsing logs or watching the process: early stop (``es_trip``) or the epoch budget
(``success``), plus the best version/val and the epochs actually run.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import EngineBackend, ModelType
from config.schema import LedgerSettings
from src.controller import ModelController
from src.data.in_memory_provider import InMemoryDataProvider
from src.data.tabular_loader import TabularCSVLoader
from src.ledger import RUN_END, FileLedgerStore


def _fit(tmp: str, *, early_stopping: bool, patience: int, epochs: int, noise: float, restore: str | None = None):
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 2))
    y = ((X[:, 0] + X[:, 1] + rng.normal(0, noise, size=n)) > 0).astype(int)
    csv = os.path.join(tmp, "d.csv")
    pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "target": y}).to_csv(csv, index=False)
    loader = TabularCSVLoader(csv, ["f1", "f2"], 0.7, 0.15, ModelType.BINARY_CLASSIFICATION, 1)
    provider = InMemoryDataProvider(loader=loader, batch_size=16, epochs=epochs, normalize_features=True)
    ctl = ModelController(data_provider=provider, learning_rate=0.05)
    ctl.initialize_network_from_dimensions(
        input_dim=2, output_dim=1, model_type=ModelType.BINARY_CLASSIFICATION,
        hidden_layers=[8], optimizer_name="adam", backend=EngineBackend.NUMPY,
    )
    ctl.fit(
        steps=provider.recomment_steps(),
        source_mode=None,
        model_type=ModelType.BINARY_CLASSIFICATION,
        early_stopping_enabled=early_stopping,
        patience=patience,
        min_delta=1e-3,
        ledger_settings=LedgerSettings(
            enabled=True, path="ledger", checkpoint_every_steps=25, restore_checkpoint_path=restore
        ),
        output_dir=tmp,
        training_manager=None,
        model_id="run-end-test",
    )
    store = FileLedgerStore(os.path.join(tmp, "ledger"))
    return [d for d in store.scan(1) if d.doc_type == RUN_END]


def test_budget_exhausted_writes_run_end_success() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        docs = _fit(tmp, early_stopping=False, patience=3, epochs=3, noise=0.1)
    assert len(docs) == 1
    body = docs[0].body
    assert body["reason"] == "success"
    assert int(body["epochs_run"]) == 3
    # The final model is recorded as a checkpoint so a ledger reader can announce it.
    assert body["final_version"] is not None


def test_early_stop_writes_run_end_es_trip_with_best_version() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # Very noisy labels + large epoch budget: validation stops improving quickly.
        docs = _fit(tmp, early_stopping=True, patience=2, epochs=200, noise=3.0)
    assert len(docs) == 1
    body = docs[0].body
    assert body["reason"] == "es_trip"
    assert body["best_version"] is not None
    assert body["best_val_loss"] is not None
    assert int(body["epochs_run"]) < 200


# --- fit contract: start a fit from a given checkpoint document ---------------------------


def test_fit_can_start_from_a_checkpoint_document_with_weights_and_adam_state() -> None:
    from src.ledger import CHECKPOINT, document_from_bytes, document_to_bytes

    with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
        _fit(tmp_a, early_stopping=False, patience=3, epochs=3, noise=0.1)
        store = FileLedgerStore(os.path.join(tmp_a, "ledger"))
        cps = [d for d in store.scan(1) if d.doc_type == CHECKPOINT and d.version and d.version > 0]
        assert cps, "first fit must have produced a non-zero checkpoint"
        src_doc = cps[-1]
        ckpt_path = os.path.join(tmp_b, "restore.ckpt")
        with open(ckpt_path, "wb") as fh:
            fh.write(document_to_bytes(src_doc))

        _fit(tmp_b, early_stopping=False, patience=3, epochs=1, noise=0.1, restore=ckpt_path)
        store_b = FileLedgerStore(os.path.join(tmp_b, "ledger"))
        v0 = [d for d in store_b.scan(1) if d.doc_type == CHECKPOINT and d.version == 0]
        assert v0, "second fit must record its starting weights as checkpoint v0"
        for a, b in zip(src_doc.body["weights"], v0[0].body["weights"]):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b))
        # ME clears Adam tracking vectors at the start of every fit (begin_fit): a restore brings
        # back the WEIGHTS; the optimizer restarts. Documented fit-contract behavior for this family.
        assert int(v0[0].body["optimizer"]["t"]) == 0
        _ = document_from_bytes


def test_every_epoch_records_train_and_validation_loss_in_the_ledger() -> None:
    """The dashboards need a validation curve; ME computes val once per epoch, so it must land in the ledger."""
    from src.ledger import STEP_METRICS

    with tempfile.TemporaryDirectory() as tmp:
        _fit(tmp, early_stopping=False, patience=3, epochs=4, noise=0.5)
        store = FileLedgerStore(os.path.join(tmp, "ledger"))
        docs = [d for d in store.scan(1) if d.doc_type == STEP_METRICS]
    assert len(docs) == 4
    for d in docs:
        assert d.body["val_loss"] is not None
        assert d.body["train_loss"] is not None


def test_a_budget_ended_fit_records_a_final_checkpoint_at_the_final_version() -> None:
    from src.ledger import CHECKPOINT

    with tempfile.TemporaryDirectory() as tmp:
        docs = _fit(tmp, early_stopping=False, patience=3, epochs=2, noise=0.1)
        store = FileLedgerStore(os.path.join(tmp, "ledger"))
        cps = {d.version for d in store.scan(1) if d.doc_type == CHECKPOINT}
    assert docs[0].body["final_version"] in cps
