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


def _fit(tmp: str, *, early_stopping: bool, patience: int, epochs: int, noise: float):
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
        ledger_settings=LedgerSettings(enabled=True, path="ledger", checkpoint_every_steps=25),
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
