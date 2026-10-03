# testing/test_validation_loss_matches_prediction.py
"""The validation loss a fit RECORDS must equal the loss of the finished model's end-to-end prediction.

Regression: ``InMemoryDataProvider.get_validation_set()`` z-scored the validation inputs, and the
session then predicted through ``controller.predict`` (which z-scores again), so the validation loss
recorded per epoch (ledger ``step.metrics``, and the number early stopping reads) was computed on
double-normalized inputs. Every consumer of ``get_validation_set()`` (diagnostics: "using RAW
data", the pipeline tests: ``controller.predict(X_val)``) treats it as RAW.
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
from src.ledger import STEP_METRICS, FileLedgerStore


def _fit_and_compare(tmp: str) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    n = 300
    # Features with a non-trivial mean/scale so z-scoring twice is clearly different from once.
    X = rng.normal(loc=[5.0, -3.0], scale=[2.0, 0.5], size=(n, 2))
    y = ((X[:, 0] - 5.0) / 2.0 + (X[:, 1] + 3.0) / 0.5 > 0).astype(int)
    csv = os.path.join(tmp, "d.csv")
    pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "target": y}).to_csv(csv, index=False)
    loader = TabularCSVLoader(csv, ["f1", "f2"], 0.7, 0.15, ModelType.BINARY_CLASSIFICATION, 1)
    provider = InMemoryDataProvider(loader=loader, batch_size=16, epochs=8, normalize_features=True)
    np.random.seed(0)
    ctl = ModelController(data_provider=provider, learning_rate=0.05)
    ctl.initialize_network_from_dimensions(
        input_dim=2, output_dim=1, model_type=ModelType.BINARY_CLASSIFICATION,
        hidden_layers=[16], optimizer_name="adam", backend=EngineBackend.NUMPY,
    )
    ctl.fit(
        steps=provider.recomment_steps(), source_mode=None, model_type=ModelType.BINARY_CLASSIFICATION,
        early_stopping_enabled=False, patience=3, min_delta=1e-4,
        ledger_settings=LedgerSettings(enabled=True, path="ledger", checkpoint_every_steps=1000),
        output_dir=tmp, training_manager=None, model_id="val-check",
    )
    docs = [d for d in FileLedgerStore(os.path.join(tmp, "ledger")).scan(1) if d.doc_type == STEP_METRICS]
    recorded = float(docs[-1].body["val_loss"])
    # The correct end-to-end value: RAW validation features through the normalizing predict.
    x_val_raw = provider.splits["X_val"]
    _, y_val = provider.get_validation_set()
    end_to_end = float(ctl.model.compute_total_loss(ctl.predict(x_val_raw), y_val))
    return recorded, end_to_end


def test_recorded_validation_loss_equals_the_end_to_end_prediction_loss() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        recorded, end_to_end = _fit_and_compare(tmp)
    np.testing.assert_allclose(recorded, end_to_end, rtol=1e-5, atol=1e-8)


def test_get_validation_set_returns_raw_features_for_a_normalizing_predict() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(1)
        X = rng.normal(loc=10.0, scale=3.0, size=(200, 2))
        y = (X[:, 0] > 10).astype(int)
        csv = os.path.join(tmp, "d.csv")
        pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "target": y}).to_csv(csv, index=False)
        loader = TabularCSVLoader(csv, ["f1", "f2"], 0.7, 0.15, ModelType.BINARY_CLASSIFICATION, 1)
        prov = InMemoryDataProvider(loader=loader, batch_size=16, epochs=1, normalize_features=True)
        x_val, _ = prov.get_validation_set()
        np.testing.assert_array_equal(x_val, prov.splits["X_val"])  # raw, as every consumer assumes
