# testing/test_checkpoint_config.py
"""Every checkpoint carries the run config and its knobs beside the model state (so Live and restore find them)."""
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
from src.ledger import CHECKPOINT, FileLedgerStore, document_from_bytes, document_to_bytes
from src.ledger_wire import pack_checkpoint_body, unpack_checkpoint_body

CFG = {"optimization": {"learning_rate": 0.05, "seed": 4, "batch_size": 16, "early_stopping_enabled": True, "patience": 3},
       "architecture": {"model_type": "binary_classification"}}


def _body(**extra):
    arr = [np.ones((2, 2), dtype=np.float32)]
    return {"version": 7, "val_loss": 0.5, "is_local_best": True, "weights": arr, "biases": arr, "gammas": None, "betas": None,
            "optimizer": {"type": "Adam", "t": 3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, **{k: None for k in (
                "ms_w", "vs_w", "ms_b", "vs_b", "ms_g", "vs_g", "ms_beta", "vs_beta")}}, **extra}


def test_config_and_knobs_round_trip_through_the_checkpoint_bytes() -> None:
    out = unpack_checkpoint_body(pack_checkpoint_body(_body(config=CFG, knobs={"learning_rate": 0.05})))
    assert out["config"] == CFG and out["knobs"] == {"learning_rate": 0.05} and out["version"] == 7


def test_a_checkpoint_without_config_still_reads() -> None:
    packed = pack_checkpoint_body(_body())
    out = unpack_checkpoint_body(packed)
    assert "config" not in out and out["version"] == 7
    assert "config" not in unpack_checkpoint_body(packed[:-4])  # bytes written before the trailer existed


def _fit(tmp: str, run_config):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    csv = os.path.join(tmp, "d.csv")
    pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "target": y}).to_csv(csv, index=False)
    loader = TabularCSVLoader(csv, ["f1", "f2"], 0.7, 0.15, ModelType.BINARY_CLASSIFICATION, 1)
    provider = InMemoryDataProvider(loader=loader, batch_size=16, epochs=3, normalize_features=True)
    ctl = ModelController(data_provider=provider, learning_rate=0.05)
    ctl.initialize_network_from_dimensions(input_dim=2, output_dim=1, model_type=ModelType.BINARY_CLASSIFICATION,
                                           hidden_layers=[8], optimizer_name="adam", backend=EngineBackend.NUMPY)
    ctl.fit(steps=provider.recomment_steps(), source_mode=None, model_type=ModelType.BINARY_CLASSIFICATION,
            early_stopping_enabled=False, patience=3, min_delta=1e-3,
            ledger_settings=LedgerSettings(enabled=True, path="ledger", checkpoint_every_steps=25, run_config=run_config),
            output_dir=tmp, training_manager=None, model_id="cfg-test")
    return [d for d in FileLedgerStore(os.path.join(tmp, "ledger")).scan(1) if d.doc_type == CHECKPOINT]


def test_a_real_fit_writes_the_config_into_every_checkpoint_and_it_survives_the_document_bytes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        docs = _fit(tmp, CFG)
    assert docs and all(d.body["config"] == CFG for d in docs)
    assert docs[-1].body["knobs"]["learning_rate"] == 0.05 and docs[-1].body["knobs"]["patience"] == 3
    again = document_from_bytes(document_to_bytes(docs[-1]))  # what the Desktop stores and TM later reads
    assert again.body["config"] == CFG and again.body["weights"] is not None


def test_a_fit_given_no_config_writes_checkpoints_without_one() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        docs = _fit(tmp, None)
    assert docs and all("config" not in d.body for d in docs)
