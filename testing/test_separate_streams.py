# testing/test_separate_streams.py
"""Data design A4 (§9 line 20): separate random streams for the plain network's starting weights and its batch order.
With neither ``init_seed`` nor ``batch_seed`` set, nothing changes: one global stream seeded by ``seed``, as for every stored run."""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config_loader import parse_tm_production_config
from config.constants import EngineBackend, ModelType
from config.schema import LedgerSettings
from src.checkpoint_knobs import knobs_from_config
from src.controller import ModelController
from src.data.in_memory_provider import InMemoryDataProvider
from src.data.tabular_loader import TabularCSVLoader
from src.ledger import CHECKPOINT, STEP_METRICS, FileLedgerStore
from utils.seeding import batch_rng, init_rng, seed_everything


def _fit(tmp: str, seed, init_seed=None, batch_seed=None, p_dropout=0.0):
    """One small plain-network fit; returns (the first epoch's batch order, the starting weights, the (step, train, val) losses)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    csv = os.path.join(tmp, "d.csv")
    pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "target": y}).to_csv(csv, index=False)
    if init_seed is None and batch_seed is None:
        seed_everything(seed)
    else:
        seed_everything(seed, init_seed, batch_seed)
    loader = TabularCSVLoader(csv, ["f1", "f2"], 0.7, 0.15, ModelType.BINARY_CLASSIFICATION, 1)
    provider = InMemoryDataProvider(loader=loader, batch_size=16, epochs=3, normalize_features=True)
    ctl = ModelController(data_provider=provider, learning_rate=0.05)
    ctl.initialize_network_from_dimensions(input_dim=2, output_dim=1, model_type=ModelType.BINARY_CLASSIFICATION, hidden_layers=[8],
                                           optimizer_name="adam", backend=EngineBackend.NUMPY, p_dropout=p_dropout)
    order = provider.indices.copy()
    w0 = [w.copy() for w in ctl.model.weights]
    ctl.fit(steps=provider.recomment_steps(), source_mode=None, model_type=ModelType.BINARY_CLASSIFICATION, early_stopping_enabled=False,
            patience=3, min_delta=1e-3, ledger_settings=LedgerSettings(enabled=True, path="ledger", checkpoint_every_steps=25),
            output_dir=tmp, training_manager=None, model_id="a4")
    docs = list(FileLedgerStore(os.path.join(tmp, "ledger")).scan(1))
    losses = [(d.body.get("step_id"), d.body.get("train_loss"), d.body.get("val_loss")) for d in docs if d.doc_type == STEP_METRICS]
    return order, w0, losses


def _run(**kw):
    with tempfile.TemporaryDirectory() as tmp:
        return _fit(tmp, **kw)


def _digest(parts) -> str:
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:16]


def test_with_neither_seed_the_streams_are_numpys_global_one() -> None:
    seed_everything(5)
    assert init_rng() is np.random and batch_rng() is np.random
    seed_everything(5, init_seed=1)
    assert init_rng() is not np.random and batch_rng() is not np.random
    seed_everything(5)  # a later plain run goes back to the legacy path, whatever the run before set
    assert init_rng() is np.random and batch_rng() is np.random


def test_a_stream_given_its_seed_draws_what_the_global_stream_draws_after_that_seed() -> None:
    seed_everything(0, init_seed=9, batch_seed=4)
    a, b = init_rng().uniform(-1, 1, 5), batch_rng().rand(5)
    np.random.seed(9)
    c = np.random.uniform(-1, 1, 5)
    np.random.seed(4)
    d = np.random.rand(5)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)


def test_a_stream_not_given_falls_back_to_seed() -> None:
    seed_everything(3, init_seed=8)
    a = batch_rng().rand(4)
    np.random.seed(3)
    np.testing.assert_array_equal(a, np.random.rand(4))
    seed_everything(3, batch_seed=8)
    a = init_rng().rand(4)
    np.random.seed(3)
    np.testing.assert_array_equal(a, np.random.rand(4))


def test_the_init_seed_changes_the_weights_and_not_the_batch_order_and_the_batch_seed_the_reverse() -> None:
    base = _run(seed=1, init_seed=10, batch_seed=20)
    other_init = _run(seed=1, init_seed=11, batch_seed=20)
    other_batch = _run(seed=1, init_seed=10, batch_seed=21)
    again = _run(seed=1, init_seed=10, batch_seed=20)
    assert _digest(again) == _digest(base)                                  # the same two seeds, the same fit
    assert np.array_equal(base[0], other_init[0]) and not all(np.array_equal(a, b) for a, b in zip(base[1], other_init[1]))
    assert not np.array_equal(base[0], other_batch[0]) and all(np.array_equal(a, b) for a, b in zip(base[1], other_batch[1]))


def test_the_old_seed_does_not_move_either_stream_once_both_are_given() -> None:
    assert _digest(_run(seed=1, init_seed=10, batch_seed=20)) == _digest(_run(seed=2, init_seed=10, batch_seed=20))


def test_dropout_masks_ride_the_batch_stream() -> None:
    a = _run(seed=1, init_seed=10, batch_seed=20, p_dropout=0.3)
    b = _run(seed=2, init_seed=10, batch_seed=20, p_dropout=0.3)
    c = _run(seed=1, init_seed=10, batch_seed=21, p_dropout=0.3)
    assert a[2] == b[2] and a[2] != c[2]


def test_the_options_reach_the_payload_the_config_and_the_knobs_only_when_set(tmp_path) -> None:
    from src.launcher.payload import build_pipeline_payload
    from testing.test_launcher import _spec

    plain = build_pipeline_payload(_spec(tmp_path), work_dir=tmp_path / "w")
    assert "init_seed" not in plain["optimization"] and "batch_seed" not in plain["optimization"]
    assert "init_seed" not in knobs_from_config(plain) and "batch_seed" not in knobs_from_config(plain)
    payload = build_pipeline_payload(_spec(tmp_path, init_seed=4, batch_seed=6), work_dir=tmp_path / "w")
    cfg = parse_tm_production_config(payload, profile="pipeline")
    assert (cfg.optimization.init_seed, cfg.optimization.batch_seed) == (4, 6)
    assert parse_tm_production_config(plain, profile="pipeline").optimization.init_seed is None
    knobs = knobs_from_config(payload)
    assert knobs["init_seed"] == 4 and knobs["batch_seed"] == 6
    later = build_pipeline_payload(_spec(tmp_path, init_seed=4, batch_seed=6, stretch_index=3), work_dir=tmp_path / "w")
    assert later["optimization"]["init_seed"] == 4 and later["optimization"]["batch_seed"] == 8      # a retry shuffles anew from the same init stream
