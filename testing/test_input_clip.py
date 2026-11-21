# testing/test_input_clip.py
"""Data design A2 (§9 row 2, 03ec5b3): standardise with the training part's numbers, clip at 5 sd, nothing after it. Off unless the caller sets it."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config_loader import parse_tm_production_config
from config.constants import EngineBackend, ModelType
from config.schema import LedgerSettings
from src import clip_stats
from src.controller import ModelController
from src.data.in_memory_provider import InMemoryDataProvider
from src.data.tabular_loader import TabularCSVLoader
from src.ledger import STEP_METRICS, FileLedgerStore
from src.launcher.payload import build_pipeline_payload
from testing.test_launcher import _spec

N = 2000


def _plate(tmp: str):
    """3 numeric columns of standard normals; the 70/15 cut is seeded, so the planted rows are found after it by their size."""
    r = np.random.default_rng(5)
    X = r.normal(size=(N, 3))
    X[10, 0] = 30.0        # a 30 sd cell
    X[20, 1] = -30.0
    X[20, 2] = 40.0        # a row with two such cells
    X[1500, 0] = 35.0
    csv = os.path.join(tmp, "d.csv")
    pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "f3": X[:, 2], "target": (X[:, 0] > 0).astype(int)}).to_csv(csv, index=False)
    return csv


def _provider(csv: str, clip_sd=None):
    loader = TabularCSVLoader(csv, ["f1", "f2", "f3"], 0.7, 0.15, ModelType.BINARY_CLASSIFICATION, 1)
    return InMemoryDataProvider(loader=loader, batch_size=32, epochs=2, normalize_features=True, clip_sd=clip_sd)


def test_the_largest_input_the_model_receives_is_5_exactly_for_every_part_and_the_cut_cells_are_counted() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        csv = _plate(tmp)
        before = _provider(csv)
        p = _provider(csv, clip_sd=5.0)      # built last: the counts a provider leaves are the last built provider's
        assert np.abs(before.normalize(np.vstack([before.X_train, before.X_val]))).max() > 5.0      # without the clip the planted cells go beyond (counted below)
        got = [p.normalize(p.X_train), p.normalize(p.X_val)]
        batches = []
        while p.has_more_batches():
            batches.append(p.normalize(p.next_batch()[0]))
        assert max(np.abs(z).max() for z in got + batches) == 5.0
        counts = clip_stats.drain()
        raw = {part: np.abs(before.normalize(X)) > 5.0 for part, X in (("train", before.X_train), ("val", before.X_val))}
        for part in ("train", "val"):
            assert counts[f"clip_cells_{part}"] == raw[part].sum() >= 0
            assert counts[f"clip_rows_multi_{part}"] == (raw[part].sum(axis=1) >= 2).sum()
        assert counts["clip_cells_train"] + counts["clip_cells_val"] >= 3      # the planted cells that fell in these two parts (the rest is in the unused 15%)


def test_the_clip_uses_the_training_numbers_for_validation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = _provider(_plate(tmp), clip_sd=5.0)
        z = p.normalize(p.X_val)
        np.testing.assert_array_equal(z, np.clip((p.X_val - p.mean) / p.std, -5.0, 5.0))


def test_without_the_switch_nothing_is_clipped_and_nothing_is_counted() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = _provider(_plate(tmp))
        np.testing.assert_array_equal(p.normalize(p.X_train), (p.X_train - p.mean) / p.std)
        assert clip_stats.drain() == {}


def test_a_onehot_column_is_not_clipped(tmp_path) -> None:
    from testing.test_onehot_exempt import _provider as lists_provider

    p = lists_provider(str(tmp_path))
    p.clip_sd = 0.5                         # a bound far below any standardised cell: only the numeric columns may move
    z = p.normalize(p.X_train)
    assert np.abs(z[:, :2]).max() == 0.5 and np.array_equal(z[:, 2:], p.X_train[:, 2:])


def test_the_counts_ride_the_first_step_document_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = _provider(_plate(tmp), clip_sd=5.0)
        ctl = ModelController(data_provider=p, learning_rate=0.05)
        ctl.initialize_network_from_dimensions(input_dim=3, output_dim=1, model_type=ModelType.BINARY_CLASSIFICATION, hidden_layers=[4], optimizer_name="adam", backend=EngineBackend.NUMPY)
        ctl.fit(steps=p.recomment_steps(), source_mode=None, model_type=ModelType.BINARY_CLASSIFICATION, early_stopping_enabled=False, patience=3, min_delta=1e-3,
                ledger_settings=LedgerSettings(enabled=True, path="ledger", checkpoint_every_steps=25), output_dir=tmp, training_manager=None, model_id="a2")
        docs = [d for d in FileLedgerStore(os.path.join(tmp, "ledger")).scan(1) if d.doc_type == STEP_METRICS]
    assert len(docs) == 2
    assert "clip_cells_train" in docs[0].body["extra_metrics"] and "clip_cells_train" not in docs[1].body.get("extra_metrics", {})


def test_the_switch_reaches_the_payload_and_the_config_only_when_set(tmp_path) -> None:
    plain = build_pipeline_payload(_spec(tmp_path), work_dir=tmp_path / "w")
    assert "input_clip_sd" not in plain["ingestion"] and parse_tm_production_config(plain, profile="pipeline").ingestion.input_clip_sd is None
    on = build_pipeline_payload(_spec(tmp_path, clip_sd=5.0), work_dir=tmp_path / "w")
    assert parse_tm_production_config(on, profile="pipeline").ingestion.input_clip_sd == 5.0
