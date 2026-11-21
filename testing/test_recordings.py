# testing/test_recordings.py
"""Data design B7 (backlog 4c46cab): per-epoch recordings for the pilots, off by default, changing no loss and no weight when on."""
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
from src.controller import ModelController
from src.data.in_memory_provider import InMemoryDataProvider
from src.data.tabular_loader import TabularCSVLoader
from src.ledger import STEP_METRICS, FileLedgerStore
from src.launcher.payload import build_pipeline_payload
from testing.test_launcher import _spec
from utils.seeding import seed_everything

CASES = [("binary", ModelType.BINARY_CLASSIFICATION, 2, 1, 0.0, False, 3),
         ("regression", ModelType.REGRESSION, 4, 1, 0.2, False, 11),
         ("multiclass", ModelType.MULTI_CLASS, 3, 3, 0.1, True, 7)]


def _fit(tmp: str, mt, nf, nout, pdrop, bn, seed, *, recordings: bool, lr: float = 0.05, epochs: int = 4, max_norm: float = 5.0, batch_size: int = 16):
    r = np.random.default_rng(100 + seed)
    X = r.normal(size=(300, nf))
    s = X.sum(1)
    y = s + 0.1 * r.normal(size=300) if mt == ModelType.REGRESSION else (np.digitize(s, np.quantile(s, [1 / 3, 2 / 3])) if nout == 3 else (s > 0).astype(int))
    csv = os.path.join(tmp, "d.csv")
    feats = [f"f{i}" for i in range(nf)]
    pd.DataFrame({**{f: X[:, i] for i, f in enumerate(feats)}, "target": y}).to_csv(csv, index=False)
    seed_everything(seed)
    prov = InMemoryDataProvider(loader=TabularCSVLoader(csv, feats, 0.7, 0.15, mt, nout), batch_size=batch_size, epochs=epochs, normalize_features=True)
    ctl = ModelController(data_provider=prov, learning_rate=lr)
    ctl.initialize_network_from_dimensions(input_dim=nf, output_dim=nout, model_type=mt, hidden_layers=[8, 4], optimizer_name="adam", backend=EngineBackend.NUMPY,
                                           p_dropout=pdrop, use_batch_norm=bn, recordings=recordings, max_norm=max_norm)
    ctl.fit(steps=prov.recomment_steps(), source_mode=None, model_type=mt, early_stopping_enabled=False, patience=3, min_delta=1e-3,
            ledger_settings=LedgerSettings(enabled=True, path="ledger", checkpoint_every_steps=25), output_dir=tmp, training_manager=None, model_id="b7")
    docs = [d for d in FileLedgerStore(os.path.join(tmp, "ledger")).scan(1) if d.doc_type == STEP_METRICS]
    wd = hashlib.sha256(b"".join(np.ascontiguousarray(w).tobytes() for w in ctl.model.weights)).hexdigest()
    return docs, wd, ctl


def _numbers(docs):
    return [(d.body["train_loss"], d.body["val_loss"]) for d in docs]


def test_with_the_switch_off_no_document_gains_a_key() -> None:
    for name, mt, nf, nout, pdrop, bn, seed in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            docs, _, _ = _fit(tmp, mt, nf, nout, pdrop, bn, seed, recordings=False)
        assert docs and not any(k.startswith("rec_") for d in docs for k in d.body.get("extra_metrics", {})), name


def test_the_losses_and_the_final_weights_are_the_same_with_the_recordings_on() -> None:
    """Three cases with dropout and batch norm, where a pass that touched the batch stream or the running numbers would show."""
    for name, mt, nf, nout, pdrop, bn, seed in CASES:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            off, w_off, _ = _fit(a, mt, nf, nout, pdrop, bn, seed, recordings=False)
            on, w_on, _ = _fit(b, mt, nf, nout, pdrop, bn, seed, recordings=True)
        assert _numbers(off) == _numbers(on) and w_off == w_on, name


def test_every_epoch_carries_every_field() -> None:
    for name, mt, nf, nout, pdrop, bn, seed in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            docs, _, _ = _fit(tmp, mt, nf, nout, pdrop, bn, seed, recordings=True)
        assert len(docs) == 4
        want = {f"rec_loss_{q}_{p}" for q in ("p10", "p50", "p90", "over3med") for p in ("train", "val")} | {"rec_dead_l0", "rec_dead_l1", "rec_dead_l0_start", "rec_dead_l1_start", "rec_clip_fires", "rec_steps", "rec_grad_agreement"}
        for d in docs:
            assert want <= set(d.body["extra_metrics"]), (name, want - set(d.body["extra_metrics"]))
            m = d.body["extra_metrics"]
            assert m["rec_loss_p10_train"] <= m["rec_loss_p50_train"] <= m["rec_loss_p90_train"] and -1.0 <= m["rec_grad_agreement"] <= 1.0


def test_the_mean_of_the_row_losses_is_the_raw_cost() -> None:
    for name, mt, nf, nout, pdrop, bn, seed in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            _, _, ctl = _fit(tmp, mt, nf, nout, pdrop, bn, seed, recordings=True)
            prov = ctl.data_provider
            X, y = prov.normalize(prov.X_train), prov.y_train_processed
            rows, dead = ctl.model.recording_pass(X, y)
            assert abs(float(np.mean(rows)) - float(ctl.model.calculate_raw_cost(ctl.model.predict(X), y))) < 1e-9, name
            assert len(dead) == 2 and all(0.0 <= s <= 1.0 for s in dead)


def test_a_high_rate_fires_the_clip_and_a_healthy_run_counts_the_same_twice() -> None:
    name, mt, nf, nout, pdrop, bn, seed = CASES[1]
    with tempfile.TemporaryDirectory() as a:
        docs, _, _ = _fit(a, mt, nf, nout, 0.0, False, seed, recordings=True, lr=50.0)
    assert sum(d.body["extra_metrics"]["rec_clip_fires"] for d in docs) > 0
    counts = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as b:
            docs, _, _ = _fit(b, mt, nf, nout, pdrop, bn, seed, recordings=True)
        counts.append([d.body["extra_metrics"]["rec_clip_fires"] for d in docs])
    assert counts[0] == counts[1]


def test_a_dead_unit_is_one_whose_pre_activation_is_not_above_zero_for_every_row() -> None:
    from src.models import RegressionNetwork
    from src.optimizers import AdamOptimizer

    net = RegressionNetwork(layer_sizes=[2, 3, 1], optimizer_instance=AdamOptimizer(lr=0.01), use_batch_norm=False, recordings=True)
    net.weights[0] = np.array([[1.0, -1.0, 0.0], [1.0, -1.0, 0.0]])      # unit 0 alive on positive rows, unit 1 negative on them, unit 2 exactly 0
    net.biases[0] = np.zeros((1, 3))
    _, dead = net.recording_pass(np.array([[1.0, 1.0], [2.0, 1.0]]), np.zeros((2, 1)))
    assert dead == [2 / 3]      # only unit 0 is above 0 on some row: units 1 and 2 are dead -> 2 of 3


def test_the_switch_reaches_the_payload_only_when_set_and_the_cnn_refuses_it(tmp_path) -> None:
    import pytest

    plain = build_pipeline_payload(_spec(tmp_path), work_dir=tmp_path / "w")
    assert "recordings" not in plain["optimization"] and parse_tm_production_config(plain, profile="pipeline").optimization.recordings is False
    on = build_pipeline_payload(_spec(tmp_path, recordings=True), work_dir=tmp_path / "w")
    assert parse_tm_production_config(on, profile="pipeline").optimization.recordings is True
    ctl = ModelController(data_provider=None, learning_rate=0.01)
    with pytest.raises(ValueError, match="plain network only"):
        ctl.initialize_network_from_dimensions(input_dim=16, output_dim=2, model_type=ModelType.CNN, hidden_layers=[4], optimizer_name="adam", backend=EngineBackend.NUMPY,
                                               cnn_config={"input_shape": [1, 4, 4], "spatial_pipeline": [], "dense_head": []}, recordings=True)


def test_the_clip_count_is_recounted_it_never_exceeds_the_steps_and_follows_the_norm() -> None:
    name, mt, nf, nout, pdrop, bn, seed = CASES[0]
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, tempfile.TemporaryDirectory() as c:
        free, _, _ = _fit(a, mt, nf, nout, 0.0, False, seed, recordings=True, max_norm=1e9)      # a clipping norm nothing reaches: no fires
        always, _, _ = _fit(b, mt, nf, nout, 0.0, False, seed, recordings=True, max_norm=1e-9)      # one nothing stays under: every step fires
        normal, _, _ = _fit(c, mt, nf, nout, 0.0, False, seed, recordings=True)
    assert all(d.body["extra_metrics"]["rec_clip_fires"] == 0 for d in free)
    assert all(d.body["extra_metrics"]["rec_clip_fires"] == d.body["extra_metrics"]["rec_steps"] > 0 for d in always)
    assert all(0 <= d.body["extra_metrics"]["rec_clip_fires"] <= d.body["extra_metrics"]["rec_steps"] for d in normal)


def test_an_epoch_of_one_step_still_carries_the_agreement_key() -> None:
    name, mt, nf, nout, pdrop, bn, seed = CASES[0]
    with tempfile.TemporaryDirectory() as tmp:
        docs, _, _ = _fit(tmp, mt, nf, nout, 0.0, False, seed, recordings=True, batch_size=10_000)      # the whole training part in one batch: one step, no pair
    assert docs and all(d.body["extra_metrics"]["rec_steps"] == 1 and d.body["extra_metrics"]["rec_grad_agreement"] == 0.0 for d in docs)


def test_the_dead_unit_sample_is_every_kth_row_and_spans_a_sorted_part() -> None:
    from src import recordings

    rows = np.arange(5000)                                   # a part sorted by time: row i is the i-th oldest
    got = rows[recordings.dead_sample(len(rows))]
    assert len(got) <= recordings.DEAD_SAMPLE and got[0] == 0 and got[-1] >= 5000 - 5 and np.all(np.diff(got) == 5)      # k = ceil(5000 / 1024) = 5: the whole file, not its first block
    part = np.arange(300)
    assert np.array_equal(part[recordings.dead_sample(300)], part)      # a part under the cap is taken whole
