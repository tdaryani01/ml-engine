# testing/test_scale_targets.py
"""Data design B15 (backlog df7792d): regression targets standardised with the training part's mean and sd, off by default, the numbers in the checkpoint's knobs."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import EngineBackend, ModelType
from config.schema import LedgerSettings
from src.controller import ModelController
from src.data.in_memory_provider import InMemoryDataProvider
from src.data.tabular_loader import TabularCSVLoader
from src.ledger import CHECKPOINT, STEP_METRICS, FileLedgerStore, document_to_bytes
from src.launcher.payload import build_pipeline_payload
from testing.test_launcher import _spec
from utils.seeding import seed_everything

N = 400


def _csv(tmp: str, scale: float = 1.0, shift: float = 0.0) -> str:
    r = np.random.default_rng(3)
    X = r.normal(size=(N, 3))
    y = (X @ np.array([1.0, -2.0, 0.5]) + 0.3 * r.normal(size=N)) * scale + shift
    path = os.path.join(tmp, f"d_{scale}_{shift}.csv")
    pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "f3": X[:, 2], "target": y}).to_csv(path, index=False)
    return path


def _provider(csv: str, scale_targets: bool, model_type=ModelType.REGRESSION):
    loader = TabularCSVLoader(csv, ["f1", "f2", "f3"], 0.7, 0.15, model_type, 1)
    return InMemoryDataProvider(loader=loader, batch_size=16, epochs=3, normalize_features=True, scale_targets=scale_targets)


def _fit(tmp: str, csv: str, scale_targets: bool, restore: str | None = None):
    seed_everything(7)
    prov = _provider(csv, scale_targets)
    ctl = ModelController(data_provider=prov, learning_rate=0.05)
    ctl.initialize_network_from_dimensions(input_dim=3, output_dim=1, model_type=ModelType.REGRESSION, hidden_layers=[6], optimizer_name="adam", backend=EngineBackend.NUMPY)
    ctl.fit(steps=prov.recomment_steps(), source_mode=None, model_type=ModelType.REGRESSION, early_stopping_enabled=False, patience=3, min_delta=1e-3,
            ledger_settings=LedgerSettings(enabled=True, path="ledger", checkpoint_every_steps=10), output_dir=tmp, training_manager=None, model_id="b15")
    led = FileLedgerStore(os.path.join(tmp, "ledger"))
    docs = list(led.scan(1))
    losses = [(d.body["train_loss"], d.body["val_loss"]) for d in docs if d.doc_type == STEP_METRICS]
    return ctl, losses, [d for d in docs if d.doc_type == CHECKPOINT]


def test_the_targets_are_standardised_with_the_training_numbers_and_off_is_the_raw_target() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        csv = _csv(tmp, scale=40.0, shift=100.0)
        raw, on = _provider(csv, False), _provider(csv, True)
        assert raw.y_mean is None and np.array_equal(raw.y_train_processed, on.loader.load_splits()[1]) and raw.y_train_processed.std() > 10
        assert abs(float(on.y_train_processed.mean())) < 1e-6 and abs(float(on.y_train_processed.std()) - 1.0) < 1e-6
        np.testing.assert_allclose(on.y_val_processed, (raw.y_val_processed - on.y_mean) / on.y_std, rtol=1e-5, atol=1e-6)      # validation with the TRAINING part's numbers
        np.testing.assert_allclose(on.y_mean, raw.y_train_processed.mean(axis=0), rtol=1e-6)
        assert on.y_train_processed.dtype == raw.y_train_processed.dtype


def test_the_option_is_for_regression_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        r = np.random.default_rng(3)
        X = r.normal(size=(N, 3))
        path = os.path.join(tmp, "c.csv")
        pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "f3": X[:, 2], "target": (X[:, 0] > 0).astype(int)}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="regression"):
            _provider(path, True, model_type=ModelType.BINARY_CLASSIFICATION)


def test_a_target_multiplied_by_100_gives_the_same_losses_and_weights_with_the_option_on_and_not_off() -> None:
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, tempfile.TemporaryDirectory() as c, tempfile.TemporaryDirectory() as d:
        ctl1, on1, _ = _fit(a, _csv(a), True)
        ctl2, on100, _ = _fit(b, _csv(b, scale=100.0), True)
        np.testing.assert_allclose(on1, on100, rtol=1e-6)                                                      # the same losses, epoch by epoch
        for w1, w2 in zip(ctl1.model.weights, ctl2.model.weights):
            np.testing.assert_allclose(w1, w2, rtol=1e-5, atol=1e-7)                                          # and the same weights
        _, off1, _ = _fit(c, _csv(c), False)
        _, off100, _ = _fit(d, _csv(d, scale=100.0), False)
        assert off100[-1][1] > 100 * off1[-1][1]                                                               # control: raw, the multiple moves the loss by about its square


def test_the_mean_predictor_scores_half_the_variance_in_the_engines_regression_cost() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        prov = _provider(_csv(tmp, scale=40.0, shift=100.0), True)
        ctl = ModelController(data_provider=prov, learning_rate=0.05)
        ctl.initialize_network_from_dimensions(input_dim=3, output_dim=1, model_type=ModelType.REGRESSION, hidden_layers=[6], optimizer_name="adam", backend=EngineBackend.NUMPY)
        y = prov.y_val_processed
        cost = float(ctl.model.calculate_raw_cost(np.zeros_like(y), y))      # the mean of the training targets is 0 after scaling: the mean predictor outputs 0
        assert cost == pytest.approx(0.5 * float(np.mean(y ** 2)), rel=1e-6) and 0.35 < cost < 0.65      # the engine's cost is half the squared error: about 0.5, the variance is about 1.0


def _run(tmp: str, csv: str, scale_targets):
    from pathlib import Path

    from src.launcher import SupervisedFitSpec, run_me_fit

    spec = SupervisedFitSpec(model_type="regression", data_path=csv, num_classes=1, hidden_layers=(6,), feature_names=("f1", "f2", "f3"), learning_rate=0.05, batch_size=16, epochs=2, patience=None, seed=7,
                             checkpoint_every=10, num_threads=1, scale_targets=scale_targets)
    snaps = list(run_me_fit(spec, work_dir=Path(tmp) / "w", should_stop=lambda: False))
    return snaps, snaps[-1]["checkpoint_bytes"]


def test_the_numbers_travel_with_the_checkpoint_only_when_set_and_a_restore_across_the_two_is_refused() -> None:
    from src.ledger import document_from_bytes

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        csv = _csv(a, scale=40.0, shift=100.0)
        on_snaps, on_ck = _run(a, csv, True)
        off_snaps, off_ck = _run(b, csv, None)
        k_on, k_off = document_from_bytes(on_ck).body["knobs"], document_from_bytes(off_ck).body["knobs"]
        assert len(k_on["y_mean"]) == 1 and k_on["y_std"][0] > 1.0 and "y_mean" not in k_off and "y_std" not in k_off      # written only when set
        assert on_snaps[-1]["loss"] < 1.0 < off_snaps[-1]["loss"]                                                      # validation loss in scaled units against raw units (sd 40)
        path = os.path.join(a, "ck.bin")
        with open(path, "wb") as fh:
            fh.write(on_ck)

        def ctl(scale):
            c = ModelController(data_provider=_provider(csv, scale), learning_rate=0.05)
            c.initialize_network_from_dimensions(input_dim=3, output_dim=1, model_type=ModelType.REGRESSION, hidden_layers=[6], optimizer_name="adam", backend=EngineBackend.NUMPY)
            return c

        with pytest.raises(ValueError, match="scaled targets"):
            ctl(False).restore_from_checkpoint_doc(path)            # weights in scaled units into a raw-target run
        assert ctl(True).restore_from_checkpoint_doc(path) > 0       # the same numbers: accepted
        off_path = os.path.join(b, "ck.bin")
        with open(off_path, "wb") as fh:
            fh.write(off_ck)
        with pytest.raises(ValueError, match="scaled targets"):
            ctl(True).restore_from_checkpoint_doc(off_path)         # a raw-target checkpoint into a scaled run
        assert ctl(False).restore_from_checkpoint_doc(off_path) > 0


def test_an_old_checkpoint_with_no_knobs_is_not_restored_into_a_scaled_run(monkeypatch) -> None:
    """Units are checked on the presence of y_mean, not on split_seed: a checkpoint from before the knobs carried the split seed is raw-unit weights."""
    import src.ledger as C

    with tempfile.TemporaryDirectory() as a:
        csv = _csv(a, scale=40.0, shift=100.0)
        _, ck = _run(a, csv, None)
        path = os.path.join(a, "old.bin")
        with open(path, "wb") as fh:
            fh.write(ck)
        real = C.document_from_bytes

        def no_knobs(data):
            d = real(data)
            d.body.pop("knobs", None)
            return d

        monkeypatch.setattr(C, "document_from_bytes", no_knobs)
        c = ModelController(data_provider=_provider(csv, True), learning_rate=0.05)
        c.initialize_network_from_dimensions(input_dim=3, output_dim=1, model_type=ModelType.REGRESSION, hidden_layers=[6], optimizer_name="adam", backend=EngineBackend.NUMPY)
        with pytest.raises(ValueError, match="scaled targets"):
            c.restore_from_checkpoint_doc(path)


def test_the_payload_key_is_written_only_when_set(tmp_path) -> None:
    assert "scale_targets" not in build_pipeline_payload(_spec(tmp_path), work_dir=tmp_path / "w")["ingestion"]
    assert build_pipeline_payload(_spec(tmp_path, scale_targets=True), work_dir=tmp_path / "w")["ingestion"]["scale_targets"] is True
