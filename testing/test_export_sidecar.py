# testing/test_export_sidecar.py
"""A fit with the ledger on writes, beside every checkpoint, the archive src/export_pytorch.py reads (a checkpoint alone has no input scaling or batch-norm statistics)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

torch = pytest.importorskip("torch")

from config.constants import EngineBackend, ModelType  # noqa: E402
from config.schema import LedgerSettings  # noqa: E402
from src import export_pytorch as ex  # noqa: E402
from src.controller import ModelController  # noqa: E402
from src.data.in_memory_provider import InMemoryDataProvider  # noqa: E402
from src.data.tabular_loader import TabularCSVLoader  # noqa: E402
from src.ledger import CHECKPOINT, FileLedgerStore, restore_model_checkpoint  # noqa: E402

RUN_CONFIG = {"architecture": {"model_type": "binary_classification", "use_batch_norm": True, "bn_momentum": 0.9},
              "optimization": {"optimizer": "adam", "learning_rate": 0.05, "seed": 4, "batch_size": 16, "early_stopping_enabled": False, "patience": 3}}


def _fit(tmp_path, run_config):
    rng = np.random.default_rng(0)
    X = rng.normal(3.0, 2.0, size=(200, 2))                        # off-centre features, so the provider's mean/std are not 0/1
    y = ((X[:, 0] + X[:, 1]) > 6.0).astype(int)
    csv = str(tmp_path / "d.csv")
    pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "target": y}).to_csv(csv, index=False)
    loader = TabularCSVLoader(csv, ["f1", "f2"], 0.7, 0.15, ModelType.BINARY_CLASSIFICATION, 1)
    provider = InMemoryDataProvider(loader=loader, batch_size=16, epochs=3, normalize_features=True)
    ctl = ModelController(data_provider=provider, learning_rate=0.05)
    ctl.initialize_network_from_dimensions(input_dim=2, output_dim=1, model_type=ModelType.BINARY_CLASSIFICATION, hidden_layers=[8], optimizer_name="adam",
                                           backend=EngineBackend.NUMPY, use_batch_norm=True, bn_momentum=0.9)
    ctl.fit(steps=provider.recomment_steps(), source_mode=None, model_type=ModelType.BINARY_CLASSIFICATION, early_stopping_enabled=False, patience=3, min_delta=1e-3,
            ledger_settings=LedgerSettings(enabled=True, path="ledger", checkpoint_every_steps=5, run_config=run_config), output_dir=str(tmp_path), training_manager=None, model_id="sidecar")
    return ctl, provider, [d for d in FileLedgerStore(str(tmp_path / "ledger")).scan(1) if d.doc_type == CHECKPOINT]


def test_every_checkpoint_gets_an_export_archive_that_exports_to_what_the_checkpoint_predicts(tmp_path):
    ctl, provider, docs = _fit(tmp_path, RUN_CONFIG)
    assert len(docs) >= 2
    ckpt_dir = tmp_path / "ledger" / "checkpoints"
    for doc in docs:
        assert (ckpt_dir / f"main_v{doc.body['version']}.export.npz").is_file()

    # the archive of the last checkpoint, exported, predicts what that checkpoint's weights predict on raw features
    last = docs[-1]
    restore_model_checkpoint(ctl.model, last.body)
    x = np.random.default_rng(3).normal(3.0, 2.0, (30, 2))
    want = ctl.model.predict((x - provider.mean) / provider.std)
    out = tmp_path / "out"
    info = ex.export(str(ckpt_dir / f"main_v{last.body['version']}.export.npz"), str(out))
    assert info["warnings"] == []
    import importlib.util
    spec = importlib.util.spec_from_file_location("exported_sidecar", str(out / "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    got = mod.load(str(out / "weights.pt"))(torch.from_numpy(x.astype(np.float32))).detach().numpy()
    np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-5)
    assert not np.allclose(provider.mean, 0.0)                     # the scaling really was part of what was reproduced


def test_a_fit_without_a_run_config_writes_no_archive_and_does_not_fail(tmp_path):
    _fit(tmp_path, None)
    assert not list((tmp_path / "ledger" / "checkpoints").glob("*.export.npz"))


def test_an_archive_that_cannot_be_written_never_fails_the_fit(tmp_path):
    bad = {"architecture": {"model_type": "binary_classification"}, "optimization": {}}     # save_model needs more than this: it raises
    _, _, docs = _fit(tmp_path, bad)
    assert docs and not list((tmp_path / "ledger" / "checkpoints").glob("*.export.npz"))
