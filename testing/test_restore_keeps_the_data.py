# testing/test_restore_keeps_the_data.py
"""Data design A3 (§9 #4): restoring a checkpoint under a given split seed gives back the same rows and the same scaling numbers.

The checkpoint holds weights, optimizer state and the config; the rows and the scaling numbers are not in it. A restoring run rebuilds them
from the data file and the split seed IT is given (loader cut, then mean and std of the training part: ``in_memory_provider.py``). The first
test is the done-when of backlog A3 as written (three split seeds, restore under the seed the checkpoint was trained under). The second
records what happens today when the restoring run is given another seed: nothing notices. It is a statement of today's behaviour, not a
wish; the fix (record the split seed in the knobs and refuse a mismatch) is owner question 4 of the gaps file and is not built."""
from __future__ import annotations

import hashlib
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
from src.ledger import CHECKPOINT, FileLedgerStore, document_to_bytes

SEEDS = (7, 8, 9)


def _h(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def _fit(tmp: str, split_seed: int, *, epochs: int, restore: str | None = None):
    """One plain-network fit on a fixed plate; returns the provider (rows, scaling) and the ledger store."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    csv = os.path.join(tmp, "d.csv")
    pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "target": y}).to_csv(csv, index=False)
    loader = TabularCSVLoader(csv, ["f1", "f2"], 0.7, 0.15, ModelType.BINARY_CLASSIFICATION, 1, random_state=split_seed)
    provider = InMemoryDataProvider(loader=loader, batch_size=16, epochs=epochs, normalize_features=True)
    ctl = ModelController(data_provider=provider, learning_rate=0.05)
    ctl.initialize_network_from_dimensions(input_dim=2, output_dim=1, model_type=ModelType.BINARY_CLASSIFICATION, hidden_layers=[8],
                                           optimizer_name="adam", backend=EngineBackend.NUMPY)
    ctl.fit(steps=provider.recomment_steps(), source_mode=None, model_type=ModelType.BINARY_CLASSIFICATION, early_stopping_enabled=False,
            patience=3, min_delta=1e-3, ledger_settings=LedgerSettings(enabled=True, path="ledger", checkpoint_every_steps=25, restore_checkpoint_path=restore),
            output_dir=tmp, training_manager=None, model_id="a3")
    return provider, FileLedgerStore(os.path.join(tmp, "ledger"))


def _data_hashes(provider) -> dict:
    return {"train": _h(provider.X_train, provider.y_train_processed), "val": _h(provider.X_val, provider.y_val_processed),
            "scaling": _h(provider.mean, provider.std)}


def _mid_run_checkpoint(store, tmp: str):
    docs = [d for d in store.scan(1) if d.doc_type == CHECKPOINT and d.version and d.version > 0]
    assert len(docs) >= 2, "the first fit must leave a checkpoint before its last"
    path = os.path.join(tmp, "restore.ckpt")
    with open(path, "wb") as fh:
        fh.write(document_to_bytes(docs[len(docs) // 2 - 1]))
    return path, docs[len(docs) // 2 - 1]


def test_a_restore_under_the_split_seed_of_the_checkpoint_gives_the_same_rows_and_scaling_for_three_seeds() -> None:
    seen = []
    for seed in SEEDS:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            before_provider, store = _fit(a, seed, epochs=3)
            ckpt, _ = _mid_run_checkpoint(store, b)
            after_provider, _ = _fit(b, seed, epochs=1, restore=ckpt)
            assert _data_hashes(after_provider) == _data_hashes(before_provider)
            seen.append(_data_hashes(before_provider))
    assert len({s["train"] for s in seen}) == 3 and len({s["scaling"] for s in seen}) == 3      # the three seeds cut three different ways


def test_today_a_restore_under_another_split_seed_is_not_noticed() -> None:
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        before_provider, store = _fit(a, 7, epochs=3)
        ckpt, doc = _mid_run_checkpoint(store, b)
        assert not any(k in doc.body for k in ("mean", "std", "rows", "indices", "split_seed")) and "split_seed" not in (doc.body.get("knobs") or {})
        after_provider, _ = _fit(b, 8, epochs=1, restore=ckpt)      # no error: the weights load into rows and scaling the checkpoint was not trained on
        assert _data_hashes(after_provider)["train"] != _data_hashes(before_provider)["train"]
        assert _data_hashes(after_provider)["scaling"] != _data_hashes(before_provider)["scaling"]
