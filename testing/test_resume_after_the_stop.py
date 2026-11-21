# testing/test_resume_after_the_stop.py
"""Data design A5 (§9 row 15, 03ec5b3): a run that early stopping ended is continued from the state it stopped in, and gives the losses
the same run with patience off gives over those epochs, bit for bit on one machine. Separate-streams mode only (row 20): each epoch's batch
order and dropout masks come from (batch seed, epoch), so no random position has to be stored. Plain network without batch norm (the
checkpoint holds no batch-norm running statistics)."""
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
from src.ledger import CHECKPOINT, RUN_END, STEP_METRICS, FileLedgerStore, document_to_bytes
from utils.seeding import seed_everything


def _fit(tmp: str, *, epochs: int, es: bool, min_delta: float = 1e-3, patience: int = 3, restore: str | None = None, modes: bool = True, dropout: float = 0.1):
    """One fit; returns (the per-epoch (train, val) losses, the ledger documents)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = ((X[:, 0] + X[:, 1] + rng.normal(0, 0.8, size=300)) > 0).astype(int)
    csv = os.path.join(tmp, "d.csv")
    pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "f3": X[:, 2], "target": y}).to_csv(csv, index=False)
    seed_everything(1, init_seed=11, batch_seed=22) if modes else seed_everything(1)
    loader = TabularCSVLoader(csv, ["f1", "f2", "f3"], 0.7, 0.15, ModelType.BINARY_CLASSIFICATION, 1)
    provider = InMemoryDataProvider(loader=loader, batch_size=16, epochs=epochs, normalize_features=True)
    ctl = ModelController(data_provider=provider, learning_rate=0.05)
    ctl.initialize_network_from_dimensions(input_dim=3, output_dim=1, model_type=ModelType.BINARY_CLASSIFICATION, hidden_layers=[8, 4],
                                           optimizer_name="adam", backend=EngineBackend.NUMPY, p_dropout=dropout)
    ctl.fit(steps=provider.recomment_steps(), source_mode=None, model_type=ModelType.BINARY_CLASSIFICATION, early_stopping_enabled=es,
            patience=patience, min_delta=min_delta,
            ledger_settings=LedgerSettings(enabled=True, path="ledger", checkpoint_every_steps=25, restore_checkpoint_path=restore),
            output_dir=tmp, training_manager=None, model_id="a5")
    docs = list(FileLedgerStore(os.path.join(tmp, "ledger")).scan(1))
    return [(d.body["train_loss"], d.body["val_loss"]) for d in docs if d.doc_type == STEP_METRICS], docs


def _stop_state(docs, tmp: str) -> str:
    states = [d for d in docs if d.doc_type == CHECKPOINT and "continuation" in d.body]
    assert len(states) == 1
    path = os.path.join(tmp, "stop.ckpt")
    with open(path, "wb") as fh:
        fh.write(document_to_bytes(states[0]))
    return path


def _stopped_run(tmp: str):
    losses, docs = _fit(tmp, epochs=12, es=True, min_delta=1.0, patience=2)      # the first epoch sets the best, nothing after improves by 1.0: stops at epoch 2
    assert len(losses) == 3
    return losses, docs


def test_a_run_stopped_on_patience_and_resumed_gives_the_losses_of_the_run_with_patience_off_bit_for_bit() -> None:
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, tempfile.TemporaryDirectory() as c:
        stopped, docs = _stopped_run(a)
        resumed, _ = _fit(b, epochs=4, es=False, restore=_stop_state(docs, a))
        whole, _ = _fit(c, epochs=3 + 4, es=False)
    assert stopped == whole[:3]                      # up to the stop the two are one run
    assert resumed == whole[3:]                      # and after it, bit for bit, with dropout 0.1


def test_the_same_continuation_in_the_legacy_mode_is_not_offered() -> None:
    with tempfile.TemporaryDirectory() as a:
        _, docs = _fit(a, epochs=12, es=True, min_delta=1.0, patience=2, modes=False)
    assert not [d for d in docs if d.doc_type == CHECKPOINT and "continuation" in d.body]      # a default run's ledger gains no document


def test_the_stopped_run_still_ends_on_its_best_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as a:
        _, docs = _stopped_run(a)
    end = [d for d in docs if d.doc_type == RUN_END][0].body
    assert end["reason"] == "es_trip" and end["final_version"] == end["best_version"] and end["best_epoch"] == 0


def test_a_resumed_run_that_never_improves_returns_the_best_from_before_the_stop() -> None:
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _, docs = _stopped_run(a)
        before = [d for d in docs if d.doc_type == RUN_END][0].body
        _, resumed = _fit(b, epochs=5, es=True, min_delta=1.0, patience=50, restore=_stop_state(docs, a))
        end = [d for d in resumed if d.doc_type == RUN_END][0].body
        v0 = [d for d in resumed if d.doc_type == CHECKPOINT and d.version == 0][0]
        best_before = [d for d in docs if d.doc_type == CHECKPOINT and d.version == before["best_version"]][0]
    assert end["best_val_loss"] == before["best_val_loss"] and end["best_epoch"] == before["best_epoch"] == 0
    for x, y in zip(best_before.body["weights"], v0.body["weights"]):
        np.testing.assert_array_equal(x, y)           # the ledger of the resumed run holds the pre-stop best weights


def test_a_resumed_run_that_improves_returns_its_own_best() -> None:
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _, docs = _stopped_run(a)
        before = [d for d in docs if d.doc_type == RUN_END][0].body
        _, resumed = _fit(b, epochs=6, es=True, min_delta=1e-9, patience=50, restore=_stop_state(docs, a))
        end = [d for d in resumed if d.doc_type == RUN_END][0].body
    assert end["best_val_loss"] < before["best_val_loss"] and end["best_epoch"] >= 3      # epochs of the resumed run count from the stop


def test_a_stop_state_checkpoint_is_refused_in_the_legacy_mode() -> None:
    import pytest

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _, docs = _stopped_run(a)
        path = _stop_state(docs, a)
        with pytest.raises(ValueError, match="separate-streams"):
            _fit(b, epochs=4, es=False, restore=path, modes=False)      # one stream: the resumed order would not continue the old one
