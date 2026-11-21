# testing/test_lists_loader.py
"""Data design A1b (§9 row 1, 03ec5b3): parts from a lists file, hashes checked, lists hash into the knobs, a restore under another lists file refused."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config_loader import parse_tm_production_config
from config.constants import EngineBackend, ModelType
from config.schema import LedgerSettings
from src.controller import ModelController
from src.data.base_loader import BaseDataLoader
from src.data.in_memory_provider import InMemoryDataProvider
from src.data.lists_loader import ListsCSVLoader
from src.data.tabular_loader import TabularCSVLoader
from src.launcher.payload import build_pipeline_payload
from src.ledger import CHECKPOINT, FileLedgerStore, document_to_bytes
from testing.test_launcher import _spec

N = 120


def _sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _write(folder, seed: int = 3, *, train=None, validation=None, tweak=None) -> str:
    """A data CSV (target last), an encoded matrix (2 numeric + 3 one-hot columns), its mask and the lists file; returns the lists path."""
    r = np.random.default_rng(1)
    num = r.normal(size=(N, 2))
    cat = r.integers(0, 3, size=N)
    y = ((num[:, 0] + (cat == 1)) > 0.3).astype(int)
    csv = os.path.join(folder, "data.csv")
    pd.DataFrame({"a": num[:, 0], "b": num[:, 1], "kind": cat, "target": y}).to_csv(csv, index=False)
    matrix = np.hstack([num, np.eye(3)[cat]])
    np.save(os.path.join(folder, f"matrix_s{seed}.npy"), matrix)
    mask = [{"column": 0, "kind": "numeric", "source": 0, "name": "a", "category": None}, {"column": 1, "kind": "numeric", "source": 1, "name": "b", "category": None}] + [
        {"column": 2 + k, "kind": "onehot", "source": 2, "name": f"kind={k}", "category": k} for k in range(3)]
    json.dump(mask, open(os.path.join(folder, f"mask_s{seed}.json"), "w"))
    perm = np.random.default_rng(seed).permutation(N)
    lists = {"split_seed": seed, "n_rows": N, "n_out": 5, "train": sorted(np.asarray(perm[:80] if train is None else train).tolist()),
             "validation": sorted(np.asarray(perm[80:100] if validation is None else validation).tolist()), "matrix": f"matrix_s{seed}.npy",
             "matrix_sha256": _sha(os.path.join(folder, f"matrix_s{seed}.npy")), "mask": f"mask_s{seed}.json",
             "mask_sha256": _sha(os.path.join(folder, f"mask_s{seed}.json")), "data_csv_sha256": _sha(csv)}
    if tweak:
        tweak(lists)
    path = os.path.join(folder, f"lists_s{seed}.json")
    json.dump(lists, open(path, "w"))
    return path


def test_the_parts_are_the_listed_rows_in_the_listed_order_and_nothing_is_cut(tmp_path) -> None:
    lp = _write(str(tmp_path))
    lists = json.load(open(lp))
    loader = ListsCSVLoader(str(tmp_path / "data.csv"), lp, ModelType.BINARY_CLASSIFICATION, 1)
    Xt, yt, Xv, yv = loader.load_splits()
    matrix = np.load(tmp_path / "matrix_s3.npy").astype(np.float32)
    df = pd.read_csv(tmp_path / "data.csv")
    np.testing.assert_array_equal(Xt, matrix[lists["train"]])
    np.testing.assert_array_equal(Xv, matrix[lists["validation"]])
    np.testing.assert_array_equal(yt.ravel(), df["target"].to_numpy()[lists["train"]])
    np.testing.assert_array_equal(yv.ravel(), df["target"].to_numpy()[lists["validation"]])
    assert (loader.n_train, loader.n_validation, loader.n_unused) == (80, 20, 20) and loader.random_state == 3
    assert loader.lists_hash == _sha(lp) and [c["kind"] for c in loader.output_mask] == ["numeric"] * 2 + ["onehot"] * 3


def test_a_file_that_does_not_match_its_hash_is_refused(tmp_path) -> None:
    lp = _write(str(tmp_path))
    for victim in ("matrix_s3.npy", "mask_s3.json", "data.csv"):
        path = tmp_path / victim
        orig = path.read_bytes()
        path.write_bytes(orig + b"\n" if victim != "matrix_s3.npy" else orig[:-1] + bytes([orig[-1] ^ 1]))
        with pytest.raises(ValueError, match="does not match"):
            ListsCSVLoader(str(tmp_path / "data.csv"), lp, ModelType.BINARY_CLASSIFICATION, 1)
        path.write_bytes(orig)


def test_overlapping_or_out_of_range_lists_are_refused(tmp_path) -> None:
    for kw, msg in (({"train": list(range(0, 50)), "validation": list(range(40, 60))}, "both"), ({"validation": [5, N]}, "outside")):
        lp = _write(str(tmp_path), **kw)
        with pytest.raises(ValueError, match=msg):
            ListsCSVLoader(str(tmp_path / "data.csv"), lp, ModelType.BINARY_CLASSIFICATION, 1).load_splits()


def test_the_config_key_builds_the_lists_loader_and_without_it_the_tabular_one_as_before(tmp_path) -> None:
    lp = _write(str(tmp_path))
    plain = build_pipeline_payload(_spec(tmp_path), work_dir=tmp_path / "w")
    assert "lists_path" not in plain["ingestion"]["splits"]
    assert type(BaseDataLoader.create_loader(parse_tm_production_config(plain, profile="pipeline"))) is TabularCSVLoader
    spec = _spec(tmp_path, lists_path=lp)
    payload = build_pipeline_payload(spec, work_dir=tmp_path / "w")
    payload["ingestion"]["data_file_path"] = str(tmp_path / "data.csv")
    cfg = parse_tm_production_config(payload, profile="pipeline")
    assert cfg.ingestion.splits.lists_path == lp and isinstance(BaseDataLoader.create_loader(cfg), ListsCSVLoader)


def _fit(tmp: str, lists_path: str, restore: str | None = None, epochs: int = 3):
    loader = ListsCSVLoader(os.path.join(os.path.dirname(lists_path), "data.csv"), lists_path, ModelType.BINARY_CLASSIFICATION, 1)
    provider = InMemoryDataProvider(loader=loader, batch_size=16, epochs=epochs, normalize_features=True)
    ctl = ModelController(data_provider=provider, learning_rate=0.05)
    ctl.initialize_network_from_dimensions(input_dim=5, output_dim=1, model_type=ModelType.BINARY_CLASSIFICATION, hidden_layers=[4], optimizer_name="adam", backend=EngineBackend.NUMPY)
    ctl.fit(steps=provider.recomment_steps(), source_mode=None, model_type=ModelType.BINARY_CLASSIFICATION, early_stopping_enabled=False, patience=3, min_delta=1e-3,
            ledger_settings=LedgerSettings(enabled=True, path="ledger", checkpoint_every_steps=5, restore_checkpoint_path=restore, run_config={"optimization": {"seed": 1}}),
            output_dir=tmp, training_manager=None, model_id="a1b")
    return [d for d in FileLedgerStore(os.path.join(tmp, "ledger")).scan(1) if d.doc_type == CHECKPOINT and d.version]


def test_the_lists_hash_and_the_parts_are_in_the_knobs_and_a_restore_under_another_lists_file_is_refused(tmp_path) -> None:
    a, b, c = (tmp_path / n for n in "abc")
    for d in (a, b, c):
        d.mkdir()
    lp_a = _write(str(a), seed=3)
    docs = _fit(str(tmp_path / "out1"), lp_a)
    knobs = docs[-1].body["knobs"]
    assert knobs["lists_hash"] == _sha(lp_a) and knobs["split_seed"] == 3
    assert (knobs["n_train"], knobs["n_validation"], knobs["n_unused"]) == (80, 20, 20)
    ckpt = str(tmp_path / "stop.ckpt")
    open(ckpt, "wb").write(document_to_bytes(docs[0]))
    _fit(str(tmp_path / "out2"), lp_a, restore=ckpt, epochs=1)                # the same lists file restores
    lp_b = _write(str(b), seed=3, train=list(range(0, 80)), validation=list(range(80, 100)))      # same split seed, other rows
    with pytest.raises(ValueError, match="lists_hash"):
        _fit(str(tmp_path / "out3"), lp_b, restore=ckpt, epochs=1)
    lp_c = _write(str(c), seed=4)
    with pytest.raises(ValueError, match="split_seed"):
        _fit(str(tmp_path / "out4"), lp_c, restore=ckpt, epochs=1)


def test_the_fraction_cut_reports_the_rows_it_leaves_out(tmp_path) -> None:
    _write(str(tmp_path))
    loader = TabularCSVLoader(str(tmp_path / "data.csv"), ["a", "b"], 0.7, 0.15, ModelType.BINARY_CLASSIFICATION, 1)
    assert loader.n_discarded is None
    Xt, _, Xv, _ = loader.load_splits()
    assert loader.n_discarded == N - len(Xt) - len(Xv) == 18
