# testing/test_split_seed.py
"""Data design A1 (first half): the seed of the train/validation cut is passed from the config to the tabular loader.
No seed given = 42 = every run so far, so a default run's arrays do not change."""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config_loader import parse_tm_production_config
from src.data.base_loader import BaseDataLoader
from src.data.tabular_loader import TabularCSVLoader
from src.launcher.payload import build_pipeline_payload
from testing.test_launcher import _spec

FEATURES = ["Feature_1", "Feature_2"]
# sha256 prefixes of (X_train, y_train, X_val, y_val) of the loader as it stood before this change (seed 42, no class shares), on the plate below
DEFAULT_HASHES = ["8a390c5545d49ea4", "f78b37daf92c950e", "66f58b70d26f59e6", "f975ae1d1aa3cd6a"]


def _plate(tmp_path):
    r = np.random.default_rng(3)
    n = 500
    df = pd.DataFrame({"Feature_1": r.normal(size=n), "Feature_2": r.normal(size=n), "Target": (r.random(n) < 0.3).astype(float)})
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    return str(path)


def _hashes(parts):
    return [hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16] for a in parts]


def test_a_default_loader_gives_the_arrays_it_gave_before(tmp_path) -> None:
    parts = TabularCSVLoader(_plate(tmp_path), FEATURES).load_splits()
    assert _hashes(parts) == DEFAULT_HASHES
    assert [p.shape for p in parts] == [(350, 2), (350, 1), (75, 2), (75, 1)]      # 70 / 15, the last 15% still left out


def test_the_seed_42_is_the_default_and_another_seed_gives_other_parts_the_same_seed_the_same_parts(tmp_path) -> None:
    plate = _plate(tmp_path)
    load = lambda s: _hashes(TabularCSVLoader(plate, FEATURES, random_state=s).load_splits())      # noqa: E731
    assert load(42) == DEFAULT_HASHES
    assert load(7) == load(7) and load(7) != DEFAULT_HASHES and load(7) != load(8)


def _config(plate, **splits):
    from testing.test_launcher import pipeline_cfg

    cfg = pipeline_cfg(plate)
    cfg["ingestion"]["splits"].update(splits)
    return cfg


def test_the_loader_factory_passes_the_split_seed_from_the_config(tmp_path) -> None:
    plate = _plate(tmp_path)
    for splits, want in (({}, 42), ({"seed": 7}, 7), ({"seed": 0}, 0)):
        payload = build_pipeline_payload(_spec_with(tmp_path, plate, **splits), work_dir=tmp_path / "w")
        cfg = parse_tm_production_config(payload, profile="pipeline")
        loader = BaseDataLoader.create_loader(cfg)
        assert loader.random_state == want and cfg.ingestion.splits.seed == want


def _spec_with(tmp_path, plate, **splits):
    from src.launcher.payload import SupervisedFitSpec

    return SupervisedFitSpec(model_type="binary_classification", data_path=plate, pipeline=_config(plate, **splits), model_id="t")


def test_the_spec_field_reaches_the_payload_and_a_default_payload_has_no_seed_key(tmp_path) -> None:
    plain = build_pipeline_payload(_spec(tmp_path), work_dir=tmp_path / "w")
    assert "seed" not in plain["ingestion"]["splits"]      # a default run's stored config is unchanged
    seeded = build_pipeline_payload(_spec(tmp_path, split_seed=9), work_dir=tmp_path / "w")
    assert seeded["ingestion"]["splits"]["seed"] == 9
    assert parse_tm_production_config(seeded, profile="pipeline").ingestion.splits.seed == 9


def test_a_spec_without_a_pipeline_also_carries_the_split_seed(tmp_path) -> None:
    from src.launcher.payload import SupervisedFitSpec

    plate = _plate(tmp_path)
    mk = lambda **kw: build_pipeline_payload(SupervisedFitSpec(model_type="binary_classification", data_path=plate, feature_names=tuple(FEATURES), **kw), work_dir=tmp_path / "w")      # noqa: E731
    assert "seed" not in mk()["ingestion"]["splits"]
    assert mk(split_seed=3)["ingestion"]["splits"]["seed"] == 3
