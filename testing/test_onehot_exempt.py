# testing/test_onehot_exempt.py
"""Data design A6 (§9 row 21, engine half, 03ec5b3): the engine reads the output mask and leaves one-hot columns unscaled."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import ModelType
from src.data.in_memory_provider import InMemoryDataProvider
from src.data.lists_loader import ListsCSVLoader
from src.data.tabular_loader import TabularCSVLoader
from testing.test_lists_loader import _write


def _provider(folder: str, **kw):
    lp = _write(folder, **kw)
    loader = ListsCSVLoader(os.path.join(folder, "data.csv"), lp, ModelType.BINARY_CLASSIFICATION, 1)
    return InMemoryDataProvider(loader=loader, batch_size=16, epochs=2, normalize_features=True)


def test_every_onehot_column_reaches_the_model_as_0_or_1_exactly_and_numeric_columns_are_standardised(tmp_path) -> None:
    p = _provider(str(tmp_path))
    for X in (p.X_train, p.X_val):
        Z = p.normalize(X)
        assert set(np.unique(Z[:, 2:])) <= {0.0, 1.0} and np.array_equal(Z[:, 2:], X[:, 2:])      # counted: every cell of the three columns
    Zt = p.normalize(p.X_train)
    np.testing.assert_allclose(Zt[:, :2].mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(Zt[:, :2].std(axis=0), 1.0, atol=1e-6)
    assert p.exempt.tolist() == [False, False, True, True, True]
    assert p.mean[2:].tolist() == [0.0] * 3 and p.std[2:].tolist() == [1.0] * 3


def test_a_provider_without_a_mask_scales_every_column_as_before(tmp_path) -> None:
    _provider(str(tmp_path))      # writes the plate
    import pandas as pd

    df = pd.read_csv(tmp_path / "data.csv")
    loader = TabularCSVLoader(str(tmp_path / "data.csv"), ["a", "b", "kind"], 0.7, 0.15, ModelType.BINARY_CLASSIFICATION, 1)
    p = InMemoryDataProvider(loader=loader, batch_size=16, epochs=2, normalize_features=True)
    np.testing.assert_array_equal(p.mean, np.mean(p.X_train, axis=0))
    np.testing.assert_array_equal(p.std, np.std(p.X_train, axis=0) + 1e-24)
    assert not p.exempt.any() and len(df) == 120


def test_a_mask_of_the_wrong_width_is_refused() -> None:
    class Stub:
        output_mask = [{"column": 0, "kind": "onehot"}]

        def load_splits(self):
            x = np.zeros((10, 3), dtype=np.float32)
            return x, np.zeros((10, 1), dtype=np.float32), x, np.zeros((10, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="output mask"):
        InMemoryDataProvider(loader=Stub(), batch_size=4, epochs=1, normalize_features=True)
