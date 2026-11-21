"""Data design A1b (§9 row 1, 03ec5b3): a run given a lists file takes its parts from row lists, and cuts nothing.

The lists file (form agreed with the data build, ``lists_file_form.md``) names, for one split seed: the 0-based row positions of the training and
the validation part in the data CSV, the encoded matrix of that seed (all rows, NOT scaled, one column per output column) and the output-column mask.
Rows in no list are unused. The engine checks the matrix, mask and data file against the hashes the lists file carries, and keeps the sha256 of the
lists file's bytes as ``lists_hash``, which goes into every checkpoint's knobs (a restore under another lists hash is refused)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Tuple

import numpy as np
import pandas as pd

from config.constants import ModelType
from src.data.tabular_loader import TabularCSVLoader

logger = logging.getLogger(__name__)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ListsCSVLoader(TabularCSVLoader):
    """Parts from row lists and an encoded matrix; the target comes from the data CSV exactly as the tabular loader reads it."""

    def __init__(self, data_file_path: str, lists_path: str, model_type: ModelType = ModelType.BINARY_CLASSIFICATION, num_classes: int = 1):
        self.lists_path = str(lists_path)
        self.lists_hash = _sha256(self.lists_path)
        with open(self.lists_path, "r", encoding="utf-8") as fh:
            self.lists: dict[str, Any] = json.load(fh)
        folder = os.path.dirname(os.path.abspath(self.lists_path))
        self.matrix_path = os.path.join(folder, self.lists["matrix"])
        self.mask_path = os.path.join(folder, self.lists["mask"])
        for label, path, key in (("matrix", self.matrix_path, "matrix_sha256"), ("mask", self.mask_path, "mask_sha256"), ("data file", data_file_path, "data_csv_sha256")):
            if _sha256(path) != self.lists[key]:
                raise ValueError(f"[Lists Loader] the {label} {path} does not match the {key} in {self.lists_path}: refusing to load")
        with open(self.mask_path, "r", encoding="utf-8") as fh:
            self.output_mask: list[dict[str, Any]] = json.load(fh)
        super().__init__(data_file_path, feature_names=[], model_type=model_type, num_classes=num_classes, random_state=int(self.lists["split_seed"]))
        self.n_unused = 0

    def load_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        matrix = np.load(self.matrix_path)
        n_rows, n_out = int(self.lists["n_rows"]), int(self.lists["n_out"])
        if matrix.shape != (n_rows, n_out) or len(self.output_mask) != n_out:
            raise ValueError(f"[Lists Loader] matrix {matrix.shape}, mask {len(self.output_mask)} columns, lists say ({n_rows}, {n_out})")
        df = pd.read_csv(self.data_file_path)
        if len(df) != n_rows:
            raise ValueError(f"[Lists Loader] the data file has {len(df)} rows, the lists say {n_rows}")
        target_column = "target" if "target" in df.columns else ("Risk_Label" if "Risk_Label" in df.columns else df.columns[-1])
        y_raw = df[target_column].to_numpy().reshape(-1, 1)
        train, val = np.asarray(self.lists["train"], dtype=np.int64), np.asarray(self.lists["validation"], dtype=np.int64)
        for name, idx in (("train", train), ("validation", val)):
            if idx.size and (idx.min() < 0 or idx.max() >= n_rows):
                raise ValueError(f"[Lists Loader] the {name} list has a position outside 0..{n_rows - 1}")
        if np.intersect1d(train, val).size:
            raise ValueError("[Lists Loader] a row is in both the train and the validation list")
        self.n_train, self.n_validation = int(train.size), int(val.size)
        self.n_unused = n_rows - self.n_train - self.n_validation
        X = matrix.astype(np.float32)
        return X[train], self._format_targets(y_raw[train]), X[val], self._format_targets(y_raw[val])
