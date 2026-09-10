# src/data/mhsa_loader.py
"""Load causal MHSA cue-recall NPZ datasets (X: N,T,D — y: N,A)."""
from __future__ import annotations

import logging
import os
from typing import Tuple

import numpy as np

from src.data.base_loader import BaseDataLoader

logger = logging.getLogger(__name__)


class MHSANpzLoader(BaseDataLoader):
    """
    Prefers pre-split keys X_train/y_train/X_val/y_val (cue_recall_*.npz).
    Falls back to X/y + random split if only a single pool is present.
    """

    def __init__(
        self,
        data_file_path: str,
        *,
        d_model: int,
        max_seq_len: int,
        action_dim: int,
        val_split: float = 0.15,
        train_split: float | None = None,
        random_state: int = 42,
    ) -> None:
        self.data_file_path = data_file_path
        self.d_model = int(d_model)
        self.max_seq_len = int(max_seq_len)
        self.action_dim = int(action_dim)
        self.val_split = float(val_split)
        self.train_split = train_split
        self.random_state = int(random_state)

    def load_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        path = self.data_file_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"[MHSA Loader] Dataset not found: {path}")

        with np.load(path) as blob:
            keys = set(blob.files)
            if {"X_train", "y_train", "X_val", "y_val"} <= keys:
                X_train = np.asarray(blob["X_train"], dtype=np.float32)
                y_train = np.asarray(blob["y_train"], dtype=np.float32)
                X_val = np.asarray(blob["X_val"], dtype=np.float32)
                y_val = np.asarray(blob["y_val"], dtype=np.float32)
            elif {"X", "y"} <= keys:
                X = np.asarray(blob["X"], dtype=np.float32)
                y = np.asarray(blob["y"], dtype=np.float32)
                X_train, y_train, X_val, y_val = self._split_pool(X, y)
            else:
                raise ValueError(
                    "[MHSA Loader] NPZ must contain X_train/y_train/X_val/y_val "
                    "or X/y. Got keys: " + ", ".join(sorted(keys))
                )

        self._validate(X_train, y_train, "train")
        self._validate(X_val, y_val, "val")
        logger.info(
            "[MHSA Loader] %s | train X=%s y=%s | val X=%s",
            path,
            X_train.shape,
            y_train.shape,
            X_val.shape,
        )
        return X_train, y_train, X_val, y_val

    def _validate(self, X: np.ndarray, y: np.ndarray, split: str) -> None:
        if X.ndim != 3:
            raise ValueError(f"[MHSA Loader] {split} X must be (N,T,D), got {X.shape}")
        if y.ndim != 2:
            raise ValueError(f"[MHSA Loader] {split} y must be (N,A), got {y.shape}")
        n, T, D = X.shape
        if y.shape[0] != n:
            raise ValueError(f"[MHSA Loader] {split} X/y length mismatch")
        if y.shape[1] != self.action_dim:
            raise ValueError(
                f"[MHSA Loader] {split} y A={y.shape[1]} != action_dim={self.action_dim}"
            )
        if D != self.d_model:
            raise ValueError(
                f"[MHSA Loader] {split} D={D} != config d_model={self.d_model}"
            )
        if T > self.max_seq_len:
            raise ValueError(
                f"[MHSA Loader] {split} T={T} exceeds max_seq_len={self.max_seq_len}"
            )

    def _split_pool(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(n)
        n_val = max(1, int(round(n * self.val_split)))
        if self.train_split is not None:
            n_train = max(1, int(round(n * float(self.train_split))))
        else:
            n_train = max(1, n - n_val)
        if n_train + n_val > n:
            n_val = n - n_train
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]
