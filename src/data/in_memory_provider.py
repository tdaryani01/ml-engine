# data/in_memory_provider.py
from typing import Tuple
import numpy as np

from src.data.base_provider import BaseDataProvider
from src.data.base_loader import BaseDataLoader
from config.constants import DataKeys
from utils.seeding import batch_epoch, batch_rng, separate_streams


class InMemoryDataProvider(BaseDataProvider):
    """
    Universal zero-overhead in-memory data provider for tabular matrices and image tensors.
    Maintains clean contiguous index slicing with zero thread/queue overhead.
    """

    def __init__(
        self,
        loader: BaseDataLoader,
        batch_size: int = 32,
        epochs: int = 300,
        normalize_features: bool = True
    ):
        self.loader = loader
        self.batch_size = batch_size
        self.epochs = epochs
        self.normalize_features = normalize_features
        self._epochs_completed = 0
        self.epoch_offset = 0      # the epoch a continued run starts at; the per-epoch batch order of the separate-streams mode depends on it
        self._batch_idx = 0
        self._has_more = True

        # Extract standard split arrays
        X_train, y_train, X_val, y_val = self.loader.load_splits()

        self.X_train = np.ascontiguousarray(X_train)
        self.y_train_processed = np.ascontiguousarray(y_train)
        self.X_val = np.ascontiguousarray(X_val)
        self.y_val_processed = np.ascontiguousarray(y_val)

        # Setup feature normalization statistics
        if self.normalize_features and self.X_train.ndim == 2:
            self.mean = np.mean(self.X_train, axis=0)
            self.std = np.std(self.X_train, axis=0) + 1e-24
            # Data design §9 row 21, engine half (03ec5b3): a column the output mask calls one-hot is not scaled (mean 0, std 1 = identity).
            mask = getattr(self.loader, "output_mask", None) or []
            if mask and len(mask) != self.mean.size:
                raise ValueError(f"[In-Memory Provider] the output mask has {len(mask)} columns, the training part has {self.mean.size}")
            self.exempt = np.array([c.get("kind") == "onehot" for c in mask], dtype=bool) if mask else np.zeros(self.mean.shape, dtype=bool)
            self.mean = np.where(self.exempt, 0.0, self.mean)
            self.std = np.where(self.exempt, 1.0, self.std)
        else:
            self.mean = np.zeros(self.X_train.shape[1:], dtype=np.float32)
            self.std = np.ones(self.X_train.shape[1:], dtype=np.float32)
            self.exempt = np.zeros(self.mean.shape, dtype=bool)

        self.n_samples = self.X_train.shape[0]
        self.num_batches = int(np.ceil(self.n_samples / self.batch_size))
        self.indices = np.arange(self.n_samples)

        self.splits = {
            DataKeys.X_TRAIN: self.X_train,
            DataKeys.Y_TRAIN: self.y_train_processed,
            DataKeys.X_VAL: self.X_val,
            DataKeys.Y_VAL: self.y_val_processed
        }

        self.reset_epoch()

    def normalize(self, data_matrix: np.ndarray) -> np.ndarray:
        """Applies z-score normalization if enabled and dimensionality is 2D tabular."""
        if self.normalize_features and data_matrix.ndim == 2:
            return (data_matrix - self.mean) / self.std
        return data_matrix

    def reset_epoch(self) -> None:
        if self._batch_idx > 0:
            self._epochs_completed += 1

        if self._epochs_completed >= self.epochs:
            self._has_more = False
            return

        if separate_streams():      # data design §9 row 20: this epoch's order, and its dropout masks, from (batch seed, epoch) alone
            batch_epoch(self.epoch_offset + self._epochs_completed)
            self.indices = np.arange(self.n_samples)
        batch_rng().shuffle(self.indices)
        self._batch_idx = 0
        self._has_more = True

    def has_more_batches(self) -> bool:
        return self._has_more and (self._batch_idx < self.num_batches)

    def next_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.has_more_batches():
            self._has_more = False
            return np.array([]), np.array([])

        start_idx = self._batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        batch_slice = self.indices[start_idx:end_idx]

        self._batch_idx += 1
        if self._batch_idx >= self.num_batches:
            self._has_more = False

        return self.X_train[batch_slice], self.y_train_processed[batch_slice]

    def get_validation_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """The validation set as RAW features.

        Every consumer predicts through ``controller.predict`` (which z-scores once) and the
        diagnostics treat it as raw, so normalizing here as well scored the validation set on
        double-normalized inputs: wrong recorded validation loss and wrong early-stop decisions
        for 2-D tabular models. Training batches are still normalized by this provider.
        """
        return self.splits[DataKeys.X_VAL], self.y_val_processed

    def recomment_steps(self) -> int:
        return self.num_batches