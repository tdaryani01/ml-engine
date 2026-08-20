# data/in_memory_provider.py
from typing import Tuple, Optional
import numpy as np

from data.base_provider import BaseDataProvider
from data.base_loader import BaseDataLoader
from data.iterator import DatasetIterator
from config.constants import DataKeys


class InMemoryDataProvider(BaseDataProvider):
    """
    Universal in-memory data provider for tabular matrices and spatial image tensors.
    Consumes a BaseDataLoader instance, manages batch iteration, shuffling, and normalization.
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
        self._has_more = True
        self.batch_generator = None

        # Execute loader to extract standard split arrays
        X_train, y_train, X_val, y_val = self.loader.load_splits()

        self.y_train_processed = y_train
        self.y_val_processed = y_val

        # Setup feature normalization statistics
        if self.normalize_features and X_train.ndim == 2:
            self.mean = np.mean(X_train, axis=0)
            self.std = np.std(X_train, axis=0) + 1e-24
        else:
            self.mean = np.zeros(X_train.shape[1:], dtype=np.float32)
            self.std = np.ones(X_train.shape[1:], dtype=np.float32)

        self.splits = {
            DataKeys.X_TRAIN: X_train,
            DataKeys.Y_TRAIN: self.y_train_processed,
            DataKeys.X_VAL: X_val,
            DataKeys.Y_VAL: self.y_val_processed
        }

        self.reset_epoch()

    def normalize(self, data_matrix: np.ndarray) -> np.ndarray:
        """Applies z-score normalization if enabled and dimensionality is 2D tabular."""
        if self.normalize_features and data_matrix.ndim == 2:
            return (data_matrix - self.mean) / self.std
        return data_matrix

    def reset_epoch(self) -> None:
        if self.batch_generator is not None:
            self._epochs_completed += 1

        if self._epochs_completed >= self.epochs:
            self._has_more = False
            return

        num_samples = self.splits[DataKeys.X_TRAIN].shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        self.splits[DataKeys.X_TRAIN] = self.splits[DataKeys.X_TRAIN][indices]
        self.y_train_processed = self.y_train_processed[indices]
        self.splits[DataKeys.Y_TRAIN] = self.y_train_processed  # <-- Synchronize dictionary key

        iterator = DatasetIterator(
            self.splits[DataKeys.X_TRAIN],
            self.y_train_processed,
            batch_size=self.batch_size,
            shuffle=False
        )
        self.batch_generator = iter(iterator)
        self._has_more = True

    def has_more_batches(self) -> bool:
        """Determines if additional batches remain in the current epoch."""
        return self._has_more

    def next_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieves the next feature and target batch from the generator."""
        if not self.has_more_batches():
            return np.array([]), np.array([])
        try:
            return next(self.batch_generator)
        except StopIteration:
            self._has_more = False
            return np.array([]), np.array([])

    def get_validation_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieves the full validation features and targets."""
        return self.splits[DataKeys.X_VAL], self.y_val_processed

    def recomment_steps(self) -> int:
        """Calculates and recommends the number of steps per epoch."""
        return int(np.ceil(len(self.splits[DataKeys.X_TRAIN]) / self.batch_size))