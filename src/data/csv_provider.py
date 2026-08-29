# data/csv_provider.py
import numpy as np
from src.data.base_provider import BaseDataProvider
from src.data.iterator import DatasetIterator
from config.constants import DataKeys

class CSVDataProvider(BaseDataProvider):
    """Data provider for loading, normalizing, and batching tabular datasets from CSV files."""
    def __init__(self, data_file_path: str, feature_names: list, batch_size: int, model_instance, epochs: int):
        from src.data.data_loader import load_pipeline_splits
        
        self.splits = load_pipeline_splits(data_file_path=data_file_path, feature_names=feature_names)
        self.batch_size = batch_size
        self.epochs = epochs
        self.iterator = None
        self.batch_generator = None
        self._has_more = True
        self._epochs_completed = 0
        
        # Static initialization matching the unified normalization interface
        self.mean = np.mean(self.splits[DataKeys.X_TRAIN], axis=0)
        self.std = np.std(self.splits[DataKeys.X_TRAIN], axis=0) + 1e-24
        
        self.y_train_processed, self.y_val_processed, _ = model_instance.preprocess_targets(
            self.splits[DataKeys.Y_TRAIN], 
            self.splits[DataKeys.Y_VAL]
        )
        
        self.reset_epoch()

    def normalize(self, data_matrix: np.ndarray) -> np.ndarray:
        """Uniform API standard matching the Stream Provider signature."""
        return (data_matrix - self.mean) / self.std

    def reset_epoch(self) -> None:
        """Resets the epoch iterator and updates completed epoch counts."""
        if self.batch_generator is not None or self.iterator is not None:
            self._epochs_completed += 1
            
        if self._epochs_completed >= self.epochs:
            self._has_more = False
            return

        # EXPLICIT SHUFFLE: Break up the seed_id groupings from GroupShuffleSplit
        # This guarantees rows are globally randomized before they are batched.
        num_samples = self.splits[DataKeys.X_TRAIN].shape[0]
        shuffle_indices = np.random.permutation(num_samples)
        
        self.splits[DataKeys.X_TRAIN] = self.splits[DataKeys.X_TRAIN][shuffle_indices]
        self.y_train_processed = self.y_train_processed[shuffle_indices]

        self.iterator = DatasetIterator(
            self.splits[DataKeys.X_TRAIN], 
            self.y_train_processed, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        self.batch_generator = iter(self.iterator)
        self._has_more = True

    def has_more_batches(self) -> bool:
        """Determines if additional batches remain in the current epoch."""
        return self._has_more

    def next_batch(self) -> tuple:
        """Retrieves the next feature and target batch from the generator."""
        if not self.has_more_batches():
            # Fixed a potential TypeError here (removed len() wrapper around the integer shape[1])
            feature_dim = self.splits[DataKeys.X_TRAIN].shape[1]
            return np.zeros((0, feature_dim), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)
        try:
            return next(self.batch_generator)
        except StopIteration:
            self._has_more = False
            return np.array([]), np.array([])

    def get_validation_set(self) -> tuple:
        """Retrieves the processed validation features and targets."""
        return self.splits[DataKeys.X_VAL], self.y_val_processed

    def recomment_steps(self) -> int:
        """Calculates and recommends the number of steps per epoch based on batch size."""
        return int(np.ceil(len(self.splits[DataKeys.X_TRAIN]) / self.batch_size))