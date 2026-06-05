# data/csv_provider.py
import numpy as np
from data.base_provider import BaseDataProvider
from data.iterator import DatasetIterator
from config.constants import DataKeys

class CSVDataProvider(BaseDataProvider):
    def __init__(self, data_file_path: str, feature_names: list, batch_size: int, model_instance):
        """
        Encapsulates local CSV file ingestion, splitting, and target preprocessing.
        Relies entirely on structural constants instead of hardcoded strings.
        """
        from data.data_loader import load_pipeline_splits
        
        # Load data splits using central key maps
        self.splits = load_pipeline_splits(data_file_path=data_file_path, feature_names=feature_names)
        self.batch_size = batch_size
        self.iterator = None
        self.batch_generator = None
        self._has_more = True
        
        # Preprocess and reshape training/validation targets upfront using explicit model rules
        self.y_train_processed, self.y_val_processed, _ = model_instance.preprocess_targets(
            self.splits[DataKeys.Y_TRAIN], 
            self.splits[DataKeys.Y_VAL]
        )
        
        self.reset_epoch()

    def reset_epoch(self) -> None:
        """Resets the cursor positions explicitly for a brand-new training epoch pass."""
        self.iterator = DatasetIterator(
            self.splits[DataKeys.X_TRAIN], 
            self.y_train_processed, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        self.batch_generator = iter(self.iterator)
        self._has_more = True  # Reset flag for the new epoch loop

    def has_more_batches(self) -> bool:
        """Tells the controller when the current epoch dataset pass has concluded."""
        return self._has_more

    def next_batch(self) -> tuple:
        """Retrieves the next chunk or safely alerts the caller that the tank is dry."""
        try:
            return next(self.batch_generator)
        except StopIteration:
            self._has_more = False
            # Return an empty array pair to gracefully signal the end of the current epoch
            return np.array([]), np.array([])

    def get_validation_set(self) -> tuple:
        """
        Returns the standardized validation inputs and preprocessed targets 
        for model evaluation steps.
        """
        return self.splits[DataKeys.X_VAL], self.y_val_processed