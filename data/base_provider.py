# data/base_provider.py
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class BaseDataProvider(ABC):
    """Abstract base class defining the required interface for data providers[cite: 9]."""
    @abstractmethod
    def has_more_batches(self) -> bool:
        """Determines whether more batches are available in the current epoch[cite: 9]."""
        pass

    @abstractmethod
    def next_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieves the next data batch containing features and targets[cite: 9]."""
        pass

    @abstractmethod
    def get_validation_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieves the full validation dataset[cite: 9]."""
        pass
        
    @abstractmethod
    def reset_epoch(self) -> None:
        """Resets provider state for a new training epoch[cite: 9]."""
        pass