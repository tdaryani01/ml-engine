# data/base_provider.py
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class BaseDataProvider(ABC):
    @abstractmethod
    def has_more_batches(self) -> bool:
        pass

    @abstractmethod
    def next_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_validation_set(self) -> Tuple[np.ndarray, np.ndarray]:
        pass
        
    @abstractmethod
    def reset_epoch(self) -> None:
        pass