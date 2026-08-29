# data/base_loader.py
from abc import ABC, abstractmethod
from typing import Tuple, Any
import numpy as np

from config.constants import ModelType


class BaseDataLoader(ABC):
    """
    Abstract interface defining the contract for all dataset loaders,
    including a factory method for instantiating concrete implementations.
    """

    @abstractmethod
    def load_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads, formats, and partitions dataset splits.
        Returns: (X_train, y_train, X_val, y_val)
        """
        pass

    @classmethod
    def create_loader(cls, cfg: Any) -> "BaseDataLoader":
        """
        Factory method to resolve and instantiate the correct BaseDataLoader 
        based on the provided pipeline configuration.
        """
        from src.data.tabular_loader import TabularCSVLoader
        from src.data.image_loader import ImageCSVLoader

        is_cnn = (cfg.architecture.model_type == ModelType.CNN)

        if is_cnn:
            cnn_cfg = getattr(cfg.architecture, "cnn", {})
            return ImageCSVLoader(
                csv_path=cfg.ingestion.data_file_path,
                input_shape=cnn_cfg.get("input_shape", [1, 28, 28]),
                num_classes=cfg.architecture.num_classes,
                val_split=cfg.ingestion.splits.val
            )

        source_mode = getattr(cfg.ingestion, "source_mode", None)
        
        # Default tabular CSV ingestion
        return TabularCSVLoader(
            data_file_path=cfg.ingestion.data_file_path,
            feature_names=cfg.ingestion.feature_names,
            train_split=cfg.ingestion.splits.train,
            val_split=cfg.ingestion.splits.val,
            model_type=cfg.architecture.model_type,
            num_classes=cfg.architecture.num_classes
        )