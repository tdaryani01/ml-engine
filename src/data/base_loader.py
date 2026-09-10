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
        from src.data.mhsa_loader import MHSANpzLoader

        is_cnn = cfg.architecture.model_type == ModelType.CNN
        is_mhsa = cfg.architecture.model_type == ModelType.MHSA

        if is_mhsa:
            mhsa_cfg = getattr(cfg.architecture, "mhsa", None) or {}
            if hasattr(mhsa_cfg, "d_model"):
                d_model = int(mhsa_cfg.d_model)
                max_seq_len = int(mhsa_cfg.max_seq_len)
                action_dim = int(mhsa_cfg.action_dim)
            else:
                d_model = int(mhsa_cfg["d_model"])
                max_seq_len = int(mhsa_cfg["max_seq_len"])
                action_dim = int(mhsa_cfg["action_dim"])
            return MHSANpzLoader(
                cfg.ingestion.data_file_path,
                d_model=d_model,
                max_seq_len=max_seq_len,
                action_dim=action_dim,
                val_split=cfg.ingestion.splits.val,
                train_split=cfg.ingestion.splits.train,
            )

        if is_cnn:
            cnn_cfg = getattr(cfg.architecture, "cnn", None) or {}
            if hasattr(cnn_cfg, "input_shape"):
                input_shape = list(cnn_cfg.input_shape)
            else:
                input_shape = list(cnn_cfg.get("input_shape", [3, 128, 128]))
            return ImageCSVLoader(
                csv_path=cfg.ingestion.data_file_path,
                input_shape=input_shape,
                num_classes=cfg.architecture.num_classes,
                val_split=cfg.ingestion.splits.val,
                train_split=cfg.ingestion.splits.train,
            )

        return TabularCSVLoader(
            data_file_path=cfg.ingestion.data_file_path,
            feature_names=cfg.ingestion.feature_names,
            train_split=cfg.ingestion.splits.train,
            val_split=cfg.ingestion.splits.val,
            model_type=cfg.architecture.model_type,
            num_classes=cfg.architecture.num_classes,
        )
