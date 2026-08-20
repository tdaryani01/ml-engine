# ==========================================
# FILE 1: data/tabular_loader.py
# ==========================================
import logging
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from data.base_loader import BaseDataLoader
from config.constants import ModelType

logger = logging.getLogger(__name__)


class TabularCSVLoader(BaseDataLoader):
    """
    Loads tabular datasets from CSV, extracts features, performs stratified group 
    or random train/validation partitioning, and formats target vectors.
    """

    def __init__(
        self,
        data_file_path: str,
        feature_names: List[str],
        train_split: float = 0.70,
        val_split: float = 0.15,
        model_type: ModelType = ModelType.BINARY_CLASSIFICATION,
        num_classes: int = 1,
        random_state: int = 42
    ):
        self.data_file_path = data_file_path
        self.feature_names = feature_names
        self.train_split = train_split
        self.val_split = val_split
        self.model_type = model_type
        self.num_classes = num_classes
        self.random_state = random_state

    def load_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        df = pd.read_csv(self.data_file_path)

        # 1. Identify target column and clean feature columns
        target_column = "target" if "target" in df.columns else ("Risk_Label" if "Risk_Label" in df.columns else df.columns[-1])
        clean_features = [f for f in self.feature_names if f not in ["seed_id", target_column]]

        # 2. Extract raw feature and target matrices
        X = df[clean_features].to_numpy().astype(np.float32)
        y_raw = df[target_column].to_numpy().reshape(-1, 1)

        remaining_split = 1.0 - self.train_split
        val_ratio = self.val_split / remaining_split if remaining_split > 0 else 0.5

        # 3. Partitioning: Stratified Group Split vs. Standard Split
        if "seed_id" in df.columns:
            logger.info("[Tabular Loader] 'seed_id' detected. Performing Stratified Group Split...")
            seed_mapping = df.groupby("seed_id")[target_column].agg(lambda s: s.mode()[0]).reset_index()
            unique_seeds = seed_mapping["seed_id"]
            seed_classes = seed_mapping[target_column]

            can_stratify = seed_classes.value_counts().min() >= 2
            stratify_labels = seed_classes if can_stratify else None

            train_seeds, temp_seeds = train_test_split(
                unique_seeds,
                train_size=self.train_split,
                stratify=stratify_labels,
                random_state=self.random_state
            )

            temp_classes = seed_classes[unique_seeds.isin(temp_seeds)]
            can_stratify_temp = temp_classes.value_counts().min() >= 2 if can_stratify else False
            stratify_temp = temp_classes if can_stratify_temp else None

            val_seeds, _ = train_test_split(
                temp_seeds,
                train_size=val_ratio,
                stratify=stratify_temp,
                random_state=self.random_state
            )

            train_idx = df.index[df["seed_id"].isin(train_seeds)].tolist()
            val_idx = df.index[df["seed_id"].isin(val_seeds)].tolist()

            leakage = set(train_seeds).intersection(set(val_seeds))
            if leakage:
                raise ValueError(f"[Tabular Loader] Data leakage detected across seed groups: {leakage}")

            X_train, y_train = X[train_idx], y_raw[train_idx]
            X_val, y_val = X[val_idx], y_raw[val_idx]
        else:
            logger.info("[Tabular Loader] Performing standard train/validation split...")
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y_raw, train_size=self.train_split, random_state=self.random_state
            )
            X_val, _, y_val, _ = train_test_split(
                X_temp, y_temp, train_size=val_ratio, random_state=self.random_state
            )

        # 4. Target Formatting
        y_train_formatted = self._format_targets(y_train)
        y_val_formatted = self._format_targets(y_val)

        return X_train, y_train_formatted, X_val, y_val_formatted

    def _format_targets(self, y: np.ndarray) -> np.ndarray:
        """Transforms targets based on the specific network task type."""
        if self.model_type == ModelType.MULTI_CLASS:
            y_flat = y.ravel().astype(int)
            y_encoded = np.zeros((len(y_flat), self.num_classes), dtype=np.float32)
            y_encoded[np.arange(len(y_flat)), y_flat] = 1.0
            return y_encoded

        return y.reshape(-1, 1).astype(np.float32)


