# data/data_loader.py
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from config.constants import DataKeys

logger = logging.getLogger(__name__)

def load_pipeline_splits(data_file_path: str, feature_names: list, train_split: float = 0.70, val_split: float = 0.15):
    """
    Explicitly accepts primitive parameters instead of a generic config object.
    Performs GroupShuffleSplit if 'seed_id' is present to eliminate data leakage.
    Ensures targets are formatted as 2D column vectors (N, 1) for uniform model preprocessing.
    """
    # 1. Read file using the clean incoming primitive path string
    df = pd.read_csv(data_file_path)
    
    # Identify target column
    target_column = "target" if "target" in df.columns else ("Risk_Label" if "Risk_Label" in df.columns else df.columns[-1])
    
    # Filter feature names to exclude metadata columns
    clean_feature_names = [f for f in feature_names if f not in ["seed_id", target_column]]
    
    # 2. Extract features and target array
    X = df[clean_feature_names].to_numpy().astype(np.float32)
    # Reshape to 2D column vector (N, 1)
    y = df[target_column].to_numpy().reshape(-1, 1).astype(np.int64)
    
    # 3. Grouped or Random Split Mechanics
    if "seed_id" in df.columns:
        logger.info("[Data Loader] 'seed_id' column detected. Executing Stratified Group Split...")
        
        # Determine the dominant class for each distinct seed_id
        seed_class_mapping = df.groupby("seed_id")[target_column].agg(lambda x: x.mode()[0]).reset_index()
        unique_seeds = seed_class_mapping["seed_id"]
        seed_classes = seed_class_mapping[target_column]
        
        # Split Train vs (Val + Test) at the SEED level, stratifying by class
        train_seeds, temp_seeds = train_test_split(
            unique_seeds, 
            train_size=train_split, 
            stratify=seed_classes, 
            random_state=42
        )
        
        # Split Val vs Test from remaining seeds
        temp_classes = seed_classes[unique_seeds.isin(temp_seeds)]
        val_seeds, test_seeds = train_test_split(
            temp_seeds, 
            train_size=0.5, 
            stratify=temp_classes, 
            random_state=42
        )
        
        # Map the chosen seeds back to the original row indices
        train_idx = df.index[df["seed_id"].isin(train_seeds)].tolist()
        val_idx = df.index[df["seed_id"].isin(val_seeds)].tolist()
        test_idx = df.index[df["seed_id"].isin(test_seeds)].tolist()

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Validation checks on group isolation
        overlap_train_val = set(train_seeds).intersection(set(val_seeds))
        overlap_train_test = set(train_seeds).intersection(set(test_seeds))
        
        logger.info(
            f"[Data Loader] Stratified Seed Groups -> Train: {len(train_seeds)} | Val: {len(val_seeds)} | Test: {len(test_seeds)}"
        )
        
        if overlap_train_val or overlap_train_test:
            logger.error("[Data Loader] Leakage detected during stratified split!")
            raise ValueError("Group leakage detected across dataset splits!")
        else:
            logger.info("[Data Loader] Verification passed: Zero group overlap detected.")
    else:
        logger.info("[Data Loader] No 'seed_id' column found. Performing standard split...")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_split, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=0.5, random_state=42)
    
    return {
        DataKeys.X_TRAIN: X_train,
        DataKeys.Y_TRAIN: y_train,
        DataKeys.X_VAL: X_val,
        DataKeys.Y_VAL: y_val,
        DataKeys.X_TEST: X_test,
        DataKeys.Y_TEST: y_test
    }