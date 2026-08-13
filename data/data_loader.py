# data/data_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # or custom matrix slicing logic[cite: 9]
from config.constants import DataKeys

def load_pipeline_splits(data_file_path: str, feature_names: list, train_split: float = 0.70, val_split: float = 0.15):
    """
    Explicitly accepts primitive parameters instead of a generic config object[cite: 9].
    """
    # 1. Read file using the clean incoming primitive path string[cite: 9]
    df = pd.read_csv(data_file_path)
    
    # 2. Extract features exactly from the explicit names list[cite: 9]
    X = df[feature_names].to_numpy()
    
    # Placeholder for targets lookup column (e.g., 'Target' or 'Risk_Label')[cite: 9]
    # Adjust this column name string to match whatever the dataset uses[cite: 9]!
    target_column = "Risk_Label" if "Risk_Label" in df.columns else df.columns[-1]
    y = df[[target_column]].to_numpy()
    
    # 3. Existing train/validation/test array splitting matrix mechanics[cite: 9]
    # Example classic slice setup[cite: 9]:
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