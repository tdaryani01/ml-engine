# data/data_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # or your custom matrix slicing logic
from config.constants import DataKeys

def load_pipeline_splits(data_file_path: str, feature_names: list, train_split: float = 0.70, val_split: float = 0.15):
    """
    Explicitly accepts primitive parameters instead of a generic config object.
    """
    # 1. Read file using the clean incoming primitive path string
    df = pd.read_csv(data_file_path)
    
    # 2. Extract features exactly from the explicit names list
    X = df[feature_names].to_numpy()
    
    # Placeholder for your targets lookup column (e.g., 'Target' or 'Risk_Label')
    # Adjust this column name string to match whatever your dataset uses!
    target_column = "Risk_Label" if "Risk_Label" in df.columns else df.columns[-1]
    y = df[[target_column]].to_numpy()
    
    # 3. Your existing train/validation/test array splitting matrix mechanics go here...
    # Example classic slice setup:
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