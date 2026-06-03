# data_loader.py
import os
import numpy as np

def load_pipeline_splits(data_config, csv_file_path):
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Target file sample '{csv_file_path}' not found.")
        
    data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1)
    
    # --- CRITICAL FIX: Shuffle rows globally before splitting ---
    # This mixes the sequential Class 0, 1, and 2 rows evenly across all splits
    np.random.seed(42)  # Fixed seed ensures reproducible matrix splitting
    np.random.shuffle(data)
    
    num_features = len(data_config["feature_names"])
    
    X = data[:, :num_features]
    y = data[:, num_features].reshape(-1, 1)
    
    total_samples = len(data)
    train_end = int(total_samples * data_config["train_split"])
    val_end = int(total_samples * (data_config["train_split"] + data_config["val_split"]))
    
    return {
        "X_train": X[:train_end], "y_train": y[:train_end],
        "X_val": X[train_end:val_end], "y_val": y[train_end:val_end],
        "X_test": X[val_end:], "y_test": y[val_end:]
    }