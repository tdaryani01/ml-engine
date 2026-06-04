# test_external_model.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def run_external_benchmark():
    # 1. Target the exact data path from your config
    data_path = r".\data\generator\chaotic_oscillator.csv"
    if not os.path.exists(data_path):
        print(f"[ERROR] Cannot find dataset at {data_path}. Run your generator script first.")
        return

    # 2. Read raw matrix values
    df = pd.read_csv(data_path)
    X = df[["Time", "Radius", "Angle"]].values
    y = df["Outcome"].values

    # 3. Match your 60/20/20 train/val/test split configuration exactly
    # Pull off 20% for testing first
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    # Take 25% of the remainder for validation (25% of 80% = 20% of total)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=101)

    print(f"[Data] Allocated {X_train.shape[0]} training rows and {X_val.shape[0]} validation rows.")

    # 4. Instantiate a high-capacity Scikit-Learn Regressor
    # Deep trees allow it to isolate high-frequency spatial regions natively
    external_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=25,
        min_samples_split=2,
        random_state=101,
        n_jobs=-1
    )

    print("\nTraining external high-capacity Random Forest model...")
    external_model.fit(X_train, y_train)

    # 5. Compute performance metrics
    train_preds = external_model.predict(X_train)
    val_preds = external_model.predict(X_val)

    train_mse = mean_squared_error(y_train, train_preds)
    val_mse = mean_squared_error(y_val, val_preds)

    print("\n==================================================")
    print("         EXTERNAL STANDALONE MODEL BENCHMARK      ")
    print("==================================================")
    print(f"Train Loss (MSE)     : {train_mse:.6f}")
    print(f"Validation Loss (MSE): {val_mse:.6f}")
    print("==================================================")

    # 6. Audit max residuals to isolate boundary gaps
    residuals = val_preds - y_val
    top_5_errors = np.sort(np.abs(residuals))[-5:]
    print(f"Top 5 Max Absolute Residual Errors: {top_5_errors}")

if __name__ == "__main__":
    run_external_benchmark()