import os
import sys
import time
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import ModelType, IngestionMode, LRHierarchy
from src.controller import ModelController
from data.csv_provider import CSVDataProvider


def run_regression_benchmark():
    config_path = os.path.join("config", "config.yaml")
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found at {config_path}")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_path = cfg["ingestion"]["data_file_path"]
    features = cfg["ingestion"]["feature_names"]
    hidden_layers = list(cfg["architecture"]["hidden_layers"])
    batch_size = int(cfg["optimization"]["batch_size"])
    lr_init = float(cfg["optimization"]["learning_rate"])
    epochs = int(cfg["optimization"]["epochs_full_dataset"])
    steps = int(cfg["optimization"]["steps_streaming"])
    lam_l2 = float(cfg["regularization"]["lam_l2"])

    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found at '{data_path}'")
        return

    print("=" * 70)
    print("       HEAD-TO-HEAD REGRESSION BENCHMARK: CUSTOM vs SKLEARN")
    print("=" * 70)

    df = pd.read_csv(data_path)
    X = df[features].values
    target_col = "Target" if "Target" in df.columns else ("Outcome" if "Outcome" in df.columns else df.columns[-1])
    y = df[target_col].values

    # 1. Scikit-Learn MLP Regressor
    print("\n[1/3] Benchmarking Scikit-Learn MLPRegressor...")
    scaler = StandardScaler()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    sklearn_mlp = MLPRegressor(
        hidden_layer_sizes=tuple(hidden_layers),
        activation="relu",
        solver="adam",
        alpha=lam_l2,
        batch_size=batch_size,
        learning_rate_init=lr_init,
        max_iter=epochs,
        random_state=42,
        early_stopping=False
    )

    t0 = time.time()
    sklearn_mlp.fit(X_train_scaled, y_train)
    sklearn_time = time.time() - t0

    sk_val_preds = sklearn_mlp.predict(X_val_scaled)
    sk_val_mse = mean_squared_error(y_val, sk_val_preds)
    sk_val_r2 = r2_score(y_val, sk_val_preds)

    # 2. Random Forest Regressor
    print("[2/3] Benchmarking Scikit-Learn Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    t0 = time.time()
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0

    rf_val_preds = rf.predict(X_val)
    rf_val_mse = mean_squared_error(y_val, rf_val_preds)
    rf_val_r2 = r2_score(y_val, rf_val_preds)

    # 3. Custom NumPy Engine
    print("[3/3] Benchmarking Custom NumPy Engine...")
    controller = ModelController(
        learning_rate=lr_init,
        lr_scheduler_type=LRHierarchy.NONE
    )
    controller.initialize_network_from_dimensions(
        input_dim=len(features),
        output_dim=1,
        model_type=ModelType.REGRESSION,
        hidden_layers=hidden_layers,
        optimizer_name="adam",
        lam_l1=0.0,
        lam_l2=lam_l2,
        p_dropout=0.0,
        use_batch_norm=False,
        bn_momentum=0.9
    )

    data_provider = CSVDataProvider(
        data_file_path=data_path,
        feature_names=features,
        batch_size=batch_size,
        epochs=epochs,
        model_instance=controller.model
    )

    t0 = time.time()
    controller.fit(
        data_provider=data_provider,
        steps=steps,
        source_mode=IngestionMode.CSV,
        model_type=ModelType.REGRESSION,
        early_stopping_enabled=False
    )
    custom_time = time.time() - t0

    X_val_custom, y_val_custom = data_provider.get_validation_set()
    custom_preds = controller.predict(X_val_custom)
    custom_val_mse = mean_squared_error(y_val_custom.ravel(), custom_preds.ravel())
    custom_val_r2 = r2_score(y_val_custom.ravel(), custom_preds.ravel())

    # 4. Summary Table
    print("\n" + "=" * 70)
    print("                    BENCHMARK SCORECARD")
    print("=" * 70)
    print(f"{'Framework / Engine':<32} | {'Val MSE':<12} | {'Val R²':<10} | {'Wall Time':<10}")
    print("-" * 70)
    print(f"{'Scikit-Learn (MLPRegressor)':<32} | {sk_val_mse:>10.6f} | {sk_val_r2:>8.4f} | {sklearn_time:>8.2f}s")
    print(f"{'Scikit-Learn (Random Forest)':<32} | {rf_val_mse:>10.6f} | {rf_val_r2:>8.4f} | {rf_time:>8.2f}s")
    print(f"{'Custom Engine (NumPy Autodiff)':<32} | {custom_val_mse:>10.6f} | {custom_val_r2:>8.4f} | {custom_time:>8.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    run_regression_benchmark()