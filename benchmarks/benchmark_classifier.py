import os
import sys
import time
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import ModelType, IngestionMode, LRHierarchy
from src.controller import ModelController
from src.data.csv_provider import CSVDataProvider


def run_classification_benchmark():
    config_path = os.path.join("config", "config.yaml")
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found at {config_path}")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_path = cfg["ingestion"]["data_file_path"]
    features = cfg["ingestion"]["feature_names"]
    hidden_layers = list(cfg["architecture"]["hidden_layers"])
    num_classes = int(cfg["architecture"].get("num_classes", 2))
    batch_size = int(cfg["optimization"]["batch_size"])
    lr_init = float(cfg["optimization"]["learning_rate"])
    epochs = int(cfg["optimization"]["epochs_full_dataset"])
    steps = int(cfg["optimization"]["steps_streaming"])
    lam_l2 = float(cfg["regularization"]["lam_l2"])

    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found at '{data_path}'")
        return

    print("=" * 70)
    print("      HEAD-TO-HEAD CLASSIFICATION BENCHMARK: CUSTOM vs SKLEARN")
    print("=" * 70)
    print(f"Dataset Path    : {data_path}")
    print(f"Feature Count   : {len(features)}")
    print(f"Topology        : {hidden_layers} -> {num_classes} classes")
    print(f"Epochs / Batch  : {epochs} / {batch_size} | LR: {lr_init} | L2: {lam_l2}")
    print("=" * 70)

    # 1. Load Data
    df = pd.read_csv(data_path)
    X = df[features].values
    
    # Infer target column
    target_col = "Outcome" if "Outcome" in df.columns else ("Target" if "Target" in df.columns else df.columns[-1])
    y_raw = df[target_col].values.astype(int)

    # 2. Benchmark Scikit-Learn MLPClassifier
    print("\n[1/3] Benchmarking Scikit-Learn MLPClassifier...")
    scaler = StandardScaler()
    X_train, X_val, y_train, y_val = train_test_split(X, y_raw, test_size=0.3, random_state=42, stratify=y_raw)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    sklearn_mlp = MLPClassifier(
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

    sk_train_preds = sklearn_mlp.predict(X_train_scaled)
    sk_val_preds = sklearn_mlp.predict(X_val_scaled)
    sk_train_acc = accuracy_score(y_train, sk_train_preds)
    sk_val_acc = accuracy_score(y_val, sk_val_preds)

    # 3. Benchmark Random Forest Baseline
    print("[2/3] Benchmarking Scikit-Learn Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    t0 = time.time()
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0
    rf_val_acc = accuracy_score(y_val, rf.predict(X_val))

    # 4. Benchmark Custom Engine
    print("[3/3] Benchmarking Custom NumPy Engine...")
    task_type = ModelType.MULTI_CLASS if num_classes > 2 else ModelType.BINARY_CLASSIFICATION
    
    controller = ModelController(
        learning_rate=lr_init,
        lr_scheduler_type=LRHierarchy.NONE
    )
    controller.initialize_network_from_dimensions(
        input_dim=len(features),
        output_dim=num_classes if task_type == ModelType.MULTI_CLASS else 1,
        model_type=task_type,
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
    train_hist, val_hist = controller.fit(
        data_provider=data_provider,
        steps=steps,
        source_mode=IngestionMode.CSV,
        model_type=task_type,
        early_stopping_enabled=False
    )
    custom_time = time.time() - t0

    X_val_custom, y_val_custom = data_provider.get_validation_set()
    custom_preds = controller.predict(X_val_custom)
    
    if task_type == ModelType.MULTI_CLASS:
        custom_pred_classes = np.argmax(custom_preds, axis=1)
        y_true_classes = np.argmax(y_val_custom, axis=1) if len(y_val_custom.shape) > 1 else y_val_custom.ravel()
    else:
        custom_pred_classes = (custom_preds > 0.5).astype(int).ravel()
        y_true_classes = y_val_custom.ravel()

    custom_val_acc = accuracy_score(y_true_classes, custom_pred_classes)

    # 5. Output Summary Comparison Table
    print("\n" + "=" * 70)
    print("                    BENCHMARK SCORECARD")
    print("=" * 70)
    print(f"{'Framework / Engine':<32} | {'Val Accuracy':<15} | {'Wall Time':<10}")
    print("-" * 70)
    print(f"{'Scikit-Learn (MLPClassifier)':<32} | {sk_val_acc * 100:>13.2f}% | {sklearn_time:>8.2f}s")
    print(f"{'Scikit-Learn (Random Forest)':<32} | {rf_val_acc * 100:>13.2f}% | {rf_time:>8.2f}s")
    print(f"{'Custom Engine (NumPy Autodiff)':<32} | {custom_val_acc * 100:>13.2f}% | {custom_time:>8.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    run_classification_benchmark()