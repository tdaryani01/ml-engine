import os
import sys
import yaml
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Silence Optuna logs for clean console output
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial):
    config_path = os.path.join("config", "config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_path = cfg["ingestion"]["data_file_path"]
    features = cfg["ingestion"]["feature_names"]
    
    df = pd.read_csv(data_path)
    X = df[features].values
    target_col = "Outcome" if "Outcome" in df.columns else ("Target" if "Target" in df.columns else df.columns[-1])
    y = df[target_col].values.astype(int)

    # Stratified Train/Val split
    scaler = StandardScaler()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Search Space
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_categorical(f"layer_{i}_units", [8, 16, 32, 64]))
    
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    l2 = trial.suggest_float("lam_l2", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    clf = MLPClassifier(
        hidden_layer_sizes=tuple(layers),
        activation="relu",
        solver="adam",
        alpha=l2,
        batch_size=batch_size,
        learning_rate_init=lr,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=15
    )

    clf.fit(X_train_scaled, y_train)
    val_preds = clf.predict(X_val_scaled)
    
    return accuracy_score(y_val, val_preds)


def run_tuning():
    print("=" * 70)
    print("        HYPERPARAMETER TUNING: PROMPT SECURITY DATASET")
    print("=" * 70)
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print("\n" + "=" * 70)
    print(f"Best Validation Accuracy: {study.best_value * 100:.2f}%")
    print("Optimal Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  • {k:<20}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    run_tuning()