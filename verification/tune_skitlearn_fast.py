# tune_sklearn_mlp_fast.py
import os
import yaml
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Set Optuna logging to clean console tracking
optuna.logging.set_verbosity(optuna.logging.INFO)

def objective(trial):
    # 1. Parse configuration file directly to match your environment paths
    config_path = "config\\config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing configuration file at {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_file = cfg["environment"]["data_file"]
    feature_names = cfg["data"]["feature_names"]
    max_epochs = cfg["training"]["epochs"]

    # 2. Load the dataset
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Missing dataset at: {data_file}")
        
    df = pd.read_csv(data_file)
    X = df[feature_names].values
    y = df["Outcome"].values

    # 3. Fast Fixed Split Geometry (60/20/20) - Mapped to match your exact splits
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=101)

    # Scale the splits cleanly
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    # 4. Target Search Bound Parameters
    num_layers = trial.suggest_int("num_layers", 2, 3)
    hidden_layer_sizes = []
    for i in range(num_layers):
        # Scan varying neuron capacities up to your max layout limits
        nodes = trial.suggest_int(f"layer_{i}_size", 64, 512, log=True)
        hidden_layer_sizes.append(nodes)
    hidden_layer_sizes = tuple(hidden_layer_sizes)

    # Optimizer Bounds
    lr_init = trial.suggest_float("learning_rate_init", 5e-4, 5e-3, log=True)
    alpha_l2 = trial.suggest_float("alpha_l2", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])

    # 5. Fast Execution Engine Instantiation
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver="adam",
        alpha=alpha_l2,
        batch_size=batch_size,
        learning_rate_init=lr_init,
        max_iter=max_epochs,
        random_state=101,
        early_stopping=True,        
        validation_fraction=0.15,
        n_iter_no_change=15,       # Reduced patience cuts off stalling trials instantly
        tol=1e-4                   # Relaxed convergence step tolerance for faster iterations
    )

    # Train a single model instance
    model.fit(X_train_norm, y_train)
    preds = model.predict(X_val_norm)
    
    # Minimize out-of-sample Validation MSE
    return mean_squared_error(y_val, preds)

def run_fast_tuning():
    print("==================================================")
    print("   LAUNCHING FAST SKLEARN OPTIMIZATION SEED       ")
    print("==================================================")
    
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=101))
    
    # Running 30 trials with the single split will finish in minutes
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print("\n==================================================")
    print("         FAST OPTIMIZATION PASS COMPLETED         ")
    print("==================================================")
    print(f"Best Configuration Validation MSE: {study.best_value:.6f}")
    print("\nOptimal Parameter Mappings Found:")
    for key, value in study.best_params.items():
        print(f"  -> {key}: {value}")
    print("==================================================")

if __name__ == "__main__":
    run_fast_tuning()