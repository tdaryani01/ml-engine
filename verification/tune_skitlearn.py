# tune_sklearn_mlp.py
import os
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Silence optuna logs if you want a clean console, or leave INFO to track trials
optuna.logging.set_verbosity(optuna.logging.INFO)

def objective(trial):
    # 1. Load the target data path
    data_path = r".\data\generator\chaotic_oscillator.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing dataset at: {data_path}")
        
    df = pd.read_csv(data_path)
    X = df[["Time", "Radius", "Angle"]].values
    y = df["Outcome"].values

    # 2. Define a High-Capacity Search Space (Prioritizing Accuracy)
    # Search up to 4 layers deep with high neuron density
    num_layers = trial.suggest_int("num_layers", 2, 4)
    hidden_layer_sizes = []
    for i in range(num_layers):
        # Allow wide layers to prevent parameter bottlenecks
        nodes = trial.suggest_int(f"layer_{i}_size", 64, 1024, log=True)
        hidden_layer_sizes.append(nodes)
    hidden_layer_sizes = tuple(hidden_layer_sizes)

    # Optimization Hyperparameters
    lr_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
    alpha_l2 = trial.suggest_float("alpha_l2", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])

    # 3. Implement Strict Cross-Validation to prevent metric overfitting
    kf = KFold(n_splits=5, shuffle=True, random_state=101)
    cv_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Standardizing raw spatial features within each fold split
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-8
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std

        # Initialize Sklearn Engine
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver="adam",
            alpha=alpha_l2,
            batch_size=batch_size,
            learning_rate_init=lr_init,
            max_iter=500,               # Given a wide runtime allowance for full convergence
            random_state=101,
            early_stopping=True,        # Use Sklearn internal validation to prevent overshooting
            validation_fraction=0.15,
            n_iter_no_change=30,        # Generous patience window to clear local minima
            tol=1e-5
        )

        model.fit(X_train_norm, y_train)
        preds = model.predict(X_val_norm)
        
        fold_mse = mean_squared_error(y_val, preds)
        cv_scores.append(fold_mse)

    # Return the mean Cross-Validation MSE to minimize
    return np.mean(cv_scores)

def run_tuning_pipeline():
    print("==================================================")
    print("     STARTING SCIKIT-LEARN ACCURACY TUNING        ")
    print("==================================================")
    
    # Using the TPESampler (Tree-structured Parzen Estimator) for global space search
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=101))
    
    # Run 50 trials. Scale this up if you want to leave it running for maximum accuracy extraction.
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print("\n==================================================")
    print("             OPTIMIZATION PASS COMPLETED          ")
    print("==================================================")
    print(f"Best Validation MSE Score: {study.best_value:.6f}")
    print("\nOptimized Architecture Parameters:")
    for key, value in study.best_params.items():
        print(f"  -> {key}: {value}")
    print("==================================================")

if __name__ == "__main__":
    run_tuning_pipeline()