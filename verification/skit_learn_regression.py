# test_sklearn.py
import os
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

def run_external_mlp_benchmark():
    config_path = "config\\config.yaml"
    if not os.path.exists(config_path):
        print(f"[ERROR] Cannot find configuration file at {config_path}")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_file = cfg["environment"]["data_file"]
    feature_names = cfg["data"]["feature_names"]
    hidden_layers = tuple(cfg["architecture"]["hidden_layers"])
    batch_size = cfg["training"]["batch_size"]
    lr_init = cfg["training"]["learning_rate"]
    alpha_l2 = cfg["regularization"]["lam_l2"]
    
    # ─── FORCE PATIENCE PAST 10 EPOCHS ─────────────────────────────────
    epochs = cfg["training"]["epochs"]
    patience_setting = max(cfg["training"]["patience"], 50)  # Guarantees a wide window
    min_delta_setting = cfg["training"]["min_delta"]
    # ───────────────────────────────────────────────────────────────────

    print("==================================================")
    print("      SKLEARN PRODUCTION MLP RUN TIME ENGINE      ")
    print("==================================================")
    print(f"[Architecture] Layout: {hidden_layers}")
    print(f"[Early Stop]   Patience Windows: {patience_setting} epochs | Min Delta: {min_delta_setting}")

    if not os.path.exists(data_file):
        print(f"[ERROR] Cannot find dataset file at {data_file}. Run your generator first.")
        return

    df = pd.read_csv(data_file)
    X = df[feature_names].values
    y = df["Outcome"].values

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=101)

    # Instantiate Scikit-Learn's engine mapping your parameters exactly
    external_mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        alpha=alpha_l2,
        batch_size=batch_size,
        learning_rate_init=lr_init,
        max_iter=epochs,
        random_state=101,
        
        # Configure external early stopping to mirror your metrics safely
        early_stopping=True,
        validation_fraction=0.2,  
        n_iter_no_change=patience_setting, 
        tol=min_delta_setting,
        verbose=True
    )

    print("\nTraining Scikit-Learn backbone network...")
    external_mlp.fit(X_train, y_train)

    train_preds = external_mlp.predict(X_train)
    val_preds = external_mlp.predict(X_val)

    print("\n==================================================")
    print("         EXTERNAL ARCHITECTURE VERIFICATION       ")
    print("==================================================")
    print(f"Sklearn Train Loss (MSE)     : {mean_squared_error(y_train, train_preds):.6f}")
    print(f"Sklearn Validation Loss (MSE): {mean_squared_error(y_val, val_preds):.6f}")
    print("==================================================")

if __name__ == "__main__":
    run_external_mlp_benchmark()