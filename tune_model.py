# tune_model.py
import optuna
import numpy as np
import logging
import yaml  # Added to handle direct file manipulation
from config.config_loader import load_production_config
from data.data_loader import load_pipeline_splits
from models.controller import ModelController

logging.getLogger().setLevel(logging.WARNING)

def pre_compute_fourier_space(X_matrix, num_frequencies=4):
    """Transforms an (N, D) matrix into a higher-dimensional periodic coordinate space."""
    features = [X_matrix]
    for i in range(num_frequencies):
        freq = 2.0 ** i
        features.append(np.sin(X_matrix * freq))
        features.append(np.cos(X_matrix * freq))
    return np.hstack(features)

def objective(trial):
    # 1. Ingest clean base configuration instance matrices
    cfg = load_production_config()
    splits = load_pipeline_splits(cfg["data"], cfg["environment"]["data_file"])
    
    # 2. Structural Hidden Topology Search Space
    num_layers = trial.suggest_int('num_layers', 2, 3)
    hidden_layers = []
    for i in range(num_layers):
        hidden_layers.append(trial.suggest_int(f'layer_{i+1}_size', 16, 48))
        
    cfg["architecture"]["hidden_layers"] = hidden_layers
    cfg["architecture"]["p_dropout"] = trial.suggest_float('p_dropout', 0.0, 0.4)
    cfg["training"]["batch_size"] = trial.suggest_categorical('batch_size', [16, 32])
    cfg["regularization"]["lam_l1"] = trial.suggest_float('lam_l1', 1e-6, 1e-3, log=True)
    cfg["regularization"]["lam_l2"] = trial.suggest_float('lam_l2', 1e-6, 1e-3, log=True)
    
    # Cap total structural evaluations per trial step
    cfg["training"]["epochs"] = 150  

    # =====================================================================
    # TUNING THE LEARNING RATE SCHEDULER
    # =====================================================================
    cfg["training"]["learning_rate"] = trial.suggest_float('lr', 1e-4, 5e-2, log=True)
    
    scheduler_strategy = trial.suggest_categorical('lr_scheduler', ['none', 'step', 'exponential'])
    cfg["training"]["lr_scheduler"] = scheduler_strategy
    
    if scheduler_strategy == 'exponential':
        cfg["training"]["scheduler_decay_rate"] = trial.suggest_float('scheduler_decay_rate', 0.90, 0.99)
    elif scheduler_strategy == 'step':
        cfg["training"]["scheduler_drop_ratio"] = trial.suggest_float('scheduler_drop_ratio', 0.2, 0.7)
        cfg["training"]["scheduler_epochs_per_drop"] = trial.suggest_int('scheduler_epochs_per_drop', 10, 50)

    # =====================================================================
    # TUNING EARLY STOPPING
    # =====================================================================
    cfg["training"]["early_stopping_enabled"] = True
    cfg["training"]["patience"] = trial.suggest_int('patience', 10, 40)
    cfg["training"]["min_delta"] = trial.suggest_float('min_delta', 1e-6, 1e-4, log=True)
    
    # Guardrails: Force structural binary writing and hydration to False during optimization trials
    cfg["environment"]["load_saved_model"] = False

    # =====================================================================
    # PRODUCTION PARITY: Pre-compute Train/Test Expansion Before Model Setup
    # =====================================================================
    training_cfg = cfg.get("training", {})
    fourier_cfg = training_cfg.get("fourier_expansion", {})
    if fourier_cfg.get("enabled", False):
        num_freqs = fourier_cfg.get("num_frequencies", 4)
        
        # Mirror production pipeline by transforming active training inputs upfront
        splits["X_train"] = pre_compute_fourier_space(splits["X_train"], num_frequencies=num_freqs)
        splits["X_test"] = pre_compute_fourier_space(splits["X_test"], num_frequencies=num_freqs)
        
        # Patch string labels in configuration copy to correctly build input weights
        original_features = list(cfg["data"]["feature_names"])
        patched_feature_names = []
        for base_name in original_features:
            patched_feature_names.append(base_name)
            for i in range(num_freqs):
                patched_feature_names.extend([f"{base_name}_sin_{i}", f"{base_name}_cos_{i}"])
        cfg['data']['feature_names'] = patched_feature_names
    
    # 3. Instantiate Controller Wrapper
    controller = ModelController(config=cfg)
    controller.setup_model(splits)
    
    # Execute trial training run
    train_history, val_history = controller.fit(splits)
    
    # 4. Score out-of-sample trial performance metrics using clean evaluation paths (training=False)
    val_preds = controller.predict(splits["X_val"])
    task_type = str(cfg["architecture"]["model_type"]).strip().lower()
    
    if task_type == "multi_class":
        score = np.mean(np.argmax(val_preds, axis=1) == splits["y_val"].ravel().astype(int))
    elif task_type == "regression":
        _, y_val_target, _ = controller.model.preprocess_targets(splits["y_train"], splits["y_val"])
        score = controller.model.compute_total_loss(val_preds, y_val_target)
    else:  # Binary Classification
        score = np.mean((val_preds > 0.5).astype(int) == splits["y_val"])
        
    return score

def update_production_config_yaml(best_params, yaml_path=".\\config\\config.yaml"):
    """Maps the flattened Optuna trial parameters back into their proper nested locations."""
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # 1. Map dynamic layer topologies
        num_layers = best_params["num_layers"]
        hidden_layers = [best_params[f"layer_{i+1}_size"] for i in range(num_layers)]
        config["architecture"]["hidden_layers"] = hidden_layers
        config["architecture"]["p_dropout"] = float(best_params["p_dropout"])
        
        # 2. Map loss penalty architectures
        config["regularization"]["lam_l1"] = float(best_params["lam_l1"])
        config["regularization"]["lam_l2"] = float(best_params["lam_l2"])
        
        # 3. Map foundational optimization blocks
        config["training"]["batch_size"] = int(best_params["batch_size"])
        config["training"]["learning_rate"] = float(best_params["lr"])
        config["training"]["lr_scheduler"] = str(best_params["lr_scheduler"])
        
        # 4. Map active learning rate scheduler configurations safely
        if best_params["lr_scheduler"] == "exponential":
            config["training"]["scheduler_decay_rate"] = float(best_params["scheduler_decay_rate"])
        elif best_params["lr_scheduler"] == "step":
            config["training"]["scheduler_drop_ratio"] = float(best_params["scheduler_drop_ratio"])
            config["training"]["scheduler_epochs_per_drop"] = int(best_params["scheduler_epochs_per_drop"])
            
        # 5. Map convergence guards
        config["training"]["patience"] = int(best_params["patience"])
        config["training"]["min_delta"] = float(best_params["min_delta"])
        
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"\n[Success] Local production file '{yaml_path}' seamlessly updated with optimized parameters.")
    except Exception as e:
        print(f"\n[Error] Failed to patch configuration file: {e}")

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARNING)
    
    cfg_preview = load_production_config()
    task_variant = str(cfg_preview["architecture"]["model_type"]).strip().lower()
    optuna_direction = 'minimize' if task_variant == 'regression' else 'maximize'
    
    print(f"[Optuna Tuning] Optimization suite initiated over active BatchNorm components. Direction: {optuna_direction}")
    
    study = optuna.create_study(direction=optuna_direction)
    study.optimize(objective, n_trials=20)
    
    print("\n" + "="*50 + "\n          OPTIMAL PARAMETERS FOUND\n" + "="*50)
    for param_key, param_val in study.best_params.items():
        print(f"  {param_key:<25}: {param_val}")
    print("="*50)

    # =====================================================================
    # INTERACTIVE POST-TUNING OVERWRITE STEP
    # =====================================================================
    user_choice = input("\nDo you want to write these optimal hyperparameters back to 'config.yaml'? (yes/no): ").strip().lower()
    if user_choice in ['yes', 'y']:
        update_production_config_yaml(study.best_params, yaml_path=".\\config\\config.yaml")
    else:
        print("\n[Tuning Intercept] Write path skipped. Your original configuration parameters remain unmodified.")