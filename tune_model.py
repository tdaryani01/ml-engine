# tune_model.py
import optuna
import numpy as np
import logging
from config.config_loader import load_production_config
from data.data_loader import load_pipeline_splits
from models.controller import ModelController

logging.getLogger().setLevel(logging.WARNING)

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
    # FIXED: Maps parameter target to 'learning_rate' to match config matrix schema exactly
    cfg["training"]["learning_rate"] = trial.suggest_float('lr', 1e-4, 5e-2, log=True)
    
    scheduler_strategy = trial.suggest_categorical('lr_scheduler', ['none', 'step', 'exponential'])
    cfg["training"]["lr_scheduler"] = scheduler_strategy
    
    # Handle conditional hyperparameter branches depending on scheduler strategy selection
    if scheduler_strategy == 'exponential':
        cfg["training"]["scheduler_decay_rate"] = trial.suggest_float('scheduler_decay_rate', 0.90, 0.99)
    elif scheduler_strategy == 'step':
        cfg["training"]["scheduler_drop_ratio"] = trial.suggest_float('scheduler_drop_ratio', 0.2, 0.7)
        cfg["training"]["scheduler_epochs_per_drop"] = trial.suggest_int('scheduler_epochs_per_drop', 10, 50)

    # =====================================================================
    # TUNING EARLY STOPPING
    # =====================================================================
    # Active convergence optimization monitors prune bad configurations early
    cfg["training"]["early_stopping_enabled"] = True
    cfg["training"]["patience"] = trial.suggest_int('patience', 10, 40)
    cfg["training"]["min_delta"] = trial.suggest_float('min_delta', 1e-6, 1e-4, log=True)
    
    # Guardrails: Force structural binary writing and hydration to False during optimization trials
    cfg["environment"]["load_saved_model"] = False
    
    # 3. Instantiate Controller Wrapper
    controller = ModelController(config=cfg)
    controller.setup_model(splits)
    
    # Execute trial training run (which seamlessly leverages Batch Normalization under the hood!)
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

if __name__ == "__main__":
    # Quiet verbose epoch logging reports to isolate print results clean
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