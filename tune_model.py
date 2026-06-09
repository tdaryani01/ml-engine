# tune_model.py
import optuna
import numpy as np
import logging
import yaml
import os
from dataclasses import replace
from config.config_loader import load_production_config
from data.data_loader import load_pipeline_splits
from models.controller import ModelController
from config.constants import ModelType, LRHierarchy

logging.getLogger().setLevel(logging.WARNING)

class DataProviderAdapter:
    """
    Lightweight adapter to wrap the raw splits dictionary 
    into the object contract expected by the ModelController.
    """
    def __init__(self, splits):
        self.splits = splits
        self.mean = np.mean(splits["X_train"], axis=0)
        self.std = np.std(splits["X_train"], axis=0) + 1e-8
        
        # Format true targets cleanly
        self.y_train_processed = splits["y_train"]

    def get_validation_set(self):
        return self.splits["X_val"], self.splits["y_val"]

    def normalize(self, X):
        return (X - self.mean) / self.std

    def reset_epoch(self):
        self._batch_idx = 0
        self._indices = np.arange(self.splits["X_train"].shape[0])
        np.random.shuffle(self._indices)

    def has_more_batches(self):
        return hasattr(self, '_indices') and self._batch_idx < len(self._indices)

    def next_batch(self, batch_size=64):
        end_idx = min(self._batch_idx + batch_size, len(self._indices))
        batch_ids = self._indices[self._batch_idx:end_idx]
        self._batch_idx = end_idx
        return self.splits["X_train"][batch_ids], self.splits["y_train"][batch_ids]


def pre_compute_fourier_space(X_matrix, num_frequencies=4):
    """Transforms an (N, D) matrix into a higher-dimensional periodic coordinate space."""
    features = [X_matrix]
    for i in range(num_frequencies):
        freq = 2.0 ** i
        features.append(np.sin(X_matrix * freq))
        features.append(np.cos(X_matrix * freq))
    return np.hstack(features)

def objective(trial):
    # 1. Ingest baseline configuration instance schema
    cfg = load_production_config()
    splits = load_pipeline_splits(cfg.ingestion.data_file_path, cfg.ingestion.feature_names)
    
    # 2. Structural Hidden Topology Search Space
    num_layers = trial.suggest_int('num_layers', 2, 3)
    hidden_layers = []
    for i in range(num_layers):
        hidden_layers.append(trial.suggest_int(f'layer_{i+1}_size', 16, 48))
        
    p_dropout = trial.suggest_float('p_dropout', 0.0, 0.4)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    lam_l1 = trial.suggest_float('lam_l1', 1e-6, 1e-3, log=True)
    lam_l2 = trial.suggest_float('lam_l2', 1e-6, 1e-3, log=True)
    learning_rate = trial.suggest_float('lr', 1e-4, 5e-2, log=True)
    
    scheduler_strategy = trial.suggest_categorical('lr_scheduler', ['none', 'step', 'exponential'])
    
    # Extract scheduler variants
    scheduler_decay_rate = 0.98
    scheduler_drop_ratio = 0.5
    scheduler_epochs_per_drop = 20
    if scheduler_strategy == 'exponential':
        scheduler_decay_rate = trial.suggest_float('scheduler_decay_rate', 0.90, 0.99)
    elif scheduler_strategy == 'step':
        scheduler_drop_ratio = trial.suggest_float('scheduler_drop_ratio', 0.2, 0.7)
        scheduler_epochs_per_drop = trial.suggest_int('scheduler_epochs_per_drop', 10, 50)

    patience = trial.suggest_int('patience', 10, 40)
    min_delta = trial.suggest_float('min_delta', 1e-6, 1e-4, log=True)

    # Convert strategy strings cleanly to Enum variants expected by controller
    lr_enum = LRHierarchy.NONE
    if scheduler_strategy == 'step':
        lr_enum = LRHierarchy.STEP
    elif scheduler_strategy == 'exponential':
        lr_enum = LRHierarchy.EXPONENTIAL

    # =====================================================================
    # PRE-COMPUTE ALL SPLITS (Train, Val, Test) Upfront
    # =====================================================================
    fourier_cfg = cfg.transformations.fourier_expansion
    if fourier_cfg.enabled:
        num_freqs = fourier_cfg.num_frequencies
        splits["X_train"] = pre_compute_fourier_space(splits["X_train"], num_frequencies=num_freqs)
        splits["X_val"]   = pre_compute_fourier_space(splits["X_val"], num_frequencies=num_freqs)
        splits["X_test"]  = pre_compute_fourier_space(splits["X_test"], num_frequencies=num_freqs)

    # Wrap raw splits dictionary in adaptive data provider interface
    provider = DataProviderAdapter(splits)

    # 3. Instantiate Controller Wrapper via updated flat constructor signature
    controller = ModelController(
        learning_rate=learning_rate,
        lr_scheduler_type=lr_enum,
        scheduler_decay_rate=scheduler_decay_rate,
        scheduler_drop_ratio=scheduler_drop_ratio,
        scheduler_epochs_per_drop=scheduler_epochs_per_drop
    )
    
    # Initialize internal layer allocations cleanly using the updated controller API contract
    m_type = ModelType.MULTI_CLASS if str(cfg.architecture.model_type).lower() == "multi_class" else ModelType.BINARY_CLASSIFICATION
    
    controller.initialize_network_from_provider(
        data_provider=provider,
        model_type=m_type,
        hidden_layers=hidden_layers,
        optimizer_name=cfg.optimization.optimizer,
        lam_l1=lam_l1,
        lam_l2=lam_l2,
        p_dropout=p_dropout,
        use_batch_norm=cfg.architecture.use_batch_norm,
        bn_momentum=cfg.architecture.bn_momentum,
        max_norm=cfg.optimization.gradient_clipping_max_norm
    )
    
    # Execute trial training run (passing the total step count and model enum)
    train_history, val_history = controller.fit(
        data_provider=provider,
        steps=cfg.optimization.steps_streaming,
        source_mode=cfg.ingestion.source_mode,
        model_type=m_type,
        early_stopping_enabled=True,
        patience=patience,
        min_delta=min_delta
    )
    
    # 4. Score out-of-sample trial performance metrics
    val_preds = controller.predict(splits["X_val"])
    
    if m_type == ModelType.MULTI_CLASS:
        if val_preds.ndim > 1 and val_preds.shape[1] > 1:
            pred_classes = np.argmax(val_preds, axis=1)
        else:
            pred_classes = val_preds.ravel()
            
        true_classes = np.argmax(splits["y_val"], axis=1) if (splits["y_val"].ndim > 1 and splits["y_val"].shape[1] > 1) else splits["y_val"].ravel().astype(int)
        score = np.mean(pred_classes == true_classes)
    else:  # Binary Classification
        if val_preds.ndim > 1 and val_preds.shape[1] > 1:
            prob_vector = val_preds[:, 1] if val_preds.shape[1] == 2 else val_preds[:, 0]
        else:
            prob_vector = val_preds.ravel()
        score = np.mean((prob_vector > 0.5).astype(int) == splits["y_val"].ravel().astype(int))
        
    return score

def update_production_config_yaml(best_params, yaml_path=".\\config\\config.yaml"):
    """Maps the flattened Optuna trial parameters back into their proper nested locations."""
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            
        num_layers = best_params["num_layers"]
        hidden_layers = [best_params[f"layer_{i+1}_size"] for i in range(num_layers)]
        config["architecture"]["hidden_layers"] = hidden_layers
        config["architecture"]["p_dropout"] = float(best_params["p_dropout"])
        
        config["regularization"]["lam_l1"] = float(best_params["lam_l1"])
        config["regularization"]["lam_l2"] = float(best_params["lam_l2"])
        
        config["optimization"]["batch_size"] = int(best_params["batch_size"])
        config["optimization"]["learning_rate"] = float(best_params["lr"])
        config["optimization"]["lr_scheduler"] = str(best_params["lr_scheduler"])
        
        for param in ["scheduler_decay_rate", "scheduler_drop_ratio", "scheduler_epochs_per_drop"]:
            config["optimization"].pop(param, None)
            
        if best_params["lr_scheduler"] == "exponential":
            config["optimization"]["scheduler_decay_rate"] = float(best_params["scheduler_decay_rate"])
        elif best_params["lr_scheduler"] == "step":
            config["optimization"]["scheduler_drop_ratio"] = float(best_params["scheduler_drop_ratio"])
            config["optimization"]["scheduler_epochs_per_drop"] = int(best_params["scheduler_epochs_per_drop"])
            
        config["optimization"]["patience"] = int(best_params["patience"])
        config["optimization"]["min_delta"] = float(best_params["min_delta"])
        
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"\n[Success] Local production file '{yaml_path}' seamlessly updated with optimized parameters.")
    except Exception as e:
        print(f"\n[Error] Failed to patch configuration file: {e}")

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARNING)
    
    cfg_preview = load_production_config()
    task_variant = str(cfg_preview.architecture.model_type).strip().lower()
    optuna_direction = 'minimize' if task_variant == 'regression' else 'maximize'
    
    print(f"[Optuna Tuning] Optimization suite initiated over active component blocks. Direction: {optuna_direction}")
    
    study = optuna.create_study(
        direction=optuna_direction,
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=101)
    )
    study.optimize(objective, n_trials=20)
    
    print("\n" + "="*50 + "\n          OPTIMAL PARAMETERS FOUND\n" + "="*50)
    for param_key, param_val in study.best_params.items():
        print(f"  {param_key:<25}: {param_val}")
    print("="*50)

    user_choice = input("\nDo you want to write these optimal hyperparameters back to 'config.yaml'? (yes/no): ").strip().lower()
    if user_choice in ['yes', 'y']:
        update_production_config_yaml(study.best_params, yaml_path=".\\config\\config.yaml")
    else:
        print("\n[Tuning Intercept] Write path skipped. Your original configuration parameters remain unmodified.")