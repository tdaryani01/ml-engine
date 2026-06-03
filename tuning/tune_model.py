import optuna
import numpy as np
from config.config_loader import load_production_config
from data.data_loader import load_pipeline_splits
from models.model_factory import ModelFactory

def objective(trial):
    # 1. Load application configurations and ingest raw data splits
    cfg = load_production_config()
    env_cfg = cfg["environment"]
    dat_cfg = cfg["data"]
    arch_cfg = cfg["architecture"]
    
    splits = load_pipeline_splits(dat_cfg, env_cfg["data_file"])
    
    # 2. Compute normalizations for input vectors (X)
    mean = np.mean(splits["X_train"], axis=0)
    std = np.std(splits["X_train"], axis=0) + 1e-24
    X_train_norm = (splits["X_train"] - mean) / std
    X_val_norm = (splits["X_val"] - mean) / std
    
    # 3. Dynamic target scaling guard for regression tracking stability
    is_regression = str(arch_cfg.get("model_type", "")).strip().lower() == "regression"
    if is_regression:
        y_mean = np.mean(splits["y_train"], axis=0)
        y_std = np.std(splits["y_train"], axis=0) + 1e-24
        y_train_target = (splits["y_train"] - y_mean) / y_std
        y_val_target = (splits["y_val"] - y_mean) / y_std
    else:
        y_train_target = splits["y_train"]
        y_val_target = splits["y_val"]
        
    # 4. Resolve topology layout dimensions based on Optuna suggest metrics
    input_dim = splits["X_train"].shape[1]
    output_dim = splits["y_train"].shape[1]
    
    num_layers = trial.suggest_int('num_layers', 1, 3)
    layout = [input_dim]
    for i in range(num_layers):
        layout.append(trial.suggest_int(f'layer_{i+1}_size', 8, 32))
    layout.append(output_dim)
    
    # 5. Build the abstract target head using your creational factory
    model = ModelFactory.create_model(
        model_type=arch_cfg["model_type"],
        layer_sizes=tuple(layout),
        lr=trial.suggest_float('lr', 1e-4, 1e-1, log=True),
        optimizer='adam',
        lam_l1=trial.suggest_float('lam_l1', 1e-5, 1e-1, log=True),
        lam_l2=trial.suggest_float('lam_l2', 1e-5, 1e-1, log=True)
    )
    
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
    m_samples = X_train_norm.shape[0]
    
    # 6. Optimization Loop
    for epoch in range(100):
        indices = np.arange(m_samples)
        np.random.shuffle(indices)
        X_s = X_train_norm[indices]
        y_s = y_train_target[indices]
        
        for i in range(0, m_samples, batch_size):
            X_b = X_s[i:i+batch_size]
            y_b = y_s[i:i+batch_size]
            
            # Subclass internally handles step activations and backwards passes
            model.backward(X_b, y_b)
            
        # Evaluate model convergence using your integrated polymorphic cost formula
        val_loss = model.compute_total_loss(model.forward(X_val_norm), y_val_target)
        
        # Report metric to optuna engine for early pruning checks
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return model.compute_total_loss(model.forward(X_val_norm), y_val_target)

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15)
    print("\nBest Parameters found:\n", study.best_params)