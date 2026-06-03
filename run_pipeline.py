# run_pipeline.py
import numpy as np
import logging
from config.config_loader import load_production_config
from data.data_loader import load_pipeline_splits
from models.controller import ModelController
from diagnostics.diagnostics import NeuralNetworkDiagnostics
from utils.logger import initialize_global_logging

def execute_training_pipeline(config, controller, splits):
    """Orchestration root isolated completely from execution mechanics."""
    env_cfg = config["environment"]
    dat_cfg = config["data"]
    trn_cfg = config["training"]
    diag_cfg = config["diagnostics"]

    # FIXED: Eliminated the duplicate load_pipeline_splits call that caused random index shifting
        
    # 2. Fire the controller's training cycle (which handles schedulers and auto-saving internally)
    train_history, val_history = controller.fit(splits)

    # 3. Handle telemetry reporting if stage flags match development requirements
    is_dev = env_cfg.get("stage", "").lower() == "development"
    if is_dev and diag_cfg.get("enabled", False):
        logging.info("Dispatching data metrics to diagnostics manager...")
        
        y_mean = getattr(controller.model, "y_mean", None)
        y_std = getattr(controller.model, "y_std", None)
        
        NeuralNetworkDiagnostics.run_diagnostics(
            metric_to_plot=diag_cfg["metric_to_plot"], model=controller.model,
            train_history=train_history, val_history=val_history,
            X_test_raw=splits["X_test"], y_test=splits["y_test"],
            feature_names=dat_cfg["feature_names"], 
            mean=controller.mean, std=controller.std,
            output_dir=env_cfg["output_dir"],
            y_mean=y_mean, y_std=y_std
        )
        NeuralNetworkDiagnostics.inspect_model_sparsity(controller.model, tolerance=trn_cfg["sparsity_tolerance"])

# =====================================================================
# Unified Composition Root Block
# =====================================================================
if __name__ == "__main__":
    # Load production configurations from the YAML matrix
    global_config = load_production_config()
    
    # Fire up global system logger and error catch configurations immediately
    initialize_global_logging(global_config)
    
    # 1. Ingest dataset splits from source paths (Single Unified Source of Truth)
    logging.info("Loading framework raw pipeline splits...")
    dataset_splits = load_pipeline_splits(global_config["data"], global_config["environment"]["data_file"])


    # Unified Framework Target Sanity Check
    logging.info("="*50)
    logging.info("          DATASTREAM TARGET MATRIX PROFILE")
    logging.info("="*50)
    logging.info(f"Unique target element footprint: {np.unique(dataset_splits['y_train'])}")
    logging.info(f"Target underlying array data type: {dataset_splits['y_train'].dtype}")
    logging.info(f"Target structural matrix shape  : {dataset_splits['y_train'].shape}")
    logging.info("="*50)
    # run_pipeline.py
    # Add this temporary debug log right after line 63
    logging.info("="*50)
    logging.info("          INPUT FEATURE SCALE PROFILE")
    logging.info("="*50)
    logging.info(f"X_train Min bounds: {np.min(dataset_splits['X_train'], axis=0)}")
    logging.info(f"X_train Max bounds: {np.max(dataset_splits['X_train'], axis=0)}")
    logging.info(f"X_train Mean vector: {np.mean(dataset_splits['X_train'], axis=0)}")
    logging.info("="*50)
    
    # 2. Instantiate the lightweight controller container
    active_controller = ModelController(config=global_config)
    
    # 3. Setup handles configuration routing or checkpoint restoration internally using the serializer
    active_controller.setup_model(dataset_splits)
    
    # 4. Fire execution pipeline with the identical data splits passed along
    execute_training_pipeline(config=global_config, controller=active_controller, splits=dataset_splits)