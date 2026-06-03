# run_pipeline.py
import numpy as np
import logging
from config.config_loader import load_production_config
from data.data_loader import load_pipeline_splits
from models.controller import ModelController
from diagnostics.diagnostics import NeuralNetworkDiagnostics
from utils.logger import initialize_global_logging

def pre_compute_fourier_space(X_matrix, num_frequencies=4):
    """
    Transforms an (N, D) matrix into a higher-dimensional coordinate space
    using sin/cos mappings at progressive frequency scales across all features.
    Runs exactly once before the epoch loop to minimize CPU overhead.
    """
    features = [X_matrix]
    for i in range(num_frequencies):
        freq = 2.0 ** i
        features.append(np.sin(X_matrix * freq))
        features.append(np.cos(X_matrix * freq))
    return np.hstack(features)

def execute_training_pipeline(config, controller, splits, X_test_clean_raw=None, original_features=None):
    """Orchestration root isolated completely from execution mechanics."""
    env_cfg = config["environment"]
    dat_cfg = config["data"]
    trn_cfg = config["training"]
    diag_cfg = config["diagnostics"]

    # Fire the controller's training cycle (which handles schedulers and auto-saving internally)
    train_history, val_history = controller.fit(splits)

    # Handle telemetry reporting if stage flags match development requirements
    is_dev = env_cfg.get("stage", "").lower() == "development"
    if is_dev and diag_cfg.get("enabled", False):
        logging.info("Dispatching data metrics to diagnostics manager...")
        
        y_mean = getattr(controller.model, "y_mean", None)
        y_std = getattr(controller.model, "y_std", None)
        
        # DYNAMIC HOOK: If the problem space was raw, match the visual diagnostic inputs 
        # to the clean unpolluted raw copy so plotting indices don't corrupt.
        fourier_cfg = trn_cfg.get("fourier_expansion", {})
        if fourier_cfg.get("enabled", False) and X_test_clean_raw is not None:
            X_diagnostic_input = X_test_clean_raw
            diagnostic_features = original_features if original_features else ["Input_X"]
        else:
            X_diagnostic_input = splits["X_test"]
            diagnostic_features = dat_cfg["feature_names"]

        NeuralNetworkDiagnostics.run_diagnostics(
            metric_to_plot=diag_cfg["metric_to_plot"], model=controller.model,
            train_history=train_history, val_history=val_history,
            X_test_raw=X_diagnostic_input, y_test=splits["y_test"],
            feature_names=diagnostic_features, 
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
    
    # Ingest dataset splits from source paths (Single Unified Source of Truth)
    logging.info("Loading framework raw pipeline splits...")
    dataset_splits = load_pipeline_splits(global_config["data"], global_config["environment"]["data_file"])

    # DYNAMIC CACHE: Preserve unpolluted features before any potential list extensions occur
    original_feature_names = list(global_config["data"]["feature_names"])
    X_test_clean_raw = dataset_splits["X_test"].copy()

    # Dynamic Preprocessing Layer Execution nested under the training config block
    training_cfg = global_config.get("training", {})
    fourier_cfg = training_cfg.get("fourier_expansion", {})
    
    # REMOVED: Slicing restriction 'shape[1] == 1'. Spans safely over arbitrary dimensions.
    if fourier_cfg.get("enabled", False):
        num_freqs = fourier_cfg.get("num_frequencies", 4)
        logging.info(f"[Preprocessing] Applying Multivariable Fourier Feature Expansion (Frequencies: {num_freqs})...")
        
        # Transform the training and testing matrices exactly once in memory
        dataset_splits["X_train"] = pre_compute_fourier_space(dataset_splits["X_train"], num_frequencies=num_freqs)
        dataset_splits["X_test"] = pre_compute_fourier_space(dataset_splits["X_test"], num_frequencies=num_freqs)
        
        # Automatically generate structural column layouts for all features dynamically
        patched_feature_names = []
        for base_name in original_feature_names:
            patched_feature_names.append(base_name)
            for i in range(num_freqs):
                patched_feature_names.extend([f"{base_name}_sin_{i}", f"{base_name}_cos_{i}"])
            
        # Patch the configuration dictionary in-memory to guide framework initialization
        global_config['data']['feature_names'] = patched_feature_names

    # Unified Framework Target Sanity Check
    logging.info("="*50)
    logging.info("          DATASTREAM TARGET MATRIX PROFILE")
    logging.info("="*50)
    logging.info(f"Unique target element footprint: {np.unique(dataset_splits['y_train'])}")
    logging.info(f"Target underlying array data type: {dataset_splits['y_train'].dtype}")
    logging.info(f"Target structural matrix shape  : {dataset_splits['y_train'].shape}")
    logging.info("="*50)
    
    # Run temporary debug log right after target validation pass
    logging.info("="*50)
    logging.info("          INPUT FEATURE SCALE PROFILE")
    logging.info("="*50)
    logging.info(f"X_train Min bounds: {np.min(dataset_splits['X_train'], axis=0)}")
    logging.info(f"X_train Max bounds: {np.max(dataset_splits['X_train'], axis=0)}")
    logging.info(f"X_train Mean vector: {np.mean(dataset_splits['X_train'], axis=0)}")
    logging.info("="*50)
    
    # Instantiate the lightweight controller container
    active_controller = ModelController(config=global_config)
    
    # Setup handles configuration routing or checkpoint restoration internally using the serializer
    active_controller.setup_model(dataset_splits)
    
    # Fire execution pipeline with the identical data splits passed along
    execute_training_pipeline(
        config=global_config, 
        controller=active_controller, 
        splits=dataset_splits, 
        X_test_clean_raw=X_test_clean_raw,
        original_features=original_feature_names
    )