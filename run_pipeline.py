# run_pipeline.py
import logging
from config.config_loader import load_production_config
from data.csv_provider import CSVDataProvider
from data.stream_provider import StreamDataProvider
from models.controller import ModelController
from config.constants import IngestionMode
from utils.logger import initialize_global_logging              # Injected custom logger
from utils.diagnostics import NeuralNetworkDiagnostics

def execute_training_pipeline():
    # 1. Hydrate the immutable, typed configuration object from your structured YAML
    cfg = load_production_config("config/config.yaml")
    
    # 2. Boot up your custom enterprise dual-destination logging architecture
    initialize_global_logging(cfg)

    # 3. Instantiate your ModelController first so it exists for down-funnel dependencies
    controller = ModelController(
        learning_rate=cfg.optimization.learning_rate,
        lr_scheduler_type=cfg.optimization.lr_scheduler,
        scheduler_decay_rate=cfg.optimization.scheduler_decay_rate,
        scheduler_drop_ratio=cfg.optimization.scheduler_drop_ratio,
        scheduler_epochs_per_drop=cfg.optimization.scheduler_epochs_per_drop
    )
    
    # 4. Build the underlying network architecture topology map upfront
    logging.info("[Pipeline Root] Initializing network topology matrix dimensions...")
    controller.initialize_network_from_dimensions(
        input_dim=len(cfg.ingestion.feature_names),
        output_dim=3,  # Mapped to match your structural multi-class tracking layout
        model_type=cfg.architecture.model_type,
        hidden_layers=cfg.architecture.hidden_layers,
        optimizer_name=cfg.optimization.optimizer,
        lam_l1=cfg.regularization.lam_l1,
        lam_l2=cfg.regularization.lam_l2,
        p_dropout=cfg.architecture.p_dropout,
        use_batch_norm=cfg.architecture.use_batch_norm,
        bn_momentum=cfg.architecture.bn_momentum,
        max_norm=cfg.optimization.gradient_clipping_max_norm
    )

    # 5. Extract the resolved Enum object from your config to route ingestion safely
    source_mode = cfg.ingestion.source_mode
    steps = cfg.optimization.steps_streaming

    if source_mode == IngestionMode.CSV:
        data_provider = CSVDataProvider(
            data_file_path=cfg.ingestion.data_file_path,
            feature_names=cfg.ingestion.feature_names,
            batch_size=cfg.optimization.batch_size,
            model_instance=controller.model,
            epochs=cfg.optimization.epochs_full_dataset
        )
        steps = data_provider.recomment_steps();

    elif source_mode == IngestionMode.STREAM:
        if not cfg.ingestion.amqp_url or not cfg.ingestion.queue_name:
            raise ValueError("[Ingestion Error] AMQP properties must be defined in config when source_mode='stream'")
            
        data_provider = StreamDataProvider(
            amqp_url=cfg.ingestion.amqp_url,
            queue_name=cfg.ingestion.queue_name,
            feature_names=cfg.ingestion.feature_names,
            batch_size=cfg.optimization.batch_size,
        )
    else:
        raise ValueError(f"[Ingestion Error] Unknown source_mode option: '{source_mode}'.")
        
    # 6. Handle pre-trained network loading if required by context rules
    if cfg.persistence.load_saved_model:
        controller.hydrate_from_asset(cfg.persistence.model_asset_path)
        
    # 7. Kick off optimization loops using the unified strategy engine interface
    train_history, val_history = controller.fit(
        data_provider=data_provider,
        steps=steps,
        source_mode=source_mode,
        model_type=cfg.architecture.model_type,
        early_stopping_enabled=cfg.optimization.early_stopping_enabled,
        patience=cfg.optimization.patience,
        min_delta=cfg.optimization.min_delta
    )
    
    # 8. Trigger pipeline diagnostics generation pass (Plots Figures 1-5 + history.csv)
    NeuralNetworkDiagnostics.run_diagnostics(
        controller=controller,
        data_provider=data_provider,
        cfg=cfg
    )
    
    # 9. Execute state serialization if continuous saving paths are active
    if cfg.persistence.load_saved_model:
        legacy_config_dict = {
            "meta": vars(cfg.meta),
            "ingestion": vars(cfg.ingestion),
            "architecture": vars(cfg.architecture),
            "optimization": vars(cfg.optimization),
            "regularization": vars(cfg.regularization),
            "persistence": vars(cfg.persistence)
        }
        controller.serialize_current_state(
            target_asset_path=cfg.persistence.model_asset_path,
            serialized_config_dict=legacy_config_dict
        )

if __name__ == "__main__":
    execute_training_pipeline()