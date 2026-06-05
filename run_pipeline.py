# run_pipeline.py
import logging
from config.config_loader import load_production_config
from data.csv_provider import CSVDataProvider
from data.stream_provider import StreamDataProvider
from models.controller import ModelController
from config.constants import IngestionMode, ModelType

def execute_training_pipeline():
    # 1. Hydrate the immutable, typed configuration object from your structured YAML
    cfg = load_production_config("config/config.yaml")
    
    # Configure baseline logging properties mapped directly from the meta sub-config
    if not cfg.meta.suppress_logging:
        log_level = getattr(logging, cfg.meta.logging_level.upper(), logging.INFO)
        logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s", force=True)
    
    logging.info(f"[Pipeline Root] Commencing pipeline initialization: {cfg.meta.pipeline_name}")

    # 2. Instantiate your ModelController first so it exists for down-funnel dependencies
    controller = ModelController(
        learning_rate=cfg.optimization.learning_rate,
        lr_scheduler_type=cfg.optimization.lr_scheduler,
        scheduler_decay_rate=cfg.optimization.scheduler_decay_rate,
        scheduler_drop_ratio=cfg.optimization.scheduler_drop_ratio,
        scheduler_epochs_per_drop=cfg.optimization.scheduler_epochs_per_drop
    )
    
    # 3. Build the underlying network architecture topology map upfront
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

    # 4. Extract the resolved Enum object from your config to route ingestion safely
    source_mode = cfg.ingestion.source_mode

    if source_mode == IngestionMode.CSV:
        data_provider = CSVDataProvider(
            data_file_path=cfg.ingestion.data_file_path,  # Explicit primitive string
            feature_names=cfg.ingestion.feature_names,     # Explicit primitive list
            batch_size=cfg.optimization.batch_size,        # Explicit primitive int
            model_instance=controller.model                # Injected to compute one-hot targets safely upfront
        )
    elif source_mode == IngestionMode.STREAM:
        if not cfg.ingestion.amqp_url or not cfg.ingestion.queue_name:
            raise ValueError("[Ingestion Error] AMQP properties must be defined in config when source_mode='stream'")
            
        data_provider = StreamDataProvider(
            amqp_url=cfg.ingestion.amqp_url,
            queue_name=cfg.ingestion.queue_name,
            feature_names=cfg.ingestion.feature_names
        )
    else:
        raise ValueError(f"[Ingestion Error] Unknown source_mode option: '{source_mode}'.")
        
    # 5. Handle pre-trained network loading if required by context rules
    if cfg.persistence.load_saved_model:
        controller.hydrate_from_asset(cfg.persistence.model_asset_path)
        
    # 6. Kick off optimization loops using the unified strategy engine interface
    train_history, val_history = controller.fit_via_provider(
        data_provider=data_provider,
        epochs=cfg.optimization.epochs,
        batch_size=cfg.optimization.batch_size,
        source_mode=source_mode,
        model_type=cfg.architecture.model_type,
        early_stopping_enabled=cfg.optimization.early_stopping_enabled,
        patience=cfg.optimization.patience,
        min_delta=cfg.optimization.min_delta
    )
    
    # 7. Execute state serialization if continuous saving paths are active
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