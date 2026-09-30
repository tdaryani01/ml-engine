# run_pipeline.py
import builtins

try:
    profile  # noqa: F821 — injected by kernprof -l
except NameError:
    builtins.profile = lambda f: f

import logging

from config.config_loader import (
    load_boot_host_context,
    load_production_config,
    parse_tm_production_config,
)
from config.config_source import (
    ConfigSource,
    detect_config_source,
    load_tm_payload,
    tm_payload_model_id,
)
from config.constants import EngineBackend
from utils.runtime import (
    apply_process_env,
    configure_runtime,
    load_runtime_settings,
    log_runtime_settings,
    training_threadpool,
)

from utils.perf_experiments import apply_im2col_gemm_perf_defaults, experiment_summary

BOOT_YAML = "config/config.yaml"

# Config provenance is fixed for the life of the process, and it is resolved
# BEFORE NumPy/SciPy BLAS first touch so thread env is pinned either way.
#   LOCAL: config.yaml is the config.
#   TRAINING_MANAGER: the TM payload ($ML_ENGINE_TM_PAYLOAD) is parsed strictly
#   (no YAML gap-filling); boot YAML supplies only host identity + output_dir.
_CONFIG_SOURCE = detect_config_source()
_TM_PAYLOAD: dict | None = None
if _CONFIG_SOURCE == ConfigSource.TRAINING_MANAGER:
    _TM_PAYLOAD = load_tm_payload()
    _host_identity, _host_output_dir = load_boot_host_context(BOOT_YAML)
    _cfg = parse_tm_production_config(
        _TM_PAYLOAD,
        host_identity=_host_identity,
        output_dir=_host_output_dir,
    )
    if _cfg.architecture.backend == EngineBackend.IM2COL_GEMM:
        apply_im2col_gemm_perf_defaults()
    # Thread budget + async knob come from the payload; runtime.yaml still
    # supplies the process env (OMP_MAX_ACTIVE_LEVELS, PYTHONUNBUFFERED, …).
    _RUNTIME = configure_runtime(
        _cfg.architecture.backend,
        config_path=BOOT_YAML,
        num_threads=_cfg.optimization.num_threads,
        native_async_submit=bool(getattr(_cfg.ledger, "native_async_submit", False)),
        config_source=_CONFIG_SOURCE,
        overwrite_env=True,
        if_unset_env=False,
        log=False,
    )
else:
    # Pin thread env + native OpenBLAS before NumPy/SciPy BLAS first touch.
    _cfg = load_production_config(BOOT_YAML)
    if _cfg.architecture.backend == EngineBackend.IM2COL_GEMM:
        apply_im2col_gemm_perf_defaults()
    apply_process_env(load_runtime_settings(), overwrite=True, if_unset=False)
    _RUNTIME = configure_runtime(_cfg.architecture.backend, log=False, overwrite_env=True)

from config.constants import IngestionMode, ModelType, DataKeys
from config.schema import PipelineConfig
from src.data.base_loader import BaseDataLoader
from src.data.in_memory_provider import InMemoryDataProvider
from src.data.stream_provider import StreamDataProvider
from src.controller import ModelController
from utils.logger import initialize_global_logging
from utils.diagnostics import NeuralNetworkDiagnostics

import numpy as np


def _build_provider_and_controller(cfg: PipelineConfig):
    """Assemble data provider + initialized network from a PipelineConfig."""
    is_cnn = cfg.architecture.model_type == ModelType.CNN
    is_mhsa = cfg.architecture.model_type == ModelType.MHSA
    source_mode = cfg.ingestion.source_mode
    cnn_cfg = getattr(cfg.architecture, "cnn", None) if is_cnn else None
    mhsa_cfg = getattr(cfg.architecture, "mhsa", None) if is_mhsa else None

    if source_mode == IngestionMode.STREAM:
        if not cfg.ingestion.amqp_url or not cfg.ingestion.queue_name:
            raise ValueError(
                "[Ingestion Error] AMQP properties must be defined in config when source_mode='stream'"
            )

        data_provider = StreamDataProvider(
            amqp_url=cfg.ingestion.amqp_url,
            queue_name=cfg.ingestion.queue_name,
            val_queue_name=cfg.ingestion.val_queue_name,
            feature_names=cfg.ingestion.feature_names,
            batch_size=cfg.optimization.batch_size,
            steps_per_epoch=cfg.optimization.steps_streaming,
            val_split_size=cfg.ingestion.splits.val,
            num_classes=cfg.architecture.num_classes,
            drain_on_empty=cfg.ingestion.drain_on_empty,
        )
        steps = cfg.optimization.steps_streaming
        input_dim = len(cfg.ingestion.feature_names)
    else:
        loader = BaseDataLoader.create_loader(cfg)

        data_provider = InMemoryDataProvider(
            loader=loader,
            batch_size=cfg.optimization.batch_size,
            epochs=cfg.optimization.epochs_full_dataset,
            normalize_features=(not is_cnn and not is_mhsa),
        )
        steps = data_provider.recomment_steps()

        if is_cnn:
            input_dim = int(
                np.prod(cnn_cfg["input_shape"] if isinstance(cnn_cfg, dict) else cnn_cfg.input_shape)
            )
        elif is_mhsa:
            X0 = data_provider.splits[DataKeys.X_TRAIN]
            input_dim = int(np.prod(X0.shape[1:]))
        else:
            input_dim = (
                len(cfg.ingestion.feature_names)
                if isinstance(cfg.ingestion.feature_names, list)
                else data_provider.splits[DataKeys.X_TRAIN].shape[1]
            )

    controller = ModelController(
        data_provider=data_provider,
        learning_rate=cfg.optimization.learning_rate,
        lr_scheduler_type=cfg.optimization.lr_scheduler,
        scheduler_decay_rate=cfg.optimization.scheduler_decay_rate,
        scheduler_drop_ratio=cfg.optimization.scheduler_drop_ratio,
        scheduler_epochs_per_drop=cfg.optimization.scheduler_epochs_per_drop,
    )

    logging.info(
        "[Pipeline Root] Initializing network topology (Type: %s)...",
        cfg.architecture.model_type,
    )
    controller.initialize_network_from_dimensions(
        input_dim=input_dim,
        output_dim=cfg.architecture.num_classes,
        model_type=cfg.architecture.model_type,
        hidden_layers=cfg.architecture.hidden_layers if not is_cnn and not is_mhsa else [],
        optimizer_name=cfg.optimization.optimizer,
        lam_l1=cfg.regularization.lam_l1,
        lam_l2=cfg.regularization.lam_l2,
        p_dropout=cfg.architecture.p_dropout,
        use_batch_norm=cfg.architecture.use_batch_norm,
        bn_momentum=cfg.architecture.bn_momentum,
        max_norm=cfg.optimization.gradient_clipping_max_norm,
        cnn_config=cnn_cfg if is_cnn else None,
        mhsa_config=mhsa_cfg if is_mhsa else None,
        backend=cfg.architecture.backend,
        contract_list_enabled=bool(getattr(cfg.ledger, "contract_list_enabled", False)),
    )

    if cfg.persistence.load_saved_model:
        controller.hydrate_from_asset(cfg.persistence.model_asset_path)

    if hasattr(data_provider, "y_train_processed"):
        unique, counts = np.unique(data_provider.y_train_processed, axis=0, return_counts=True)
        logging.info("\n=== Training Class Balance ===")
        for u, c in zip(unique, counts):
            logging.info("Target Vector: %s | Count: %s", u, c)
        logging.info("=============================\n")

    return controller, data_provider, steps, source_mode


def _fit_assembled(
    cfg: PipelineConfig,
    controller: ModelController,
    data_provider,
    steps: int,
    source_mode,
    *,
    manager_heartbeat=None,
    model_id: str | None = None,
) -> None:
    controller.fit(
        steps=steps,
        source_mode=source_mode,
        model_type=cfg.architecture.model_type,
        early_stopping_enabled=cfg.optimization.early_stopping_enabled,
        patience=cfg.optimization.patience,
        min_delta=cfg.optimization.min_delta,
        ledger_settings=cfg.ledger,
        training_manager=cfg.training_manager,
        output_dir=cfg.meta.output_dir,
        manager_heartbeat=manager_heartbeat,
        model_id=model_id,
    )

    if cfg.architecture.backend == EngineBackend.IM2COL_GEMM:
        from utils.conv_dispatch import log_im2col_telemetry

        log_im2col_telemetry()

    NeuralNetworkDiagnostics.run_diagnostics(
        controller=controller,
        data_provider=data_provider,
        cfg=cfg,
    )

    if cfg.persistence.load_saved_model:
        legacy_config_dict = {
            "meta": vars(cfg.meta),
            "ingestion": vars(cfg.ingestion),
            "architecture": vars(cfg.architecture),
            "optimization": vars(cfg.optimization),
            "regularization": vars(cfg.regularization),
            "persistence": vars(cfg.persistence),
        }
        controller.serialize_current_state(
            target_asset_path=cfg.persistence.model_asset_path,
            serialized_config_dict=legacy_config_dict,
        )


def _run_oneshot(
    cfg: PipelineConfig, runtime, *, manager_heartbeat=None, model_id: str | None = None
) -> None:
    """Direct run: assemble from the resolved config and train once."""
    controller, data_provider, steps, source_mode = _build_provider_and_controller(cfg)
    with training_threadpool(runtime, cfg.architecture.backend):
        _fit_assembled(
            cfg,
            controller,
            data_provider,
            steps,
            source_mode,
            manager_heartbeat=manager_heartbeat,
            model_id=model_id,
        )


def execute_training_pipeline():
    """Boot → resolve config (local YAML or strict TM payload) → train directly."""
    # Resolved at import (before BLAS first touch); see the top of this module.
    cfg = _cfg
    model_id = tm_payload_model_id(_TM_PAYLOAD)
    initialize_global_logging(cfg)
    if _CONFIG_SOURCE == ConfigSource.TRAINING_MANAGER:
        runtime = _RUNTIME
        log_runtime_settings(runtime, cfg.architecture.backend)
        logging.info(
            "[pipeline] TM-sourced config (model_id=%s, threads=%d)",
            model_id or "-",
            runtime.num_threads,
        )
    else:
        runtime = configure_runtime(
            cfg.architecture.backend,
            config_path=BOOT_YAML,
            overwrite_env=True,
            if_unset_env=False,
        )
    logging.warning(experiment_summary())

    tm = cfg.training_manager
    hb = None
    if getattr(tm, "enabled", False):
        from src.manager_heartbeat import maybe_from_settings

        hb = maybe_from_settings(tm, ledger_enabled=bool(cfg.ledger.enabled))
        if hb is None:
            raise RuntimeError(
                "training_manager.enabled but heartbeat client failed to build"
            )

    # Direct run: the heartbeat (when present) is control/metrics/ledger only;
    # training_manager.authorize_on_boot decides train-now vs wait-for-Start.
    _run_oneshot(cfg, runtime, manager_heartbeat=hb, model_id=model_id)


if __name__ == "__main__":
    execute_training_pipeline()
