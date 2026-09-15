# run_pipeline.py
import builtins

try:
    profile  # noqa: F821 — injected by kernprof -l
except NameError:
    builtins.profile = lambda f: f

import logging

from config.config_loader import load_production_config
from config.constants import EngineBackend
from utils.runtime import apply_process_env, configure_runtime, load_runtime_settings, training_threadpool

from utils.perf_experiments import apply_im2col_gemm_perf_defaults, experiment_summary

# Pin thread env + native OpenBLAS before NumPy/SciPy BLAS first touch.
_cfg = load_production_config("config/config.yaml")
if _cfg.architecture.backend == EngineBackend.IM2COL_GEMM:
    apply_im2col_gemm_perf_defaults()
apply_process_env(load_runtime_settings(), overwrite=True, if_unset=False)
configure_runtime(_cfg.architecture.backend, log=False, overwrite_env=True)

from config.constants import IngestionMode, ModelType, DataKeys
from config.schema import PipelineConfig
from src.data.base_loader import BaseDataLoader
from src.data.in_memory_provider import InMemoryDataProvider
from src.data.stream_provider import StreamDataProvider
from src.controller import ModelController
from src.pool_worker import restore_job_weights, run_pool_worker_loop
from utils.logger import initialize_global_logging
from utils.diagnostics import NeuralNetworkDiagnostics

import numpy as np

BOOT_YAML = "config/config.yaml"


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
    adopt_job=None,
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
        adopt_job=adopt_job,
        job_scoped=adopt_job is not None,
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


def _run_oneshot(cfg: PipelineConfig, runtime) -> None:
    """Local / TM-disabled: assemble from boot YAML and train once."""
    controller, data_provider, steps, source_mode = _build_provider_and_controller(cfg)
    with training_threadpool(runtime, cfg.architecture.backend):
        _fit_assembled(cfg, controller, data_provider, steps, source_mode)


def _run_claimed_job(cfg: PipelineConfig, hb, job: dict, runtime) -> None:
    """Claim path: config already materialized; build weights, train this lease only."""
    controller, data_provider, steps, source_mode = _build_provider_and_controller(cfg)
    restore_job_weights(hb, controller.model, job)
    with training_threadpool(runtime, cfg.architecture.backend):
        _fit_assembled(
            cfg,
            controller,
            data_provider,
            steps,
            source_mode,
            manager_heartbeat=hb,
            adopt_job=job,
        )


def execute_training_pipeline():
    """Boot → (pool idle wait | oneshot assemble+train)."""
    cfg = load_production_config(BOOT_YAML)
    initialize_global_logging(cfg)
    runtime = configure_runtime(
        cfg.architecture.backend,
        config_path=BOOT_YAML,
        overwrite_env=True,
        if_unset_env=False,
    )
    logging.warning(experiment_summary())

    tm = cfg.training_manager
    if not getattr(tm, "enabled", False):
        _run_oneshot(cfg, runtime)
        return

    from src.manager_heartbeat import maybe_from_settings
    from examples.closed_loop_draw.run_lease import tm_dict_from_heartbeat

    hb = maybe_from_settings(tm, ledger_enabled=bool(cfg.ledger.enabled))
    if hb is None:
        raise RuntimeError("training_manager.enabled but heartbeat client failed to build")

    def _run_supervised(job_cfg: PipelineConfig, heartbeat, job: dict) -> None:
        _run_claimed_job(job_cfg, heartbeat, job, runtime)

    run_pool_worker_loop(
        boot_yaml=BOOT_YAML,
        boot_cfg=cfg,
        hb=hb,
        run_supervised_job=_run_supervised,
        boot_tm=tm_dict_from_heartbeat(hb),
    )


if __name__ == "__main__":
    execute_training_pipeline()
