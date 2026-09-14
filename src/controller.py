# src/controller.py
import os
import logging
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from src.model_factory import ModelFactory
from src.serializer import ModelSerializer
from src.schedulers import StepDecay, ExponentialDecay
from src.training_session import TrainingSession
from config.constants import ModelType, LRHierarchy, EngineBackend
from config.schema import LedgerSettings


class ModelController:
    """
    Orchestrates the neural network lifecycle: compilation, training loops, 
    metric evaluation, early stopping checkpoints, and asset serialization.
    """
    def __init__(
        self,
        learning_rate: float = 0.01,
        lr_scheduler_type: LRHierarchy = LRHierarchy.NONE,
        data_provider: Optional[Any] = None,
        scheduler_decay_rate: float = 0.98,
        scheduler_drop_ratio: float = 0.5,
        scheduler_epochs_per_drop: int = 20
    ):
        self.initial_lr = learning_rate
        self.data_provider = data_provider
        self.model = None
        self.train_history: List[float] = []
        self.val_history: List[float] = []
        self.steps_completed = 0

        # Configure learning rate scheduler
        if lr_scheduler_type == LRHierarchy.STEP:
            self.scheduler = StepDecay(
                learning_rate, 
                drop_ratio=scheduler_drop_ratio, 
                epochs_per_drop=scheduler_epochs_per_drop
            )
        elif lr_scheduler_type == LRHierarchy.EXPONENTIAL:
            self.scheduler = ExponentialDecay(
                learning_rate, 
                decay_rate=scheduler_decay_rate
            )
        else:
            self.scheduler = None

    def initialize_network_from_dimensions(
        self,
        model_type: ModelType,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        hidden_layers: Optional[List[int]] = None,
        optimizer_name: str = "adam",
        lam_l1: float = 0.0,
        lam_l2: float = 0.0,
        p_dropout: float = 0.0,
        use_batch_norm: bool = False,
        bn_momentum: float = 0.9,
        max_norm: float = 5.0,
        cnn_config: Optional[Dict[str, Any]] = None,
        mhsa_config: Optional[Dict[str, Any]] = None,
        backend: EngineBackend = EngineBackend.NATIVE,
        contract_list_enabled: bool = False,
    ) -> None:
        """
        Builds and initializes network topology. 
        If input_dim or output_dim are omitted, infers them automatically from the bound data_provider.
        """
        self.backend = backend
        hidden_layers = hidden_layers or []

        # Auto-infer dimensions from data_provider if not explicitly passed
        if input_dim is None or output_dim is None:
            if self.data_provider is None:
                raise ValueError(
                    "[Model Controller] Cannot infer dimensions: data_provider is not set and explicit dimensions were not provided."
                )
            X_val, y_val = self.data_provider.get_validation_set()
            
            if input_dim is None:
                input_dim = int(np.prod(X_val.shape[1:]))

            if output_dim is None:
                if model_type == ModelType.BINARY_CLASSIFICATION:
                    output_dim = 1
                elif model_type in (ModelType.MULTI_CLASS, ModelType.CNN):
                    output_dim = y_val.shape[1] if y_val.ndim > 1 else len(np.unique(y_val))
                else:
                    output_dim = y_val.shape[1] if y_val.ndim > 1 else 1

        resolved_topology = tuple([input_dim] + hidden_layers + [output_dim])
        logging.info(f"[Model Controller] Resolved Layer Architecture Sequence: {resolved_topology}")

        type_str = model_type.name.lower() if hasattr(model_type, "name") else str(model_type).lower()

        self.model = ModelFactory.create_model(
            model_type=type_str,
            layer_sizes=resolved_topology,
            lr=self.initial_lr,
            optimizer=optimizer_name,
            lam_l1=lam_l1,
            lam_l2=lam_l2,
            p_dropout=p_dropout,
            use_batch_norm=use_batch_norm,
            bn_momentum=bn_momentum,
            max_norm=max_norm,
            backend=self.backend,
            cnn_config=cnn_config,
            mhsa_config=mhsa_config,
            contract_list_enabled=contract_list_enabled,
        )

        # FORENSIC HOOK: INIT TRACE
        if hasattr(self.model, "weights") and hasattr(self.model, "biases") and len(self.model.weights) > 0:
            initial_w_norm = np.linalg.norm(self.model.weights[0])
            initial_b = self.model.biases[-1][0] if len(self.model.biases[-1].shape) > 1 else self.model.biases[-1]
            logging.info("=== FORENSIC INIT TRACE ===")
            logging.info(f"Layer 0 Starting Weight Norm: {initial_w_norm:.6f}")
            logging.info(f"Final Layer Starting Biases : {np.round(initial_b, 4).tolist()}")
            logging.info("===========================")

    def hydrate_from_asset(self, asset_path: str) -> bool:
        """Loads a serialized model asset from disk if present."""
        if os.path.exists(asset_path):
            logging.info(f"[Model Controller] Discovered serialized asset at {asset_path}. Routing hydration to Serializer...")
            self.model = ModelSerializer.load_model(asset_path)
            return True
        return False

    def predict(self, raw_data_matrix: np.ndarray) -> np.ndarray:
        """Normalizes features (if provider attached) and runs forward inference."""
        if self.model is None:
            raise ValueError("[Model Controller] Cannot run prediction before model is fitted or loaded.")

        if self.data_provider is not None:
            raw_data_matrix = self.data_provider.normalize(raw_data_matrix)

        return self.model.predict(raw_data_matrix)

    def fit(
        self,
        steps: int,
        source_mode: Any,
        model_type: ModelType,
        early_stopping_enabled: bool = True,
        patience: int = 15,
        min_delta: float = 1e-5,
        ledger_settings: LedgerSettings | None = None,
        training_manager: Any | None = None,
        output_dir: str = "diagnostics_output",
        max_epochs: int | None = None,
        manager_heartbeat: Any | None = None,
        adopt_job: dict[str, Any] | None = None,
        job_scoped: bool = False,
    ) -> Tuple[List[float], List[float]]:
        """Executes the training loop via TrainingSession (Phase C boundary)."""
        if self.model is None:
            raise ValueError("[Model Controller] Execution Error: Cannot call fit before initializing the network.")
        if self.data_provider is None:
            raise ValueError("[Model Controller] Execution Error: No data_provider bound to controller.")

        engine = None
        ledger_on = ledger_settings is not None and ledger_settings.enabled
        from src.manager_heartbeat import maybe_from_settings
        from src.training_engine import create_training_engine

        hb = manager_heartbeat
        if hb is None:
            hb = maybe_from_settings(training_manager, ledger_enabled=ledger_on)
        if ledger_on or hb is not None:
            import os

            from src.ledger import LedgerConfig

            ls = ledger_settings or LedgerSettings()
            ledger_dir = os.path.join(output_dir, ls.path if ledger_on else "training_ledger_noop")
            arch_id = model_type.name if hasattr(model_type, "name") else str(model_type)
            # Ledger model id is the TM agent id. Until a job is claimed/adopted it is
            # unbound; claim/adopt stamps job.model_id (never the pool worker id).
            model_instance_id = "unbound"
            if adopt_job is not None:
                mid = str(adopt_job.get("model_id") or "").strip()
                if mid:
                    model_instance_id = mid

            if ledger_on:
                eng_cfg = LedgerConfig(
                    checkpoint_every_steps=ls.checkpoint_every_steps,
                    checkpoint_on_local_best=ls.checkpoint_on_local_best,
                    contract_list_enabled=ls.contract_list_enabled,
                    native_async_submit=ls.native_async_submit,
                    store_backend=ls.store_backend,
                )
            else:
                # TM connect without a journal: noop store, no contract-list path.
                eng_cfg = LedgerConfig(store_backend="noop")

            store_kwargs: dict = {}
            backend_key = str(eng_cfg.store_backend).strip().lower()
            if backend_key in ("http_tm", "tm_http", "training_manager"):
                uri = ""
                timeout_s = 0.5
                if training_manager is not None:
                    uri = str(getattr(training_manager, "uri", "") or "")
                    if not uri and isinstance(training_manager, dict):
                        uri = str(training_manager.get("uri") or "")
                    timeout_s = float(
                        getattr(training_manager, "timeout_s", 0.5)
                        if not isinstance(training_manager, dict)
                        else training_manager.get("timeout_s", 0.5)
                    )
                if not uri and hb is not None:
                    cfg = getattr(hb, "_cfg", None) or getattr(hb, "cfg", None)
                    uri = str(getattr(cfg, "uri", "") or "") if cfg is not None else ""
                if not uri:
                    raise ValueError(
                        "ledger.store_backend=http_tm requires training_manager.uri"
                    )
                store_kwargs = {"http_tm_uri": uri, "timeout_s": timeout_s}

            engine = create_training_engine(
                ledger_dir,
                branch_id=ls.branch_id,
                model_instance_id=model_instance_id,
                architecture_id=arch_id,
                config=eng_cfg,
                manager_heartbeat=hb,
                store_kwargs=store_kwargs or None,
            )
            if ledger_on:
                logging.info(
                    "[Model Controller] Training ledger enabled: %s backend=%s instance_id=%s",
                    ledger_dir,
                    eng_cfg.store_backend,
                    model_instance_id,
                )
            elif hb is not None:
                logging.info(
                    "[Model Controller] Training Manager heartbeats enabled (ledger off): %s",
                    getattr(training_manager, "uri", None),
                )
            try:
                session = engine.start_session(
                    model=self.model,
                    data_provider=self.data_provider,
                    initial_lr=self.initial_lr,
                    scheduler=self.scheduler,
                    predict_fn=self.predict,
                    steps_completed=self.steps_completed,
                    steps=steps,
                    source_mode=source_mode,
                    model_type=model_type,
                    early_stopping_enabled=early_stopping_enabled,
                    patience=patience,
                    min_delta=min_delta,
                    compute_r2_score=self.compute_r2_score,
                    max_epochs=max_epochs,
                )
                if adopt_job is not None:
                    engine.adopt_claimed_job(dict(adopt_job))
                results = engine.run(job_scoped=bool(job_scoped or adopt_job is not None))
                hist = results.get(session.session_id)
                if hist is None:
                    hist = (list(session.train_history), list(session.val_history))
                self.train_history, self.val_history = hist
                self.steps_completed = session.steps_completed
            finally:
                engine.close()
            return self.train_history, self.val_history

        session = TrainingSession(
            model=self.model,
            data_provider=self.data_provider,
            initial_lr=self.initial_lr,
            scheduler=self.scheduler,
            predict_fn=self.predict,
        )
        session.steps_completed = self.steps_completed

        self.train_history, self.val_history = session.fit(
            steps=steps,
            source_mode=source_mode,
            model_type=model_type,
            early_stopping_enabled=early_stopping_enabled,
            patience=patience,
            min_delta=min_delta,
            compute_r2_score=self.compute_r2_score,
            engine=None,
            max_epochs=max_epochs,
        )
        self.steps_completed = session.steps_completed
        return self.train_history, self.val_history

    def serialize_current_state(self, target_asset_path: str, serialized_config_dict: dict) -> None:
        """Writes active weights and hyperparameter state out to disk."""
        logging.info(f"[Model Controller] Executing state preservation write out to: {target_asset_path}")
        ModelSerializer.save_model(
            self.model,
            serialized_config_dict,
            data_provider=self.data_provider,
            file_path=target_asset_path
        )

    @staticmethod
    def compute_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes R² coefficient of determination."""
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()

        y_mean = np.mean(y_true_flat)
        ss_tot = np.sum((y_true_flat - y_mean) ** 2)
        ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)

        if ss_tot < 1e-10:
            return 0.0

        return float(1.0 - (ss_res / ss_tot))