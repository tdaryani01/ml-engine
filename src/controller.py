# src/controller.py
import os
import logging
import copy
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from src.model_factory import ModelFactory
from src.serializer import ModelSerializer
from src.schedulers import StepDecay, ExponentialDecay
from config.constants import DataKeys, ModelType, LRHierarchy, EngineBackend


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
        backend: EngineBackend = EngineBackend.NATIVE
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
            cnn_config=cnn_config
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

    def _set_train_batch_caps(self, model_type: ModelType) -> None:
        if model_type != ModelType.CNN:
            return
        if not hasattr(self.model, "set_train_batch_cap"):
            return
        cap = int(getattr(self.data_provider, "batch_size", 32))
        self.model.set_train_batch_cap(cap)
        logging.debug(f"[Model Controller] Train buffer cap set to N={cap}.")

    def fit(
        self,
        steps: int,
        source_mode: Any,
        model_type: ModelType,
        early_stopping_enabled: bool = True,
        patience: int = 15,
        min_delta: float = 1e-5
    ) -> Tuple[List[float], List[float]]:
        """Executes the training loop over epochs using the bound self.data_provider."""
        if self.model is None:
            raise ValueError("[Model Controller] Execution Error: Cannot call fit before initializing the network.")
        if self.data_provider is None:
            raise ValueError("[Model Controller] Execution Error: No data_provider bound to controller.")

        epoch = 0
        is_classification = model_type in (ModelType.BINARY_CLASSIFICATION, ModelType.MULTI_CLASS, ModelType.CNN)

        X_val, y_val_target = self.data_provider.get_validation_set()

        # FORENSIC HOOK: Static Validation Distribution
        if is_classification:
            val_class_dist = (
                np.sum(y_val_target, axis=0).tolist()
                if hasattr(y_val_target, "ndim") and y_val_target.ndim > 1
                else np.unique(y_val_target, return_counts=True)[1].tolist()
            )
            logging.info(f"[Forensic Trace] Static Validation Set Class Distribution: {val_class_dist}")

        if steps <= 0:
            logging.info("[Model Controller] Steps count set to 0. Skipping training execution loops.")
            return self.train_history, self.val_history

        self._set_train_batch_caps(model_type)

        if hasattr(self.model.optimizer, "_setup_done"):
            self.model.optimizer._setup_done = False
            self.model.optimizer.t = 0
            logging.info("[Model Controller] Patched Adam state: Tracking vectors cleared for new execution pass.")

        es_state = {
            "best_val_loss": float("inf"),
            "best_epoch": 0,
            "patience_counter": 0,
            "weights": None,
            "biases": None
        }

        while True:
            active_lr = self.scheduler.step(epoch) if self.scheduler else self.initial_lr

            epoch_train_loss, batch_forensics = self._run_epoch_training_pass(active_lr, steps, epoch, is_classification)
            if epoch_train_loss == 0.0:
                break
            self.train_history.append(epoch_train_loss)

            val_preds = self.predict(X_val)
            current_val_loss = self.model.compute_total_loss(val_preds, y_val_target)
            current_val_raw_cost = self.model.calculate_raw_cost(val_preds, y_val_target)

            self.val_history.append(current_val_loss)

            self._evaluate_epoch_performance(
                epoch, epoch_train_loss, val_preds, y_val_target,
                current_val_loss, active_lr, is_classification
            )

            if early_stopping_enabled:
                if self._handle_early_stopping(epoch, current_val_raw_cost, min_delta, patience, es_state):
                    break
            epoch += 1

        self._generate_final_summary_report(
            X_val, y_val_target, source_mode, is_classification, model_type, es_state, early_stopping_enabled
        )
        return self.train_history, self.val_history

    def _run_epoch_training_pass(
        self,
        active_lr: float,
        steps: int,
        epoch: int,
        is_classification: bool
    ) -> Tuple[float, Dict[str, Any]]:
        """Processes mini-batches for one epoch."""
        self.data_provider.reset_epoch()
        batch_losses = []
        batch_idx = 0
        forensic_data: Dict[str, Any] = {}

        while self.data_provider.has_more_batches():
            X_b, y_b = self.data_provider.next_batch()
            if X_b.size == 0:
                break

            X_b_norm = self.data_provider.normalize(X_b)
            batch_loss = self.model.backward(X_b_norm, y_b, active_lr=active_lr)

            if batch_loss is None:
                preds_b = self.model.predict(X_b_norm)
                batch_loss = self.model.compute_total_loss(preds_b, y_b)

            batch_losses.append(batch_loss)

            # FORENSIC HOOK: Track batch 0 dynamics every 10 epochs
            if batch_idx == 0 and epoch % 10 == 0:
                if is_classification:
                    y_classes = np.sum(y_b, axis=0) if len(y_b.shape) > 1 else np.unique(y_b, return_counts=True)[1]
                else:
                    y_classes = "N/A"

                if hasattr(self.model, "activations") and len(self.model.activations) > 0:
                    preds = self.model.activations[-1]
                    if is_classification and hasattr(preds, "shape") and len(preds.shape) > 1:
                        pred_classes = np.argmax(preds, axis=1)
                        pred_spread = [int(np.sum(pred_classes == c)) for c in range(preds.shape[1])]
                    else:
                        pred_spread = "N/A"

                    l1_act = self.model.activations[1] if len(self.model.activations) > 1 else None
                    dead_pct = float(np.mean(l1_act <= 0.0) * 100) if l1_act is not None else 0.0
                else:
                    pred_spread = []
                    dead_pct = 0.0

                forensic_data = {
                    "batch_target_dist": y_classes.tolist() if isinstance(y_classes, np.ndarray) else y_classes,
                    "batch_pred_spread": pred_spread,
                    "dead_zone_pct": dead_pct
                }

            batch_idx += 1
            self.steps_completed += 1
            if self.steps_completed >= steps:
                self.steps_completed = 0
                break

        return float(np.mean(batch_losses)) if batch_losses else 0.0, forensic_data

    def _evaluate_epoch_performance(
        self,
        epoch: int,
        train_loss: float,
        val_preds: np.ndarray,
        y_val_target: np.ndarray,
        current_val_loss: float,
        active_lr: float,
        is_classification: bool
    ) -> None:
        """Logs evaluation metrics at epoch intervals."""
        is_info_enabled = logging.getLogger().isEnabledFor(logging.INFO)
        if not is_info_enabled:
            return

        if epoch % 10 == 0 or not self.data_provider.has_more_batches():
            metric_name = "Acc" if is_classification else "R²"

            if hasattr(self.data_provider, "splits") and DataKeys.X_TRAIN in self.data_provider.splits:
                X_train = self.data_provider.splits[DataKeys.X_TRAIN]
                train_preds = self.predict(X_train)
                y_train_target = self.data_provider.y_train_processed

                if is_classification:
                    train_score = np.mean(np.argmax(train_preds, axis=1) == np.argmax(y_train_target, axis=1))
                    val_score = np.mean(np.argmax(val_preds, axis=1) == np.argmax(y_val_target, axis=1))
                else:
                    train_score = self.compute_r2_score(y_train_target, train_preds)
                    val_score = self.compute_r2_score(y_val_target, val_preds)
            else:
                train_score = 0.0
                val_score = (
                    np.mean(np.argmax(val_preds, axis=1) == np.argmax(y_val_target, axis=1))
                    if is_classification
                    else self.compute_r2_score(y_val_target, val_preds)
                )

            train_score_str = f"{train_score * 100:.2f}%" if is_classification else f"{train_score:.4f}"
            val_score_str = f"{val_score * 100:.2f}%" if is_classification else f"{val_score:.4f}"

            logging.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {current_val_loss:.4f} | "
                f"Train {metric_name}: {train_score_str} | "
                f"Val {metric_name}: {val_score_str} | "
                f"Active LR: {active_lr:.6f}"
            )

    def _handle_early_stopping(
        self,
        epoch: int,
        current_val_raw_cost: float,
        min_delta: float,
        patience: int,
        es_state: Dict[str, Any]
    ) -> bool:
        """Validates convergence criteria and restores best checkpoints on divergence."""
        if current_val_raw_cost < (es_state["best_val_loss"] - min_delta):
            es_state["best_val_loss"] = current_val_raw_cost
            es_state["best_epoch"] = epoch
            es_state["patience_counter"] = 0
            es_state["weights"] = copy.deepcopy(self.model.weights)
            es_state["biases"] = copy.deepcopy(self.model.biases)
            return False
        else:
            es_state["patience_counter"] += 1
            if es_state["patience_counter"] >= patience:
                logging.info(
                    f"[Early Stopping] Validation divergence at epoch {epoch}. "
                    f"Restoring best checkpoint from epoch {es_state['best_epoch']}."
                )
                self.model.weights = es_state["weights"]
                self.model.biases = es_state["biases"]
                if hasattr(self.model, "_sync_restored_weights"):
                    self.model._sync_restored_weights()
                return True
        return False

    def _generate_final_summary_report(
        self,
        X_val_raw: np.ndarray,
        y_val_target: np.ndarray,
        source_mode: Any,
        is_classification: bool,
        model_type: ModelType,
        es_state: Dict[str, Any],
        es_enabled: bool
    ) -> None:
        """Logs the final execution benchmark table."""
        final_val_preds = self.predict(X_val_raw)
        val_loss = self.val_history[-1] if self.val_history else float("inf")

        if hasattr(self.data_provider, "splits") and DataKeys.X_TRAIN in self.data_provider.splits:
            X_train = self.data_provider.splits[DataKeys.X_TRAIN]
            final_train_preds = self.predict(X_train)
            y_train_target = self.data_provider.y_train_processed

            train_acc = np.mean(np.argmax(final_train_preds, axis=1) == np.argmax(y_train_target, axis=1))
            val_acc = np.mean(np.argmax(final_val_preds, axis=1) == np.argmax(y_val_target, axis=1))
            train_r2 = self.compute_r2_score(y_train_target, final_train_preds)
            val_r2 = self.compute_r2_score(y_val_target, final_val_preds)
        else:
            train_acc, train_r2 = 0.0, 0.0
            val_acc = (
                np.mean(np.argmax(final_val_preds, axis=1) == np.argmax(y_val_target, axis=1))
                if is_classification
                else 0.0
            )
            val_r2 = self.compute_r2_score(y_val_target, final_val_preds)

        total_weights_count = 0
        zeroed_weights_count = 0
        sparsity_tolerance = 1e-5

        if hasattr(self.model, "weights"):
            for w in self.model.weights:
                total_weights_count += w.size
                zeroed_weights_count += np.sum(np.abs(w) <= sparsity_tolerance)

        sparsity_percentage = (zeroed_weights_count / total_weights_count * 100) if total_weights_count > 0 else 0.0
        mode_str = source_mode.name.upper() if hasattr(source_mode, "name") else str(source_mode).upper()
        type_str = model_type.name if hasattr(model_type, "name") else str(model_type)

        performance_metrics_block = ""
        if is_classification:
            performance_metrics_block += f"  • Final Training Accuracy : {train_acc * 100:.2f}%\n"
            performance_metrics_block += f"  • Final Target Accuracy   : {val_acc * 100:.2f}%\n"
            performance_metrics_block += f"  • In-Sample R² Score      : {train_r2:.6f} (Train Alignment)\n"
            performance_metrics_block += f"  • Out-of-Sample R² Score  : {val_r2:.6f} (Validation Alignment)\n"
        else:
            performance_metrics_block += f"  • Final Training R² Score : {train_r2:.6f}\n"
            performance_metrics_block += f"  • Final Target R² Score   : {val_r2:.6f}\n"

        divider = "═" * 70
        sub_divider = "─" * 70

        report = (
            f"\n{divider}\n"
            f"                     TRAINING RUN EXECUTION REPORT\n"
            f"{divider}\n"
            f"  [ PIPELINE ENVIRONMENT ]\n"
            f"  • Ingestion Strategy Mode : {mode_str}\n"
            f"  • Network Task Profile    : {type_str}\n"
            f"  • Total Epochs Executed   : {len(self.val_history)} / 300\n"
            f"  • Early Stopping State    : {'ACTIVE' if es_enabled else 'DISABLED'}\n"
            f"  • Best Operational Epoch  : {es_state['best_epoch'] if es_enabled else 'N/A'}\n"
            f"{sub_divider}\n"
            f"  [ MATHEMATICAL OPTIMIZATION BENCHMARKS ]\n"
            f"  • Final Validation Loss   : {val_loss:.6f}\n"
            f"  • Best Validation Raw Cost: {es_state['best_val_loss']:.6f}\n"
            f"{performance_metrics_block}"
            f"{sub_divider}\n"
            f"  [ WEIGHT MATRIX STRUCTURAL METRICS ]\n"
            f"  • Active Synaptic Tensors : {total_weights_count:,} parameters\n"
            f"  • Dead/Sparse Connections : {zeroed_weights_count:,} elements\n"
            f"  • Total Network Sparsity  : {sparsity_percentage:.4f}% Density Suppression\n"
            f"{divider}\n"
        )
        logging.info(report)

        # FORENSIC HOOK: FINAL TRACE
        if hasattr(self.model, "weights") and hasattr(self.model, "biases") and len(self.model.weights) > 0:
            final_w_norm = np.linalg.norm(self.model.weights[0])
            final_biases = np.round(
                self.model.biases[-1][0] if len(self.model.biases[-1].shape) > 1 else self.model.biases[-1],
                4
            )
            logging.info("=== FORENSIC FINAL TRACE ===")
            logging.info(f"Layer 0 Final Weight Norm: {final_w_norm:.6f}")
            logging.info(f"Final Layer Output Biases: {final_biases.tolist()}")
            logging.info("============================")

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