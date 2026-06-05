# models/controller.py
import os
import logging
import copy
import numpy as np
from models.model_factory import ModelFactory
from models.serializer import ModelSerializer
from models.schedulers import StepDecay, ExponentialDecay
from config.constants import DataKeys, ModelType, LRHierarchy

class ModelController:
    def __init__(self, learning_rate: float, lr_scheduler_type: LRHierarchy, scheduler_decay_rate: float = 0.98, 
                 scheduler_drop_ratio: float = 0.5, scheduler_epochs_per_drop: int = 20):
        self.initial_lr = learning_rate
        self.model = None
        self.train_history = []
        self.val_history = []
        self.steps_completed = 0
        
        if lr_scheduler_type == LRHierarchy.STEP:
            self.scheduler = StepDecay(learning_rate, drop_ratio=scheduler_drop_ratio, epochs_per_drop=scheduler_epochs_per_drop)
        elif lr_scheduler_type == LRHierarchy.EXPONENTIAL:
            self.scheduler = ExponentialDecay(learning_rate, decay_rate=scheduler_decay_rate)
        else:
            self.scheduler = None

    def initialize_network_from_dimensions(self, input_dim: int, output_dim: int, model_type: ModelType, 
                                           hidden_layers: list, optimizer_name: str, lam_l1: float, lam_l2: float, 
                                           p_dropout: float = 0.0, use_batch_norm: bool = True, 
                                           bn_momentum: float = 0.9, max_norm: float = 5.0) -> None:
        resolved_topology = tuple([input_dim] + hidden_layers + [output_dim])
        logging.info(f"[Model Controller] Resolved Layer Architecture Sequence: {resolved_topology}")
        
        self.model = ModelFactory.create_model(
            model_type=model_type.name.lower(),
            layer_sizes=resolved_topology,
            lr=self.initial_lr,
            optimizer=optimizer_name,
            lam_l1=lam_l1,
            lam_l2=lam_l2,
            p_dropout=p_dropout,
            use_batch_norm=use_batch_norm,
            bn_momentum=bn_momentum,
            max_norm=max_norm
        )

    def initialize_network_from_provider(self, data_provider, model_type: ModelType, hidden_layers: list, 
                                        optimizer_name: str, lam_l1: float, lam_l2: float, 
                                        p_dropout: float = 0.0, use_batch_norm: bool = True, 
                                        bn_momentum: float = 0.9, max_norm: float = 5.0) -> None:
        X_val, y_val = data_provider.get_validation_set()
        input_dim = X_val.shape[1]
        
        if model_type in (ModelType.MULTI_CLASS, ModelType.BINARY_CLASSIFICATION):
            distinct_classes = len(np.unique(y_val))
            if model_type == ModelType.BINARY_CLASSIFICATION and len(y_val.shape) == 2 and y_val.shape[1] == 1:
                output_dim = 1
            else:
                output_dim = distinct_classes
        else:
            output_dim = y_val.shape[1]
            
        self.initialize_network_from_dimensions(
            input_dim=input_dim,
            output_dim=output_dim,
            model_type=model_type,
            hidden_layers=hidden_layers,
            optimizer_name=optimizer_name,
            lam_l1=lam_l1,
            lam_l2=lam_l2,
            p_dropout=p_dropout,
            use_batch_norm=use_batch_norm,
            bn_momentum=bn_momentum,
            max_norm=max_norm
        )

    def hydrate_from_asset(self, asset_path: str) -> bool:
        if os.path.exists(asset_path):
            logging.info(f"[Model Controller] Discovered serialized asset at {asset_path}. Routing hydration to Serializer...")
            self.model = ModelSerializer.load_model(asset_path)
            return True
        return False

    def fit(self, data_provider, steps: int, source_mode, model_type: ModelType,
                         early_stopping_enabled: bool = True, patience: int = 15, min_delta: float = 1e-5) -> tuple:
        if self.model is None:
            raise ValueError("[Model Controller] Execution Error: Cannot call fit before initializing the network.")

        epoch = 0
        is_classification = model_type in (ModelType.BINARY_CLASSIFICATION, ModelType.MULTI_CLASS)
        
        # Ingest validation sets from our data strategy provider
        X_val, y_val_target = data_provider.get_validation_set()
        
        # 🚨 Changed: Routing standardization call straight to the provider's native normalization layer
        X_val_norm = data_provider.normalize(X_val)
        
        if steps <= 0:
            logging.info("[Model Controller] Steps count set to 0. Skipping training execution loops.")
            return self.train_history, self.val_history

        if hasattr(self.model.optimizer, '_setup_done'):
            self.model.optimizer._setup_done = False
            self.model.optimizer.t = 0
            logging.info("[Model Controller] Patched Adam state: Tracking vectors cleared for new execution pass.")

        es_state = {"best_val_loss": float("inf"), "best_epoch": 0, "patience_counter": 0, "weights": None, "biases": None}

        while True:
            active_lr = self.scheduler.step(epoch) if self.scheduler else self.initial_lr
            
            epoch_train_loss = self._run_epoch_training_pass(data_provider, active_lr, steps)
            if epoch_train_loss == 0.0:
                break
            self.train_history.append(epoch_train_loss)
            
            # Re-normalize validation window space to ensure it adapts to rolling streaming bounds
            X_val_norm = data_provider.normalize(X_val)
            val_preds = self.model.forward(X_val_norm, training=False)
            current_val_loss = self.model.compute_total_loss(val_preds, y_val_target)
            current_val_raw_cost = self.model.calculate_raw_cost(val_preds, y_val_target)
            
            self.val_history.append(current_val_loss)
            
            self._evaluate_epoch_performance(epoch, data_provider, epoch_train_loss, val_preds, y_val_target, current_val_loss, active_lr, is_classification, model_type)
            
            if early_stopping_enabled:
                should_stop = self._handle_early_stopping(epoch, current_val_raw_cost, min_delta, patience, es_state)
                if should_stop:
                    break
            epoch += 1

        self._generate_final_summary_report(data_provider, X_val_norm, y_val_target, source_mode, is_classification, model_type, es_state, early_stopping_enabled)
        return self.train_history, self.val_history

    def _run_epoch_training_pass(self, data_provider, active_lr: float, steps: int) -> float:
        data_provider.reset_epoch()
        batch_losses = []
        
        while data_provider.has_more_batches():
            X_b, y_b = data_provider.next_batch()
            if X_b.size == 0:
                break
            
            # 🚨 Changed: Delegate normalization straight to provider
            X_b_norm = data_provider.normalize(X_b)
            batch_loss = self.model.backward(X_b_norm, y_b, active_lr=active_lr)
            
            if batch_loss is None:
                preds_b = self.model.forward(X_b_norm, training=False)
                batch_loss = self.model.compute_total_loss(preds_b, y_b)
                
            batch_losses.append(batch_loss)
            self.steps_completed += 1
            if self.steps_completed >= steps:
                self.steps_completed = 0
                break
        return float(np.mean(batch_losses)) if batch_losses else 0.0

    def _evaluate_epoch_performance(self, epoch: int, data_provider, train_loss: float, val_preds: np.ndarray, y_val_target: np.ndarray, 
                                    current_val_loss: float, active_lr: float, is_classification: bool, model_type: ModelType) -> None:
        if epoch % 10 == 0 or not data_provider.has_more_batches():
            metric_name_step = "Acc" if is_classification else "R²"
            
            if hasattr(data_provider, "splits") and DataKeys.X_TRAIN in data_provider.splits:
                X_train = data_provider.splits[DataKeys.X_TRAIN]
                # 🚨 Changed: Delegate normalization straight to provider
                X_train_norm = data_provider.normalize(X_train)
                train_preds = self.model.forward(X_train_norm, training=False)
                y_train_target = data_provider.y_train_processed
                
                if is_classification:
                    train_score = np.mean(np.argmax(train_preds, axis=1) == np.argmax(y_train_target, axis=1))
                    val_score = np.mean(np.argmax(val_preds, axis=1) == np.argmax(y_val_target, axis=1))
                else:
                    train_score = self.compute_r2_score(y_train_target, train_preds)
                    val_score = self.compute_r2_score(y_val_target, val_preds)
            else:
                train_score = 0.0
                val_score = np.mean(np.argmax(val_preds, axis=1) == np.argmax(y_val_target, axis=1)) if is_classification else self.compute_r2_score(y_val_target, val_preds)

            logging.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {current_val_loss:.4f} | "
                f"Train {metric_name_step}: {train_score * 100:.2f}% | " if is_classification else f"Train {metric_name_step}: {train_score:.4f} | "
                f"Val {metric_name_step}: {val_score * 100:.2f}% | " if is_classification else f"Val {metric_name_step}: {val_score:.4f} | "
                f"Active LR: {active_lr:.6f}"
            )

    def _handle_early_stopping(self, epoch: int, current_val_raw_cost: float, min_delta: float, patience: int, es_state: dict) -> bool:
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
                logging.info(f"[Early Stopping] True validation data divergence detected at epoch {epoch}. Restoring checkpoint maps from epoch {es_state['best_epoch']}.")
                self.model.weights = es_state["weights"]
                self.model.biases = es_state["biases"]
                return True
        return False

    def _generate_final_summary_report(self, data_provider, X_val_norm: np.ndarray, y_val_target: np.ndarray, source_mode, 
                                       is_classification: bool, model_type: ModelType, es_state: dict, es_enabled: bool) -> None:
        final_val_preds = self.model.forward(X_val_norm, training=False)
        val_loss = self.val_history[-1] if self.val_history else float('inf')
        
        if hasattr(data_provider, "splits") and DataKeys.X_TRAIN in data_provider.splits:
            X_train = data_provider.splits[DataKeys.X_TRAIN]
            # 🚨 Changed: Delegate normalization straight to provider
            X_train_norm = data_provider.normalize(X_train)
            final_train_preds = self.model.forward(X_train_norm, training=False)
            y_train_target = data_provider.y_train_processed
            
            train_acc = np.mean(np.argmax(final_train_preds, axis=1) == np.argmax(y_train_target, axis=1))
            val_acc = np.mean(np.argmax(final_val_preds, axis=1) == np.argmax(y_val_target, axis=1))
            
            train_r2 = self.compute_r2_score(y_train_target, final_train_preds)
            val_r2 = self.compute_r2_score(y_val_target, final_val_preds)
        else:
            train_acc, train_r2 = 0.0, 0.0
            val_acc = np.mean(np.argmax(final_val_preds, axis=1) == np.argmax(y_val_target, axis=1)) if is_classification else 0.0
            val_r2 = self.compute_r2_score(y_val_target, final_val_preds)

        total_weights_count = 0
        zeroed_weights_count = 0
        sparsity_tolerance = 1e-5
        
        if hasattr(self.model, 'weights'):
            for weight_matrix in self.model.weights:
                total_weights_count += weight_matrix.size
                zeroed_weights_count += np.sum(np.abs(weight_matrix) <= sparsity_tolerance)
        
        sparsity_percentage = (zeroed_weights_count / total_weights_count * 100) if total_weights_count > 0 else 0.0
        mode_str = source_mode.name.upper() if hasattr(source_mode, 'name') else str(source_mode).upper()
        
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
            f"  • Network Task Profile    : {model_type.name}\n"
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

    def serialize_current_state(self, target_asset_path: str, serialized_config_dict: dict) -> None:
        logging.info(f"[Model Controller] Executing state preservation write out to: {target_asset_path}")
        ModelSerializer.save_model(self.model, serialized_config_dict, file_path=target_asset_path)

    def predict(self, raw_data_matrix: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("[Model Controller] Cannot run prediction before model is fitted or loaded.")
        # Predictions bypass internal namespaces, relying on direct model execution
        return self.model.predict(raw_data_matrix)

    def compute_r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        
        y_mean = np.mean(y_true_flat)
        ss_tot = np.sum((y_true_flat - y_mean) ** 2)
        ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
        
        if ss_tot < 1e-10:
            return 0.0
        
        return float(1.0 - (ss_res / ss_tot))