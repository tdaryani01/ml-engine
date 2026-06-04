# models/controller.py
import os
import logging
import copy
import numpy as np
from models.model_factory import ModelFactory
from models.serializer import ModelSerializer
from models.schedulers import StepDecay, ExponentialDecay
from data.iterator import DatasetIterator 

class ModelController:
    def __init__(self, config):
        self.config = config
        self.trn_cfg = config["training"]
        self.env_cfg = config["environment"]
        self.model = None
        self.train_history = []
        self.val_history = []
        self.mean = None
        self.std = None
        self.scheduler = None

    def _apply_fourier_expansion_if_needed(self, X_matrix):
        """Reconstructs identical Fourier features for evaluation splits."""
        fourier_cfg = self.trn_cfg.get("fourier_expansion", {})
        
        if fourier_cfg.get("enabled", False):
            num_freqs = fourier_cfg.get("num_frequencies", 4)
            features = [X_matrix]
            for i in range(num_freqs):
                freq = 2.0 ** i
                features.append(np.sin(X_matrix * freq))
                features.append(np.cos(X_matrix * freq))
            return np.hstack(features)
        return X_matrix

    def setup_model(self, splits):
        """Drives setup orchestration by resolving configuration settings or cold-starting layouts."""
        load_flag = self.env_cfg.get("load_saved_model", False)
        file_path = self.env_cfg.get("model_asset_path", "deployed_model.npz")
        
        self.mean = np.mean(splits["X_train"], axis=0)
        self.std = np.std(splits["X_train"], axis=0) + 1e-24
        
        init_lr = self.trn_cfg["learning_rate"]
        sched_type = self.trn_cfg.get("lr_scheduler", "exponential").strip().lower()
        
        if sched_type == "step":
            drop_ratio = self.trn_cfg.get("scheduler_drop_ratio", 0.5)
            epochs_drop = self.trn_cfg.get("scheduler_epochs_per_drop", 20)
            self.scheduler = StepDecay(init_lr, drop_ratio=drop_ratio, epochs_per_drop=epochs_drop)
        elif sched_type == "exponential":
            decay_rate = self.trn_cfg.get("scheduler_decay_rate", 0.98)
            self.scheduler = ExponentialDecay(init_lr, decay_rate=decay_rate)
        else:
            self.scheduler = None
        
        if load_flag and os.path.exists(file_path):
            logging.info(f"[Model Controller] Discovered serialized asset. Routing hydration to Serializer...")
            self.model = ModelSerializer.load_model(file_path)
            self.model.config = self.config
            return

        logging.info("[Model Controller] Cold start execution path selected. Computing network topology layers...")
        
        input_dim = splits["X_train"].shape[1]
        task_type = str(self.config["architecture"]["model_type"]).strip().lower()
        
        if task_type in ["multi_class", "binary_classification"]:
            all_blocks = [splits["y_train"], splits["y_val"]]
            if "y_test" in splits:
                all_blocks.append(splits["y_test"])
            global_y = np.vstack(all_blocks)
            distinct_classes = len(np.unique(global_y))
            
            if task_type == "binary_classification" and len(splits["y_train"].shape) == 2 and splits["y_train"].shape[1] == 1:
                output_dim = 1
            elif task_type == "binary_classification":
                output_dim = 1 
            else:
                output_dim = distinct_classes
        else:
            output_dim = splits["y_train"].shape[1]
            
        resolved_topology = tuple([input_dim] + self.config["architecture"]["hidden_layers"] + [output_dim])
        logging.info(f"[Model Controller] Resolved Layer Architecture Sequence: {resolved_topology}")
        
        self.model = ModelFactory.create_model(
            model_type=self.config["architecture"]["model_type"],
            layer_sizes=resolved_topology,
            lr=init_lr,
            optimizer=self.config["training"]["optimizer"],
            lam_l1=self.config["regularization"]["lam_l1"],
            lam_l2=self.config["regularization"]["lam_l2"],
            p_dropout=self.config["architecture"].get("p_dropout", 0.0),
            use_batch_norm=self.config["architecture"].get("use_batch_norm", True),
            bn_momentum=self.config["architecture"].get("bn_momentum", 0.9)
        )

        self.model.config = self.config

    def fit(self, splits):
        """Main training loop interface with adaptive learning rates and early stopping tracking."""
        if self.model is None:
            raise ValueError("[Model Controller] Execution Error: Cannot call fit before calling setup_model.")

        X_val_expanded = self._apply_fourier_expansion_if_needed(splits["X_val"])

        X_train_norm = (splits["X_train"] - self.mean) / self.std
        X_val_norm = (X_val_expanded - self.mean) / self.std
        
        y_train_target, y_val_target, _ = self.model.preprocess_targets(
            splits["y_train"], splits["y_val"]
        )
        
        epochs_to_run = self.trn_cfg["epochs"]
        batch_size = self.trn_cfg["batch_size"]

        if epochs_to_run <= 0:
            logging.info("[Model Controller] Epoch count set to 0. Skipping training execution loops.")
            return self.train_history, self.val_history

        es_enabled = self.trn_cfg.get("early_stopping_enabled", True)
        patience = self.trn_cfg.get("patience", 15)
        min_delta = self.trn_cfg.get("min_delta", 1e-5)
        
        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = None
        best_biases = None

        if hasattr(self.model.optimizer, '_setup_done'):
            self.model.optimizer._setup_done = False
            self.model.optimizer.t = 0
            logging.info("[Model Controller] Patched Adam state: Tracking vectors cleared for new execution pass.")

        logging.info(f"[Model Controller] Running optimization loops for {epochs_to_run} epochs...")
        for epoch in range(epochs_to_run):

            active_lr = self.scheduler.step(epoch) if self.scheduler else self.trn_cfg["learning_rate"]
            
            train_iterator = DatasetIterator(X_train_norm, y_train_target, batch_size=batch_size, shuffle=True)
            
            for batch_idx, (X_b, y_b) in enumerate(train_iterator):
                self.model.backward(X_b, y_b, active_lr=active_lr)
                
            # ─── 🛠️ FIXED: Assign forward passes to clean scoped variable names ───
            train_preds = self.model.forward(X_train_norm, training=False)
            val_preds = self.model.forward(X_val_norm, training=False)
            
            current_train_loss = self.model.compute_total_loss(train_preds, y_train_target)
            current_val_loss = self.model.compute_total_loss(val_preds, y_val_target)
            
            self.train_history.append(current_train_loss)
            self.val_history.append(current_val_loss)
            
            if epoch % 10 == 0 or epoch == epochs_to_run - 1:
                # FIXED: Called self reference on class method pass
                val_r2_score = self.compute_r2_score(y_val_target, val_preds)
        
                logging.info(
                    f"Epoch {epoch:3d}/{epochs_to_run} | "
                    f"Train Loss: {current_train_loss:.6f} | "
                    f"Val Loss: {current_val_loss:.6f} | "
                    f"Val Score (R²): {val_r2_score:.6f} | "
                    f"Active LR: {active_lr:.6f}"
                )
            
            if es_enabled:
                if current_val_loss < (best_val_loss - min_delta):
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    best_weights = copy.deepcopy(self.model.weights)
                    best_biases = copy.deepcopy(self.model.biases)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(f"[Early Stopping] Validation divergence detected. Loss has failed to improve for {patience} sequential epochs.")
                        logging.info(f"[Early Stopping] Halting optimization process at epoch {epoch}. Restoring best checkpoint parameters (Val Loss: {best_val_loss:.6f}).")
                        self.model.weights = best_weights
                        self.model.biases = best_biases
                        break
        
        logging.info("[Model Controller] Training sequence completed. Triggering data serialization...")
        target_path = self.env_cfg.get("model_asset_path", "deployed_model.npz")
        
        if self.env_cfg.get("load_saved_model", False):
            logging.info("[Model Controller] Continuous learning cycle complete. Overwriting serialized asset...")
            ModelSerializer.save_model(self.model, self.config, file_path=target_path)
        else:
            logging.info("[Model Controller] Cold-start / Tuning detected. Skipping disk write.")
            
        return self.train_history, self.val_history

    def predict(self, raw_data):
        if self.mean is None or self.std is None:
            raise ValueError("[Model Controller] Execution Error: Cannot run prediction before model is fitted or loaded.")
        expanded_data = self._apply_fourier_expansion_if_needed(raw_data)
        return self.model.predict(expanded_data, self.mean, self.std)

    # ─── 🛠️ FIXED: Added missing 'self' method signature ─────────────────────
    def compute_r2_score(self, y_true, y_pred):
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        
        y_mean = np.mean(y_true_flat)
        ss_tot = np.sum((y_true_flat - y_mean) ** 2)
        ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
        
        if ss_tot < 1e-10:
            return 0.0
        
        return float(1.0 - (ss_res / ss_tot))