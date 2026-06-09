# utils/diagnostics.py
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from config.schema import PipelineConfig
from config.constants import DataKeys
from mpl_toolkits.mplot3d import Axes3D

class NeuralNetworkDiagnostics:
    @staticmethod
    def run_diagnostics(controller, data_provider, cfg: PipelineConfig) -> None:
        """
        Adapts legacy diagnostic suites to run dynamically using strongly-typed configuration 
        schemas, active controller performance histories, and strategy data providers.
        """
        diag_cfg = cfg.diagnostics
        meta_cfg = cfg.meta
        
        # 1. Enforce output target directory creation boundaries
        os.makedirs(meta_cfg.output_dir, exist_ok=True)
        
        # 2. Extract out-of-sample arrays cleanly from our providers (using RAW data)
        X_test_raw, y_test = data_provider.get_validation_set()
        feature_names = cfg.ingestion.feature_names
        
        # Pull parameters dynamically from configuration hooks
        selection = str(diag_cfg.metric_to_plot).strip().lower()

        # =====================================================================
        # DYNAMIC RECONSTRUCTION: Handle Fourier expansions if enabled
        # =====================================================================
        fourier_cfg = cfg.transformations.fourier_expansion
        mean = data_provider.mean
        
        if fourier_cfg.enabled and X_test_raw.shape[1] < mean.shape[0]:
            num_freqs = fourier_cfg.num_frequencies
            features = [X_test_raw]
            for i in range(num_freqs):
                freq = 2.0 ** i
                features.append(np.sin(X_test_raw * freq))
                features.append(np.cos(X_test_raw * freq))
            X_processing = np.hstack(features)
        else:
            X_processing = X_test_raw

        # 🚨 FIXED: Route raw values directly through the clean controller.predict() framework
        raw_preds = controller.predict(X_processing)
        
        # 🚨 FIXED: Collapse probabilities (450, 3) down to class indices (450,) via argmax to prevent ValueError broadcast crashes
        if raw_preds.ndim > 1 and raw_preds.shape[1] > 1:
            pred_classes = np.argmax(raw_preds, axis=1)
        else:
            pred_classes = raw_preds.ravel()
        
        # Pull topology boundaries using the correct property name 'layer_sizes'
        num_classes = controller.model.layer_sizes[-1]

        if num_classes > 1:
            true_classes = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 and y_test.shape[1] > 1 else y_test.ravel().astype(int)
            accuracy = np.mean(pred_classes == true_classes) * 100
            logging.info(f"[Diagnostics Check] Out-of-Sample Final Verification Accuracy: {accuracy:.2f}%")
            NeuralNetworkDiagnostics.plot_confusion_matrix(true_classes, pred_classes, num_classes, meta_cfg.output_dir, diag_cfg)

        # 3. Route to configured figure metrics
        if selection in ['1', 'loss', 'all']:
            NeuralNetworkDiagnostics.plot_metric_1_loss(controller.train_history, meta_cfg.output_dir, diag_cfg)
        if selection in ['2', 'gap', 'all']:
            NeuralNetworkDiagnostics.plot_metric_2_gap(controller.train_history, controller.val_history, meta_cfg.output_dir, diag_cfg)
        if selection in ['3', 'efficiency', 'all']:
            NeuralNetworkDiagnostics.plot_metric_3_efficiency(controller.train_history, meta_cfg.output_dir, diag_cfg)
        if selection in ['4', 'space', 'all']:
            
            # Re-architect closure logic to use your clean controller layer
            def predict_unscaled(X_raw_input):
                if fourier_cfg.enabled and X_raw_input.shape[1] < mean.shape[0]:
                    num_freqs = fourier_cfg.num_frequencies
                    features = [X_raw_input]
                    for i in range(num_freqs):
                        freq = 2.0 ** i
                        features.append(np.sin(X_raw_input * freq))
                        features.append(np.cos(X_raw_input * freq))
                    X_eval_processing = np.hstack(features)
                else:
                    X_eval_processing = X_raw_input

                # Returns clean class predictions or distributions seamlessly via the gateway
                return controller.predict(X_eval_processing)

            NeuralNetworkDiagnostics.plot_metric_4_space(X_test_raw, y_test, feature_names, predict_unscaled, meta_cfg.output_dir, diag_cfg)

        # 4. Handle Raw Performance Metric Log Output
        if diag_cfg.save_raw_logs:
            log_path = os.path.join(meta_cfg.output_dir, f"{meta_cfg.pipeline_name}_history.csv")
            with open(log_path, "w") as f:
                f.write("epoch,train_loss,val_loss\n")
                for i in range(len(controller.val_history)):
                    train_l = controller.train_history[i] if i < len(controller.train_history) else 0.0
                    f.write(f"{i},{train_l:.6f},{controller.val_history[i]:.6f}\n")
            logging.info(f"[Diagnostics Log] Raw epoch histories archived to: {log_path}")

    @staticmethod
    def _apply_configured_style(diag_cfg):
        """Safely configures custom Matplotlib parameters on the fly."""
        try:
            plt.style.use(diag_cfg.plot_style)
        except Exception:
            plt.style.use('default')

    @staticmethod
    def plot_metric_1_loss(history, out_dir, diag_cfg):
        NeuralNetworkDiagnostics._apply_configured_style(diag_cfg)
        plt.figure(figsize=(diag_cfg.figure_width, diag_cfg.figure_height), dpi=100)
        plt.plot(history, color="#1f77b4", linewidth=2)
        plt.title("Metric 1: Loss Curve Geometry", fontweight="bold")
        plt.xlabel("Epoch Loops")
        plt.ylabel("Loss Magnitude")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Figure_1.{diag_cfg.output_format.lower()}"))
        plt.close()

    @staticmethod
    def plot_metric_2_gap(train_h, val_h, out_dir, diag_cfg):
        NeuralNetworkDiagnostics._apply_configured_style(diag_cfg)
        plt.figure(figsize=(diag_cfg.figure_width, diag_cfg.figure_height), dpi=100)
        plt.plot(train_h, color="#2ca02c", label="Train Loss")
        plt.plot(val_h, color="#d62728", linestyle="--", label="Validation Loss")
        plt.title("Metric 2: Generalization Gap Profiles", fontweight="bold")
        plt.xlabel("Epoch Loops")
        plt.ylabel("Loss Core Range")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Figure_2.{diag_cfg.output_format.lower()}"))
        plt.close()

    @staticmethod
    def plot_metric_3_efficiency(adam_h, out_dir, diag_cfg):
        NeuralNetworkDiagnostics._apply_configured_style(diag_cfg)
        plt.figure(figsize=(diag_cfg.figure_width, diag_cfg.figure_height), dpi=100)
        plt.plot(adam_h, color="#9467bd", label="Adam Engine Processing Curve")
        plt.title("Metric 3: Optimization Convergence Progress", fontweight="bold")
        plt.xlabel("Epoch Loops")
        plt.ylabel("Running Objective Target Scalar")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Figure_3.{diag_cfg.output_format.lower()}"))
        plt.close()

    @staticmethod
    def plot_metric_4_space(X, y, features, predict_fn, out_dir, diag_cfg):
        NeuralNetworkDiagnostics._apply_configured_style(diag_cfg)
        num_features = X.shape[1]
        fig = plt.figure(figsize=(diag_cfg.figure_width, diag_cfg.figure_height), dpi=100)

        # 1D FEATURE INPUT WINDOW
        if num_features == 1:
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            xx = np.linspace(x_min, x_max, 500).reshape(-1, 1)
            raw_preds = predict_fn(xx)
            if raw_preds.ndim > 1 and raw_preds.shape[1] > 1:
                classes = np.argmax(raw_preds, axis=1)
            else:
                classes = raw_preds.ravel()
            plt.plot(xx, classes, color="#ff7f0e", linewidth=2.5, label="Model Prediction")
            plt.scatter(X[:, 0], y.ravel(), color="#1f77b4", alpha=0.6, edgecolor="k", label="Actual Data")
            plt.xlabel(features[0])
            plt.ylabel("Target Value")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)

        # 2D FEATURE INPUT WINDOW (Contour Landscape Topology)
        elif num_features == 2:
            x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
            y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
            raw_preds = predict_fn(np.c_[xx.ravel(), yy.ravel()])
            
            y_eval_flat = np.argmax(y, axis=1) if len(y.shape) > 1 and y.shape[1] > 1 else y.ravel()
            
            if raw_preds.ndim > 1 and raw_preds.shape[1] > 1:
                zz = np.argmax(raw_preds, axis=1).reshape(xx.shape)
                num_classes = raw_preds.shape[1]
                levels = np.arange(-0.5, num_classes + 0.5, 1)
                cmap_to_use = "Set1"
            else:
                zz = raw_preds.reshape(xx.shape)
                num_classes = len(np.unique(y_eval_flat))
                levels = 50
                cmap_to_use = "coolwarm"
            
            contour = plt.contourf(xx, yy, zz, levels=levels, cmap=cmap_to_use, alpha=0.6)
            if num_classes > 1 and raw_preds.ndim > 1 and raw_preds.shape[1] > 1:
                cbar = plt.colorbar(contour, ticks=list(range(num_classes)))
                cbar.set_label("Assigned Class Category", rotation=270, labelpad=15, fontweight="bold")
                cbar.ax.set_yticklabels([f'Class {i}' for i in range(num_classes)])
                plt.scatter(X[:, 0], X[:, 1], c=y_eval_flat, cmap="Set1", edgecolor="k", linewidth=0.5)
            else:
                cbar = plt.colorbar(contour)
                cbar.set_label("Predicted Value", rotation=270, labelpad=15, fontweight="bold")
                plt.scatter(X[:, 0], X[:, 1], c=y_eval_flat, cmap="coolwarm", edgecolor="k", linewidth=0.5)
            plt.xlabel(features[0])
            plt.ylabel(features[1])

        # 3D FEATURE INPUT WINDOW (Stereoscopic Projection)
        elif num_features == 3:
            ax = fig.add_subplot(111, projection='3d')
            raw_preds = predict_fn(X)
            y_eval_flat = np.argmax(y, axis=1) if len(y.shape) > 1 and y.shape[1] > 1 else y.ravel()
            
            if raw_preds.ndim > 1 and raw_preds.shape[1] > 1:
                display_values = np.argmax(raw_preds, axis=1)
                cmap_to_use = "Set1"
            else:
                display_values = raw_preds.ravel()
                cmap_to_use = "coolwarm"
                
            sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=display_values.ravel(), cmap=cmap_to_use, edgecolor='k', s=40, alpha=0.8)
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])
            ax.set_zlabel(features[2] if len(features) > 2 else "Feature 3")
            cbar = plt.colorbar(sc, pad=0.1)
            cbar.set_label("Model Prediction Matrix Mapping", rotation=270, labelpad=15, fontweight="bold")

        # HIGH-DIMENSIONAL FALLBACK (>3 FEATURES UPPER BOUND RESIDUALS)
        else:
            raw_preds = predict_fn(X)
            y_eval_flat = np.argmax(y, axis=1) if len(y.shape) > 1 and y.shape[1] > 1 else y.ravel()
            
            if raw_preds.ndim > 1 and raw_preds.shape[1] > 1:
                display_preds = np.argmax(raw_preds, axis=1)
                plt.scatter(range(len(y_eval_flat)), y_eval_flat, color="#2ca02c", alpha=0.6, label="True Categorical Labels", marker="o")
                plt.scatter(range(len(display_preds)), display_preds, color="#d62728", alpha=0.6, label="Predicted Class Matrix", marker="x")
                plt.ylabel("Discrete Class Index")
            else:
                display_preds = raw_preds.ravel()
                plt.scatter(y_eval_flat, display_preds, color="#9467bd", alpha=0.6, edgecolor="k")
                ideal_line = np.linspace(min(y_eval_flat.min(), display_preds.min()), max(y_eval_flat.max(), display_preds.max()), 100)
                plt.plot(ideal_line, ideal_line, color="black", linestyle="--", linewidth=1.5, label="Ideal Performance Horizon")
                plt.xlabel("Ground Truth Target Landscape")
                plt.ylabel("Predicted Target Output")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)

        plt.title(f"Metric 4: Target Prediction Landscape ({num_features}D Space)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Figure_4.{diag_cfg.output_format.lower()}"))
        plt.close()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, num_classes, out_dir, diag_cfg):
        NeuralNetworkDiagnostics._apply_configured_style(diag_cfg)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        fig, ax = plt.subplots(figsize=(diag_cfg.figure_width // 2 + 1, diag_cfg.figure_height - 1), dpi=100)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(num_classes), yticks=np.arange(num_classes),
               xticklabels=[f"P:{i}" for i in range(num_classes)], 
               yticklabels=[f"T:{i}" for i in range(num_classes)],
               title="Multi-Class Confusion Matrix Mapping",
               ylabel="True Category", xlabel="Predicted Category")

        thresh = cm.max() / 2.
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontweight="bold")
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Figure_5.{diag_cfg.output_format.lower()}"))
        plt.close()