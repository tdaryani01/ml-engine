# diagnostics.py
import os
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetworkDiagnostics:
    @staticmethod
    def run_diagnostics(metric_to_plot, model, train_history, val_history, X_test_raw, y_test, feature_names, mean, std, output_dir, y_mean=None, y_std=None):
        os.makedirs(output_dir, exist_ok=True)
        selection = str(metric_to_plot).strip().lower()

        # =====================================================================
        # DYNAMIC RECONSTRUCTION: Scale raw coordinates to match statistics
        # =====================================================================
        config = getattr(model, "config", {}) if hasattr(model, "config") else {}
        training_cfg = config.get("training", {})
        fourier_cfg = training_cfg.get("fourier_expansion", {})
        
        # If Fourier engineering is active but inputs are still raw, project them up
        if fourier_cfg.get("enabled", False) and X_test_raw.shape[1] < mean.shape[0]:
            num_freqs = fourier_cfg.get("num_frequencies", 4)
            features = [X_test_raw]
            for i in range(num_freqs):
                freq = 2.0 ** i
                features.append(np.sin(X_test_raw * freq))
                features.append(np.cos(X_test_raw * freq))
            X_processing = np.hstack(features)
        else:
            X_processing = X_test_raw

        # Align normalization vectors flawlessly against the correct matrix space
        X_test_norm = (X_processing - mean) / std

        # Calculate and print raw accuracy metrics for classification tasks (forcing inference mode)
        raw_preds = model.forward(X_test_norm, training=False)
        if len(raw_preds.shape) > 1 and raw_preds.shape[1] > 1:
            pred_classes = np.argmax(raw_preds, axis=1)
            true_classes = y_test.ravel().astype(int)
            accuracy = np.mean(pred_classes == true_classes) * 100
            print(f"\n[Diagnostics] Out-of-Sample Test Accuracy: {accuracy:.2f}%")
            NeuralNetworkDiagnostics.plot_confusion_matrix(true_classes, pred_classes, raw_preds.shape[1], output_dir)

        if selection in ['1', 'all']:
            NeuralNetworkDiagnostics.plot_metric_1_loss(train_history, output_dir)
        if selection in ['2', 'all']:
            NeuralNetworkDiagnostics.plot_metric_2_gap(train_history, val_history, output_dir)
        if selection in ['3', 'all']:
            # Cleared the ad-hoc training loop. 
            # If you want to plot efficiency, pass an independent historical baseline dictionary here.
            NeuralNetworkDiagnostics.plot_metric_3_efficiency(train_history, output_dir)
        if selection in ['4', 'all']:
            def predict_unscaled(X_raw_input):
                # Dynamically apply multi-feature spatial expansion to evaluation slices
                if fourier_cfg.get("enabled", False) and X_raw_input.shape[1] < mean.shape[0]:
                    num_freqs = fourier_cfg.get("num_frequencies", 4)
                    
                    features = [X_raw_input]
                    for i in range(num_freqs):
                        freq = 2.0 ** i
                        features.append(np.sin(X_raw_input * freq))
                        features.append(np.cos(X_raw_input * freq))
                    X_eval_processing = np.hstack(features)
                else:
                    X_eval_processing = X_raw_input

                # Align normalization vectors against the processed space
                X_input_norm = (X_eval_processing - mean) / std
                scaled_pred = model.forward(X_input_norm, training=False)
                if y_mean is not None and y_std is not None:
                    return (scaled_pred * y_std) + y_mean
                return scaled_pred

            NeuralNetworkDiagnostics.plot_metric_4_space(X_test_raw, y_test, feature_names, predict_unscaled, output_dir)

    @staticmethod
    def plot_metric_1_loss(history, out_dir):
        plt.figure(figsize=(8, 4), dpi=100)
        plt.plot(history, color="#1f77b4", linewidth=2)
        plt.title("Metric 1: Loss Curve Geometry", fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "Figure_1.png")); plt.close()

    @staticmethod
    def plot_metric_2_gap(train_h, val_h, out_dir):
        plt.figure(figsize=(8, 4), dpi=100)
        plt.plot(train_h, color="#2ca02c", label="Train Loss")
        plt.plot(val_h, color="#d62728", linestyle="--", label="Validation Loss")
        plt.title("Metric 2: Generalization Gap", fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "Figure_2.png")); plt.close()

    @staticmethod
    def plot_metric_3_efficiency(adam_h, out_dir):
        plt.figure(figsize=(8, 4), dpi=100)
        plt.plot(adam_h, color="#9467bd", label="Adam Engine")
        plt.title("Metric 3: Optimization Convergence Progress", fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "Figure_3.png")); plt.close()

    @staticmethod
    def plot_metric_4_space(X, y, features, predict_fn, out_dir):
        num_features = X.shape[1]
        fig = plt.figure(figsize=(10, 5), dpi=100)

        # =====================================================================
        # 1D FEATURE INPUT WINDOW (e.g., Single Feature Hard Regression)
        # =====================================================================
        if num_features == 1:
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            xx = np.linspace(x_min, x_max, 500).reshape(-1, 1)
            raw_preds = predict_fn(xx)
            
            plt.plot(xx, raw_preds, color="#ff7f0e", linewidth=2.5, label="Model Prediction")
            plt.scatter(X[:, 0], y.ravel(), color="#1f77b4", alpha=0.6, edgecolor="k", linewidth=0.5, label="Actual Data")
            plt.xlabel(features[0])
            plt.ylabel("Target Value" if len(features) < 2 else features[1])
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)

        # =====================================================================
        # 2D FEATURE INPUT WINDOW (Original Contour Landscape Topology)
        # =====================================================================
        elif num_features == 2:
            x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
            y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
            
            raw_preds = predict_fn(np.c_[xx.ravel(), yy.ravel()])
            if len(raw_preds.shape) > 1 and raw_preds.shape[1] > 1:
                zz = np.argmax(raw_preds, axis=1).reshape(xx.shape)
                num_classes = raw_preds.shape[1]
                levels = np.arange(-0.5, num_classes + 0.5, 1)
                cmap_to_use = "Set1"
            else:
                zz = raw_preds.reshape(xx.shape)
                num_classes = 1
                levels = 50
                cmap_to_use = "coolwarm"
            
            contour = plt.contourf(xx, yy, zz, levels=levels, cmap=cmap_to_use, alpha=0.6)
            
            if num_classes > 1:
                cbar = plt.colorbar(contour, ticks=list(range(num_classes)))
                cbar.set_label("Assigned Class Category", rotation=270, labelpad=15, fontweight="bold")
                cbar.ax.set_yticklabels([f'Class {i}' for i in range(num_classes)])
                plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="Set1", edgecolor="k", linewidth=0.5)
            else:
                cbar = plt.colorbar(contour)
                cbar.set_label("Predicted Continuous Value", rotation=270, labelpad=15, fontweight="bold")
                plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="coolwarm", edgecolor="k", linewidth=0.5)
            
            plt.xlabel(features[0])
            plt.ylabel(features[1])

        # =====================================================================
        # 3D FEATURE INPUT WINDOW (Stereoscopic Hypervolume Projection)
        # =====================================================================
        elif num_features == 3:
            ax = fig.add_subplot(111, projection='3d')
            raw_preds = predict_fn(X)
            
            if len(raw_preds.shape) > 1 and raw_preds.shape[1] > 1:
                display_values = np.argmax(raw_preds, axis=1)
                cmap_to_use = "Set1"
            else:
                display_values = raw_preds.ravel()
                cmap_to_use = "coolwarm"
                
            sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=display_values, cmap=cmap_to_use, edgecolor='k', s=40, alpha=0.8)
            
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])
            ax.set_zlabel(features[2] if len(features) > 2 else "Feature 3")
            
            cbar = plt.colorbar(sc, pad=0.1)
            cbar.set_label("Model Prediction Matrix Mapping", rotation=270, labelpad=15, fontweight="bold")

        # =====================================================================
        # HIGH-DIMENSIONAL FALLBACK (>3 FEATURES UPPER BOUND RESIDUALS)
        # =====================================================================
        else:
            raw_preds = predict_fn(X)
            if len(raw_preds.shape) > 1 and raw_preds.shape[1] > 1:
                display_preds = np.argmax(raw_preds, axis=1)
                plt.scatter(range(len(y)), y.ravel(), color="#2ca02c", alpha=0.6, label="True Categorical Labels", marker="o")
                plt.scatter(range(len(display_preds)), display_preds, color="#d62728", alpha=0.6, label="Predicted Class Matrix", marker="x")
                plt.ylabel("Discrete Class Index")
            else:
                display_preds = raw_preds.ravel()
                plt.scatter(y.ravel(), display_preds, color="#9467bd", alpha=0.6, edgecolor="k", linewidth=0.5)
                ideal_line = np.linspace(min(y.min(), display_preds.min()), max(y.max(), display_preds.max()), 100)
                plt.plot(ideal_line, ideal_line, color="black", linestyle="--", linewidth=1.5, label="Ideal Performance Horizon")
                plt.xlabel("Ground Truth Target Landscape")
                plt.ylabel("Predicted Target Output")
            
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)

        plt.title(f"Metric 4: Dynamic Target Prediction Landscape ({num_features}D Space)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "Figure_4.png")); plt.close()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, num_classes, out_dir):
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
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
        plt.savefig(os.path.join(out_dir, "Figure_5.png")); plt.close()

    @staticmethod
    def inspect_model_sparsity(model, tolerance=1e-5):
        total_w, total_p = 0, 0
        print("\n" + "="*50 + "\n          SPARSITY SCANNER REPORT\n" + "="*50)
        for idx, W in enumerate(model.weights):
            p = np.sum(np.abs(W) < tolerance)
            total_w += W.size; total_p += p
            print(f"Layer {idx + 1}: {p:4d} / {W.size:4d} elements zeroed out ({ (p/W.size)*100 :6.2f}% Sparse)")
        print(f"Global Sparsity: {(total_p/total_w)*100:.2f}%\n" + "="*50)