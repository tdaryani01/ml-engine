# diagnostics.py
import os
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetworkDiagnostics:
    @staticmethod
    def run_diagnostics(metric_to_plot, model, train_history, val_history, X_test_raw, y_test, feature_names, mean, std, output_dir, y_mean=None, y_std=None):
        os.makedirs(output_dir, exist_ok=True)
        selection = str(metric_to_plot).strip().lower()

        X_test_norm = (X_test_raw - mean) / std

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
                X_input_norm = (X_raw_input - mean) / std
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
        x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
        y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        
        raw_preds = predict_fn(np.c_[xx.ravel(), yy.ravel()])
        if len(raw_preds.shape) > 1 and raw_preds.shape[1] > 1:
            zz = np.argmax(raw_preds, axis=1).reshape(xx.shape)
            num_classes = raw_preds.shape[1]
            
            # Keep original discrete map settings for classification tasks
            levels = np.arange(-0.5, num_classes + 0.5, 1)
            cmap_to_use = "Set1"
        else:
            zz = raw_preds.reshape(xx.shape)
            num_classes = 1
            
            # Dynamic continuous normalization resolution bounds for regression
            levels = 50
            cmap_to_use = "coolwarm"
        
        plt.figure(figsize=(10, 5), dpi=100)
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
        
        plt.title("Metric 4: Dynamic Target Prediction Landscape", fontweight="bold")
        plt.xlabel(features[0]); plt.ylabel(features[1])
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