import argparse
import json
import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.runtime import get_docker_section

import numpy as np
from config.constants import DataKeys
from benchmarks.benchmark_cnn import (
    load_benchmark_data,
    run_pytorch_benchmark,
    run_custom_engine_benchmark,
    resolve_model_type,
    resolve_backend,
    extract_layer_specs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Docker Isolated ML Engine Benchmark Runner")
    parser.add_argument(
        "--target",
        type=str,
        choices=["pytorch", "custom", "both"],
        required=True,
        help="Target framework to benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=get_docker_section().get(
            "results_file", "benchmark_diagnostics/docker_benchmark_results.json"
        ),
        help="Path to output JSON results",
    )
    return parser.parse_args()


def load_existing_results(output_path: str) -> dict:
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_results(output_path: str, data: dict):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def print_terminal_report(
    results: dict,
    epochs: int,
    num_train_samples: int,
    num_val_samples: int,
    backend_name: str,
    cnn_dict: dict,
    batch_size: int,
    sample_shape: tuple,
):
    t_res = results.get("pytorch")
    c_res = results.get("custom")

    if not t_res or not c_res:
        return

    torch_threads = t_res.get("threads_verified", 4)
    custom_threads = c_res.get("threads_verified", 4)

    torch_col = f"PyTorch CNN ({torch_threads}T)"
    custom_col = f"Custom [{backend_name}] ({custom_threads}T)"

    torch_time = max(t_res.get("train_time", 1e-6), 1e-6)
    custom_time = max(c_res.get("train_time", 1e-6), 1e-6)

    t_epochs_comp = t_res.get("epochs_completed", epochs)
    c_epochs_comp = c_res.get("epochs_completed", epochs)

    torch_throughput = (num_train_samples * t_epochs_comp) / torch_time
    custom_throughput = (num_train_samples * c_epochs_comp) / custom_time

    # Accurately extract layer geometry from spatial pipeline or root config
    specs = extract_layer_specs(cnn_dict)
    if specs:
        kernel_size = specs[0].get("kernel_size", 3)
        padding = specs[0].get("padding", 1)
        stride = specs[0].get("stride", 1)
        filter_desc = " -> ".join(str(s.get("out_channels", 32)) for s in specs)
    else:
        kernel_size = cnn_dict.get("kernel_size", 3) if cnn_dict else 3
        padding = cnn_dict.get("padding", 1) if cnn_dict else 1
        stride = cnn_dict.get("stride", 1) if cnn_dict else 1
        num_filters = cnn_dict.get("num_filters", [32]) if cnn_dict else [32]
        filter_desc = " -> ".join(map(str, num_filters)) if isinstance(num_filters, list) else str(num_filters)

    print("\n" + "=" * 88)
    print("                    HEAD-TO-HEAD CONVERGENCE & PERFORMANCE REPORT")
    print("=" * 88)
    
    # 1. Architecture & Execution Profile
    print(f"{'ARCHITECTURAL & ALGORITHM SPECIFICATIONS':^88}")
    print("-" * 88)
    print(f"{'Input Spatial Geometry':<36} | {str(sample_shape):<23} | {str(sample_shape):<23}")
    print(f"{'Convolution Kernels (H x W)':<36} | {f'{kernel_size}x{kernel_size}':<23} | {f'{kernel_size}x{kernel_size}':<23}")
    print(f"{'Padding / Stride':<36} | {f'pad={padding}, stride={stride}':<23} | {f'pad={padding}, stride={stride}':<23}")
    print(f"{'Channel Topology':<36} | {filter_desc:<23} | {filter_desc:<23}")
    print(f"{'Underlying Engine Algorithm':<36} | {'oneDNN / MKL (mkldnn)':<23} | {'AVX2 / im2col (OpenMP)':<23}")
    print(f"{'Data Batch Size':<36} | {batch_size:<23d} | {batch_size:<23d}")
    print(f"{'Active Hardware Threads':<36} | {torch_threads:<23d} | {custom_threads:<23d}")
    print(f"{'Total Trainable Parameters':<36} | {t_res.get('params', 0):<23,d} | {c_res.get('params', 0):<23,d}")
    print("-" * 88)

    # 2. Convergence Metrics
    print(f"{'CONVERGENCE & ACCURACY PROFILES':^88}")
    print("-" * 88)
    print(f"{'Target Epochs / Completed':<36} | {f'{epochs} / {t_epochs_comp}':<23} | {f'{epochs} / {c_epochs_comp}':<23}")
    print(f"{'Best Validation Epoch':<36} | {t_res.get('best_epoch', 0):<23d} | {c_res.get('best_epoch', 0):<23d}")
    print(f"{'Early Stopping Triggered':<36} | {str(t_res.get('early_stopped', False)):<23} | {str(c_res.get('early_stopped', False)):<23}")
    print(f"{'Total Forward Pass Dispatches':<36} | {t_res.get('forward_counts', 0):<23,d} | {c_res.get('forward_counts', 0):<23,d}")
    print(f"{'Total Backward Pass Dispatches':<36} | {t_res.get('backward_counts', 0):<23,d} | {c_res.get('backward_counts', 0):<23,d}")
    print(f"{'Final Training Loss':<36} | {t_res.get('train_loss', 0.0):<23.6f} | {c_res.get('train_loss', 0.0):<23.6f}")
    print(f"{'Final Validation Loss':<36} | {t_res.get('val_loss', 0.0):<23.6f} | {c_res.get('val_loss', 0.0):<23.6f}")
    print(f"{'Final Validation Accuracy':<36} | {t_res.get('val_acc', 0.0) * 100:>22.2f}% | {c_res.get('val_acc', 0.0) * 100:>22.2f}%")
    print("-" * 88)

    # 3. Latency & Throughput Metrics
    print(f"{'LATENCY & HARDWARE THROUGHPUT':^88}")
    print("-" * 88)
    print(f"{'Total Training Runtime':<36} | {torch_time:>21.3f} s | {custom_time:>21.3f} s")
    print(f"{'Training Throughput':<36} | {torch_throughput:>17.1f} smp/s | {custom_throughput:>17.1f} smp/s")
    print(f"{'Average Latency per Epoch':<36} | {(torch_time / max(t_epochs_comp, 1)) * 1000:>19.2f} ms | {(custom_time / max(c_epochs_comp, 1)) * 1000:>19.2f} ms")
    print(f"{'Validation Batch Latency':<36} | {t_res.get('inf_time', 0.0) * 1000:>19.3f} ms | {c_res.get('inf_time', 0.0) * 1000:>19.3f} ms")
    print(f"{'Per-Sample Inference Latency':<36} | {(t_res.get('inf_time', 0.0) / max(num_val_samples, 1)) * 1000:>19.4f} ms | {(c_res.get('inf_time', 0.0) / max(num_val_samples, 1)) * 1000:>19.4f} ms")
    print("=" * 88 + "\n")


def main():
    args = parse_args()

    print(f"[Docker Runner] Loading benchmark dataset and configuration...")
    data_provider, cfg_dict = load_benchmark_data()

    raw_model_type = cfg_dict["architecture"]["model_type"]
    task_type = resolve_model_type(raw_model_type)

    raw_backend = cfg_dict["architecture"].get("backend", "native")
    backend = resolve_backend(raw_backend)

    num_classes = int(cfg_dict["architecture"].get("num_classes", 1))
    batch_size = int(cfg_dict["optimization"]["batch_size"])
    lr_init = float(cfg_dict["optimization"]["learning_rate"])
    epochs = int(cfg_dict["optimization"]["epochs_full_dataset"])
    num_threads = int(cfg_dict["optimization"].get("num_threads", 4))
    lam_l2 = float(cfg_dict["regularization"].get("lam_l2", 0.0))
    lam_l1 = float(cfg_dict["regularization"].get("lam_l1", 0.0))
    cnn_dict = cfg_dict["architecture"].get("cnn", None)

    early_stopping_enabled = bool(cfg_dict["optimization"].get("early_stopping_enabled", False))
    patience = int(cfg_dict["optimization"].get("patience", 10))
    min_delta = float(cfg_dict["optimization"].get("min_delta", 1e-4))

    X_train = data_provider.splits[DataKeys.X_TRAIN]
    y_train = data_provider.splits[DataKeys.Y_TRAIN]
    X_val, y_val = data_provider.get_validation_set()

    y_train_classes = np.argmax(y_train, axis=1) if y_train.ndim > 1 else y_train.ravel()
    y_val_classes = np.argmax(y_val, axis=1) if y_val.ndim > 1 else y_val.ravel()

    results = load_existing_results(args.output)

    if args.target in ["pytorch", "both"]:
        print("\n[Docker Runner] Running PyTorch Baseline Benchmark...")
        pt_metrics = run_pytorch_benchmark(
            X_train=X_train,
            y_train_classes=y_train_classes,
            X_val=X_val,
            y_val_classes=y_val_classes,
            num_classes=num_classes,
            batch_size=batch_size,
            epochs=epochs,
            lr_init=lr_init,
            lam_l2=lam_l2,
            early_stopping_enabled=early_stopping_enabled,
            patience=patience,
            min_delta=min_delta,
            cnn_dict=cnn_dict,
            num_threads=num_threads,
        )
        results["pytorch"] = pt_metrics

    if args.target in ["custom", "both"]:
        print("\n[Docker Runner] Running Custom ML Engine Benchmark...")
        custom_metrics = run_custom_engine_benchmark(
            data_provider=data_provider,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            y_val_classes=y_val_classes,
            cnn_dict=cnn_dict,
            num_classes=num_classes,
            task_type=task_type,
            epochs=epochs,
            lr_init=lr_init,
            lam_l1=lam_l1,
            lam_l2=lam_l2,
            early_stopping_enabled=early_stopping_enabled,
            patience=patience,
            min_delta=min_delta,
            backend=backend,
            num_threads=num_threads,
        )
        results["custom"] = custom_metrics

    save_results(args.output, results)
    print(f"[Docker Runner] Benchmark metrics saved to: {args.output}")

    # Only print the comparative table if running both OR if executing the second (custom) stage
    if args.target in ["custom", "both"] and "pytorch" in results and "custom" in results:
        print_terminal_report(
            results=results,
            epochs=epochs,
            num_train_samples=len(X_train),
            num_val_samples=len(X_val),
            backend_name=backend.value,
            cnn_dict=cnn_dict,
            batch_size=batch_size,
            sample_shape=X_train.shape[1:],
        )


if __name__ == "__main__":
    main()