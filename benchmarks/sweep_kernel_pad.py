# benchmarks/sweep_kernel_pad.py
"""Sweep conv kernel size vs PyTorch with a full per-kernel log report."""
import copy
import datetime
import math
import os
import sys
from typing import IO, List, Optional, TextIO, Tuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from benchmarks.benchmark_cnn import (
    extract_layer_specs,
    format_head_to_head_report,
    format_system_banner,
    load_benchmark_data,
    run_custom_engine_benchmark,
    run_pytorch_benchmark,
    resolve_backend,
    resolve_model_type,
)
from config.constants import DataKeys
from utils.docker_omp_env import apply_docker_omp_env


class _Tee(TextIO):
    """Write benchmark output to stdout and a log file."""

    def __init__(self, *streams: IO[str]):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def patch_cnn_kernel(cnn_dict: dict, kernel_size: int, pad: int) -> dict:
    cfg = copy.deepcopy(cnn_dict)
    for item in cfg.get("spatial_pipeline", []):
        if isinstance(item, dict) and item.get("type") == "conv":
            item["kernel_size"] = int(kernel_size)
            item["pad"] = int(pad)
    return cfg


def conv_out_hw(h: int, w: int, k: int, pad: int, stride: int = 1) -> Tuple[int, int]:
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    return oh, ow


def pool_out_hw(h: int, w: int, pool: int, stride: int) -> Tuple[int, int]:
    oh = (h - pool) // stride + 1
    ow = (w - pool) // stride + 1
    return oh, ow


def expected_native_dispatch(kernel: int, pad: int) -> Tuple[str, str]:
    """All conv traffic routes through generic fallback + Stride1Specialist plugins."""
    del kernel, pad
    return "GENERIC_FALLBACK", "GENERIC_FALLBACK"


def format_geometry_section(
    *,
    kernel: int,
    pad: int,
    input_shape: tuple,
    specs: list,
    n_train: int,
    batch_size: int,
    epochs: int,
) -> str:
    c, h, w = input_shape
    steps = int(math.ceil(n_train / batch_size))
    conv_layers = len(specs)
    model_backward_per_epoch = steps
    conv_block_backward_per_epoch = steps * conv_layers

    lines = [
        "",
        "-" * 80,
        f"GEOMETRY & WORKLOAD — kernel={kernel}, pad={pad}",
        "-" * 80,
        f"Input shape (C,H,W)           : ({c}, {h}, {w})",
        f"Train samples / batch / steps : {n_train} / {batch_size} / {steps} per epoch",
        f"Target epochs (ES off)        : {epochs}",
        f"Model backward calls / epoch  : {model_backward_per_epoch}",
        f"Conv-block backward / epoch   : {conv_block_backward_per_epoch}",
        (
            f"Extrap. conv-block bwd (full): "
            f"{conv_block_backward_per_epoch * epochs} "
            f"(= {steps} batches × {conv_layers} blocks × {epochs} epochs)"
        ),
        "",
        "Layer spatial trace (stride=1 conv, then pool):",
    ]

    cur_h, cur_w = h, w
    for i, sp in enumerate(specs, start=1):
        k = sp["kernel_size"]
        p = sp["padding"]
        conv_h, conv_w = conv_out_hw(cur_h, cur_w, k, p, sp["stride"])
        fwd_path, bwd_path = expected_native_dispatch(k, p)
        lines.append(
            f"  Layer {i}: {sp['out_channels']}ch k={k} pad={p} "
            f"conv_out={conv_h}x{conv_w} | native FWD: {fwd_path}"
        )
        lines.append(f"           native BWD: {bwd_path}")
        if sp["pool_size"] > 0:
            cur_h, cur_w = pool_out_hw(conv_h, conv_w, sp["pool_size"], sp["pool_stride"])
            lines.append(
                f"           pool {sp['pool_size']}x{sp['pool_size']} stride={sp['pool_stride']} "
                f"-> {cur_h}x{cur_w}"
            )
        else:
            cur_h, cur_w = conv_h, conv_w

    lines.append("-" * 80)
    return "\n".join(lines)


def format_derived_metrics(
    *,
    kernel: int,
    t_res: dict,
    c_res: dict,
    steps: int,
    conv_layers: int,
) -> str:
    t_bwd_per_model = t_res["train_time"] / max(t_res["backward_counts"], 1) * 1000.0
    c_bwd_per_model = c_res["train_time"] / max(c_res["backward_counts"], 1) * 1000.0
    t_block_calls = t_res["backward_counts"] * conv_layers
    c_block_calls = c_res["backward_counts"] * conv_layers
    t_bwd_per_block = t_res["train_time"] / max(t_block_calls, 1) * 1000.0
    c_bwd_per_block = c_res["train_time"] / max(c_block_calls, 1) * 1000.0

    lines = [
        "",
        "-" * 80,
        f"DERIVED TIMING — kernel={kernel}",
        "-" * 80,
        f"PyTorch  train / model.backward call : {t_bwd_per_model:8.3f} ms",
        f"Custom   train / model.backward call : {c_bwd_per_model:8.3f} ms",
        (
            f"PyTorch  approx conv_block_backward  : {t_bwd_per_block:8.3f} ms "
            f"(train_time / ({t_res['backward_counts']} × {conv_layers}) conv blocks; conv+overhead)"
        ),
        (
            f"Custom   approx conv_block_backward  : {c_bwd_per_block:8.3f} ms "
            f"(train_time / ({c_res['backward_counts']} × {conv_layers}) conv blocks; conv+overhead)"
        ),
        (
            f"PyTorch  forward count breakdown     : "
            f"{t_res['forward_counts']} total "
            f"(~{steps} train + ~1 val per epoch × {t_res['epochs_completed']} epochs)"
        ),
        (
            f"Custom   forward count breakdown     : "
            f"{c_res['forward_counts']} total "
            f"(includes train + val predict hooks)"
        ),
        "-" * 80,
    ]
    return "\n".join(lines)


def format_summary_table(rows: List[dict]) -> str:
    lines = [
        "",
        "=" * 110,
        "SWEEP SUMMARY",
        "=" * 110,
        (
            f"{'k':>3} | {'PyTorch(s)':>10} | {'Custom(s)':>10} | {'Ratio':>7} | "
            f"{'Torch ms/ep':>11} | {'Custom ms/ep':>12} | "
            f"{'Torch bwd':>10} | {'Custom bwd':>10} | {'Val acc T/C':>14}"
        ),
        "-" * 110,
    ]
    for row in rows:
        lines.append(
            f"{row['kernel']:3d} | {row['torch_train_s']:10.3f} | {row['custom_train_s']:10.3f} | "
            f"{row['ratio']:7.2f}x | {row['torch_ms_per_epoch']:11.2f} | {row['custom_ms_per_epoch']:12.2f} | "
            f"{row['torch_bwd_ms']:10.3f} | {row['custom_bwd_ms']:10.3f} | {row['val_acc']:>14}"
        )
    lines.append("=" * 110)
    lines.append("Ratio = Custom / PyTorch total train time (>1 means custom slower)")
    lines.append("bwd ms = train_time / backward_counts (model.backward granularity)")
    return "\n".join(lines)


def run_sweep(
    kernel_sizes: List[int],
    pad: int,
    config_path: Optional[str] = None,
    output_path: Optional[str] = None,
    backend_override: Optional[str] = None,
) -> List[dict]:
    if config_path is None:
        config_path = os.path.join(project_root, "config", "config.yaml")

    data_provider, cfg_dict = load_benchmark_data(config_path)
    cfg_dict["optimization"]["early_stopping_enabled"] = False
    if backend_override:
        cfg_dict["architecture"]["backend"] = backend_override

    cnn_base = cfg_dict["architecture"]["cnn"]
    input_shape = tuple(cnn_base.get("input_shape", [3, 28, 28]))
    task_type = resolve_model_type(cfg_dict["architecture"]["model_type"])
    backend = resolve_backend(cfg_dict["architecture"].get("backend", "native"))

    batch_size = int(cfg_dict["optimization"]["batch_size"])
    epochs = int(cfg_dict["optimization"]["epochs_full_dataset"])
    lr_init = float(cfg_dict["optimization"]["learning_rate"])
    lam_l2 = float(cfg_dict["regularization"].get("lam_l2", 0.0))
    lam_l1 = float(cfg_dict["regularization"].get("lam_l1", 0.0))
    num_threads = int(cfg_dict["optimization"].get("num_threads", 4))
    num_classes = int(cfg_dict["architecture"]["num_classes"])
    patience = int(cfg_dict["optimization"].get("patience", 10))
    min_delta = float(cfg_dict["optimization"].get("min_delta", 1e-4))
    lr_scheduler_type = cfg_dict["optimization"].get("lr_scheduler", "none")
    data_path = cfg_dict["ingestion"]["data_file_path"]

    X_train = data_provider.splits[DataKeys.X_TRAIN]
    y_train = data_provider.splits[DataKeys.Y_TRAIN]
    X_val, y_val = data_provider.get_validation_set()
    y_train_classes = y_train.argmax(axis=1) if y_train.ndim > 1 else y_train.ravel()
    y_val_classes = y_val.argmax(axis=1) if getattr(y_val, "ndim", 1) > 1 else y_val.ravel()
    n_train = len(X_train)
    n_val = len(X_val)
    steps = int(math.ceil(n_train / batch_size))

    if output_path is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(project_root, "benchmark_diagnostics")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"kernel_sweep_pad{pad}_{stamp}.log")

    summary_rows: List[dict] = []

    with open(output_path, "w", encoding="utf-8") as log_file:
        tee = _Tee(sys.stdout, log_file)
        old_stdout = sys.stdout
        sys.stdout = tee
        try:
            print(format_system_banner(
                data_path=data_path,
                backend=backend,
                epochs=epochs,
                batch_size=batch_size,
                lr_init=lr_init,
                lam_l2=lam_l2,
                lam_l1=lam_l1,
                lr_scheduler_type=lr_scheduler_type,
                early_stopping_enabled=False,
                patience=patience,
                min_delta=min_delta,
                num_threads=num_threads,
                specs=extract_layer_specs(patch_cnn_kernel(cnn_base, kernel_sizes[0], pad)),
                title="KERNEL SWEEP: CUSTOM ENGINE vs PYTORCH CNN",
            ))
            print("")
            print(f"Sweep parameters : kernel={kernel_sizes[0]}..{kernel_sizes[-1]}, pad={pad}")
            print(f"Config file      : {config_path}")
            print(f"Log file         : {output_path}")
            print(f"Started (UTC)    : {datetime.datetime.now(datetime.timezone.utc).isoformat()}")

            for k in kernel_sizes:
                cnn_dict = patch_cnn_kernel(cnn_base, k, pad)
                specs = extract_layer_specs(cnn_dict)
                conv_layers = len(specs)

                print("")
                print("#" * 80)
                print(f"# KERNEL SIZE {k} × {k}  |  pad={pad}")
                print("#" * 80)
                print(format_geometry_section(
                    kernel=k,
                    pad=pad,
                    input_shape=input_shape,
                    specs=specs,
                    n_train=n_train,
                    batch_size=batch_size,
                    epochs=epochs,
                ))

                apply_docker_omp_env("pytorch", num_threads, overwrite=True)
                t_res = run_pytorch_benchmark(
                    X_train=X_train,
                    y_train_classes=y_train_classes,
                    X_val=X_val,
                    y_val_classes=y_val_classes,
                    num_classes=num_classes,
                    batch_size=batch_size,
                    epochs=epochs,
                    lr_init=lr_init,
                    lam_l2=lam_l2,
                    early_stopping_enabled=False,
                    patience=patience,
                    min_delta=min_delta,
                    cnn_dict=cnn_dict,
                    num_threads=num_threads,
                )

                apply_docker_omp_env("custom", num_threads, overwrite=True)
                c_res = run_custom_engine_benchmark(
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
                    early_stopping_enabled=False,
                    patience=patience,
                    min_delta=min_delta,
                    backend=backend,
                    num_threads=num_threads,
                )

                print(format_head_to_head_report(
                    t_res, c_res,
                    epochs=epochs,
                    n_train=n_train,
                    n_val=n_val,
                    backend=backend,
                    kernel_label=f"kernel={k}, pad={pad}",
                ))
                print(format_derived_metrics(
                    kernel=k,
                    t_res=t_res,
                    c_res=c_res,
                    steps=steps,
                    conv_layers=conv_layers,
                ))

                ratio = c_res["train_time"] / t_res["train_time"] if t_res["train_time"] > 0 else float("inf")
                summary_rows.append({
                    "kernel": k,
                    "torch_train_s": t_res["train_time"],
                    "custom_train_s": c_res["train_time"],
                    "ratio": ratio,
                    "torch_ms_per_epoch": (t_res["train_time"] / t_res["epochs_completed"]) * 1000.0,
                    "custom_ms_per_epoch": (c_res["train_time"] / c_res["epochs_completed"]) * 1000.0,
                    "torch_bwd_ms": t_res["train_time"] / max(t_res["backward_counts"], 1) * 1000.0,
                    "custom_bwd_ms": c_res["train_time"] / max(c_res["backward_counts"], 1) * 1000.0,
                    "val_acc": f"{t_res['val_acc'] * 100:.1f}%/{c_res['val_acc'] * 100:.1f}%",
                })

            print(format_summary_table(summary_rows))
            print(f"Finished (UTC)   : {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
            print(f"Log written to   : {output_path}")
        finally:
            sys.stdout = old_stdout

    return summary_rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sweep conv kernel size vs PyTorch with per-kernel log reports"
    )
    parser.add_argument("--pad", type=int, default=1)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=7)
    parser.add_argument("--backend", default=None, help="Override config backend (native | im2col+gemm)")
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument(
        "--output",
        default=None,
        help="Log file path (default: benchmark_diagnostics/kernel_sweep_pad{N}_<ts>.log)",
    )
    args = parser.parse_args()

    run_sweep(
        kernel_sizes=list(range(args.k_min, args.k_max + 1)),
        pad=args.pad,
        config_path=args.config,
        output_path=args.output,
        backend_override=args.backend,
    )
