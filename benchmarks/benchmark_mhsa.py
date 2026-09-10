# benchmarks/benchmark_mhsa.py
"""Causal MHSA e2e A/B: native contract engine vs PyTorch Pre-LN twin."""
from __future__ import annotations

import gc
import getpass
import multiprocessing as mp
import os
import platform
import sys
import time
import warnings

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.runtime import (
    apply_process_env,
    load_runtime_settings,
    log_runtime_settings,
    training_threadpool,
)

_RUNTIME = load_runtime_settings()
apply_process_env(_RUNTIME, if_unset=True)

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")

from config.config_loader import load_production_config
from config.constants import DataKeys, EngineBackend, IngestionMode, LRHierarchy, ModelType
from config.schema import LedgerSettings, MHSAConfig
from src.controller import ModelController
from src.data.base_loader import BaseDataLoader
from src.data.in_memory_provider import InMemoryDataProvider
from utils.engine_ops import create_engine_context

from benchmarks.benchmark_cnn import (
    extract_custom_engine_param_count,
    load_native_telemetry_lib,
    reset_benchmark_data_provider,
    resolve_backend,
)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true.astype(np.float64).reshape(-1)
    yp = y_pred.astype(np.float64).reshape(-1)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true.astype(np.float64) - y_pred.astype(np.float64)) ** 2))


def _mhsa_as_dict(mhsa) -> dict:
    if mhsa is None:
        raise ValueError("architecture.mhsa is required for MHSA benchmark")
    if isinstance(mhsa, MHSAConfig) or hasattr(mhsa, "d_model"):
        return {
            "d_model": int(mhsa.d_model),
            "num_heads": int(mhsa.num_heads),
            "max_seq_len": int(mhsa.max_seq_len),
            "action_dim": int(mhsa.action_dim),
            "ffn_mult": int(getattr(mhsa, "ffn_mult", 4)),
            "num_layers": int(getattr(mhsa, "num_layers", 1)),
            "use_pos_encoding": bool(getattr(mhsa, "use_pos_encoding", True)),
        }
    return {
        "d_model": int(mhsa["d_model"]),
        "num_heads": int(mhsa["num_heads"]),
        "max_seq_len": int(mhsa["max_seq_len"]),
        "action_dim": int(mhsa["action_dim"]),
        "ffn_mult": int(mhsa.get("ffn_mult", 4)),
        "num_layers": int(mhsa.get("num_layers", 1)),
        "use_pos_encoding": bool(mhsa.get("use_pos_encoding", True)),
    }


def create_torch_mhsa_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CausalPreLNBlock(nn.Module):
        def __init__(self, d_model: int, num_heads: int, ffn_mult: int):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
            self.attn = nn.MultiheadAttention(
                d_model, num_heads, batch_first=True, bias=True
            )
            self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
            hidden = d_model * ffn_mult
            self.ff1 = nn.Linear(d_model, hidden)
            self.ff2 = nn.Linear(hidden, d_model)

        def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
            x = x + a
            h = self.ln2(x)
            # Match native tanh-approx GELU (not erf).
            x = x + self.ff2(F.gelu(self.ff1(h), approximate="tanh"))
            return x

    class TorchCausalMHSA(nn.Module):
        """Pre-LN causal MHSA stack + last-token tanh action head (native twin)."""

        def __init__(self, mhsa: dict):
            super().__init__()
            d_model = int(mhsa["d_model"])
            num_heads = int(mhsa["num_heads"])
            ffn_mult = int(mhsa.get("ffn_mult", 4))
            num_layers = int(mhsa.get("num_layers", 1))
            action_dim = int(mhsa["action_dim"])
            if d_model % num_heads != 0:
                raise ValueError(f"d_model={d_model} not divisible by num_heads={num_heads}")
            self.blocks = nn.ModuleList(
                [CausalPreLNBlock(d_model, num_heads, ffn_mult) for _ in range(num_layers)]
            )
            self.action = nn.Linear(d_model, action_dim)
            self._max_seq_len = int(mhsa["max_seq_len"])
            self.use_pos = bool(mhsa.get("use_pos_encoding", True))
            if self.use_pos:
                self.pos_embed = nn.Parameter(
                    torch.randn(self._max_seq_len, d_model) * 0.02
                )
            else:
                self.register_parameter("pos_embed", None)
            self.register_buffer(
                "_causal_mask",
                torch.triu(
                    torch.ones(self._max_seq_len, self._max_seq_len, dtype=torch.bool),
                    diagonal=1,
                ),
                persistent=False,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, D)
            t = x.shape[1]
            if self.use_pos and self.pos_embed is not None:
                x = x + self.pos_embed[:t]
            mask = self._causal_mask[:t, :t]
            for block in self.blocks:
                x = block(x, mask)
            return torch.tanh(self.action(x[:, -1, :]))

    return TorchCausalMHSA


def _mhsa_param_count(controller: ModelController) -> int:
    model = getattr(controller, "model", None)
    if model is None:
        return extract_custom_engine_param_count(controller)
    total = 0
    for attr in ("weights", "biases", "ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta"):
        mats = getattr(model, attr, None)
        if mats is None:
            continue
        for m in mats:
            if m is not None:
                total += int(np.asarray(m).size)
    pos = getattr(model, "pos_embed", None)
    if pos is not None:
        total += int(np.asarray(pos).size)
    return total if total > 0 else extract_custom_engine_param_count(controller)


def load_mhsa_benchmark_data(config_path: str | None = None):
    if config_path is None:
        config_path = os.path.join(project_root, "config", "config.yaml")
    cfg = load_production_config(config_path)
    if cfg.architecture.model_type != ModelType.MHSA:
        raise ValueError(
            f"benchmark_mhsa expects model_type=mhsa, got {cfg.architecture.model_type}"
        )

    data_path = cfg.ingestion.data_file_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(project_root, data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"MHSA dataset missing: {data_path}. "
            "Generate with: .venv/bin/python data/generators/mhsa/generate_cue_recall.py --preset quick"
        )

    # Re-bind absolute path for the loader without mutating frozen cfg fields via object replace.
    from dataclasses import replace

    cfg = replace(cfg, ingestion=replace(cfg.ingestion, data_file_path=data_path))

    mhsa = _mhsa_as_dict(cfg.architecture.mhsa)
    batch_size = int(cfg.optimization.batch_size)
    epochs = int(cfg.optimization.epochs_full_dataset)
    loader = BaseDataLoader.create_loader(cfg)
    data_provider = InMemoryDataProvider(
        loader=loader,
        batch_size=batch_size,
        epochs=epochs,
        normalize_features=False,
    )
    return data_provider, cfg, mhsa, data_path


def run_pytorch_mhsa_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    mhsa: dict,
    batch_size: int,
    epochs: int,
    lr_init: float,
    lam_l2: float,
    early_stopping_enabled: bool,
    patience: int,
    min_delta: float,
    num_threads: int = 4,
) -> dict:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    verified_threads = torch.get_num_threads()
    print(
        f"\n[1/2] Setting up and executing PyTorch MHSA benchmark "
        f"({verified_threads} Threads active)..."
    )
    torch.manual_seed(42)

    TorchCls = create_torch_mhsa_class()
    torch_model = TorchCls(mhsa)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(torch_model.parameters(), lr=lr_init, weight_decay=lam_l2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    train_x = torch.tensor(np.ascontiguousarray(X_train), dtype=torch.float32)
    train_y = torch.tensor(np.ascontiguousarray(y_train), dtype=torch.float32)
    val_x = torch.tensor(np.ascontiguousarray(X_val), dtype=torch.float32)
    val_y = torch.tensor(np.ascontiguousarray(y_val), dtype=torch.float32)

    train_ds = TensorDataset(train_x, train_y)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    torch_total_params = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)

    t0_train = time.perf_counter()
    torch_epochs_completed = 0
    final_torch_train_loss = 0.0
    final_torch_val_loss = 0.0
    final_torch_val_r2 = 0.0
    best_torch_val_loss = float("inf")
    best_torch_epoch = 1
    torch_patience_counter = 0
    torch_early_stopped = False
    torch_forward_counts = 0
    torch_backward_counts = 0

    settings = load_runtime_settings(num_threads=num_threads, native_async_submit=False)
    log_runtime_settings(settings, EngineBackend.NUMPY, prefix="[Benchmark PyTorch MHSA]")

    with training_threadpool(settings, EngineBackend.NUMPY):
        for ep in range(epochs):
            torch_model.train()
            running_loss = torch.tensor(0.0)
            for bx, by in train_dl:
                optimizer.zero_grad(set_to_none=True)
                out = torch_model(bx)
                torch_forward_counts += 1
                loss = criterion(out, by)
                loss.backward()
                torch_backward_counts += 1
                optimizer.step()
                running_loss += loss.detach() * len(bx)

            final_torch_train_loss = (running_loss / len(train_ds)).item()
            scheduler.step()
            torch_epochs_completed += 1

            torch_model.eval()
            with torch.no_grad():
                val_out = torch_model(val_x)
                torch_forward_counts += 1
                current_val_loss = criterion(val_out, val_y).item()
                final_torch_val_loss = current_val_loss
                final_torch_val_r2 = _r2_score(y_val, val_out.cpu().numpy())

            if best_torch_val_loss - current_val_loss > min_delta:
                best_torch_val_loss = current_val_loss
                best_torch_epoch = ep + 1
                torch_patience_counter = 0
            else:
                torch_patience_counter += 1
                if current_val_loss < best_torch_val_loss:
                    best_torch_val_loss = current_val_loss
                    best_torch_epoch = ep + 1
                if early_stopping_enabled and torch_patience_counter >= patience:
                    torch_early_stopped = True
                    break

        torch_train_time = time.perf_counter() - t0_train

        torch_model.eval()
        t0_inf = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = torch_model(val_x)
        torch_inf_time = (time.perf_counter() - t0_inf) / 100.0

    del torch_model, optimizer, scheduler, criterion, train_dl, train_ds
    del val_x, val_y, train_x, train_y
    gc.collect()

    return {
        "params": torch_total_params,
        "threads_verified": verified_threads,
        "epochs_completed": torch_epochs_completed,
        "best_epoch": best_torch_epoch,
        "early_stopped": torch_early_stopped,
        "forward_counts": torch_forward_counts,
        "backward_counts": torch_backward_counts,
        "train_loss": float(final_torch_train_loss),
        "val_loss": float(final_torch_val_loss),
        "val_acc": float(final_torch_val_r2),  # R² (report labels say R²)
        "train_time": float(torch_train_time),
        "inf_time": float(torch_inf_time),
    }


def run_custom_mhsa_benchmark(
    data_provider: InMemoryDataProvider,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    mhsa: dict,
    epochs: int,
    lr_init: float,
    lam_l1: float,
    lam_l2: float,
    early_stopping_enabled: bool,
    patience: int,
    min_delta: float,
    backend: EngineBackend = EngineBackend.NATIVE,
    num_threads: int = 4,
    config_path: str | None = None,
    ledger_settings: LedgerSettings | None = None,
    output_dir: str | None = None,
    contract_list_enabled: bool = True,
) -> dict:
    if config_path is None:
        config_path = os.path.join(project_root, "config", "config.yaml")
    if ledger_settings is None or output_dir is None:
        cfg = load_production_config(config_path)
        if ledger_settings is None:
            ledger_settings = cfg.ledger
        if output_dir is None:
            output_dir = cfg.meta.output_dir

    engine_ctx = create_engine_context(backend)
    native_lib = engine_ctx.native_lib or (
        load_native_telemetry_lib() if backend == EngineBackend.NATIVE else None
    )
    settings = load_runtime_settings(config_path=config_path, num_threads=num_threads)
    planned_threads = settings.omp_threads_for(backend)

    print(
        f"[2/2] Setting up and executing Custom MHSA [{backend.value}] benchmark "
        f"({planned_threads} Threads configured)..."
    )
    if native_lib and hasattr(native_lib, "reset_thread_execution_stats"):
        native_lib.reset_thread_execution_stats()

    reset_benchmark_data_provider(data_provider)
    np.random.seed(42)

    controller = ModelController(
        data_provider=data_provider,
        learning_rate=lr_init,
        lr_scheduler_type=LRHierarchy.EXPONENTIAL,
        scheduler_decay_rate=0.98,
        scheduler_drop_ratio=0.5,
        scheduler_epochs_per_drop=10,
    )
    controller.initialize_network_from_dimensions(
        input_dim=int(np.prod(X_train.shape[1:])),
        output_dim=int(mhsa["action_dim"]),
        model_type=ModelType.MHSA,
        hidden_layers=[],
        optimizer_name="adam",
        lam_l1=lam_l1,
        lam_l2=lam_l2,
        p_dropout=0.0,
        use_batch_norm=False,
        bn_momentum=0.9,
        max_norm=5.0,
        mhsa_config=mhsa,
        backend=backend,
        contract_list_enabled=contract_list_enabled,
    )

    model = controller.model
    forward_counter = [0]
    backward_counter = [0]

    if hasattr(model, "run_contract_train_step"):
        _orig_contract_step = model.run_contract_train_step

        def _counted_contract_step(*args, **kwargs):
            forward_counter[0] += 1
            backward_counter[0] += 1
            return _orig_contract_step(*args, **kwargs)

        model.run_contract_train_step = _counted_contract_step

    _orig_predict = model.predict

    def _counted_predict(processed_data, *args, **kwargs):
        forward_counter[0] += 1
        return _orig_predict(processed_data, *args, **kwargs)

    model.predict = _counted_predict

    custom_total_params = _mhsa_param_count(controller)

    with training_threadpool(settings, backend):
        log_runtime_settings(settings, backend, prefix="[Benchmark Custom MHSA]")
        t0_train = time.perf_counter()
        train_history, val_history = controller.fit(
            steps=data_provider.recomment_steps(),
            source_mode=IngestionMode.CSV,
            model_type=ModelType.MHSA,
            early_stopping_enabled=early_stopping_enabled,
            patience=patience,
            min_delta=min_delta,
            ledger_settings=ledger_settings,
            output_dir=output_dir,
        )
        custom_train_time = time.perf_counter() - t0_train
        custom_forward_counts = forward_counter[0]
        custom_backward_counts = backward_counter[0]

        t0_inf = time.perf_counter()
        for _ in range(100):
            custom_raw_val_preds = controller.predict(X_val)
        custom_inf_time = (time.perf_counter() - t0_inf) / 100.0

        final_custom_val_loss = _mse(y_val, custom_raw_val_preds)
        final_custom_val_r2 = _r2_score(y_val, custom_raw_val_preds)
        custom_raw_train_preds = controller.predict(X_train)
        final_custom_train_loss = _mse(y_train, custom_raw_train_preds)

    if val_history and len(val_history) > 0:
        custom_epochs_completed = len(val_history)
        custom_best_epoch = int(np.argmin(val_history) + 1)
    elif hasattr(controller, "val_loss_history") and len(controller.val_loss_history) > 0:
        custom_epochs_completed = len(controller.val_loss_history)
        custom_best_epoch = int(np.argmin(controller.val_loss_history) + 1)
    else:
        custom_epochs_completed = getattr(controller, "epochs_completed", epochs)
        custom_best_epoch = getattr(controller, "best_epoch", custom_epochs_completed)

    custom_early_stopped = custom_epochs_completed < epochs

    if backend == EngineBackend.NATIVE and native_lib and hasattr(
        native_lib, "log_thread_execution_stats"
    ):
        native_lib.log_thread_execution_stats()

    del controller, model, train_history
    gc.collect()

    return {
        "params": custom_total_params,
        "threads_verified": planned_threads,
        "epochs_completed": custom_epochs_completed,
        "best_epoch": custom_best_epoch,
        "early_stopped": custom_early_stopped,
        "forward_counts": custom_forward_counts,
        "backward_counts": custom_backward_counts,
        "train_loss": float(final_custom_train_loss),
        "val_loss": float(final_custom_val_loss),
        "val_acc": float(final_custom_val_r2),
        "train_time": float(custom_train_time),
        "inf_time": float(custom_inf_time),
    }


def format_system_banner_mhsa(
    *,
    data_path: str,
    backend,
    epochs: int,
    batch_size: int,
    lr_init: float,
    lam_l2: float,
    lam_l1: float,
    lr_scheduler_type: str,
    early_stopping_enabled: bool,
    patience: int,
    min_delta: float,
    num_threads: int,
    mhsa: dict,
) -> str:
    user_name = getpass.getuser()
    system_node = platform.node()
    os_name = f"{platform.system()} {platform.release()} ({platform.machine()})"
    cpu_model = platform.processor() or "AMD x86_64 Family"
    logical_cores = os.cpu_count()
    backend_value = backend.value if hasattr(backend, "value") else str(backend)
    geom = (
        f"L={mhsa['num_layers']} D={mhsa['d_model']} H={mhsa['num_heads']} "
        f"T<={mhsa['max_seq_len']} A={mhsa['action_dim']} ffn×{mhsa['ffn_mult']}"
    )
    lines = [
        "=" * 80,
        "      CONVERGENCE BENCHMARK: CUSTOM ENGINE vs PYTORCH MHSA",
        "=" * 80,
        f"User / Host         : {user_name}@{system_node}",
        f"OS / Architecture   : {os_name}",
        f"CPU Model           : {cpu_model}",
        f"Logical CPU Cores   : {logical_cores}",
        f"Dataset Path        : {data_path}",
        f"Active Backend      : {backend_value}",
        f"Epochs / Batch Size : {epochs} / {batch_size}",
        f"Learning Rate / L2  : {lr_init} / {lam_l2} (L1: {lam_l1})",
        f"LR Scheduler Type   : {lr_scheduler_type}",
        (
            f"Early Stopping      : Enabled={early_stopping_enabled} "
            f"(Patience={patience}, Min Delta={min_delta})"
        ),
        "-" * 80,
        f"Configured Threads  : {num_threads} Threads (Enforced via OMP/MKL/PyTorch)",
        f"MHSA Geometry       : {geom}",
        "=" * 80,
    ]
    return "\n".join(lines)


def format_mhsa_head_to_head(
    t_res: dict,
    c_res: dict,
    *,
    epochs: int,
    n_train: int,
    n_val: int,
    backend,
) -> str:
    backend_value = backend.value if hasattr(backend, "value") else str(backend)
    torch_col = f"PyTorch MHSA ({t_res['threads_verified']}T)"
    custom_col = f"Custom [{backend_value}] ({c_res['threads_verified']}T)"
    torch_tp = (n_train * t_res["epochs_completed"]) / t_res["train_time"]
    custom_tp = (n_train * c_res["epochs_completed"]) / c_res["train_time"]
    ratio = (
        c_res["train_time"] / t_res["train_time"] if t_res["train_time"] > 0 else float("inf")
    )
    lines = [
        "",
        "=" * 80,
        "HEAD-TO-HEAD MHSA BENCHMARK REPORT".center(80),
        "=" * 80,
        f"{'Performance Metric':<32} | {torch_col:<20} | {custom_col:<20}",
        "-" * 80,
        f"{'Active Hardware Threads':<32} | {t_res['threads_verified']:<20d} | {c_res['threads_verified']:<20d}",
        f"{'Total Trainable Parameters':<32} | {t_res['params']:<20,d} | {c_res['params']:<20,d}",
        f"{'Target Epochs':<32} | {epochs:<20d} | {epochs:<20d}",
        f"{'Epochs Completed':<32} | {t_res['epochs_completed']:<20d} | {c_res['epochs_completed']:<20d}",
        f"{'Best Validation Epoch':<32} | {t_res['best_epoch']:<20d} | {c_res['best_epoch']:<20d}",
        f"{'Early Stopping Triggered':<32} | {str(t_res['early_stopped']):<20} | {str(c_res['early_stopped']):<20}",
        f"{'Forward Pass Count':<32} | {t_res['forward_counts']:<20,d} | {c_res['forward_counts']:<20,d}",
        f"{'Backward Pass Count':<32} | {t_res['backward_counts']:<20,d} | {c_res['backward_counts']:<20,d}",
        f"{'Final Training Loss (MSE)':<32} | {t_res['train_loss']:<20.6f} | {c_res['train_loss']:<20.6f}",
        f"{'Final Validation Loss (MSE)':<32} | {t_res['val_loss']:<20.6f} | {c_res['val_loss']:<20.6f}",
        f"{'Final Validation R²':<32} | {t_res['val_acc']:<20.4f} | {c_res['val_acc']:<20.4f}",
        "-" * 80,
        f"{'Total Training Time':<32} | {t_res['train_time']:>18.3f} s | {c_res['train_time']:>18.3f} s",
        f"{'Training Throughput':<32} | {torch_tp:>14.1f} smp/s | {custom_tp:>14.1f} smp/s",
        (
            f"{'Time per Epoch':<32} | "
            f"{(t_res['train_time'] / t_res['epochs_completed']) * 1000:>16.2f} ms | "
            f"{(c_res['train_time'] / c_res['epochs_completed']) * 1000:>16.2f} ms"
        ),
        (
            f"{'Val Inference Latency (Batch)':<32} | "
            f"{t_res['inf_time'] * 1000:>16.3f} ms | {c_res['inf_time'] * 1000:>16.3f} ms"
        ),
        (
            f"{'Per-Sample Inference Latency':<32} | "
            f"{(t_res['inf_time'] / n_val) * 1000:>16.4f} ms | "
            f"{(c_res['inf_time'] / n_val) * 1000:>16.4f} ms"
        ),
        "-" * 80,
        f"{'Custom / PyTorch Train Ratio':<32} | {ratio:>18.3f}x | {'(>1 = custom slower)':>20}",
        "=" * 80,
    ]
    return "\n".join(lines)


def format_mhsa_single_engine(
    res: dict,
    *,
    engine: str,
    epochs: int,
    n_train: int,
    n_val: int,
    backend,
) -> str:
    backend_value = backend.value if hasattr(backend, "value") else str(backend)
    if engine == "pytorch":
        col = f"PyTorch MHSA ({res['threads_verified']}T)"
        title = "PYTORCH-ONLY MHSA BENCHMARK REPORT"
    else:
        col = f"Custom [{backend_value}] ({res['threads_verified']}T)"
        title = "CUSTOM-ONLY MHSA BENCHMARK REPORT"
    throughput = (n_train * res["epochs_completed"]) / res["train_time"]
    lines = [
        "",
        "=" * 80,
        title.center(80),
        "=" * 80,
        f"{'Performance Metric':<32} | {col:<20}",
        "-" * 80,
        f"{'Active Hardware Threads':<32} | {res['threads_verified']:<20d}",
        f"{'Total Trainable Parameters':<32} | {res['params']:<20,d}",
        f"{'Target Epochs':<32} | {epochs:<20d}",
        f"{'Epochs Completed':<32} | {res['epochs_completed']:<20d}",
        f"{'Best Validation Epoch':<32} | {res['best_epoch']:<20d}",
        f"{'Early Stopping Triggered':<32} | {str(res['early_stopped']):<20}",
        f"{'Forward Pass Count':<32} | {res['forward_counts']:<20,d}",
        f"{'Backward Pass Count':<32} | {res['backward_counts']:<20,d}",
        f"{'Final Training Loss (MSE)':<32} | {res['train_loss']:<20.6f}",
        f"{'Final Validation Loss (MSE)':<32} | {res['val_loss']:<20.6f}",
        f"{'Final Validation R²':<32} | {res['val_acc']:<20.4f}",
        "-" * 80,
        f"{'Total Training Time':<32} | {res['train_time']:>18.3f} s",
        f"{'Training Throughput':<32} | {throughput:>14.1f} smp/s",
        (
            f"{'Time per Epoch':<32} | "
            f"{(res['train_time'] / res['epochs_completed']) * 1000:>16.2f} ms"
        ),
        (
            f"{'Val Inference Latency (Batch)':<32} | "
            f"{res['inf_time'] * 1000:>16.3f} ms"
        ),
        (
            f"{'Per-Sample Inference Latency':<32} | "
            f"{(res['inf_time'] / n_val) * 1000:>16.4f} ms"
        ),
        "=" * 80,
    ]
    return "\n".join(lines)


def _benchmark_common_from_config(config_path: str | None):
    data_provider, cfg, mhsa, data_path = load_mhsa_benchmark_data(config_path)
    X_train = data_provider.splits[DataKeys.X_TRAIN]
    y_train = data_provider.splits[DataKeys.Y_TRAIN]
    X_val = data_provider.splits[DataKeys.X_VAL]
    y_val = data_provider.splits[DataKeys.Y_VAL]
    backend = cfg.architecture.backend
    if isinstance(backend, str):
        backend = resolve_backend(backend)
    return {
        "data_provider": data_provider,
        "cfg": cfg,
        "mhsa": mhsa,
        "data_path": data_path,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "backend": backend,
        "epochs": int(cfg.optimization.epochs_full_dataset),
        "batch_size": int(cfg.optimization.batch_size),
        "lr_init": float(cfg.optimization.learning_rate),
        "lam_l1": float(cfg.regularization.lam_l1),
        "lam_l2": float(cfg.regularization.lam_l2),
        "num_threads": int(cfg.optimization.num_threads),
        "early_stopping_enabled": bool(cfg.optimization.early_stopping_enabled),
        "patience": int(cfg.optimization.patience),
        "min_delta": float(cfg.optimization.min_delta),
        "lr_scheduler_type": str(cfg.optimization.lr_scheduler),
        "ledger_settings": cfg.ledger,
        "output_dir": cfg.meta.output_dir,
        "contract_list_enabled": bool(
            getattr(cfg.ledger, "contract_list_enabled", True)
        ),
    }


def pytorch_mhsa_child(config_path, result_queue) -> None:
    try:
        c = _benchmark_common_from_config(config_path)
        t_res = run_pytorch_mhsa_benchmark(
            X_train=c["X_train"],
            y_train=c["y_train"],
            X_val=c["X_val"],
            y_val=c["y_val"],
            mhsa=c["mhsa"],
            batch_size=c["batch_size"],
            epochs=c["epochs"],
            lr_init=c["lr_init"],
            lam_l2=c["lam_l2"],
            early_stopping_enabled=c["early_stopping_enabled"],
            patience=c["patience"],
            min_delta=c["min_delta"],
            num_threads=c["num_threads"],
        )
        result_queue.put(("ok", t_res))
    except Exception as exc:
        result_queue.put(("err", f"{type(exc).__name__}: {exc}"))


def custom_mhsa_child(config_path, result_queue) -> None:
    try:
        c = _benchmark_common_from_config(config_path)
        c_res = run_custom_mhsa_benchmark(
            data_provider=c["data_provider"],
            X_train=c["X_train"],
            y_train=c["y_train"],
            X_val=c["X_val"],
            y_val=c["y_val"],
            mhsa=c["mhsa"],
            epochs=c["epochs"],
            lr_init=c["lr_init"],
            lam_l1=c["lam_l1"],
            lam_l2=c["lam_l2"],
            early_stopping_enabled=c["early_stopping_enabled"],
            patience=c["patience"],
            min_delta=c["min_delta"],
            backend=c["backend"],
            num_threads=c["num_threads"],
            config_path=config_path,
            ledger_settings=c["ledger_settings"],
            output_dir=c["output_dir"],
            contract_list_enabled=c["contract_list_enabled"],
        )
        result_queue.put(("ok", c_res))
    except Exception as exc:
        result_queue.put(("err", f"{type(exc).__name__}: {exc}"))


def run_benchmark_child(target, config_path, label: str):
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=target, args=(config_path, result_queue), name=label)
    proc.start()
    status, payload = result_queue.get()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"{label} exited with code {proc.exitcode}")
    if status != "ok":
        raise RuntimeError(f"{label} failed: {payload}")
    return payload


def run_mhsa_benchmark(config_path: str | None = None, target: str = "both"):
    if target not in ("pytorch", "custom", "both"):
        raise ValueError(f"target must be pytorch|custom|both, got {target!r}")

    c = _benchmark_common_from_config(config_path)
    print(
        format_system_banner_mhsa(
            data_path=c["data_path"],
            backend=c["backend"],
            epochs=c["epochs"],
            batch_size=c["batch_size"],
            lr_init=c["lr_init"],
            lam_l2=c["lam_l2"],
            lam_l1=c["lam_l1"],
            lr_scheduler_type=c["lr_scheduler_type"],
            early_stopping_enabled=c["early_stopping_enabled"],
            patience=c["patience"],
            min_delta=c["min_delta"],
            num_threads=c["num_threads"],
            mhsa=c["mhsa"],
        )
    )
    print(f"[Benchmark] target={target} (each engine in its own spawned process)")

    t_res = None
    c_res = None
    if target in ("pytorch", "both"):
        t_res = run_benchmark_child(pytorch_mhsa_child, config_path, "benchmark-mhsa-pytorch")
    if target in ("custom", "both"):
        c_res = run_benchmark_child(custom_mhsa_child, config_path, "benchmark-mhsa-custom")

    if t_res is not None and c_res is not None:
        print(
            format_mhsa_head_to_head(
                t_res,
                c_res,
                epochs=c["epochs"],
                n_train=len(c["X_train"]),
                n_val=len(c["X_val"]),
                backend=c["backend"],
            )
        )
    elif t_res is not None:
        print(
            format_mhsa_single_engine(
                t_res,
                engine="pytorch",
                epochs=c["epochs"],
                n_train=len(c["X_train"]),
                n_val=len(c["X_val"]),
                backend=c["backend"],
            )
        )
    elif c_res is not None:
        print(
            format_mhsa_single_engine(
                c_res,
                engine="custom",
                epochs=c["epochs"],
                n_train=len(c["X_train"]),
                n_val=len(c["X_val"]),
                backend=c["backend"],
            )
        )
    return t_res, c_res


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MHSA convergence benchmark: custom engine vs PyTorch"
    )
    parser.add_argument(
        "--config",
        default=os.path.join(project_root, "config", "config.yaml"),
        help="Pipeline YAML (must be model_type=mhsa)",
    )
    parser.add_argument(
        "--target",
        choices=("pytorch", "custom", "both"),
        default="both",
        help="Which engine(s) to run (default: both, spawn-isolated)",
    )
    args = parser.parse_args()
    run_mhsa_benchmark(config_path=args.config, target=args.target)
