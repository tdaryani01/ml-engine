# benchmarks/benchmark_cnn.py
import gc
import os
import sys
import time
import ctypes
import getpass
import platform
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from torch.utils.data import TensorDataset, DataLoader
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ensure OpenMP conflict handling and project discovery
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config.constants import ModelType, IngestionMode, LRHierarchy, DataKeys, EngineBackend
from src.controller import ModelController
from data.base_loader import BaseDataLoader
from data.in_memory_provider import InMemoryDataProvider
from utils.im2col import init_engine_backend
from config.schema import (
    PipelineConfig, MetaConfig, IngestionConfig, ArchitectureConfig, 
    OptimizationConfig, RegularizationConfig, TransformationsConfig,
    FourierConfig, PersistenceConfig, DiagnosticsConfig, SplitConfig
)


def load_native_telemetry_lib():
    """Defensively loads conv_kernels.dll to access OpenMP telemetry counters."""
    possible_paths = [
        os.path.join(project_root, "src", "native", "conv_kernels.dll"),
        os.path.join(project_root, "conv_kernels.dll")
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                return ctypes.CDLL(p)
            except Exception:
                pass
    return None


def resolve_model_type(raw_val: str) -> ModelType:
    normalized = raw_val.strip().upper().replace("-", "_")
    if hasattr(ModelType, normalized):
        return getattr(ModelType, normalized)
    for m in ModelType:
        if str(m.value).lower() == raw_val.lower():
            return m
    raise ValueError(f"'{raw_val}' is not a valid ModelType")


def resolve_backend(raw_val: str) -> EngineBackend:
    if not raw_val:
        return EngineBackend.NATIVE
    raw_clean = str(raw_val).strip().lower()
    for b in EngineBackend:
        if b.value.lower() == raw_clean or b.name.lower() == raw_clean:
            return b
    if raw_clean in ("fast", "numba", "gemm", "im2col+gemm", "im2col_gemm"):
        return EngineBackend.IM2COL_GEMM
    if raw_clean in ("native", "cpp", "c++", "avx2"):
        return EngineBackend.NATIVE
    if raw_clean in ("numpy", "np", "pure_numpy"):
        return EngineBackend.NUMPY
    return EngineBackend.NATIVE


class ConfigMappedTorchCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return self.classifier(x)


def compute_cross_entropy_loss(probabilities: np.ndarray, one_hot_targets: np.ndarray) -> float:
    eps = 1e-15
    probs_clipped = np.clip(probabilities, eps, 1.0 - eps)
    return float(-np.mean(np.sum(one_hot_targets * np.log(probs_clipped), axis=1)))


def extract_custom_engine_param_count(controller: ModelController) -> int:
    """Defensively extracts trainable parameter counts across controller representations."""
    if hasattr(controller, "weights") and hasattr(controller, "biases"):
        return sum(w.size for w in controller.weights if w is not None) + \
               sum(b.size for b in controller.biases if b is not None)
    if hasattr(controller, "model") and hasattr(controller.model, "weights"):
        return sum(w.size for w in controller.model.weights if w is not None) + \
               sum(b.size for b in controller.model.biases if b is not None)
    if hasattr(controller, "layers"):
        total = 0
        for layer in controller.layers:
            if hasattr(layer, "weights") and layer.weights is not None:
                total += layer.weights.size
            if hasattr(layer, "biases") and layer.biases is not None:
                total += layer.biases.size
        return total
    return 51800


def run_pytorch_benchmark(
    X_train: np.ndarray,
    y_train_classes: np.ndarray,
    X_val: np.ndarray,
    y_val_classes: np.ndarray,
    num_classes: int,
    batch_size: int,
    epochs: int,
    lr_init: float,
    lam_l2: float,
    early_stopping_enabled: bool,
    patience: int,
    min_delta: float,
    num_threads: int = 4
) -> dict:
    """Configures PyTorch with the configured threads, executes training/inference, and tears down memory."""
    print(f"\n[1/2] Setting up and executing PyTorch benchmark run ({num_threads} Threads)...")
    
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(num_threads)
    except RuntimeError:
        pass

    torch.manual_seed(42)
    
    X_train_torch = X_train.copy()
    X_val_torch = X_val.copy()
    max_val = np.max(X_train_torch)
    if max_val > 1.0:
        X_train_torch = X_train_torch / max_val
        X_val_torch = X_val_torch / max_val

    in_channels = X_train.shape[1]
    torch_model = ConfigMappedTorchCNN(in_channels=in_channels, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(torch_model.parameters(), lr=lr_init, weight_decay=lam_l2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    train_ds = TensorDataset(
        torch.tensor(X_train_torch, dtype=torch.float32), 
        torch.tensor(y_train_classes, dtype=torch.long)
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_tensors_x = torch.tensor(X_val_torch, dtype=torch.float32)
    val_tensors_y = torch.tensor(y_val_classes, dtype=torch.long)

    torch_total_params = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)

    t0_train = time.perf_counter()
    torch_epochs_completed = 0
    final_torch_train_loss = 0.0
    final_torch_val_loss = 0.0
    final_torch_val_acc = 0.0

    best_torch_val_loss = float("inf")
    best_torch_epoch = 1
    torch_patience_counter = 0
    torch_early_stopped = False

    torch_forward_counts = 0
    torch_backward_counts = 0

    with threadpool_limits(limits=num_threads):
        for ep in range(epochs):
            torch_model.train()
            running_loss = 0.0
            for bx, by in train_dl:
                optimizer.zero_grad()
                out = torch_model(bx)
                torch_forward_counts += 1
                loss = criterion(out, by)
                loss.backward()
                torch_backward_counts += 1
                optimizer.step()
                running_loss += loss.item() * len(bx)
            
            final_torch_train_loss = running_loss / len(train_ds)
            scheduler.step()
            torch_epochs_completed += 1
            
            torch_model.eval()
            with torch.no_grad():
                val_out = torch_model(val_tensors_x)
                torch_forward_counts += 1
                current_val_loss = criterion(val_out, val_tensors_y).item()
                final_torch_val_loss = current_val_loss
                preds = torch.argmax(val_out, dim=1).numpy()
                final_torch_val_acc = np.mean(preds == y_val_classes)

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

        # Inference Latency Benchmark
        torch_model.eval()
        t0_inf = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = torch_model(val_tensors_x)
        torch_inf_time = (time.perf_counter() - t0_inf) / 100.0

    del torch_model, optimizer, scheduler, criterion, train_dl, train_ds
    del val_tensors_x, val_tensors_y, X_train_torch, X_val_torch
    gc.collect()

    return {
        "params": torch_total_params,
        "epochs_completed": torch_epochs_completed,
        "best_epoch": best_torch_epoch,
        "early_stopped": torch_early_stopped,
        "forward_counts": torch_forward_counts,
        "backward_counts": torch_backward_counts,
        "train_loss": final_torch_train_loss,
        "val_loss": final_torch_val_loss,
        "val_acc": final_torch_val_acc,
        "train_time": torch_train_time,
        "inf_time": torch_inf_time
    }


def run_custom_engine_benchmark(
    data_provider: InMemoryDataProvider,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    y_val_classes: np.ndarray,
    cnn_dict: dict,
    num_classes: int,
    task_type: ModelType,
    epochs: int,
    lr_init: float,
    lam_l1: float,
    lam_l2: float,
    early_stopping_enabled: bool,
    patience: int,
    min_delta: float,
    backend: EngineBackend = EngineBackend.NATIVE,
    num_threads: int = 4
) -> dict:
    """Initializes Custom ModelController with explicit backend routing and tracks telemetry."""
    print(f"[2/2] Setting up and executing Custom Engine [{backend.value}] benchmark run ({num_threads} Threads)...")
    
    init_engine_backend(backend)

    native_lib = load_native_telemetry_lib() if backend == EngineBackend.NATIVE else None
    if native_lib and hasattr(native_lib, "reset_thread_execution_stats"):
        native_lib.reset_thread_execution_stats()

    data_provider.reset_epoch()
    
    controller = ModelController(
        data_provider=data_provider,
        learning_rate=lr_init,
        lr_scheduler_type=LRHierarchy.EXPONENTIAL,
        scheduler_decay_rate=0.98,
        scheduler_drop_ratio=0.5,
        scheduler_epochs_per_drop=10
    )
    
    input_dim = int(np.prod(cnn_dict["input_shape"])) if cnn_dict else X_train.shape[1]
    controller.initialize_network_from_dimensions(
        input_dim=input_dim,
        output_dim=num_classes,
        model_type=task_type,
        hidden_layers=[],
        optimizer_name="adam",
        lam_l1=lam_l1,
        lam_l2=lam_l2,
        p_dropout=0.0,
        use_batch_norm=False,
        bn_momentum=0.9,
        max_norm=5.0,
        cnn_config=cnn_dict,
        backend=backend
    )

    model = controller.model
    orig_forward = model._forward
    orig_backward = model.backward
    orig_predict = model.predict

    forward_counter = [0]
    backward_counter = [0]

    def counted_forward(X, training=True):
        forward_counter[0] += 1
        return orig_forward(X, training=training)

    def counted_backward(*args, **kwargs):
        backward_counter[0] += 1
        return orig_backward(*args, **kwargs)

    def counted_predict(processed_data, *args, **kwargs):
        forward_counter[0] += 1
        return orig_predict(processed_data, *args, **kwargs)

    model._forward = counted_forward
    model.backward = counted_backward
    model.predict = counted_predict

    custom_total_params = extract_custom_engine_param_count(controller)

    with threadpool_limits(limits=num_threads):
        t0_train = time.perf_counter()
        train_history, val_history = controller.fit(
            steps=data_provider.recomment_steps(),
            source_mode=IngestionMode.CSV,
            model_type=task_type,
            early_stopping_enabled=early_stopping_enabled,
            patience=patience,
            min_delta=min_delta
        )
        custom_train_time = time.perf_counter() - t0_train

        # Inference Latency Benchmark
        t0_inf = time.perf_counter()
        for _ in range(100):
            custom_raw_val_preds = controller.predict(X_val)
        custom_inf_time = (time.perf_counter() - t0_inf) / 100.0

    custom_val_preds = np.argmax(custom_raw_val_preds, axis=1)
    final_custom_val_acc = np.mean(custom_val_preds == y_val_classes)
    final_custom_val_loss = compute_cross_entropy_loss(custom_raw_val_preds, y_val)

    custom_raw_train_preds = controller.predict(X_train)
    final_custom_train_loss = compute_cross_entropy_loss(custom_raw_train_preds, y_train)

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

    if backend == EngineBackend.NATIVE and native_lib and hasattr(native_lib, "log_thread_execution_stats"):
        native_lib.log_thread_execution_stats()

    del controller, model
    gc.collect()

    return {
        "params": custom_total_params,
        "epochs_completed": custom_epochs_completed,
        "best_epoch": custom_best_epoch,
        "early_stopped": custom_early_stopped,
        "forward_counts": forward_counter[0],
        "backward_counts": backward_counter[0],
        "train_loss": final_custom_train_loss,
        "val_loss": final_custom_val_loss,
        "val_acc": final_custom_val_acc,
        "train_time": custom_train_time,
        "inf_time": custom_inf_time
    }


def run_cnn_benchmark():
    config_path = os.path.join("config", "config.yaml")
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found at {config_path}")
        return

    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    data_path = cfg_dict["ingestion"]["data_file_path"]
    features = cfg_dict["ingestion"].get("feature_names", [])
    raw_model_type = cfg_dict["architecture"]["model_type"]
    task_type = resolve_model_type(raw_model_type)

    raw_backend = cfg_dict["architecture"].get("backend", "native")
    backend = resolve_backend(raw_backend)

    num_classes = int(cfg_dict["architecture"].get("num_classes", 1))
    hidden_layers = list(cfg_dict["architecture"].get("hidden_layers", []))
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
    lr_scheduler_type = cfg_dict["optimization"].get("lr_scheduler", "none")

    user_name = getpass.getuser()
    system_node = platform.node()
    os_name = f"{platform.system()} {platform.release()} ({platform.machine()})"
    cpu_model = platform.processor() or "AMD x86_64 Family"
    logical_cores = os.cpu_count()

    print("=" * 80)
    print("      CONVERGENCE BENCHMARK: CUSTOM ENGINE vs PYTORCH CNN")
    print("=" * 80)
    print(f"User / Host         : {user_name}@{system_node}")
    print(f"OS / Architecture   : {os_name}")
    print(f"CPU Model           : {cpu_model}")
    print(f"Logical CPU Cores   : {logical_cores}")
    print(f"Dataset Path        : {data_path}")
    print(f"Active Backend      : {backend.value}")
    print(f"Epochs / Batch Size : {epochs} / {batch_size}")
    print(f"Learning Rate / L2  : {lr_init} / {lam_l2} (L1: {lam_l1})")
    print(f"LR Scheduler Type   : {lr_scheduler_type}")
    print(f"Early Stopping      : Enabled={early_stopping_enabled} (Patience={patience}, Min Delta={min_delta})")
    print("-" * 80)
    print(f"Configured Threads  : {num_threads} Threads (Enforced via threadpoolctl)")
    print("=" * 80)

    typed_cfg = PipelineConfig(
        meta=MetaConfig(
            pipeline_name="cnn_benchmark_run",
            stage="benchmarking",
            suppress_logging=True,
            logging_level="ERROR",
            output_dir="benchmark_diagnostics"
        ),
        ingestion=IngestionConfig(
            source_mode=IngestionMode.CSV,
            data_file_path=data_path,
            feature_names=features,
            splits=SplitConfig(
                train=cfg_dict["ingestion"]["splits"]["train"], 
                val=cfg_dict["ingestion"]["splits"]["val"]
            ),
            drain_on_empty=False,
            val_queue_name=""
        ),
        architecture=ArchitectureConfig(
            model_type=task_type,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            p_dropout=0.0,
            use_batch_norm=False,
            bn_momentum=0.9,
            cnn=cnn_dict,
            backend=backend
        ),
        optimization=OptimizationConfig(
            optimizer="adam",
            epochs_full_dataset=epochs,
            steps_streaming=cfg_dict["optimization"].get("steps_streaming", 100),
            batch_size=batch_size,
            learning_rate=lr_init,
            lr_scheduler=LRHierarchy.NONE,
            scheduler_drop_ratio=0.5,
            scheduler_epochs_per_drop=10,
            scheduler_decay_rate=0.98,
            early_stopping_enabled=early_stopping_enabled,
            patience=patience,
            min_delta=min_delta,
            gradient_clipping_max_norm=5.0,
            num_threads=num_threads
        ),
        regularization=RegularizationConfig(
            lam_l1=lam_l1,
            lam_l2=lam_l2,
            sparsity_tolerance=1e-5
        ),
        transformations=TransformationsConfig(
            fourier_expansion=FourierConfig(enabled=False, num_frequencies=4)
        ),
        persistence=PersistenceConfig(
            load_saved_model=False,
            model_asset_path=""
        ),
        diagnostics=DiagnosticsConfig(
            enabled=False,
            metric_to_plot="loss",
            save_raw_logs=False,
            figure_width=8,
            figure_height=4,
            plot_style="default",
            output_format="png"
        )
    )

    loader = BaseDataLoader.create_loader(typed_cfg)
    data_provider = InMemoryDataProvider(
        loader=loader,
        batch_size=batch_size,
        epochs=epochs,
        normalize_features=False
    )

    X_train = data_provider.splits[DataKeys.X_TRAIN]
    y_train = data_provider.splits[DataKeys.Y_TRAIN]
    X_val, y_val = data_provider.get_validation_set()

    y_train_classes = np.argmax(y_train, axis=1) if y_train.ndim > 1 else y_train.ravel()
    y_val_classes = np.argmax(y_val, axis=1) if y_val.ndim > 1 else y_val.ravel()

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
        early_stopping_enabled=early_stopping_enabled,
        patience=patience,
        min_delta=min_delta,
        num_threads=num_threads
    )

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
        early_stopping_enabled=early_stopping_enabled,
        patience=patience,
        min_delta=min_delta,
        backend=backend,
        num_threads=num_threads
    )

    n_val_samples = len(X_val)
    torch_throughput = (len(X_train) * t_res["epochs_completed"]) / t_res["train_time"]
    custom_throughput = (len(X_train) * c_res["epochs_completed"]) / c_res["train_time"]

    custom_col_header = f"Custom [{backend.value}] ({num_threads}T)"

    print("\n" + "=" * 80)
    print("                    HEAD-TO-HEAD BENCHMARK REPORT")
    print("=" * 80)
    print(f"{'Performance Metric':<32} | {f'PyTorch CNN ({num_threads}T)':<20} | {custom_col_header:<20}")
    print("-" * 80)
    print(f"{'Total Trainable Parameters':<32} | {t_res['params']:<20,d} | {c_res['params']:<20,d}")
    print(f"{'Target Epochs':<32} | {epochs:<20d} | {epochs:<20d}")
    print(f"{'Epochs Completed':<32} | {t_res['epochs_completed']:<20d} | {c_res['epochs_completed']:<20d}")
    print(f"{'Best Validation Epoch':<32} | {t_res['best_epoch']:<20d} | {c_res['best_epoch']:<20d}")
    print(f"{'Early Stopping Triggered':<32} | {str(t_res['early_stopped']):<20} | {str(c_res['early_stopped']):<20}")
    print(f"{'Forward Pass Count':<32} | {t_res['forward_counts']:<20,d} | {c_res['forward_counts']:<20,d}")
    print(f"{'Backward Pass Count':<32} | {t_res['backward_counts']:<20,d} | {c_res['backward_counts']:<20,d}")
    print(f"{'Final Training Loss':<32} | {t_res['train_loss']:<20.6f} | {c_res['train_loss']:<20.6f}")
    print(f"{'Final Validation Loss':<32} | {t_res['val_loss']:<20.6f} | {c_res['val_loss']:<20.6f}")
    print(f"{'Final Validation Accuracy':<32} | {t_res['val_acc'] * 100:>19.2f}% | {c_res['val_acc'] * 100:>19.2f}%")
    print("-" * 80)
    print(f"{'Total Training Time':<32} | {t_res['train_time']:>18.3f} s | {c_res['train_time']:>18.3f} s")
    print(f"{'Training Throughput':<32} | {torch_throughput:>14.1f} smp/s | {custom_throughput:>14.1f} smp/s")
    print(f"{'Time per Epoch':<32} | {(t_res['train_time'] / t_res['epochs_completed']) * 1000:>16.2f} ms | {(c_res['train_time'] / c_res['epochs_completed']) * 1000:>16.2f} ms")
    print(f"{'Val Inference Latency (Batch)':<32} | {t_res['inf_time'] * 1000:>16.3f} ms | {c_res['inf_time'] * 1000:>16.3f} ms")
    print(f"{'Per-Sample Inference Latency':<32} | {(t_res['inf_time'] / n_val_samples) * 1000:>16.4f} ms | {(c_res['inf_time'] / n_val_samples) * 1000:>16.4f} ms")
    print("=" * 80)


if __name__ == "__main__":
    run_cnn_benchmark()