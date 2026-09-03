# benchmarks/benchmark_cnn.py
import gc
import os
import sys
import time
import ctypes
import getpass
import platform
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

# Apply threading env from config/runtime.yaml before NumPy/BLAS init.
_RUNTIME = load_runtime_settings()
apply_process_env(_RUNTIME, if_unset=True)

import yaml
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")

from config.constants import ModelType, IngestionMode, LRHierarchy, DataKeys, EngineBackend
from src.controller import ModelController
from src.data.base_loader import BaseDataLoader
from src.data.in_memory_provider import InMemoryDataProvider
from utils.engine_ops import create_engine_context
from config.schema import (
    PipelineConfig, MetaConfig, IngestionConfig, ArchitectureConfig, 
    OptimizationConfig, RegularizationConfig, TransformationsConfig,
    FourierConfig, PersistenceConfig, DiagnosticsConfig, SplitConfig
)


def load_native_telemetry_lib():
    possible_paths = [
        os.path.join(project_root, "src", "native", "conv_kernels.so"),
        os.path.join(project_root, "bin", "conv_kernels.so"),
        os.path.join(project_root, "conv_kernels.so"),
        os.path.join(project_root, "src", "native", "conv_kernels.dll"),
        os.path.join(project_root, "conv_kernels.dll"),
        os.path.join(project_root, "bin", "conv_kernels.dll"),
        os.path.join(project_root, "native", "conv_kernels.dll")
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


def extract_layer_specs(cnn_config: dict) -> list:
    if not cnn_config:
        return []
    
    spatial_pipe = cnn_config.get("spatial_pipeline", [])
    specs = []
    
    for item in spatial_pipe:
        if isinstance(item, dict) and item.get("type") == "conv":
            out_c = int(item.get("out_channels", 8))
            specs.append({
                "out_channels": out_c,
                "kernel_size": int(item.get("kernel_size", 3)),
                "stride": int(item.get("stride", 1)),
                "padding": int(item.get("pad", item.get("padding", 0))),
                "pool_size": 0,
                "pool_stride": 0
            })
        elif isinstance(item, dict) and item.get("type") == "pool" and specs:
            specs[-1]["pool_size"] = int(item.get("pool_size", 2))
            specs[-1]["pool_stride"] = int(item.get("stride", 2))
            
    if specs:
        return specs

    raw_layers = cnn_config.get("conv_layers") or cnn_config.get("layers") or []
    if raw_layers:
        for l in raw_layers:
            specs.append({
                "out_channels": int(l.get("out_channels", l.get("filters", 8))),
                "kernel_size": int(l.get("kernel_size", l.get("filter_size", 3))),
                "stride": int(l.get("stride", 1)),
                "padding": int(l.get("padding", l.get("pad", 0))),
                "pool_size": int(l.get("pool_size", l.get("pool", 2))),
                "pool_stride": int(l.get("pool_stride", l.get("pool_s", 2)))
            })
        return specs

    return []


def validate_cnn_spatial_geometry(input_shape: tuple, specs: list) -> None:
    """Fail fast when conv/pool stacks collapse spatial dims (before PyTorch init)."""
    if not specs:
        return

    if len(input_shape) == 3:
        h, w = int(input_shape[1]), int(input_shape[2])
    elif len(input_shape) == 2:
        h, w = int(input_shape[0]), int(input_shape[1])
    else:
        return

    for i, sp in enumerate(specs):
        k = int(sp["kernel_size"])
        p = int(sp["padding"])
        s = int(sp["stride"])
        h = (h + 2 * p - k) // s + 1
        w = (w + 2 * p - k) // s + 1
        if h < 1 or w < 1:
            raise ValueError(
                f"CNN layer {i + 1} conv output invalid ({h}x{w}) for input_shape={input_shape} "
                f"and kernel_size={k}, stride={s}, pad={p}."
            )

        pool_size = int(sp.get("pool_size") or 0)
        pool_stride = int(sp.get("pool_stride") or 0)
        if pool_size > 0:
            h = (h - pool_size) // pool_stride + 1
            w = (w - pool_size) // pool_stride + 1
            if h < 1 or w < 1:
                raise ValueError(
                    f"CNN layer {i + 1} pool output invalid ({h}x{w}) for input_shape={input_shape}. "
                    f"Reduce kernel/stride or increase input size."
                )


def ensure_nchw_format(data: np.ndarray, target_shape: tuple) -> np.ndarray:
    if len(target_shape) == 3:
        c, h, w = target_shape
    elif len(target_shape) == 2:
        c, h, w = 1, target_shape[0], target_shape[1]
    else:
        raise ValueError(f"Unsupported input shape configuration: {target_shape}")

    if data.ndim == 4:
        if data.shape[1] == c and data.shape[2] == h and data.shape[3] == w:
            return data
        if data.shape[3] == c and data.shape[1] == h and data.shape[2] == w:
            return np.transpose(data, (0, 3, 1, 2))
        if data.shape[2] == h and data.shape[3] >= w:
            return np.ascontiguousarray(data[:, :c, :h, :w])
        return data.reshape(data.shape[0], c, h, w)

    if data.ndim == 3:
        if data.shape[1] == h and data.shape[2] >= w:
            return np.ascontiguousarray(data[:, :h, :w]).reshape(data.shape[0], 1, h, w)
        return data.reshape(data.shape[0], 1, h, w)

    if data.ndim == 2:
        num_samples, num_features = data.shape
        if num_features == c * h * w:
            return data.reshape(num_samples, c, h, w)
            
        w_stride = num_features // (c * h)
        if num_features % (c * h) == 0 and w_stride >= w:
            reshaped = data.reshape(num_samples, c, h, w_stride)
            return np.ascontiguousarray(reshaped[:, :, :, :w])
            
        if num_features % (h * w) == 0:
            actual_c = num_features // (h * w)
            return data.reshape(num_samples, actual_c, h, w)
            
        side = int(np.sqrt(num_features // c))
        return data.reshape(num_samples, c, side, side)

    return data.reshape(-1, c, h, w)


def create_torch_model_class():
    import torch
    import torch.nn as nn

    class DynamicConfigTorchCNN(nn.Module):
        def __init__(self, in_channels: int, num_classes: int, cnn_config: dict, input_shape: tuple):
            super().__init__()
            specs = extract_layer_specs(cnn_config)
            
            feature_modules = []
            c_in = in_channels

            for sp in specs:
                feature_modules.append(
                    nn.Conv2d(c_in, sp["out_channels"], kernel_size=sp["kernel_size"], stride=sp["stride"], padding=sp["padding"])
                )
                feature_modules.append(nn.ReLU())
                if sp["pool_size"] > 0:
                    feature_modules.append(nn.MaxPool2d(kernel_size=sp["pool_size"], stride=sp["pool_stride"]))
                c_in = sp["out_channels"]

            self.features = nn.Sequential(*feature_modules)

            dummy = torch.zeros(1, in_channels, input_shape[1], input_shape[2])
            with torch.no_grad():
                feat_out = self.features(dummy)
                flattened_dim = feat_out.numel()

            dense_hidden = cnn_config.get("dense_head", [64]) if cnn_config else [64]
            if isinstance(dense_hidden, int):
                dense_hidden = [dense_hidden]

            classifier_modules = [nn.Flatten()]
            c_dense_in = flattened_dim
            for dh in dense_hidden:
                classifier_modules.append(nn.Linear(c_dense_in, dh))
                classifier_modules.append(nn.ReLU())
                c_dense_in = dh
            classifier_modules.append(nn.Linear(c_dense_in, num_classes))

            self.classifier = nn.Sequential(*classifier_modules)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            return self.classifier(self.features(x))

    return DynamicConfigTorchCNN


def compute_cross_entropy_loss(probabilities: np.ndarray, one_hot_targets: np.ndarray) -> float:
    eps = 1e-15
    probs_clipped = np.clip(probabilities, eps, 1.0 - eps)
    return float(-np.mean(np.sum(one_hot_targets * np.log(probs_clipped), axis=1)))


def extract_custom_engine_param_count(controller: ModelController) -> int:
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
    return 0


def load_benchmark_data(config_path: str = None):
    if config_path is None:
        config_path = os.path.join(project_root, "config", "config.yaml")

    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    data_path = cfg_dict["ingestion"]["data_file_path"]
    if not os.path.isabs(data_path):
        data_path = os.path.join(project_root, data_path)

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

    return data_provider, cfg_dict

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
    cnn_dict: dict,
    num_threads: int = 4
) -> dict:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    verified_threads = torch.get_num_threads()
    print(f"\n[1/2] Setting up and executing PyTorch benchmark run ({verified_threads} Threads active)...")
    
    torch.manual_seed(42)

    raw_shape = cnn_dict.get("input_shape", (1, 28, 28)) if cnn_dict else (1, 28, 28)
    if len(raw_shape) == 2:
        input_shape = (1, raw_shape[0], raw_shape[1])
    else:
        input_shape = tuple(raw_shape)

    if X_train.ndim == 2:
        num_features = X_train.shape[1]
        c, h, w = input_shape
        if num_features % (c * h * w) != 0:
            if num_features % (h * w) == 0:
                input_shape = (num_features // (h * w), h, w)
            else:
                side = int(np.sqrt(num_features))
                input_shape = (1, side, side)

    X_train_torch = ensure_nchw_format(X_train.copy(), input_shape)
    X_val_torch = ensure_nchw_format(X_val.copy(), input_shape)

    max_val = np.max(X_train_torch)
    if max_val > 1.0:
        X_train_torch = X_train_torch / max_val
        X_val_torch = X_val_torch / max_val

    TorchCNNClass = create_torch_model_class()
    torch_model = TorchCNNClass(
        in_channels=input_shape[0],
        num_classes=num_classes,
        cnn_config=cnn_dict or {},
        input_shape=input_shape
    )
    
    # 1. Align model weights to channels-last layout
    torch_model = torch_model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(torch_model.parameters(), lr=lr_init, weight_decay=lam_l2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # 2. Convert training and validation tensors to contiguous channels-last format
    train_x_tensor = torch.tensor(X_train_torch, dtype=torch.float32).to(memory_format=torch.channels_last).contiguous()
    train_y_tensor = torch.tensor(y_train_classes, dtype=torch.long)
    val_tensors_x = torch.tensor(X_val_torch, dtype=torch.float32).to(memory_format=torch.channels_last).contiguous()
    val_tensors_y = torch.tensor(y_val_classes, dtype=torch.long)

    train_ds = TensorDataset(train_x_tensor, train_y_tensor)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

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

    settings = load_runtime_settings(num_threads=num_threads)
    log_runtime_settings(settings, EngineBackend.NUMPY, prefix="[Benchmark PyTorch]")

    with training_threadpool(settings, EngineBackend.NUMPY):
        for ep in range(epochs):
            torch_model.train()
            running_loss = torch.tensor(0.0)
            
            for bx, by in train_dl:
                bx = bx.to(memory_format=torch.channels_last)
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
                val_out = torch_model(val_tensors_x)
                torch_forward_counts += 1
                current_val_loss = criterion(val_out, val_tensors_y).item()
                final_torch_val_loss = current_val_loss
                preds = torch.argmax(val_out, dim=1).cpu().numpy()
                final_torch_val_acc = float(np.mean(preds == y_val_classes))

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
                _ = torch_model(val_tensors_x)
        torch_inf_time = (time.perf_counter() - t0_inf) / 100.0

    del torch_model, optimizer, scheduler, criterion, train_dl, train_ds
    del val_tensors_x, val_tensors_y, X_train_torch, X_val_torch
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
        "val_acc": float(final_torch_val_acc),
        "train_time": float(torch_train_time),
        "inf_time": float(torch_inf_time)
    }

def reset_benchmark_data_provider(data_provider) -> None:
    """Reset epoch cursor so a shared InMemoryDataProvider can run another benchmark."""
    data_provider._epochs_completed = 0
    data_provider._batch_idx = 0
    data_provider._has_more = True
    data_provider.reset_epoch()


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
    engine_ctx = create_engine_context(backend)

    native_lib = engine_ctx.native_lib or (
        load_native_telemetry_lib() if backend == EngineBackend.NATIVE else None
    )
    verified_threads = num_threads
    if native_lib and hasattr(native_lib, "get_omp_threads"):
        try:
            verified_threads = native_lib.get_omp_threads()
        except Exception:
            pass

    print(f"[2/2] Setting up and executing Custom Engine [{backend.value}] benchmark run ({verified_threads} Threads active)...")
    
    if native_lib and hasattr(native_lib, "reset_thread_execution_stats"):
        native_lib.reset_thread_execution_stats()

    # Fresh epoch budget when the provider is reused across benchmark runs.
    reset_benchmark_data_provider(data_provider)

    np.random.seed(42)
    
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

    def counted_forward(X, training=True, **kwargs):
        forward_counter[0] += 1
        return orig_forward(X, training=training, **kwargs)

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

    settings = load_runtime_settings(num_threads=num_threads)
    log_runtime_settings(settings, backend, prefix="[Benchmark Custom]")

    with training_threadpool(settings, backend):
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

        t0_inf = time.perf_counter()
        for _ in range(100):
            custom_raw_val_preds = controller.predict(X_val)
        custom_inf_time = (time.perf_counter() - t0_inf) / 100.0

        custom_val_preds = np.argmax(custom_raw_val_preds, axis=1)
        final_custom_val_acc = float(np.mean(custom_val_preds == y_val_classes))
        final_custom_val_loss = float(compute_cross_entropy_loss(custom_raw_val_preds, y_val))

        custom_raw_train_preds = controller.predict(X_train)
        final_custom_train_loss = float(compute_cross_entropy_loss(custom_raw_train_preds, y_train))

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
        "threads_verified": verified_threads,
        "epochs_completed": custom_epochs_completed,
        "best_epoch": custom_best_epoch,
        "early_stopped": custom_early_stopped,
        "forward_counts": forward_counter[0],
        "backward_counts": backward_counter[0],
        "train_loss": float(final_custom_train_loss),
        "val_loss": float(final_custom_val_loss),
        "val_acc": float(final_custom_val_acc),
        "train_time": float(custom_train_time),
        "inf_time": float(custom_inf_time)
    }


def format_system_banner(
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
    specs: list,
    title: str = "CONVERGENCE BENCHMARK: CUSTOM ENGINE vs PYTORCH CNN",
) -> str:
    user_name = getpass.getuser()
    system_node = platform.node()
    os_name = f"{platform.system()} {platform.release()} ({platform.machine()})"
    cpu_model = platform.processor() or "AMD x86_64 Family"
    logical_cores = os.cpu_count()
    backend_value = backend.value if hasattr(backend, "value") else str(backend)

    lines = [
        "=" * 80,
        f"      {title}",
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
        f"CNN Layer Specs     : {specs}",
        "=" * 80,
    ]
    return "\n".join(lines)


def format_head_to_head_report(
    t_res: dict,
    c_res: dict,
    *,
    epochs: int,
    n_train: int,
    n_val: int,
    backend,
    kernel_label: str = "",
) -> str:
    backend_value = backend.value if hasattr(backend, "value") else str(backend)
    torch_col_header = f"PyTorch CNN ({t_res['threads_verified']}T)"
    custom_col_header = f"Custom [{backend_value}] ({c_res['threads_verified']}T)"
    torch_throughput = (n_train * t_res["epochs_completed"]) / t_res["train_time"]
    custom_throughput = (n_train * c_res["epochs_completed"]) / c_res["train_time"]
    ratio = c_res["train_time"] / t_res["train_time"] if t_res["train_time"] > 0 else float("inf")

    title = "HEAD-TO-HEAD BENCHMARK REPORT"
    if kernel_label:
        title = f"HEAD-TO-HEAD REPORT — {kernel_label}"

    lines = [
        "",
        "=" * 80,
        title.center(80),
        "=" * 80,
        f"{'Performance Metric':<32} | {torch_col_header:<20} | {custom_col_header:<20}",
        "-" * 80,
        f"{'Active Hardware Threads':<32} | {t_res['threads_verified']:<20d} | {c_res['threads_verified']:<20d}",
        f"{'Total Trainable Parameters':<32} | {t_res['params']:<20,d} | {c_res['params']:<20,d}",
        f"{'Target Epochs':<32} | {epochs:<20d} | {epochs:<20d}",
        f"{'Epochs Completed':<32} | {t_res['epochs_completed']:<20d} | {c_res['epochs_completed']:<20d}",
        f"{'Best Validation Epoch':<32} | {t_res['best_epoch']:<20d} | {c_res['best_epoch']:<20d}",
        f"{'Early Stopping Triggered':<32} | {str(t_res['early_stopped']):<20} | {str(c_res['early_stopped']):<20}",
        f"{'Forward Pass Count':<32} | {t_res['forward_counts']:<20,d} | {c_res['forward_counts']:<20,d}",
        f"{'Backward Pass Count':<32} | {t_res['backward_counts']:<20,d} | {c_res['backward_counts']:<20,d}",
        f"{'Final Training Loss':<32} | {t_res['train_loss']:<20.6f} | {c_res['train_loss']:<20.6f}",
        f"{'Final Validation Loss':<32} | {t_res['val_loss']:<20.6f} | {c_res['val_loss']:<20.6f}",
        f"{'Final Validation Accuracy':<32} | {t_res['val_acc'] * 100:>19.2f}% | {c_res['val_acc'] * 100:>19.2f}%",
        "-" * 80,
        f"{'Total Training Time':<32} | {t_res['train_time']:>18.3f} s | {c_res['train_time']:>18.3f} s",
        f"{'Training Throughput':<32} | {torch_throughput:>14.1f} smp/s | {custom_throughput:>14.1f} smp/s",
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


def print_head_to_head_report(*args, **kwargs) -> str:
    report = format_head_to_head_report(*args, **kwargs)
    print(report)
    return report


def run_cnn_benchmark(config_path: str = None):
    data_provider, cfg_dict = load_benchmark_data(config_path)
    
    data_path = cfg_dict["ingestion"]["data_file_path"]
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
    lr_scheduler_type = cfg_dict["optimization"].get("lr_scheduler", "none")

    specs = extract_layer_specs(cnn_dict)
    if cnn_dict:
        raw_shape = tuple(cnn_dict.get("input_shape", (1, 28, 28)))
        if len(raw_shape) == 2:
            bench_input_shape = (1, raw_shape[0], raw_shape[1])
        else:
            bench_input_shape = raw_shape
        validate_cnn_spatial_geometry(bench_input_shape, specs)

    print(format_system_banner(
        data_path=data_path,
        backend=backend,
        epochs=epochs,
        batch_size=batch_size,
        lr_init=lr_init,
        lam_l2=lam_l2,
        lam_l1=lam_l1,
        lr_scheduler_type=lr_scheduler_type,
        early_stopping_enabled=early_stopping_enabled,
        patience=patience,
        min_delta=min_delta,
        num_threads=num_threads,
        specs=specs,
    ))

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
        cnn_dict=cnn_dict,
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
    print_head_to_head_report(
        t_res, c_res,
        epochs=epochs,
        n_train=len(X_train),
        n_val=n_val_samples,
        backend=backend,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CNN convergence benchmark: custom engine vs PyTorch")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config (default: config/config.yaml)",
    )
    args = parser.parse_args()
    run_cnn_benchmark(config_path=args.config)