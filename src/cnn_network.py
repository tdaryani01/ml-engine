# src/cnn_network.py
import logging
from typing import Callable

import numpy as np
from config.constants import EngineBackend
from src.scratch_arena import ScratchArena
from src.spatial_layers import Conv2D, MaxPool2D, Flatten, ConvBlock
from src.training_cache import ForwardCache, new_forward_cache
from utils.engine_ops import create_engine_context


class CNNNetwork:
    """
    Modular Convolutional Neural Network engine.
    Automatically identifies and constructs fused ConvBlocks (Conv2D -> ReLU -> MaxPool2D)
    to minimize Python-to-C++ FFI dispatch transitions.
    """
    def __init__(self, conv_configs: list, dense_sizes: list, optimizer_instance,
                 backend: EngineBackend = EngineBackend.NATIVE,
                 engine_ctx=None,
                 lam_l1: float = 0.01, lam_l2: float = 0.01, p_dropout: float = 0.0,
                 max_norm: float = 5.0, task_type: str = "multiclass", **kwargs):
        self.engine_ctx = engine_ctx or create_engine_context(backend)
        self.backend = self.engine_ctx.backend
        self.scratch_arena = ScratchArena(self.backend)
        self.optimizer = optimizer_instance
        self.lam_l1 = lam_l1
        self.lam_l2 = lam_l2
        self.p_dropout = p_dropout
        self.max_norm = max_norm
        self.task_type = task_type
        self.diagnostic_counter = 0

        self.layers = []
        self.weights = []
        self.biases = []
        self.param_layers = []

        self._build_spatial_layers(conv_configs)
        self._build_dense_head(dense_sizes)

        logging.info("\n--- CONV LAYER SHAPES ---")
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ConvBlock):
                 logging.info(f"Layer {i} (ConvBlock): W={layer.W.shape}, conv_stride={layer.conv_stride}, pool_size={layer.pool_size}")
            elif isinstance(layer, Conv2D):
                 logging.info(f"Layer {i} (Conv2D): W={layer.W.shape}, stride={layer.stride}, pad={layer.pad}")
            elif isinstance(layer, MaxPool2D):
                 logging.info(f"Layer {i} (MaxPool2D): pool_size={layer.pool_size}, stride={layer.stride}")
        logging.info("-------------------------\n")

        self._dense_z_bufs = []
        self._dense_delta_bufs = []
        self._grad_weights_bufs = []
        self._grad_biases_bufs = []
        self._cached_batch_size = 0
        self._cached_dtype = None
        self._train_batch_cap = 0
        self._eval_batch_size = 0
        self._eval_cached_dtype = None
        self._eval_dense_z_bufs = []

        self._dense_w_indices = [i for i, l in enumerate(self.param_layers) if l == "dense"]
        self._layer_param_idx: dict[int, int] = {}
        for li, layer in enumerate(self.layers):
            if isinstance(layer, (ConvBlock, Conv2D)):
                self._layer_param_idx[li] = self.param_layers.index(layer)
        self._train_cache: ForwardCache | None = None
        self.contract_list_enabled = bool(kwargs.pop("contract_list_enabled", False))
        self._contract_runtime = None
        if self.contract_list_enabled:
            self._init_contract_path()

    def enable_contract_list(self) -> None:
        """Opt-in contract path after construction (e.g. from training engine at fit time)."""
        if self._contract_runtime is not None:
            return
        self.contract_list_enabled = True
        self._init_contract_path()

    def _init_contract_path(self) -> None:
        from src.contract import compile_cnn_training_step
        from src.contract_runtime import ContractRuntime
        from src.spatial_layers import ConvBlock

        if self.backend != EngineBackend.NATIVE:
            raise ValueError("Contract list requires NATIVE backend")
        if len(self._dense_w_indices) != 1:
            raise ValueError("Contract path requires a single dense head (no hidden dense layers)")
        for layer in self.layers:
            if not isinstance(layer, (ConvBlock, Flatten)) and layer != "relu":
                from src.spatial_layers import Conv2D, MaxPool2D
                if isinstance(layer, (Conv2D, MaxPool2D)):
                    raise ValueError("Contract path requires fused ConvBlock spatial stack")

        contract = compile_cnn_training_step(
            self.layers,
            layer_param_idx=self._layer_param_idx,
            dense_w_indices=self._dense_w_indices,
        )
        self._contract = contract
        self._contract_runtime = ContractRuntime(self, contract)
        logging.info("[CNN] Contract list enabled: %d ops", contract.op_count)

    
    def add_training_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        *,
        apply_adam: bool = False,
        step_token: int | None = None,
    ) -> str:
        """Manager handshake: OK if accepted, BUSY if native queue occupied."""
        if self._contract_runtime is None:
            raise RuntimeError("Contract path not initialized")
        if self._contract_runtime.try_submit_step(
            X, y, lr, apply_adam=apply_adam, step_token=step_token
        ):
            return "OK"
        return "BUSY"

    def contract_busy(self) -> bool:
        if self._contract_runtime is None:
            return False
        return self._contract_runtime.is_busy()

    def submit_contract_train_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        *,
        apply_adam: bool = False,
        step_token: int | None = None,
    ) -> str:
        return self.add_training_step(
            X, y, lr, apply_adam=apply_adam, step_token=step_token
        )

    def try_reap_contract_train_step(
        self,
    ) -> tuple[float, list, list, int] | None:
        if self._contract_runtime is None:
            return None
        return self._contract_runtime.try_reap_step()

    def run_contract_train_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        *,
        apply_adam: bool = False,
        step_token: int | None = None,
        tick_fn: Callable[[], None] | None = None,
    ) -> tuple[float, list, list, int]:
        if self._contract_runtime is None:
            raise RuntimeError("Contract path not initialized")
        return self._contract_runtime.run_step(
            X, y, lr, apply_adam=apply_adam, step_token=step_token, tick_fn=tick_fn
        )

    @property
    def layer_sizes(self) -> list:
        sizes = []
        for w in self.weights:
            if w.ndim == 4:
                sizes.append(w.shape[1] * w.shape[2] * w.shape[3])
            elif w.ndim == 2:
                sizes.append(w.shape[0])
        if len(self.weights) > 0:
            sizes.append(self.weights[-1].shape[-1] if self.weights[-1].ndim == 2 else self.weights[-1].shape[0])
        return sizes

    @property
    def num_classes(self) -> int:
        return self.weights[-1].shape[-1] if len(self.weights) > 0 else 0

    @property
    def output_dim(self) -> int:
        return self.num_classes

    def _build_spatial_layers(self, conv_configs: list):
        i = 0
        n = len(conv_configs)
        while i < n:
            cfg = conv_configs[i]
            l_type = cfg["type"].lower()

            if (l_type == "conv" and i + 2 < n and
                conv_configs[i + 1]["type"].lower() == "relu" and
                conv_configs[i + 2]["type"].lower() == "pool"):
                
                pool_cfg = conv_configs[i + 2]
                block = ConvBlock(
                    in_channels=cfg["in_channels"],
                    out_channels=cfg["out_channels"],
                    kernel_size=cfg.get("kernel_size", 3),
                    conv_stride=cfg.get("stride", 1),
                    conv_pad=cfg.get("pad", 0),
                    pool_size=pool_cfg.get("pool_size", 2),
                    pool_stride=pool_cfg.get("stride", 2),
                    engine_ctx=self.engine_ctx,
                )
                self.layers.append(block)
                self.param_layers.append(block)
                self.weights.append(block.W)
                self.biases.append(block.b)
                i += 3
                continue

            if l_type == "conv":
                layer = Conv2D(
                    in_channels=cfg["in_channels"],
                    out_channels=cfg["out_channels"],
                    kernel_size=cfg.get("kernel_size", 3),
                    stride=cfg.get("stride", 1),
                    pad=cfg.get("pad", 0),
                    engine_ctx=self.engine_ctx,
                )
                self.layers.append(layer)
                self.param_layers.append(layer)
                self.weights.append(layer.W)
                self.biases.append(layer.b)
            elif l_type == "pool":
                self.layers.append(MaxPool2D(
                    pool_size=cfg.get("pool_size", 2),
                    stride=cfg.get("stride", 2),
                    engine_ctx=self.engine_ctx,
                ))
            elif l_type == "flatten":
                self.layers.append(Flatten())
            elif l_type == "relu":
                self.layers.append("relu")
            i += 1

    def _build_dense_head(self, dense_sizes: list):
        for i in range(len(dense_sizes) - 1):
            fan_in, fan_out = dense_sizes[i], dense_sizes[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            W = np.random.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)
            b = np.zeros((1, fan_out), dtype=np.float32)
            self.weights.append(W)
            self.biases.append(b)
            self.param_layers.append("dense")

    def _sync_restored_weights(self) -> None:
        """Re-link ConvBlock/Conv2D params after early-stopping restores model.weights."""
        for w_idx, layer in enumerate(self.param_layers):
            if isinstance(layer, (ConvBlock, Conv2D)):
                layer.W = self.weights[w_idx]
                layer.b = self.biases[w_idx]

    def set_train_batch_cap(self, cap: int) -> None:
        self._train_batch_cap = int(cap)
        self.scratch_arena.set_train_batch_cap(cap)

    def _ensure_dense_buffers(self, m: int, dtype, inference: bool = False):
        if inference:
            if self._eval_cached_dtype == dtype and m <= self._eval_batch_size and self._eval_dense_z_bufs:
                return

            self._eval_batch_size = max(self._eval_batch_size, m)
            self._eval_cached_dtype = dtype
            self._eval_dense_z_bufs = []

            for w_idx in self._dense_w_indices:
                W = self.weights[w_idx]
                fan_in, fan_out = W.shape
                cap = self._eval_batch_size
                self._eval_dense_z_bufs.append(np.empty((cap, fan_out), dtype=dtype))
            return

        target_m = max(self._cached_batch_size, m)
        if self._train_batch_cap > 0:
            if self._cached_batch_size > self._train_batch_cap:
                target_m = self._train_batch_cap
            else:
                target_m = min(target_m, self._train_batch_cap)

        if self._cached_dtype == dtype and m <= self._cached_batch_size and self._cached_batch_size == target_m and self._dense_z_bufs:
            return

        self._cached_batch_size = target_m
        self._cached_dtype = dtype
        self._dense_z_bufs = []
        self._dense_delta_bufs = []
        self._grad_weights_bufs = []
        self._grad_biases_bufs = []

        for w_idx in self._dense_w_indices:
            W = self.weights[w_idx]
            fan_in, fan_out = W.shape
            cap = self._cached_batch_size
            self._dense_z_bufs.append(np.empty((cap, fan_out), dtype=dtype))
            self._dense_delta_bufs.append(np.empty((cap, fan_in), dtype=dtype))
            self._grad_weights_bufs.append(np.empty((fan_in, fan_out), dtype=dtype))
            self._grad_biases_bufs.append(np.empty((1, fan_out), dtype=dtype))

    def _forward(
        self,
        X: np.ndarray,
        training: bool = True,
        cache: ForwardCache | None = None,
    ) -> np.ndarray:
        arena = self.scratch_arena

        if training and cache is None:
            cache = ForwardCache()

        if cache is not None:
            cache.activations = [X]
            if len(cache.spatial_inputs) != len(self.layers):
                cache.spatial_inputs = [None] * len(self.layers)
                cache.spatial_logical_ws = [None] * len(self.layers)
            if len(cache.dense_inputs) != len(self._dense_w_indices):
                cache.dense_inputs = [None] * len(self._dense_w_indices)
                cache.masks = [None] * len(self._dense_w_indices)

        for i, layer in enumerate(self.param_layers):
            if isinstance(layer, (Conv2D, ConvBlock)):
                if layer.W is not self.weights[i]:
                    layer.W = self.weights[i]
                if layer.b is not self.biases[i]:
                    layer.b = self.biases[i]

        current_act = X
        current_logical_w = getattr(self, "input_logical_w", None)
        if current_logical_w is None:
            current_logical_w = 28 if (X.ndim == 4 and X.shape[3] == 32) else (X.shape[3] if X.ndim == 4 else None)

        for layer_idx, layer in enumerate(self.layers):
            if layer == "relu":
                if training and cache is not None:
                    cache.spatial_inputs[layer_idx] = current_act
                    cache.spatial_logical_ws[layer_idx] = current_logical_w
                current_act = self.engine_ctx.conv.relu_forward(current_act)
            elif isinstance(layer, Flatten):
                if training and cache is not None:
                    cache.spatial_inputs[layer_idx] = current_act
                    cache.spatial_logical_ws[layer_idx] = current_logical_w
                current_act = layer.forward(
                    current_act,
                    logical_w=current_logical_w,
                    cache=cache,
                    layer_idx=layer_idx,
                )
            elif isinstance(layer, (Conv2D, ConvBlock, MaxPool2D)):
                if training and cache is not None:
                    cache.spatial_inputs[layer_idx] = current_act
                    cache.spatial_logical_ws[layer_idx] = current_logical_w
                if isinstance(layer, ConvBlock):
                    current_act = layer.forward(
                        current_act,
                        W_logical=current_logical_w,
                        inference=not training,
                        cache=cache,
                        arena=arena,
                        layer_idx=layer_idx,
                    )
                elif isinstance(layer, Conv2D):
                    current_act = layer.forward(
                        current_act,
                        W_logical=current_logical_w,
                        cache=cache,
                        arena=arena,
                        layer_idx=layer_idx,
                    )
                else:
                    current_act = layer.forward(
                        current_act,
                        W_logical=current_logical_w,
                        cache=cache,
                        arena=arena,
                        layer_idx=layer_idx,
                    )
                current_logical_w = layer.out_w
            else:
                if training and cache is not None:
                    cache.spatial_inputs[layer_idx] = current_act
                    cache.spatial_logical_ws[layer_idx] = current_logical_w
                current_act = layer.forward(current_act)
            if cache is not None:
                cache.activations.append(current_act)

        m = current_act.shape[0]
        self._ensure_dense_buffers(m, current_act.dtype, inference=not training)
        num_dense = len(self._dense_w_indices)
        z_buf_source = self._eval_dense_z_bufs if not training else self._dense_z_bufs

        for idx, w_idx in enumerate(self._dense_w_indices):
            if training and cache is not None:
                cache.dense_inputs[idx] = current_act

            z_buf = z_buf_source[idx]
            W_mat = self.weights[w_idx]
            if W_mat.dtype != current_act.dtype:
                W_mat = W_mat.astype(current_act.dtype)
            b_vec = self.biases[w_idx]
            if b_vec.dtype != current_act.dtype:
                b_vec = b_vec.astype(current_act.dtype)

            active_z = z_buf[:m]
            np.dot(current_act, W_mat, out=active_z)
            active_z += b_vec

            if idx == num_dense - 1:
                current_act = self.apply_output_activation(active_z)
                if training and cache is not None:
                    cache.masks[idx] = None
            else:
                current_act = np.maximum(0.0, active_z)
                if training and self.p_dropout > 0.0:
                    mask = (np.random.rand(*current_act.shape) >= self.p_dropout).astype(current_act.dtype) / (1.0 - self.p_dropout)
                    current_act *= mask
                    if cache is not None:
                        cache.masks[idx] = mask
                elif training and cache is not None:
                    cache.masks[idx] = None

            if cache is not None:
                cache.activations.append(current_act)

        return current_act

    def forward_train(self, X: np.ndarray) -> tuple[np.ndarray, ForwardCache]:
        """One training forward; returns output and cache for explicit backward."""
        if self._train_cache is None:
            self._train_cache = new_forward_cache(len(self.layers), len(self._dense_w_indices))
        output = self._forward(X, training=True, cache=self._train_cache)
        return output, self._train_cache

    def apply_output_activation(self, z_out: np.ndarray) -> np.ndarray:
        if self.task_type == "multiclass":
            shift_z = z_out - np.max(z_out, axis=1, keepdims=True)
            exps = np.exp(shift_z)
            return exps / np.sum(exps, axis=1, keepdims=True)
        elif self.task_type == "binary":
            return 1.0 / (1.0 + np.exp(-np.clip(z_out, -500, 500)))
        return z_out

    def compute_output_delta(self, output: np.ndarray, y: np.ndarray) -> np.ndarray:
        return output - y

    def calculate_raw_cost(self, output: np.ndarray, y: np.ndarray) -> float:
        m = y.shape[0]
        eps = 1e-15
        if self.task_type == "multiclass":
            clipped = np.clip(output, eps, 1.0 - eps)
            return float(-np.sum(y * np.log(clipped)) / m)
        elif self.task_type == "binary":
            clipped = np.clip(output, eps, 1.0 - eps)
            return float(-np.sum(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)) / m)
        return float(np.sum((output - y) ** 2) / (2.0 * m))

    
    def compute_total_loss(self, output: np.ndarray, y: np.ndarray) -> float:
        raw_cost = self.calculate_raw_cost(output, y)
        m = y.shape[0]
        l2_sum = 0.0
        l1_sum = 0.0
        if self.lam_l2 > 0.0:
            for w in self.weights:
                l2_sum += float(np.sum(w * w))
        if self.lam_l1 > 0.0:
            for w in self.weights:
                l1_sum += float(np.sum(np.abs(w)))
        return raw_cost + (self.lam_l2 / (2.0 * m)) * l2_sum + (self.lam_l1 / m) * l1_sum

    def _compute_grads_from_cache(self, cache: ForwardCache, y: np.ndarray):
        """Compute gradients from a prior forward cache; does not update weights."""
        m = cache.batch_size
        inv_m = 1.0 / float(m)
        output = cache.output
        delta = self.compute_output_delta(output, y)
        arena = self.scratch_arena

        num_params = len(self.weights)
        grad_weights = [None] * num_params
        grad_biases = [None] * num_params

        num_dense = len(self._dense_w_indices)

        for local_idx in reversed(range(num_dense)):
            w_idx = self._dense_w_indices[local_idx]
            act_in = cache.dense_inputs[local_idx]

            gw_buf = self._grad_weights_bufs[local_idx]
            gb_buf = self._grad_biases_bufs[local_idx]

            W_mat = self.weights[w_idx]
            if W_mat.dtype != delta.dtype:
                W_mat = W_mat.astype(delta.dtype)

            np.dot(act_in.T, delta, out=gw_buf)
            gw_buf *= inv_m
            grad_weights[w_idx] = gw_buf

            np.sum(delta, axis=0, keepdims=True, out=gb_buf)
            gb_buf *= inv_m
            grad_biases[w_idx] = gb_buf

            next_delta_buf = self._dense_delta_bufs[local_idx][:m]
            np.dot(delta, W_mat.T, out=next_delta_buf)
            delta = next_delta_buf

            if local_idx > 0:
                if cache.masks[local_idx - 1] is not None:
                    delta *= cache.masks[local_idx - 1]
                delta *= (act_in > 0.0)

        spatial_grad = delta
        arena.zero_dx_buffers(m)
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            in_act = cache.spatial_inputs[i]

            if layer == "relu":
                spatial_grad = self.engine_ctx.conv.relu_backward(spatial_grad, in_act)
            elif isinstance(layer, ConvBlock):
                w_log = cache.spatial_logical_ws[i]
                spatial_grad = layer.backward(
                    spatial_grad,
                    W_logical=w_log,
                    cache=cache,
                    arena=arena,
                    layer_idx=i,
                )
                c_idx = self._layer_param_idx[i]
                grad_weights[c_idx] = layer.dW
                grad_biases[c_idx] = layer.db
            elif isinstance(layer, Conv2D):
                w_log = cache.spatial_logical_ws[i]
                spatial_grad = layer.backward(
                    spatial_grad,
                    W_logical=w_log,
                    cache=cache,
                    arena=arena,
                    layer_idx=i,
                )
                c_idx = self._layer_param_idx[i]
                grad_weights[c_idx] = layer.dW
                grad_biases[c_idx] = layer.db
            elif isinstance(layer, MaxPool2D):
                spatial_grad = layer.backward(
                    spatial_grad,
                    cache=cache,
                    arena=arena,
                    layer_idx=i,
                )
            elif isinstance(layer, Flatten):
                spatial_grad = layer.backward(spatial_grad, cache=cache, layer_idx=i)
            else:
                spatial_grad = layer.backward(spatial_grad)

        if self.lam_l1 > 0.0:
            scale = self.lam_l1 * inv_m
            for i in range(num_params):
                grad_weights[i] += scale * np.sign(self.weights[i])

        total_sq = 0.0
        for gw, gb in zip(grad_weights, grad_biases):
            if gw is not None:
                total_sq += float(np.sum(gw.astype(np.float64) ** 2))
            if gb is not None:
                total_sq += float(np.sum(gb.astype(np.float64) ** 2))
        total_norm = np.sqrt(total_sq)

        if total_norm > self.max_norm:
            scaling_factor = self.max_norm / (total_norm + 1e-15)
            for i in range(num_params):
                if grad_weights[i] is not None:
                    grad_weights[i] *= scaling_factor
                if grad_biases[i] is not None:
                    grad_biases[i] *= scaling_factor

        loss = self.compute_total_loss(output, y)
        return loss, grad_weights, grad_biases, m, None, None

    
    def _apply_grads(
        self,
        grad_weights,
        grad_biases,
        m_samples: int,
        active_lr: float,
        grad_gammas=None,
        grad_betas=None,
    ) -> None:
        self.optimizer.update(
            weights=self.weights, biases=self.biases,
            grad_weights=grad_weights, grad_biases=grad_biases,
            m_samples=m_samples, lam_l2=self.lam_l2, active_lr=active_lr,
            gammas=None, betas=None, grad_gammas=grad_gammas, grad_betas=grad_betas,
        )
        self.diagnostic_counter += 1

    def _backward_from_cache(self, cache: ForwardCache, y: np.ndarray, active_lr: float) -> float:
        """Backward pass using a prior forward cache (no re-forward)."""
        loss, gw, gb, m, _, _ = self._compute_grads_from_cache(cache, y)
        self._apply_grads(gw, gb, m, active_lr)
        return loss

    def backward(self, X: np.ndarray, y: np.ndarray, active_lr: float) -> float:
        """Training step: forward once, then backward from cache."""
        output, cache = self.forward_train(X)
        return self._backward_from_cache(cache, y, active_lr)

    def predict(self, processed_data: np.ndarray) -> np.ndarray:
        return self._forward(processed_data, training=False)
