# src/cnn_network.py
import builtins
import logging
import numpy as np
from config.constants import EngineBackend
from src.spatial_layers import Conv2D, MaxPool2D, Flatten, ConvBlock
from utils.im2col import relu_spatial_forward, relu_spatial_backward

if 'profile' not in builtins.__dict__:
    builtins.__dict__['profile'] = lambda x: x


class CNNNetwork:
    """
    Modular Convolutional Neural Network engine.
    Automatically identifies and constructs fused ConvBlocks (Conv2D -> ReLU -> MaxPool2D)
    to minimize Python-to-C++ FFI dispatch transitions.
    """
    def __init__(self, conv_configs: list, dense_sizes: list, optimizer_instance,
                 backend: EngineBackend = EngineBackend.NATIVE,
                 lam_l1: float = 0.01, lam_l2: float = 0.01, p_dropout: float = 0.0,
                 max_norm: float = 5.0, task_type: str = "multiclass", **kwargs):
        self.backend = backend
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

        self.spatial_inputs = []
        self.dense_inputs = []
        self.masks = []
        self.activations = []

        # Zero-allocation buffer caches for dense head
        self._dense_z_bufs = []
        self._dense_delta_bufs = []
        self._grad_weights_bufs = []
        self._grad_biases_bufs = []
        self._cached_batch_size = 0
        self._cached_dtype = None

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

            # Pattern Match: [Conv2D -> ReLU -> MaxPool2D] -> Fuse into ConvBlock
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
                    backend=self.backend
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
                    backend=self.backend
                )
                self.layers.append(layer)
                self.param_layers.append(layer)
                self.weights.append(layer.W)
                self.biases.append(layer.b)
            elif l_type == "pool":
                self.layers.append(MaxPool2D(
                    pool_size=cfg.get("pool_size", 2),
                    stride=cfg.get("stride", 2),
                    backend=self.backend
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

    def _ensure_dense_buffers(self, m: int, dtype):
        if self._cached_batch_size == m and self._cached_dtype == dtype and self._dense_z_bufs:
            return

        self._cached_batch_size = m
        self._cached_dtype = dtype
        self._dense_z_bufs = []
        self._dense_delta_bufs = []
        self._grad_weights_bufs = []
        self._grad_biases_bufs = []

        dense_w_indices = [i for i, l in enumerate(self.param_layers) if l == "dense"]
        for w_idx in dense_w_indices:
            W = self.weights[w_idx]
            fan_in, fan_out = W.shape
            self._dense_z_bufs.append(np.empty((m, fan_out), dtype=dtype))
            self._dense_delta_bufs.append(np.empty((m, fan_in), dtype=dtype))
            self._grad_weights_bufs.append(np.empty((fan_in, fan_out), dtype=dtype))
            self._grad_biases_bufs.append(np.empty((1, fan_out), dtype=dtype))

    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self.spatial_inputs.clear()
        self.dense_inputs.clear()
        self.masks.clear()
        self.activations = [X]

        # Sync spatial parameter pointers
        for i, layer in enumerate(self.param_layers):
            if isinstance(layer, (Conv2D, ConvBlock)):
                if layer.W is not self.weights[i]:
                    layer.W = self.weights[i]
                if layer.b is not self.biases[i]:
                    layer.b = self.biases[i]

        current_act = X

        # 1. Spatial Forward Pass (Tracking logical width across layers)
        current_logical_w = getattr(self, "input_logical_w", None)
        if current_logical_w is None:
            # If input is (N, C, H, W_stride), derive logical width
            # Standard MNIST/benchmark logical width is 28 if stride-padded to 32
            current_logical_w = 28 if (X.ndim == 4 and X.shape[3] == 32) else (X.shape[3] if X.ndim == 4 else None)

        for layer in self.layers:
            if layer == "relu":
                if training:
                    self.spatial_inputs.append(current_act)
                current_act = relu_spatial_forward(current_act)
            elif isinstance(layer, Flatten):
                if training:
                    self.spatial_inputs.append(current_act)
                current_act = layer.forward(current_act, logical_w=current_logical_w)
            elif isinstance(layer, (Conv2D, ConvBlock, MaxPool2D)):
                if training:
                    self.spatial_inputs.append(current_act)
                current_act = layer.forward(current_act, W_logical=current_logical_w)
                current_logical_w = layer.out_w
            else:
                if training:
                    self.spatial_inputs.append(current_act)
                current_act = layer.forward(current_act)
            self.activations.append(current_act)

        # 2. Dense Forward Pass
        m = current_act.shape[0]
        self._ensure_dense_buffers(m, current_act.dtype)
        dense_w_indices = [i for i, l in enumerate(self.param_layers) if l == "dense"]
        num_dense = len(dense_w_indices)

        for idx, w_idx in enumerate(dense_w_indices):
            if training:
                self.dense_inputs.append(current_act)
            
            z_buf = self._dense_z_bufs[idx]
            W_mat = self.weights[w_idx]
            if W_mat.dtype != current_act.dtype:
                W_mat = W_mat.astype(current_act.dtype)
            b_vec = self.biases[w_idx]
            if b_vec.dtype != current_act.dtype:
                b_vec = b_vec.astype(current_act.dtype)

            np.dot(current_act, W_mat, out=z_buf)
            z_buf += b_vec

            if idx == num_dense - 1:
                current_act = self.apply_output_activation(z_buf)
                if training:
                    self.masks.append(None)
            else:
                current_act = np.maximum(0.0, z_buf)
                if training and self.p_dropout > 0.0:
                    mask = (np.random.rand(*current_act.shape) >= self.p_dropout).astype(current_act.dtype) / (1.0 - self.p_dropout)
                    current_act *= mask
                    self.masks.append(mask)
                elif training:
                    self.masks.append(None)

            self.activations.append(current_act)

        return current_act

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

    
    def backward(self, X: np.ndarray, y: np.ndarray, active_lr: float) -> float:
        m = X.shape[0]
        inv_m = 1.0 / float(m)
        output = self._forward(X, training=True)
        delta = self.compute_output_delta(output, y)

        num_params = len(self.weights)
        grad_weights = [None] * num_params
        grad_biases = [None] * num_params

        dense_w_indices = [i for i, l in enumerate(self.param_layers) if l == "dense"]
        num_dense = len(dense_w_indices)

        # 1. Backprop Dense Head
        for local_idx in reversed(range(num_dense)):
            w_idx = dense_w_indices[local_idx]
            act_in = self.dense_inputs[local_idx]

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

            next_delta_buf = self._dense_delta_bufs[local_idx]
            np.dot(delta, W_mat.T, out=next_delta_buf)
            delta = next_delta_buf

            if local_idx > 0:
                if self.masks[local_idx - 1] is not None:
                    delta *= self.masks[local_idx - 1]
                delta *= (act_in > 0.0)

        # 2. Backprop Spatial Pipeline
        spatial_grad = delta
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            in_act = self.spatial_inputs[i]

            if layer == "relu":
                spatial_grad = relu_spatial_backward(spatial_grad, in_act)
            else:
                spatial_grad = layer.backward(spatial_grad)
                if isinstance(layer, (Conv2D, ConvBlock)):
                    c_idx = self.param_layers.index(layer)
                    grad_weights[c_idx] = layer.dW
                    grad_biases[c_idx] = layer.db

        # 3. L1 Penalty
        if self.lam_l1 > 0.0:
            scale = self.lam_l1 * inv_m
            for i in range(num_params):
                grad_weights[i] += scale * np.sign(self.weights[i])

        # 4. Fast Gradient Clipping
        total_sq = 0.0
        for gw in grad_weights:
            total_sq += float(np.sum(gw * gw))
        total_norm = np.sqrt(total_sq)

        if total_norm > self.max_norm:
            scaling_factor = self.max_norm / (total_norm + 1e-15)
            for i in range(num_params):
                grad_weights[i] *= scaling_factor
                grad_biases[i] *= scaling_factor

        # 5. Optimizer Update Step
        self.optimizer.update(
            weights=self.weights, biases=self.biases,
            grad_weights=grad_weights, grad_biases=grad_biases,
            m_samples=m, lam_l2=self.lam_l2, active_lr=active_lr,
            gammas=None, betas=None, grad_gammas=None, grad_betas=None
        )

        self.diagnostic_counter += 1
        return self.compute_total_loss(output, y)

    def predict(self, processed_data: np.ndarray) -> np.ndarray:
        return self._forward(processed_data, training=False)