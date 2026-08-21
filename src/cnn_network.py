# src/cnn_network.py
import builtins
import logging
import numpy as np
from src.spatial_layers import Conv2D, MaxPool2D, Flatten
from utils.im2col import relu_spatial_forward, relu_spatial_backward

if 'profile' not in builtins.__dict__:
    builtins.__dict__['profile'] = lambda x: x


class CNNNetwork:
    """
    Modular Convolutional Neural Network engine.
    Composes spatial layers with a dense classification/regression head while
    exposing standard interfaces for ModelController, Optimizers, and Diagnostics.
    """
    def __init__(self, conv_configs: list, dense_sizes: list, optimizer_instance,
                 lam_l1: float = 0.01, lam_l2: float = 0.01, p_dropout: float = 0.0,
                 max_norm: float = 5.0, task_type: str = "multiclass"):
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

        self.spatial_inputs = []
        self.dense_inputs = []
        self.masks = []
        self.activations = []

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
        for cfg in conv_configs:
            l_type = cfg["type"].lower()
            if l_type == "conv":
                layer = Conv2D(
                    in_channels=cfg["in_channels"],
                    out_channels=cfg["out_channels"],
                    kernel_size=cfg.get("kernel_size", 3),
                    stride=cfg.get("stride", 1),
                    pad=cfg.get("pad", 0)
                )
                self.layers.append(layer)
                self.param_layers.append(layer)
                self.weights.append(layer.W)
                self.biases.append(layer.b)
            elif l_type == "pool":
                self.layers.append(MaxPool2D(
                    pool_size=cfg.get("pool_size", 2),
                    stride=cfg.get("stride", 2)
                ))
            elif l_type == "flatten":
                self.layers.append(Flatten())
            elif l_type == "relu":
                self.layers.append("relu")

    def _build_dense_head(self, dense_sizes: list):
        for i in range(len(dense_sizes) - 1):
            fan_in, fan_out = dense_sizes[i], dense_sizes[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            W = np.random.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)
            b = np.zeros((1, fan_out), dtype=np.float32)
            self.weights.append(W)
            self.biases.append(b)
            self.param_layers.append("dense")

    @profile
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self.spatial_inputs.clear()
        self.dense_inputs.clear()
        self.masks.clear()
        self.activations = [X]

        # Sync spatial parameter pointers if model.weights/model.biases were re-assigned externally
        for i, layer in enumerate(self.param_layers):
            if isinstance(layer, Conv2D):
                if layer.W is not self.weights[i]:
                    layer.W = self.weights[i]
                if layer.b is not self.biases[i]:
                    layer.b = self.biases[i]

        current_act = X

        # 1. Spatial Forward Pass
        for layer in self.layers:
            if training:
                self.spatial_inputs.append(current_act)
            if layer == "relu":
                current_act = relu_spatial_forward(current_act.copy() if training else current_act)
            else:
                current_act = layer.forward(current_act)
            self.activations.append(current_act)

        # 2. Dense Forward Pass
        dense_w_indices = [i for i, l in enumerate(self.param_layers) if l == "dense"]
        num_dense = len(dense_w_indices)

        for idx, w_idx in enumerate(dense_w_indices):
            if training:
                self.dense_inputs.append(current_act)
            z = np.dot(current_act, self.weights[w_idx]) + self.biases[w_idx]

            if idx == num_dense - 1:
                current_act = self.apply_output_activation(z)
                if training:
                    self.masks.append(None)
            else:
                current_act = np.maximum(0, z)
                if training and self.p_dropout > 0.0:
                    mask = (np.random.rand(*current_act.shape) >= self.p_dropout).astype(current_act.dtype) / (1.0 - self.p_dropout)
                    current_act = current_act * mask
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
            output = np.clip(output, eps, 1.0 - eps)
            return float(-np.sum(y * np.log(output)) / m)
        elif self.task_type == "binary":
            output = np.clip(output, eps, 1.0 - eps)
            return float(-np.sum(y * np.log(output) + (1.0 - y) * np.log(1.0 - output)) / m)
        return float(np.sum((output - y) ** 2) / (2 * m))

    def compute_total_loss(self, output: np.ndarray, y: np.ndarray) -> float:
        raw_cost = self.calculate_raw_cost(output, y)
        m = y.shape[0]
        l2_penalty = (self.lam_l2 / (2 * m)) * sum(np.sum(w**2) for w in self.weights)
        l1_penalty = (self.lam_l1 / m) * sum(np.sum(np.abs(w)) for w in self.weights)
        return raw_cost + l2_penalty + l1_penalty

    @profile
    def backward(self, X: np.ndarray, y: np.ndarray, active_lr: float) -> float:
        m = X.shape[0]
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

            grad_weights[w_idx] = np.dot(act_in.T, delta) / m
            grad_biases[w_idx] = np.sum(delta, axis=0, keepdims=True) / m

            delta = np.dot(delta, self.weights[w_idx].T)

            if local_idx > 0:
                if self.masks[local_idx - 1] is not None:
                    delta *= self.masks[local_idx - 1]
                delta *= (act_in > 0)

        # 2. Backprop Spatial Pipeline
        spatial_grad = delta
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            in_act = self.spatial_inputs[i]

            if layer == "relu":
                spatial_grad = relu_spatial_backward(spatial_grad, in_act)
            else:
                spatial_grad = layer.backward(spatial_grad)
                if isinstance(layer, Conv2D):
                    c_idx = self.param_layers.index(layer)
                    grad_weights[c_idx] = layer.dW
                    grad_biases[c_idx] = layer.db

        # 3. L1 Penalty
        if self.lam_l1 > 0.0:
            scale = self.lam_l1 / m
            for i in range(num_params):
                grad_weights[i] += scale * np.sign(self.weights[i])

        # 4. Gradient Clipping
        total_norm = np.sqrt(sum(np.sum(gw**2) for gw in grad_weights))
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