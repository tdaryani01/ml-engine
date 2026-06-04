# models/base_network.py
from abc import ABC, abstractmethod
import logging
import numpy as np

class BaseNeuralNetwork(ABC):
    def __init__(self, layer_sizes, optimizer_instance, lam_l1=0.01, lam_l2=0.01, p_dropout=0.0, use_batch_norm=True, bn_momentum=0.9, max_norm=5.0) -> None:
        self.optimizer = optimizer_instance  
        self.lam_l1 = lam_l1
        self.lam_l2 = lam_l2
        self.p_dropout = p_dropout
        self.layer_sizes = layer_sizes
        self.use_batch_norm = use_batch_norm
        self.bn_momentum = bn_momentum
        self.max_norm = max_norm
        self.eps = 1e-5
        
        self.weights = []
        self.biases = []
        
        # --- BATCH NORMALIZATION VECTORS ---
        self.gammas = []
        self.betas = []
        self.running_means = []
        self.running_vars = []
        
        self.diagnostic_counter = 0
        
        for i in range(len(layer_sizes) - 1):
            input_dim, output_dim = layer_sizes[i], layer_sizes[i+1]
            # scale = np.sqrt(2.0 / input_dim)
            scale = np.sqrt(2.0 / (input_dim))
            # self.weights.append(np.random.randn(input_dim, output_dim) * scale)
            limit = np.sqrt(6.0 / (input_dim + output_dim))
            self.weights.append(np.random.uniform(-limit, limit, (input_dim, output_dim)))
            self.biases.append(np.zeros((1, output_dim)))
            
            # Conditionally populate BN properties based on config flag
            if self.use_batch_norm and (i < len(layer_sizes) - 2):
                self.gammas.append(np.random.uniform(0.8, 1.2, size=(1, output_dim)))
                self.betas.append(np.random.uniform(-0.1, 0.1, size=(1, output_dim)))
    
                self.running_means.append(np.zeros((1, output_dim)))
                self.running_vars.append(np.ones((1, output_dim)))

    def preprocess_targets(self, y_train, y_val, y_test=None):
        return y_train, y_val, y_test

    def leaky_relu(self, z): 
        return np.where(z > 0, z, z * 0.01)
        
    def leaky_relu_der(self, z): 
        return np.where(z > 0, 1.0, 0.01)

    # =====================================================================
    # ISOLATED BATCH NORMALIZATION UTILITIES
    # =====================================================================
    def _bn_forward(self, z, layer_idx, training):
        if training:
            mean = np.mean(z, axis=0, keepdims=True)
            var = np.var(z, axis=0, keepdims=True)
            
            # Debug pre-normalized input variance anomalies
            logging.debug(f"[BN Layer {layer_idx}] PRE-NORM - z shape: {z.shape} | mean global: {np.mean(mean):.4f} | var global: {np.mean(var):.4f}")
            if np.any(var < 1e-8):
                logging.debug(f"[BN Layer {layer_idx}] WARNING: Near-zero variance detected in raw activations! Minimum variance column: {np.min(var):.4e}")
            
            z_hat = (z - mean) / np.sqrt(var + self.eps)
            z_bn = self.gammas[layer_idx] * z_hat + self.betas[layer_idx]
            
            self.running_means[layer_idx] = self.bn_momentum * self.running_means[layer_idx] + (1 - self.bn_momentum) * mean
            self.running_vars[layer_idx] = self.bn_momentum * self.running_vars[layer_idx] + (1 - self.bn_momentum) * var
            
            self.batch_means.append(mean)
            self.batch_vars.append(var)
            self.z_hats.append(z_hat)
            
            # Track if gammas/betas are blowing up or collapsing
            logging.debug(f"[BN Layer {layer_idx}] POST-NORM - z_bn mean: {np.mean(z_bn):.4f} | z_bn var: {np.var(z_bn):.4f}")
            logging.debug(f"[BN Layer {layer_idx}] PARAM METRICS - gammas range: [{np.min(self.gammas[layer_idx]):.4f}, {np.max(self.gammas[layer_idx]):.4f}] | betas range: [{np.min(self.betas[layer_idx]):.4f}, {np.max(self.betas[layer_idx]):.4f}]")
            
            if self.diagnostic_counter < 3:
                logging.debug(f"[TRACK FORWARD] Layer {layer_idx} | Mean range: [{np.min(mean):.4f}, {np.max(mean):.4f}] | Var range: [{np.min(var):.4f}, {np.max(var):.4f}]")
        else:
            z_hat = (z - self.running_means[layer_idx]) / np.sqrt(self.running_vars[layer_idx] + self.eps)
            z_bn = self.gammas[layer_idx] * z_hat + self.betas[layer_idx]
            
            # Monitor inference path scaling mismatch
            logging.debug(f"[BN Layer {layer_idx}] INFERENCE EVAL - running_mean global: {np.mean(self.running_means[layer_idx]):.4f} | running_var global: {np.mean(self.running_vars[layer_idx]):.4f}")

        return z_bn

    def _bn_backward(self, delta, layer_idx, m):
        """Computes balanced backpropagation gradients mapped to engine-scaled delta states."""
        z_hat = self.z_hats[layer_idx]
        var = self.batch_vars[layer_idx]
        
        dgamma = np.sum(delta * z_hat, axis=0, keepdims=True) / m
        dbeta = np.sum(delta, axis=0, keepdims=True) / m
        
        inv_std = 1.0 / np.sqrt(var + self.eps)
        
        dx = self.gammas[layer_idx] * inv_std * (
            delta - np.mean(delta, axis=0, keepdims=True) - z_hat * np.mean(delta * z_hat, axis=0, keepdims=True)
        )
        
        if self.diagnostic_counter < 3:
            logging.debug(f"[TRACK BACKWARD] Layer {layer_idx} | Incoming delta norm: {np.linalg.norm(delta):.4f} -> Outgoing dx norm: {np.linalg.norm(dx):.4f}")
            
        return dx, dgamma, dbeta

    # =====================================================================
    # CORE EXECUTION PIPELINE INTERFACES
    # =====================================================================
    def forward_hidden(self, X, training=True):
        self.zs = []            
        self.z_bns = []         
        self.z_hats = []        
        self.batch_means = []   
        self.batch_vars = []    
        self.activations = [X]
        self.masks = []
        
        current_act = X
        num_hidden_layers = len(self.weights) - 1
        
        # --- ENGINE TRACE: CAPTURE MATRIX ENTRANCE PROFILE ---
        if self.diagnostic_counter < 1:
            logging.info(f"[ENGINE TRACE] Hidden Input Shape: {X.shape} | Max: {np.max(X):.4f} | Min: {np.min(X):.4f} | Mean: {np.mean(X):.4f}")
        
        for i in range(num_hidden_layers):
            z = np.dot(current_act, self.weights[i]) + self.biases[i]
            self.zs.append(z)
            
            if self.use_batch_norm:
                z_layer = self._bn_forward(z, layer_idx=i, training=training)
                self.z_bns.append(z_layer)
            else:
                z_layer = z
            
            current_act = self.leaky_relu(z_layer)
            
            # --- ENGINE TRACE: HIDDEN ACTIVATION SATURATION ---
            if self.diagnostic_counter < 1:
                logging.info(f"[ENGINE TRACE] Layer {i} Pre-Activation (z) -> Max: {np.max(z):.4f} | Min: {np.min(z):.4f} | Mean: {np.mean(z):.4f}")
                logging.info(f"[ENGINE TRACE] Layer {i} Post-Activation (a) -> Max: {np.max(current_act):.4f} | Min: {np.min(current_act):.4f} | Mean: {np.mean(current_act):.4f}")
            
            if training and self.p_dropout > 0.0:
                mask = (np.random.rand(*current_act.shape) >= self.p_dropout) / (1.0 - self.p_dropout)
                current_act = current_act * mask
                self.masks.append(mask)
            else:
                self.masks.append(None)
                
            self.activations.append(current_act)
            
        z_out = np.dot(current_act, self.weights[-1]) + self.biases[-1]
        self.zs.append(z_out)
        return z_out

    def forward(self, X, training=True):
        z_out = self.forward_hidden(X, training=training)
        output = self.apply_output_activation(z_out)
        
        # --- ENGINE TRACE: OUTPUT HEAD COMPOSITION ---
        if self.diagnostic_counter < 1:
            logging.info(f"[ENGINE TRACE] Final Out Pre-Activation (z_out) Shape: {z_out.shape} | Max: {np.max(z_out):.4f} | Min: {np.min(z_out):.4f}")
            logging.info(f"[ENGINE TRACE] Final Out Activation Layer (output) Shape: {output.shape} | Max: {np.max(output):.4f} | Min: {np.min(output):.4f} | Mean: {np.mean(output):.4f}")
            
        self.activations.append(output)
        return output

    @abstractmethod
    def apply_output_activation(self, z_out): pass

    @abstractmethod
    def compute_output_delta(self, output, y): pass

    @abstractmethod
    def calculate_raw_cost(self, output, y): pass

    def compute_total_loss(self, output, y):
        raw_cost = self.calculate_raw_cost(output, y)
        m = y.shape[0]
        l2_penalty = (self.lam_l2 / (2 * m)) * sum(np.sum(w**2) for w in self.weights)
        l1_penalty = (self.lam_l1 / m) * sum(np.sum(np.abs(w)) for w in self.weights)
        return raw_cost + l2_penalty + l1_penalty

    def backward(self, X, y, active_lr):
        m = X.shape[0]
        num_layers = len(self.weights)
        grad_weights, grad_biases = [None] * num_layers, [None] * num_layers
        
        grad_gammas = [None] * (num_layers - 1)
        grad_betas = [None] * (num_layers - 1)
        
        # Explicitly clear activation memory arrays before building mini-batch states
        self.zs = []            
        self.z_bns = []         
        self.z_hats = []        
        self.batch_means = []   
        self.batch_vars = []    
        self.activations = [X]
        self.masks = []
        
        # Compute forward pass explicitly for THIS specific mini-batch
        output = self.forward(X, training=True)
        delta = self.compute_output_delta(output, y)
        
        # Step backward through the network layers
        for i in reversed(range(num_layers)):
            # 1. Compute gradients for the weights and biases of the current layer
            grad_weights[i] = np.dot(self.activations[i].T, delta) / m
            grad_biases[i] = np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0:
                # 2. Backpropagate delta through the weights to the layer below (i-1)
                delta = np.dot(delta, self.weights[i].T)
                
                # 3. Apply the activation derivative belonging strictly to layer i-1
                activation_z = self.z_bns[i-1] if self.use_batch_norm else self.zs[i-1]
                delta = delta * self.leaky_relu_der(activation_z)
                
                if self.masks[i-1] is not None:
                    delta = delta * self.masks[i-1]
                
                # 4. FIX: Apply BN backward using the exact index matching the layer (i-1)
                # Ensure the layer index actually possesses BN parameters (excluding output head)
                if self.use_batch_norm and (i - 1 < len(self.gammas)):
                    # Pass the correctly matched layer index (i-1)
                    delta, dgamma, dbeta = self._bn_backward(delta, layer_idx=i-1, m=m)
                    grad_gammas[i-1] = dgamma
                    grad_betas[i-1] = dbeta
                
        # Apply L1 weight regularization adjustments
        for i in range(num_layers):
            grad_weights[i] += (self.lam_l1 / m) * np.sign(self.weights[i])

        # =====================================================================
        # 🛡️ DEFENSIVE ENGINEERING: GLOBAL GRADIENT CLIPPING
        # =====================================================================
        # Calculate the square of the L2 norm across all weight gradients combined
        total_norm = np.sqrt(sum(np.sum(gw**2) for gw in grad_weights))
        
        if total_norm > self.max_norm:  # <-- Changed from max_norm to self.max_norm
            scaling_factor = self.max_norm / (total_norm + 1e-15)
            logging.debug(f"[Backward Stability] Exploding gradient threat detected! Total Norm: {total_norm:.4f} | Scaling down by factor: {scaling_factor:.4f}")
            for i in range(num_layers):
                grad_weights[i] *= scaling_factor
                if grad_biases[i] is not None:
                    grad_biases[i] *= scaling_factor

        # Ship aligned gradients to your Adam optimizer instance
        self.optimizer.update(
            weights=self.weights, biases=self.biases,
            grad_weights=grad_weights, grad_biases=grad_biases,
            m_samples=m, lam_l2=self.lam_l2, active_lr=active_lr,
            gammas=self.gammas if self.use_batch_norm else None,
            betas=self.betas if self.use_batch_norm else None,
            grad_gammas=grad_gammas if self.use_batch_norm else None,
            grad_betas=grad_betas if self.use_batch_norm else None
        )
        
        self.diagnostic_counter += 1
        
    def predict(self, raw_data, mean, std):
        return self.forward((raw_data - mean) / std, training=False)