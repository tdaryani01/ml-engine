# src/optimizers.py
from abc import ABC, abstractmethod
import numpy as np
import logging

class Optimizer(ABC):
    """Abstract base class defining the required interface for optimization algorithms."""
    @abstractmethod
    def setup(self, weights, biases, gammas=None, betas=None):
        """Initializes internal hyperparameter state tracking arrays."""
        pass

    @abstractmethod
    def update(self, weights, biases, grad_weights, grad_biases, m_samples, lam_l2, active_lr, 
               gammas=None, betas=None, grad_gammas=None, grad_betas=None):
        """Mutates parameter matrices in-place using the current epoch's learning rate."""
        pass

class AdamOptimizer(Optimizer):
    """Adam (Adaptive Moment Estimation) optimizer implementation with support for weight decay and batch normalization parameters."""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.initial_lr = lr  
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self._setup_done = False

    def setup(self, weights, biases, gammas=None, betas=None):
        """Allocates moment tracking arrays for weights, biases, and optional batch normalization parameters."""
        self.ms_w = [np.zeros_like(w) for w in weights]
        self.vs_w = [np.zeros_like(w) for w in weights]
        self.ms_b = [np.zeros_like(b) for b in biases]
        self.vs_b = [np.zeros_like(b) for b in biases]
        
        if gammas is not None:
            self.ms_g = [np.zeros_like(g) for g in gammas]
            self.vs_g = [np.zeros_like(g) for g in gammas]
        else:
            self.ms_g = None
            self.vs_g = None

        if betas is not None:
            self.ms_beta = [np.zeros_like(b) for b in betas]
            self.vs_beta = [np.zeros_like(b) for b in betas]
        else:
            self.ms_beta = None
            self.vs_beta = None
            
        self._setup_done = True

    def update(self, weights, biases, grad_weights, grad_biases, m_samples, lam_l2, active_lr, 
               gammas=None, betas=None, grad_gammas=None, grad_betas=None):
        """Performs an Adam optimization step, updating weights, biases, and optional batch normalization parameters."""
        if not self._setup_done:
            self.setup(weights, biases, gammas, betas)
            
        self.t += 1

        # Precompute scalar step multipliers once per step instead of per tensor
        bias_correction1 = 1.0 - (self.beta1 ** self.t)
        bias_correction2 = 1.0 - (self.beta2 ** self.t)
        step_scale = active_lr * (np.sqrt(bias_correction2) / bias_correction1)
        eps_corrected = self.eps * np.sqrt(bias_correction2)
        decay_factor = active_lr * (lam_l2 / float(m_samples)) if lam_l2 > 0.0 else 0.0

        one_minus_beta1 = 1.0 - self.beta1
        one_minus_beta2 = 1.0 - self.beta2
        
        # 1. Update Core Weights and Biases
        for i in range(len(weights)):
            gw = grad_weights[i]
            gb = grad_biases[i]

            # First moment updates
            self.ms_w[i] = self.beta1 * self.ms_w[i] + one_minus_beta1 * gw
            self.ms_b[i] = self.beta1 * self.ms_b[i] + one_minus_beta1 * gb
            
            # Second moment updates
            self.vs_w[i] = self.beta2 * self.vs_w[i] + one_minus_beta2 * (gw * gw)
            self.vs_b[i] = self.beta2 * self.vs_b[i] + one_minus_beta2 * (gb * gb)
            
            # In-place step application
            if decay_factor > 0.0:
                weights[i] -= decay_factor * weights[i]

            weights[i] -= step_scale * (self.ms_w[i] / (np.sqrt(self.vs_w[i]) + eps_corrected))
            biases[i] -= step_scale * (self.ms_b[i] / (np.sqrt(self.vs_b[i]) + eps_corrected))

        # 2. Update Batch Normalization Scaling Parameters via Adam
        if gammas is not None and grad_gammas is not None:
            for i in range(len(gammas)):
                gg = grad_gammas[i]
                if gg is not None:
                    self.ms_g[i] = self.beta1 * self.ms_g[i] + one_minus_beta1 * gg
                    self.vs_g[i] = self.beta2 * self.vs_g[i] + one_minus_beta2 * (gg * gg)
                    gammas[i] -= step_scale * (self.ms_g[i] / (np.sqrt(self.vs_g[i]) + eps_corrected))

        if betas is not None and grad_betas is not None:
            for i in range(len(betas)):
                gb_val = grad_betas[i]
                if gb_val is not None:
                    self.ms_beta[i] = self.beta1 * self.ms_beta[i] + one_minus_beta1 * gb_val
                    self.vs_beta[i] = self.beta2 * self.vs_beta[i] + one_minus_beta2 * (gb_val * gb_val)
                    betas[i] -= step_scale * (self.ms_beta[i] / (np.sqrt(self.vs_beta[i]) + eps_corrected))


class SGDOptimizer(Optimizer):
    """Stochastic Gradient Descent optimizer with momentum and weight decay support."""
    def __init__(self, lr=0.05, momentum=0.9):
        self.initial_lr = lr
        self.momentum = momentum
        self._setup_done = False

    def setup(self, weights, biases, gammas=None, betas=None):
        """Allocates velocity tracking buffers for SGD with momentum."""
        self.vs_w = [np.zeros_like(w) for w in weights]
        self.vs_b = [np.zeros_like(b) for b in biases]
        
        if gammas is not None:
            self.vs_g = [np.zeros_like(g) for g in gammas]
        else:
            self.vs_g = None

        if betas is not None:
            self.vs_beta = [np.zeros_like(b) for b in betas]
        else:
            self.vs_beta = None
            
        self._setup_done = True

    def update(self, weights, biases, grad_weights, grad_biases, m_samples, lam_l2, active_lr, 
               gammas=None, betas=None, grad_gammas=None, grad_betas=None):
        """Performs an SGD update step using momentum."""
        if not self._setup_done:
            self.setup(weights, biases, gammas, betas)
            
        decay_term = (lam_l2 / float(m_samples)) if lam_l2 > 0.0 else 0.0

        # 1. Update Weights and Biases
        for i in range(len(weights)):
            gw = grad_weights[i]
            if decay_term > 0.0:
                gw = gw + decay_term * weights[i]

            self.vs_w[i] = (self.momentum * self.vs_w[i]) + active_lr * gw
            self.vs_b[i] = (self.momentum * self.vs_b[i]) + active_lr * grad_biases[i]
            
            weights[i] -= self.vs_w[i]
            biases[i] -= self.vs_b[i]

        # 2. Update Batch Normalization Tracking vectors via classical momentum
        if gammas is not None and grad_gammas is not None:
            for i in range(len(gammas)):
                if grad_gammas[i] is not None:
                    self.vs_g[i] = (self.momentum * self.vs_g[i]) + active_lr * grad_gammas[i]
                    gammas[i] -= self.vs_g[i]

        if betas is not None and grad_betas is not None:
            for i in range(len(betas)):
                if grad_betas[i] is not None:
                    self.vs_beta[i] = (self.momentum * self.vs_beta[i]) + active_lr * grad_betas[i]
                    betas[i] -= self.vs_beta[i]