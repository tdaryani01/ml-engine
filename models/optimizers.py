# models/optimizers.py
from abc import ABC, abstractmethod
import numpy as np
import logging

class Optimizer(ABC):
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
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.initial_lr = lr  # Kept as fallback or reference initial tracking bounds
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self._setup_done = False

    def setup(self, weights, biases, gammas=None, betas=None):
        # --- OPTIMIZER TRACE: CAPTURE STATE CREATION DEFAULTS ---
        logging.info(f"[OPTIMIZER TRACE] Running Adam setup allocation pass...")
        logging.info(f"[OPTIMIZER TRACE] Gammas parameter passed to setup: {type(gammas)} | Betas: {type(betas)}")
        
        self.ms_w = [np.zeros_like(w) for w in weights]
        self.vs_w = [np.zeros_like(w) for w in weights]
        self.ms_b = [np.zeros_like(b) for b in biases]
        self.vs_b = [np.zeros_like(b) for b in biases]
        
        # Setup tracking arrays for Batch Normalization if present
        if gammas is not None:
            logging.info(f"[OPTIMIZER TRACE] Allocating Adam Batch Normalization Gamma tracking vectors (Count: {len(gammas)})")
            self.ms_g = [np.zeros_like(g) for g in gammas]
            self.vs_g = [np.zeros_like(g) for g in gammas]
        if betas is not None:
            logging.info(f"[OPTIMIZER TRACE] Allocating Adam Batch Normalization Beta tracking vectors (Count: {len(betas)})")
            self.ms_beta = [np.zeros_like(b) for b in betas]
            self.vs_beta = [np.zeros_like(b) for b in betas]
            
        self._setup_done = True

    def update(self, weights, biases, grad_weights, grad_biases, m_samples, lam_l2, active_lr, 
               gammas=None, betas=None, grad_gammas=None, grad_betas=None):
        if not self._setup_done:
            self.setup(weights, biases, gammas, betas)
            
        self.t += 1
        
        # --- OPTIMIZER TRACE: MONITOR ENTRANCE METRICS ON FIRST STEP ---
        if self.t == 1:
            logging.info("=" * 60)
            logging.info("   OPTIMIZER STEP 1 CONSOLE TELEMETRY")
            logging.info("=" * 60)
            logging.info(f"[OPTIMIZER TRACE] Active Learning Rate: {active_lr:.6f} | L2 Regularization Parameter: {lam_l2}")
            logging.info(f"[OPTIMIZER TRACE] Incoming Core Weights Layers Count: {len(weights)}")
            logging.info(f"[OPTIMIZER TRACE] Incoming Gammas: {type(gammas)} | Grad Gammas: {type(grad_gammas)}")
            logging.info(f"[OPTIMIZER TRACE] Incoming Betas: {type(betas)} | Grad Betas: {type(grad_betas)}")
        
        # 1. Update Core Weights and Biases
        for i in range(len(weights)):
            # Update biased first moment estimate
            self.ms_w[i] = self.beta1 * self.ms_w[i] + (1 - self.beta1) * grad_weights[i]
            self.ms_b[i] = self.beta1 * self.ms_b[i] + (1 - self.beta1) * grad_biases[i]
            
            # Update biased second raw moment estimate
            self.vs_w[i] = self.beta2 * self.vs_w[i] + (1 - self.beta2) * (grad_weights[i] ** 2)
            self.vs_b[i] = self.beta2 * self.vs_b[i] + (1 - self.beta2) * (grad_biases[i] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_w_hat = self.ms_w[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.ms_b[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_w_hat = self.vs_w[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.vs_b[i] / (1 - self.beta2 ** self.t)
            
            # Compute the delta change component to see what adjustments are applied
            w_step = ((active_lr * m_w_hat) / (np.sqrt(v_w_hat) + self.eps)) + active_lr * ((lam_l2 / m_samples) * weights[i])
            b_step = (active_lr * m_b_hat) / (np.sqrt(v_b_hat) + self.eps)
            
            if self.t == 1:
                logging.info(f"[OPTIMIZER TRACE] Layer {i} Input Grad Weights Norm: {np.linalg.norm(grad_weights[i]):.6f} | Grad Biases Norm: {np.linalg.norm(grad_biases[i]):.6f}")
                logging.info(f"[OPTIMIZER TRACE] Layer {i} Computed Adam Step Weight Norm: {np.linalg.norm(w_step):.6f} | Step Bias Norm: {np.linalg.norm(b_step):.6f}")
            
            # Apply updates with L2 regularization step decoupled cleanly using the scheduled active learning rate
            weights[i] -= w_step
            biases[i] -= b_step

        # 2. Update Batch Normalization Scaling Parameters via Adam
        if gammas is not None and grad_gammas is not None:
            if self.t == 1:
                logging.info(f"[OPTIMIZER TRACE] Gamma block hit. Processing updates across {len(gammas)} vectors.")
            for i in range(len(gammas)):
                if grad_gammas[i] is not None:
                    self.ms_g[i] = self.beta1 * self.ms_g[i] + (1 - self.beta1) * grad_gammas[i]
                    self.vs_g[i] = self.beta2 * self.vs_g[i] + (1 - self.beta2) * (grad_gammas[i] ** 2)
                    
                    mg_hat = self.ms_g[i] / (1 - self.beta1 ** self.t)
                    vg_hat = self.vs_g[i] / (1 - self.beta2 ** self.t)
                    
                    g_step = (active_lr * mg_hat) / (np.sqrt(vg_hat) + self.eps)
                    if self.t == 1:
                        logging.info(f"[OPTIMIZER TRACE] Gamma Vector {i} Adam Step Norm: {np.linalg.norm(g_step):.6f}")
                    gammas[i] -= g_step
        elif self.t == 1:
            logging.info("[OPTIMIZER TRACE] Gamma block skipped. (Either gammas or grad_gammas is None)")

        if betas is not None and grad_betas is not None:
            if self.t == 1:
                logging.info(f"[OPTIMIZER TRACE] Beta block hit. Processing updates across {len(betas)} vectors.")
            for i in range(len(betas)):
                if grad_betas[i] is not None:
                    self.ms_beta[i] = self.beta1 * self.ms_beta[i] + (1 - self.beta1) * grad_betas[i]
                    self.vs_beta[i] = self.beta2 * self.vs_beta[i] + (1 - self.beta2) * (grad_betas[i] ** 2)
                    
                    mbeta_hat = self.ms_beta[i] / (1 - self.beta1 ** self.t)
                    vbeta_hat = self.vs_beta[i] / (1 - self.beta2 ** self.t)
                    
                    beta_step = (active_lr * mbeta_hat) / (np.sqrt(vbeta_hat) + self.eps)
                    if self.t == 1:
                        logging.info(f"[OPTIMIZER TRACE] Beta Vector {i} Adam Step Norm: {np.linalg.norm(beta_step):.6f}")
                    betas[i] -= beta_step
        elif self.t == 1:
            logging.info("[OPTIMIZER TRACE] Beta block skipped. (Either betas or grad_betas is None)")
            logging.info("=" * 60)


class SGDOptimizer(Optimizer):
    def __init__(self, lr=0.05, momentum=0.9):
        self.initial_lr = lr
        self.momentum = momentum
        self._setup_done = False

    def setup(self, weights, biases, gammas=None, betas=None):
        self.vs_w = [np.zeros_like(w) for w in weights]
        self.vs_b = [np.zeros_like(b) for b in biases]
        
        if gammas is not None:
            self.vs_g = [np.zeros_like(g) for g in gammas]
        if betas is not None:
            self.vs_beta = [np.zeros_like(b) for b in betas]
            
        self._setup_done = True

    def update(self, weights, biases, grad_weights, grad_biases, m_samples, lam_l2, active_lr, 
               gammas=None, betas=None, grad_gammas=None, grad_betas=None):
        if not self._setup_done:
            self.setup(weights, biases, gammas, betas)
            
        # 1. Update Weights and Biases
        for i in range(len(weights)):
            # Compute classical momentum vectors scaled dynamically with the scheduled active learning rate
            self.vs_w[i] = (self.momentum * self.vs_w[i]) + active_lr * (grad_weights[i] + (lam_l2 / m_samples) * weights[i])
            self.vs_b[i] = (self.momentum * self.vs_b[i]) + active_lr * grad_biases[i]
            
            # Update physical parameter coordinate spaces in-place
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