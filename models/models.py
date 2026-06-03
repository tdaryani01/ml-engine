# models/models.py
import numpy as np
import logging
from models.base_network import BaseNeuralNetwork

class BinaryClassificationNetwork(BaseNeuralNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.y_mean = None
        self.y_std = None
    def preprocess_targets(self, y_train, y_val, y_test=None):
        """FIXED: Enforces explicit 2D column vector shapes to prevent implicit numpy broadcasting corruption."""
        logging.info("[Model Engine: Binary] Enforcing column shape vector transformations on targets...")
        
        y_train_fixed = y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train
        y_val_fixed = y_val.reshape(-1, 1) if len(y_val.shape) == 1 else y_val
        y_test_fixed = (y_test.reshape(-1, 1) if len(y_test.shape) == 1 else y_test) if y_test is not None else None
        
        return y_train_fixed, y_val_fixed, y_test_fixed

    def apply_output_activation(self, z_out):
        return 1.0 / (1.0 + np.exp(-np.clip(z_out, -500, 500)))

    def compute_output_delta(self, output, y):
        # --- MODEL PASS TRACE: INTERCEPT DELTA VALUES ---
        if self.diagnostic_counter < 1:
            logging.info(f"[MODEL PASS] Error Delta Math -> Output Shape: {output.shape} | Target Shape: {y.shape}")
            logging.info(f"[MODEL PASS] Error Delta Array Bounds -> Max: {np.max(output - y):.4f} | Min: {np.min(output - y):.4f} | Mean: {np.mean(output - y):.4f}")
        return output - y

    def calculate_raw_cost(self, output, y):
        m = y.shape[0]
        eps = 1e-15
        output = np.clip(output, eps, 1.0 - eps)
        
        # Continuous tracking logs to monitor sigmoid saturation thresholds
        logging.debug(f"[COST DIAGNOSTIC: Binary] Output Range: [{np.min(output):.4f}, {np.max(output):.4f}] | Target Mean: {np.mean(y):.4f}")
        
        # --- MODEL PASS TRACE: BREAK DOWN ENTROPY COMPONENTS ---
        if self.diagnostic_counter < 2:
            pos_loss_component = y * np.log(output)
            neg_loss_component = (1.0 - y) * np.log(1.0 - output)
            logging.info(f"[MODEL PASS] Cost Math -> Total Samples (m): {m}")
            logging.info(f"[MODEL PASS] Binary Cross-Entropy Breakdown -> Positive Target Log Sum: {np.sum(pos_loss_component):.4f} | Negative Target Log Sum: {np.sum(neg_loss_component):.4f}")
        
        return -np.sum(y * np.log(output) + (1.0 - y) * np.log(1.0 - output)) / m


class RegressionNetwork(BaseNeuralNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.y_mean = None
        self.y_std = None

    def preprocess_targets(self, y_train, y_val, y_test=None):
        print("[Model Engine: Regression] Normalizing continuous target arrays...")
        self.y_mean = np.mean(y_train, axis=0)
        self.y_std = np.std(y_train, axis=0) + 1e-24
        
        y_train_norm = (y_train - self.y_mean) / self.y_std
        y_val_norm = (y_val - self.y_mean) / self.y_std
        y_test_norm = (y_test - self.y_mean) / self.y_std if y_test is not None else None
        
        return y_train_norm, y_val_norm, y_test_norm

    def apply_output_activation(self, z_out):
        return z_out  # Linear activation head

    def compute_output_delta(self, output, y):
        return output - y

    def calculate_raw_cost(self, output, y):
        m = y.shape[0]
        return np.sum((output - y) ** 2) / (2 * m)


class MultiClassNetwork(BaseNeuralNetwork):
    def preprocess_targets(self, y_train, y_val, y_test=None):
        print("[Model Engine: Multi-Class] Transmuting discrete class vector states to one-hot matrices...")
        
        # Combine matrices safely to discover true global distinct class footprint boundaries
        all_blocks = [y_train, y_val]
        if y_test is not None:
            all_blocks.append(y_test)
        global_y = np.vstack(all_blocks)
        num_classes = len(np.unique(global_y))
        
        def to_one_hot(y_vec, classes):
            encoded = np.zeros((len(y_vec), classes))
            encoded[np.arange(len(y_vec)), y_vec.ravel().astype(int)] = 1
            return encoded
            
        return to_one_hot(y_train, num_classes), to_one_hot(y_val, num_classes), (to_one_hot(y_test, num_classes) if y_test is not None else None)

    def apply_output_activation(self, z_out):
        shift_z = z_out - np.max(z_out, axis=1, keepdims=True)
        exps = np.exp(shift_z)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def compute_output_delta(self, output, y):
        return output - y

    def calculate_raw_cost(self, output, y):
        m = y.shape[0]
        eps = 1e-15
        output = np.clip(output, eps, 1.0 - eps)
        return -np.sum(y * np.log(output)) / m