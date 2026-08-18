import sys
import os
import numpy as np

# Ensure project root is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_factory import ModelFactory


class MockOptimizer:
    """Intercepts gradients from backward() and simulates optimizer weight decay."""
    def __init__(self):
        self.grad_weights = None
        self.grad_biases = None
        self.grad_gammas = None
        self.grad_betas = None

    def update(self, weights, biases, grad_weights, grad_biases, m_samples=1, lam_l2=0.0, grad_gammas=None, grad_betas=None, **kwargs):
        if lam_l2 > 0.0:
            for i in range(len(grad_weights)):
                grad_weights[i] += (lam_l2 / m_samples) * weights[i]
                
        self.grad_weights = grad_weights
        self.grad_biases = grad_biases
        self.grad_gammas = grad_gammas
        self.grad_betas = grad_betas


class GradientChecker:
    def __init__(self, model, epsilon: float = 1e-7, tolerance: float = 1e-5):
        self.model = model
        self.eps = epsilon
        self.tol = tolerance

    @staticmethod
    def _compute_relative_error(grad_analytic: float, grad_numeric: float) -> float:
        abs_diff = abs(grad_analytic - grad_numeric)
        # Absolute tolerance threshold for values mathematically approaching zero
        if abs(grad_analytic) < 1e-7 and abs(grad_numeric) < 1e-7:
            return abs_diff
        denominator = max(abs(grad_analytic) + abs(grad_numeric), 1e-12)
        return abs_diff / denominator

    def check_tensor(self, param_list: list, grad_list: list, tensor_name: str, X: np.ndarray, y: np.ndarray, fixed_seed: int = 42) -> bool:
        passed = True
        max_error = 0.0
        worst_coord = None
        worst_analytic = 0.0
        worst_numeric = 0.0

        if param_list is None or grad_list is None:
            return True

        for layer_idx, (tensor, grad_tensor) in enumerate(zip(param_list, grad_list)):
            it = np.nditer(tensor, flags=["multi_index"], op_flags=["readwrite"])
            while not it.finished:
                coord = it.multi_index
                original_value = float(tensor[coord])

                # 1. Evaluate J(theta + eps) with frozen RNG seed for identical dropout masks
                tensor[coord] = original_value + self.eps
                np.random.seed(fixed_seed)
                out_pos = self.model._forward(X, training=True)
                j_plus = self.model.compute_total_loss(out_pos, y)

                # 2. Evaluate J(theta - eps) with frozen RNG seed for identical dropout masks
                tensor[coord] = original_value - self.eps
                np.random.seed(fixed_seed)
                out_neg = self.model._forward(X, training=True)
                j_minus = self.model.compute_total_loss(out_neg, y)

                # 3. Restore original parameter
                tensor[coord] = original_value

                # 4. Compute finite difference gradient
                g_num = (j_plus - j_minus) / (2.0 * self.eps)
                g_ana = float(grad_tensor[coord])

                # 5. Measure relative error
                rel_error = self._compute_relative_error(g_ana, g_num)

                if rel_error > max_error:
                    max_error = rel_error
                    worst_coord = (layer_idx, coord)
                    worst_analytic = g_ana
                    worst_numeric = g_num

                if rel_error > self.tol:
                    passed = False

                it.iternext()

        status_tag = "PASSED" if passed else "FAILED"
        print(f"[{status_tag}] {tensor_name:<12} | Max Rel Error: {max_error:.2e} (Threshold: {self.tol:.0e})")
        
        if not passed and worst_coord:
            print(f"  └── Failure Coordinate: Layer {worst_coord[0]} Index {worst_coord[1]}")
            print(f"      • Analytical Grad : {worst_analytic:+.8e}")
            print(f"      • Numerical Grad  : {worst_numeric:+.8e}")
            print(f"      • Absolute Delta  : {abs(worst_analytic - worst_numeric):.8e}")

        return passed


def run_gradient_check(task_type: str, use_batch_norm: bool, p_dropout: float) -> int:
    seed_val = 42
    np.random.seed(seed_val)
    
    print("\n" + "=" * 70)
    print(f" RUNNING CHECK: Task='{task_type}' | BN={use_batch_norm} | Dropout={p_dropout}")
    print("=" * 70)

    # 1. Deterministic Micro-Topology tailored to the task head
    output_dim = 1 if task_type == "binary_classification" else 3
    layer_dims = (4, 6, 4, output_dim)
    batch_size = 4
    n_features = layer_dims[0]

    # 2. Generate double-precision Mock Data based on the task type
    X_mock = np.random.randn(batch_size, n_features).astype(np.float64)
    
    if task_type == "multi_class":
        raw_labels = np.random.randint(0, output_dim, size=batch_size)
        y_mock = np.zeros((batch_size, output_dim), dtype=np.float64)
        for i, label in enumerate(raw_labels):
            y_mock[i, label] = 1.0
    elif task_type == "binary_classification":
        y_mock = np.random.randint(0, 2, size=(batch_size, 1)).astype(np.float64)
    elif task_type == "regression":
        y_mock = np.random.randn(batch_size, output_dim).astype(np.float64)

    # 3. Instantiate concrete model via factory
    model = ModelFactory.create_model(
        model_type=task_type,
        layer_sizes=layer_dims,
        optimizer="sgd",
        lr=0.01,
        lam_l1=0.01,
        lam_l2=0.01,
        p_dropout=p_dropout,
        use_batch_norm=use_batch_norm,
        max_norm=1000.0
    )

    # Cast initial tensors to float64 precision
    model.weights = [w.astype(np.float64) for w in model.weights]
    model.biases = [b.astype(np.float64) for b in model.biases]
    if model.use_batch_norm:
        model.gammas = [g.astype(np.float64) for g in model.gammas]
        model.betas = [b.astype(np.float64) for b in model.betas]

    # 4. Intercept Analytical Gradients
    mock_opt = MockOptimizer()
    model.optimizer = mock_opt
    
    # Freeze the random seed immediately before the analytical pass to anchor the dropout masks
    np.random.seed(seed_val)
    model.backward(X_mock, y_mock, active_lr=0.01)
    
    grad_weights = mock_opt.grad_weights
    grad_biases = mock_opt.grad_biases

    # 5. Execute Numerical Verification
    checker = GradientChecker(model=model, epsilon=1e-7, tolerance=1e-5)
    
    passed_all = True
    passed_all &= checker.check_tensor(model.weights, grad_weights, "Weights (W)", X_mock, y_mock, seed_val)
    passed_all &= checker.check_tensor(model.biases, grad_biases, "Biases (b)", X_mock, y_mock, seed_val)
    
    if use_batch_norm:
        passed_all &= checker.check_tensor(model.gammas, mock_opt.grad_gammas, "Gammas (γ)", X_mock, y_mock, seed_val)
        passed_all &= checker.check_tensor(model.betas, mock_opt.grad_betas, "Betas (β)", X_mock, y_mock, seed_val)

    print("-" * 70)
    if passed_all:
        print(f"[SUCCESS] Test passed for {task_type}.")
        return 0
    else:
        print(f"[ERROR] Discrepancy detected in {task_type}.")
        return 1


def run_all_checks():
    # Matrix of Edge Cases to satisfy framework-grade requirements
    test_cases = [
        {"task": "multi_class", "bn": False, "p_dropout": 0.0},           # Base core engine
        {"task": "multi_class", "bn": True, "p_dropout": 0.0},            # Batch Norm isolated
        {"task": "binary_classification", "bn": False, "p_dropout": 0.5}, # Binary BCE + Dropout freeze
        {"task": "regression", "bn": True, "p_dropout": 0.2},             # Regression MSE + Combined features
    ]
    
    overall_status = 0
    for case in test_cases:
        exit_code = run_gradient_check(case["task"], case["bn"], case["p_dropout"])
        if exit_code != 0:
            overall_status = 1
            
    print("\n" + "=" * 70)
    if overall_status == 0:
        print("[GLOBAL SUCCESS] Entire Engine Mathematically Verified Across All Tasks!")
        return 0
    else:
        print("[GLOBAL ERROR] One or more gradient checks failed.")
        return 1


# Test fixture for pytest (CI/CD integration)
def test_autodiff_gradients():
    assert run_all_checks() == 0, "Analytical gradients do not match numerical finite differences."


if __name__ == "__main__":
    sys.exit(run_all_checks())