# testing/test_gradient_check.py
import sys
import os
import numpy as np

# Ensure project root is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_factory import ModelFactory
from config.constants import EngineBackend
from utils.im2col import conv2d_forward, conv2d_backward_fused, init_engine_backend


CONV_DX_KERNEL_PAD_CASES = [
    (3, 1),
    (6, 1),
]


def _relative_grad_error(analytic: float, numeric: float) -> float:
    abs_diff = abs(analytic - numeric)
    if abs(analytic) < 1e-7 and abs(numeric) < 1e-7:
        return abs_diff
    return abs_diff / max(abs(analytic) + abs(numeric), 1e-12)


def check_conv2d_input_gradient(
    backend: EngineBackend,
    kernel: int,
    pad: int,
    *,
    fuse_relu: bool = False,
    epsilon: float = 1e-4,
    tolerance: float = 8e-2,
) -> bool:
    """
    Finite-difference check for conv2d dX (input gradient).
    Catches native fallback dX regressions that weight-only CNN grad checks miss.
    """
    init_engine_backend(backend)
    rng = np.random.default_rng(1000 + kernel * 17 + pad)
    n, c_in, h, w = 1, 2, 10, 10
    c_out = 3
    x = (rng.standard_normal((n, c_in, h, w), dtype=np.float32) * 0.1).copy()
    weight = (rng.standard_normal((c_out, c_in, kernel, kernel), dtype=np.float32) * 0.1).copy()
    bias = np.zeros((1, c_out), dtype=np.float32)
    out_h = (h + 2 * pad - kernel) + 1
    out_w = (w + 2 * pad - kernel) + 1

    out = np.zeros((n, c_out, out_h, out_w), dtype=np.float32)
    conv2d_forward(x, weight, bias, stride=1, pad=pad, out_buf=out, fuse_relu=fuse_relu)

    # Linear loss L = sum(out * dout_fixed) => dL/d(out) = dout_fixed (constant).
    dout_fixed = (rng.standard_normal(out.shape, dtype=np.float32) * 0.1).copy()
    dx_analytic = np.zeros_like(x)
    dW = np.zeros_like(weight)
    conv2d_backward_fused(
        dout=dout_fixed,
        x=x,
        W=weight,
        dx_buf=dx_analytic,
        dW_buf=dW,
        stride=1,
        pad=pad,
        inv_m=1.0 / float(n),
        in_act=out if fuse_relu else None,
        fuse_relu=fuse_relu,
    )

    max_error = 0.0
    worst = None
    for c in range(c_in):
        for ih in range(h):
            for iw in range(w):
                orig = float(x[0, c, ih, iw])

                x[0, c, ih, iw] = orig + epsilon
                out_pos = np.zeros_like(out)
                conv2d_forward(x, weight, bias, stride=1, pad=pad, out_buf=out_pos, fuse_relu=fuse_relu)
                loss_pos = float(np.sum(out_pos * dout_fixed))

                x[0, c, ih, iw] = orig - epsilon
                out_neg = np.zeros_like(out)
                conv2d_forward(x, weight, bias, stride=1, pad=pad, out_buf=out_neg, fuse_relu=fuse_relu)
                loss_neg = float(np.sum(out_neg * dout_fixed))

                x[0, c, ih, iw] = orig
                grad_num = (loss_pos - loss_neg) / (2.0 * epsilon)
                grad_ana = float(dx_analytic[0, c, ih, iw])
                rel_error = _relative_grad_error(grad_ana, grad_num)
                if rel_error > max_error:
                    max_error = rel_error
                    worst = (c, ih, iw, grad_ana, grad_num)

    relu_tag = "relu" if fuse_relu else "linear"
    status = "PASSED" if max_error <= tolerance else "FAILED"
    print(
        f"[{status}] conv2d dX [{backend.value}] k={kernel} pad={pad} {relu_tag} "
        f"| Max Rel Error: {max_error:.2e} (Threshold: {tolerance:.0e})"
    )
    if status == "FAILED" and worst is not None:
        c, ih, iw, grad_ana, grad_num = worst
        print(f"  └── Worst at (c,h,w)=({c},{ih},{iw}) ana={grad_ana:+.6e} num={grad_num:+.6e}")
    return status == "PASSED"


def run_conv_dx_gradient_check(backend: EngineBackend = EngineBackend.NATIVE) -> int:
    print("\n" + "=" * 70)
    print(f" RUNNING CHECK: conv2d dX | Backend='{backend.value}'")
    print("=" * 70)

    passed_all = True
    for kernel, pad in CONV_DX_KERNEL_PAD_CASES:
        passed_all &= check_conv2d_input_gradient(backend, kernel, pad, fuse_relu=False)
        # ReLU fused dX needs a separate numeric setup; linear check catches native regressions.

    print("-" * 70)
    if passed_all:
        print(f"[SUCCESS] conv2d dX gradient check passed for backend '{backend.value}'.")
        return 0
    print(f"[ERROR] conv2d dX gradient check failed for backend '{backend.value}'.")
    return 1


class MockOptimizer:
    """Intercepts gradients from backward() and simulates optimizer weight decay."""
    def __init__(self):
        self.grad_weights = None
        self.grad_biases = None
        self.grad_gammas = None
        self.grad_betas = None

    def update(self, weights, biases, grad_weights, grad_biases, m_samples=1, lam_l2=0.0, grad_gammas=None, grad_betas=None, **kwargs):
        if lam_l2 > 0.0 and grad_weights is not None:
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

                tensor[coord] = original_value + self.eps
                np.random.seed(fixed_seed)
                out_pos = self.model._forward(X, training=True)
                j_plus = self.model.compute_total_loss(out_pos, y)

                tensor[coord] = original_value - self.eps
                np.random.seed(fixed_seed)
                out_neg = self.model._forward(X, training=True)
                j_minus = self.model.compute_total_loss(out_neg, y)

                tensor[coord] = original_value

                g_num = (j_plus - j_minus) / (2.0 * self.eps)
                g_ana = float(grad_tensor[coord])

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
        print(f"[{status_tag}] {tensor_name:<28} | Max Rel Error: {max_error:.2e} (Threshold: {self.tol:.0e})")
        
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

    output_dim = 1 if task_type == "binary_classification" else 3
    layer_dims = (4, 6, 4, output_dim)
    batch_size = 4
    n_features = layer_dims[0]

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

    model.weights = [w.astype(np.float64) for w in model.weights]
    model.biases = [b.astype(np.float64) for b in model.biases]
    if getattr(model, "use_batch_norm", False):
        model.gammas = [g.astype(np.float64) for g in model.gammas]
        model.betas = [b.astype(np.float64) for b in model.betas]

    mock_opt = MockOptimizer()
    model.optimizer = mock_opt
    
    np.random.seed(seed_val)
    model.backward(X_mock, y_mock, active_lr=0.01)

    checker = GradientChecker(model=model, epsilon=1e-7, tolerance=1e-5)
    
    passed_all = True
    passed_all &= checker.check_tensor(model.weights, mock_opt.grad_weights, "Weights (W)", X_mock, y_mock, seed_val)
    passed_all &= checker.check_tensor(model.biases, mock_opt.grad_biases, "Biases (b)", X_mock, y_mock, seed_val)
    
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


def run_cnn_gradient_check(backend: EngineBackend = EngineBackend.NATIVE) -> int:
    """Dedicated finite-difference check for spatial CNN layers across all compute backends."""
    seed_val = 42
    np.random.seed(seed_val)
    
    print("\n" + "=" * 70)
    print(f" RUNNING CHECK: Task='cnn' | Backend='{backend.value}' (Spatial Conv2D + Dense Head)")
    print("=" * 70)

    batch_size = 2
    in_channels, in_h, in_w = 1, 6, 6
    num_classes = 3

    X_mock = np.random.randn(batch_size, in_channels, in_h, in_w).astype(np.float64)
    raw_labels = np.random.randint(0, num_classes, size=batch_size)
    y_mock = np.zeros((batch_size, num_classes), dtype=np.float64)
    for i, label in enumerate(raw_labels):
        y_mock[i, label] = 1.0

    cnn_config = {
        "input_shape": [in_channels, in_h, in_w],
        "spatial_pipeline": [
            {"type": "conv", "in_channels": in_channels, "out_channels": 2, "kernel_size": 3, "stride": 1, "pad": 0},
            {"type": "relu"},
            {"type": "flatten"}
        ],
        "dense_head": [4]
    }

    model = ModelFactory.create_model(
        model_type="cnn",
        layer_sizes=[in_channels * in_h * in_w, num_classes],
        optimizer="sgd",
        lr=0.01,
        lam_l1=0.01,
        lam_l2=0.01,
        p_dropout=0.0,
        use_batch_norm=False,
        max_norm=1000.0,
        cnn_config=cnn_config,
        backend=backend
    )

    model.weights = [w.astype(np.float64) for w in model.weights]
    model.biases = [b.astype(np.float64) for b in model.biases]

    mock_opt = MockOptimizer()
    model.optimizer = mock_opt

    np.random.seed(seed_val)
    model.backward(X_mock, y_mock, active_lr=0.01)

    checker = GradientChecker(model=model, epsilon=1e-7, tolerance=1e-5)
    passed_all = True
    passed_all &= checker.check_tensor(model.weights, mock_opt.grad_weights, f"CNN Weights (W) [{backend.value}]", X_mock, y_mock, seed_val)
    passed_all &= checker.check_tensor(model.biases, mock_opt.grad_biases, f"CNN Biases (b) [{backend.value}]", X_mock, y_mock, seed_val)

    print("-" * 70)
    if passed_all:
        print(f"[SUCCESS] Test passed for CNN spatial architecture with backend '{backend.value}'.")
        return 0
    else:
        print(f"[ERROR] Discrepancy detected in CNN spatial architecture with backend '{backend.value}'.")
        return 1


def run_all_checks():
    test_cases = [
        {"task": "multi_class", "bn": False, "p_dropout": 0.0},
        {"task": "multi_class", "bn": True, "p_dropout": 0.0},
        {"task": "binary_classification", "bn": False, "p_dropout": 0.5},
        {"task": "regression", "bn": True, "p_dropout": 0.2},
    ]
    
    overall_status = 0
    for case in test_cases:
        exit_code = run_gradient_check(case["task"], case["bn"], case["p_dropout"])
        if exit_code != 0:
            overall_status = 1

    backends_to_test = [EngineBackend.NATIVE, EngineBackend.IM2COL_GEMM, EngineBackend.NUMPY]
    for backend in backends_to_test:
        cnn_exit_code = run_cnn_gradient_check(backend=backend)
        if cnn_exit_code != 0:
            overall_status = 1

    for backend in backends_to_test:
        dx_exit_code = run_conv_dx_gradient_check(backend=backend)
        if dx_exit_code != 0:
            overall_status = 1
            
    print("\n" + "=" * 70)
    if overall_status == 0:
        print("[GLOBAL SUCCESS] Entire Engine (MLP + CNN) Mathematically Verified Across All Tasks & Backends!")
        return 0
    else:
        print("[GLOBAL ERROR] One or more gradient checks failed.")
        return 1


def test_autodiff_gradients():
    assert run_all_checks() == 0, "Analytical gradients do not match numerical finite differences."


if __name__ == "__main__":
    sys.exit(run_all_checks())