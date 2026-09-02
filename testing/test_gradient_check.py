# testing/test_gradient_check.py
import sys
import os
import itertools
import numpy as np

# Ensure project root is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_factory import ModelFactory
from config.constants import EngineBackend
from utils.engine_ops import create_engine_context
from utils.conv_dispatch import conv2d_forward, conv2d_backward_fused, init_engine_backend


CONV_KERNELS = tuple(range(1, 8))
CONV_STRIDES = (1, 2)
CONV_PADS = (1, 2)

# Full matrix: k=1..7, stride=1..2, pad=1..2 (28 cases).
CONV_GRAD_CASES = [
    (k, stride, pad)
    for k, stride, pad in itertools.product(CONV_KERNELS, CONV_STRIDES, CONV_PADS)
]

# Backward-compatible alias used by test_native_conv.py (stride-1 only).
CONV_DX_KERNEL_PAD_CASES = [(k, pad) for k, stride, pad in CONV_GRAD_CASES if stride == 1]


def _conv_output_size(h: int, w: int, kernel: int, stride: int, pad: int) -> tuple[int, int]:
    out_h = (h + 2 * pad - kernel) // stride + 1
    out_w = (w + 2 * pad - kernel) // stride + 1
    return out_h, out_w


def conv_grad_fixture_size(kernel: int, stride: int, pad: int, min_out: int = 3) -> tuple[int, int]:
    """Pick spatial dims large enough for stable finite-difference checks."""
    for size in range(max(kernel, min_out + kernel), 128):
        out_h, out_w = _conv_output_size(size, size, kernel, stride, pad)
        if out_h >= min_out and out_w >= min_out:
            return size, size
    raise ValueError(
        f"No valid fixture size for kernel={kernel}, stride={stride}, pad={pad}."
    )


def _relative_grad_error(analytic: float, numeric: float) -> float:
    abs_diff = abs(analytic - numeric)
    # FD noise floor: treat tiny analytic values with zero numeric as absolute error.
    if abs(numeric) < 1e-6 and abs(analytic) < 1e-4:
        return abs_diff
    if abs(analytic) < 1e-7 and abs(numeric) < 1e-7:
        return abs_diff
    return abs_diff / max(abs(analytic) + abs(numeric), 1e-12)


def _make_conv_grad_fixture(
    backend: EngineBackend,
    kernel: int,
    stride: int,
    pad: int,
    *,
    fuse_relu: bool = False,
):
    ctx = create_engine_context(backend)
    rng = np.random.default_rng(1000 + kernel * 17 + stride * 31 + pad * 53)
    h, w = conv_grad_fixture_size(kernel, stride, pad)
    n, c_in = 1, 2
    c_out = 3
    x = (rng.standard_normal((n, c_in, h, w), dtype=np.float32) * 0.1).copy()
    weight = (rng.standard_normal((c_out, c_in, kernel, kernel), dtype=np.float32) * 0.1).copy()
    bias = np.zeros((1, c_out), dtype=np.float32)
    out_h, out_w = _conv_output_size(h, w, kernel, stride, pad)
    if out_h < 1 or out_w < 1:
        raise ValueError(
            f"Invalid conv geometry for fixture: k={kernel}, stride={stride}, pad={pad}, input={h}x{w}"
        )

    out = np.zeros((n, c_out, out_h, out_w), dtype=np.float32)
    conv2d_forward(
        x, weight, bias, stride=stride, pad=pad, out_buf=out,
        fuse_relu=fuse_relu, ctx=ctx,
    )
    dout_fixed = (rng.standard_normal(out.shape, dtype=np.float32) * 0.1).copy()
    return x, weight, bias, out, dout_fixed, stride, pad, fuse_relu, ctx


def check_conv2d_input_gradient(
    backend: EngineBackend,
    kernel: int,
    pad: int,
    *,
    stride: int = 1,
    fuse_relu: bool = False,
    epsilon: float = 1e-4,
    tolerance: float = 8e-2,
) -> bool:
    """
    Finite-difference check for conv2d dX (input gradient).
    Catches native fallback dX regressions that weight-only CNN grad checks miss.
    """
    x, weight, bias, out, dout_fixed, stride, pad, fuse_relu, ctx = _make_conv_grad_fixture(
        backend, kernel, stride, pad, fuse_relu=fuse_relu
    )

    dx_analytic = np.zeros_like(x)
    dW = np.zeros_like(weight)
    conv2d_backward_fused(
        dout=dout_fixed,
        x=x,
        W=weight,
        dx_buf=dx_analytic,
        dW_buf=dW,
        stride=stride,
        pad=pad,
        inv_m=1.0 / float(x.shape[0]),
        in_act=out if fuse_relu else None,
        fuse_relu=fuse_relu,
        ctx=ctx,
    )

    _, _, h, w = x.shape
    c_in = x.shape[1]
    max_error = 0.0
    worst = None
    for c in range(c_in):
        for ih in range(h):
            for iw in range(w):
                orig = float(x[0, c, ih, iw])

                x[0, c, ih, iw] = orig + epsilon
                out_pos = np.zeros_like(out)
                conv2d_forward(
                    x, weight, bias, stride=stride, pad=pad, out_buf=out_pos,
                    fuse_relu=fuse_relu, ctx=ctx,
                )
                loss_pos = float(np.sum(out_pos * dout_fixed))

                x[0, c, ih, iw] = orig - epsilon
                out_neg = np.zeros_like(out)
                conv2d_forward(
                    x, weight, bias, stride=stride, pad=pad, out_buf=out_neg,
                    fuse_relu=fuse_relu, ctx=ctx,
                )
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
        f"[{status}] conv2d dX [{backend.value}] k={kernel} s={stride} pad={pad} {relu_tag} "
        f"| Max Rel Error: {max_error:.2e} (Threshold: {tolerance:.0e})"
    )
    if status == "FAILED" and worst is not None:
        c, ih, iw, grad_ana, grad_num = worst
        print(f"  └── Worst at (c,h,w)=({c},{ih},{iw}) ana={grad_ana:+.6e} num={grad_num:+.6e}")
    return status == "PASSED"


def check_conv2d_weight_gradient(
    backend: EngineBackend,
    kernel: int,
    pad: int,
    *,
    stride: int = 1,
    fuse_relu: bool = False,
    epsilon: float = 1e-4,
    tolerance: float = 8e-2,
) -> bool:
    """Finite-difference check for conv2d dW (weight gradient)."""
    x, weight, bias, out, dout_fixed, stride, pad, fuse_relu, ctx = _make_conv_grad_fixture(
        backend, kernel, stride, pad, fuse_relu=fuse_relu
    )

    dx_analytic = np.zeros_like(x)
    dW_analytic = np.zeros_like(weight)
    conv2d_backward_fused(
        dout=dout_fixed,
        x=x,
        W=weight,
        dx_buf=dx_analytic,
        dW_buf=dW_analytic,
        stride=stride,
        pad=pad,
        inv_m=1.0 / float(x.shape[0]),
        in_act=out if fuse_relu else None,
        fuse_relu=fuse_relu,
        ctx=ctx,
    )

    max_error = 0.0
    worst = None
    it = np.nditer(weight, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        coord = it.multi_index
        orig = float(weight[coord])

        weight[coord] = orig + epsilon
        out_pos = np.zeros_like(out)
        conv2d_forward(
            x, weight, bias, stride=stride, pad=pad, out_buf=out_pos,
            fuse_relu=fuse_relu, ctx=ctx,
        )
        loss_pos = float(np.sum(out_pos * dout_fixed))

        weight[coord] = orig - epsilon
        out_neg = np.zeros_like(out)
        conv2d_forward(
            x, weight, bias, stride=stride, pad=pad, out_buf=out_neg,
            fuse_relu=fuse_relu, ctx=ctx,
        )
        loss_neg = float(np.sum(out_neg * dout_fixed))

        weight[coord] = orig
        grad_num = (loss_pos - loss_neg) / (2.0 * epsilon)
        grad_ana = float(dW_analytic[coord])
        rel_error = _relative_grad_error(grad_ana, grad_num)
        if rel_error > max_error:
            max_error = rel_error
            worst = (coord, grad_ana, grad_num)
        it.iternext()

    relu_tag = "relu" if fuse_relu else "linear"
    status = "PASSED" if max_error <= tolerance else "FAILED"
    print(
        f"[{status}] conv2d dW [{backend.value}] k={kernel} s={stride} pad={pad} {relu_tag} "
        f"| Max Rel Error: {max_error:.2e} (Threshold: {tolerance:.0e})"
    )
    if status == "FAILED" and worst is not None:
        coord, grad_ana, grad_num = worst
        print(f"  └── Worst at W{coord} ana={grad_ana:+.6e} num={grad_num:+.6e}")
    return status == "PASSED"


def check_conv2d_gradients_native_vs_numpy(
    kernel: int,
    stride: int,
    pad: int,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> bool:
    """Compare native conv backward against NumPy reference (dX and dW)."""
    x, weight, bias, out, dout, stride, pad, _, _ = _make_conv_grad_fixture(
        EngineBackend.NATIVE, kernel, stride, pad, fuse_relu=False
    )
    inv_m = 1.0 / float(x.shape[0])

    dx_ref = np.zeros_like(x)
    dW_ref = np.zeros_like(weight)
    numpy_ctx = create_engine_context(EngineBackend.NUMPY)
    conv2d_backward_fused(
        dout, x, weight, dx_ref, dW_ref, stride=stride, pad=pad, inv_m=inv_m, ctx=numpy_ctx
    )

    dx_native = np.zeros_like(x)
    dW_native = np.zeros_like(weight)
    native_ctx = create_engine_context(EngineBackend.NATIVE)
    conv2d_backward_fused(
        dout, x, weight, dx_native, dW_native, stride=stride, pad=pad, inv_m=inv_m, ctx=native_ctx
    )

    dx_diff = float(np.max(np.abs(dx_native - dx_ref)))
    dW_diff = float(np.max(np.abs(dW_native - dW_ref)))
    dx_ok = np.allclose(dx_native, dx_ref, rtol=rtol, atol=atol)
    dW_ok = np.allclose(dW_native, dW_ref, rtol=rtol, atol=atol)

    for label, ok, diff in (("dX", dx_ok, dx_diff), ("dW", dW_ok, dW_diff)):
        status = "PASSED" if ok else "FAILED"
        print(
            f"[{status}] conv2d {label} [native] k={kernel} s={stride} pad={pad} "
            f"| Max Abs Error vs ref: {diff:.2e} (rtol={rtol:.0e}, atol={atol:.0e})"
        )
    return dx_ok and dW_ok


def check_conv2d_gradients_im2col_gemm_vs_numpy(
    kernel: int,
    stride: int,
    pad: int,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> bool:
    """Compare im2col+GEMM conv backward against NumPy reference (dX and dW)."""
    x, weight, bias, out, dout, stride, pad, _, _ = _make_conv_grad_fixture(
        EngineBackend.IM2COL_GEMM, kernel, stride, pad, fuse_relu=False
    )
    inv_m = 1.0 / float(x.shape[0])

    dx_ref = np.zeros_like(x)
    dW_ref = np.zeros_like(weight)
    numpy_ctx = create_engine_context(EngineBackend.NUMPY)
    conv2d_backward_fused(
        dout, x, weight, dx_ref, dW_ref, stride=stride, pad=pad, inv_m=inv_m, ctx=numpy_ctx
    )

    dx_gemm = np.zeros_like(x)
    dW_gemm = np.zeros_like(weight)
    gemm_ctx = create_engine_context(EngineBackend.IM2COL_GEMM)
    conv2d_backward_fused(
        dout, x, weight, dx_gemm, dW_gemm, stride=stride, pad=pad, inv_m=inv_m, ctx=gemm_ctx
    )

    dx_diff = float(np.max(np.abs(dx_gemm - dx_ref)))
    dW_diff = float(np.max(np.abs(dW_gemm - dW_ref)))
    dx_ok = np.allclose(dx_gemm, dx_ref, rtol=rtol, atol=atol)
    dW_ok = np.allclose(dW_gemm, dW_ref, rtol=rtol, atol=atol)

    for label, ok, diff in (("dX", dx_ok, dx_diff), ("dW", dW_ok, dW_diff)):
        status = "PASSED" if ok else "FAILED"
        print(
            f"[{status}] conv2d {label} [im2col+gemm] k={kernel} s={stride} pad={pad} "
            f"| Max Abs Error vs ref: {diff:.2e} (rtol={rtol:.0e}, atol={atol:.0e})"
        )
    return dx_ok and dW_ok


def check_conv2d_gradients(
    backend: EngineBackend,
    kernel: int,
    stride: int,
    pad: int,
    *,
    fuse_relu: bool = False,
) -> bool:
    if backend == EngineBackend.NATIVE:
        return check_conv2d_gradients_native_vs_numpy(kernel, stride, pad)
    if backend == EngineBackend.IM2COL_GEMM:
        return check_conv2d_gradients_im2col_gemm_vs_numpy(kernel, stride, pad)
    return (
        check_conv2d_input_gradient(
            backend, kernel, pad, stride=stride, fuse_relu=fuse_relu
        )
        and check_conv2d_weight_gradient(
            backend, kernel, pad, stride=stride, fuse_relu=fuse_relu
        )
    )


def run_conv_gradient_check(
    backend: EngineBackend = EngineBackend.NATIVE,
    *,
    cases: list[tuple[int, int, int]] | None = None,
) -> int:
    """Run conv backward checks across geometry matrix (native vs NumPy reference)."""
    cases = cases or CONV_GRAD_CASES
    print("\n" + "=" * 70)
    print(
        f" RUNNING CHECK: conv2d dX+dW matrix | Backend='{backend.value}' "
        f"| {len(cases)} cases (k=1..7, stride=1..2, pad=1..2)"
    )
    if backend == EngineBackend.NATIVE:
        print(" Reference: NumPy im2col backward (not finite-difference)")
    elif backend == EngineBackend.IM2COL_GEMM:
        print(" Reference: NumPy im2col backward (not finite-difference)")
    print("=" * 70)

    passed_all = True
    for kernel, stride, pad in cases:
        passed_all &= check_conv2d_gradients(
            backend, kernel, stride, pad, fuse_relu=False
        )

    print("-" * 70)
    if passed_all:
        print(f"[SUCCESS] conv2d gradient matrix passed for backend '{backend.value}'.")
        return 0
    print(f"[ERROR] conv2d gradient matrix failed for backend '{backend.value}'.")
    return 1


def run_conv_dx_gradient_check(backend: EngineBackend = EngineBackend.NATIVE) -> int:
    print("\n" + "=" * 70)
    print(f" RUNNING CHECK: conv2d dX | Backend='{backend.value}'")
    print("=" * 70)

    passed_all = True
    for kernel, pad in CONV_DX_KERNEL_PAD_CASES:
        passed_all &= check_conv2d_input_gradient(backend, kernel, pad, stride=1, fuse_relu=False)
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

    # Conv kernel matrix: native vs NumPy; im2col+GEMM vs NumPy (Phase D3).
    conv_exit_code = run_conv_gradient_check(backend=EngineBackend.NATIVE)
    if conv_exit_code != 0:
        overall_status = 1
        print(
            "[HINT] Native conv grad check failed. Rebuild the DLL first: .\\build_native.ps1"
        )

    conv_gemm_exit = run_conv_gradient_check(backend=EngineBackend.IM2COL_GEMM)
    if conv_gemm_exit != 0:
        overall_status = 1
        print("[HINT] IM2COL+GEMM conv grad check failed vs NumPy reference.")
            
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