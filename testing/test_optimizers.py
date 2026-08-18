import sys
import os
import numpy as np

# Ensure project root is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.optimizers import AdamOptimizer, SGDOptimizer


def test_sgd_optimizer():
    print("Running SGD Optimizer Unit Test...")
    lr = 0.1
    # Test SGD with zero momentum baseline to assert direct gradient steps
    optimizer = SGDOptimizer(lr=lr, momentum=0.0)
    
    weights = [np.array([[1.0, 2.0], [3.0, 4.0]])]
    biases = [np.array([[0.5, -0.5]])]
    
    grad_weights = [np.array([[0.1, 0.2], [0.3, 0.4]])]
    grad_biases = [np.array([[0.05, -0.05]])]
    
    # Pass exact signature parameters
    optimizer.update(
        weights=weights,
        biases=biases,
        grad_weights=grad_weights,
        grad_biases=grad_biases,
        m_samples=1,
        lam_l2=0.0,
        active_lr=lr
    )
    
    # Expected: W_new = W_old - lr * grad_weight
    expected_w = np.array([[1.0, 2.0], [3.0, 4.0]]) - 0.1 * np.array([[0.1, 0.2], [0.3, 0.4]])
    expected_b = np.array([[0.5, -0.5]]) - 0.1 * np.array([[0.05, -0.05]])
    
    assert np.allclose(weights[0], expected_w), f"SGD Weight update mismatch! Got {weights[0]}"
    assert np.allclose(biases[0], expected_b), f"SGD Bias update mismatch! Got {biases[0]}"
    print("[PASSED] SGD Optimizer update logic verified.")


def test_adam_optimizer():
    print("Running Adam Optimizer Unit Test...")
    lr = 0.001
    optimizer = AdamOptimizer(lr=lr, beta1=0.9, beta2=0.999, eps=1e-8)
    
    weights = [np.array([[1.0, 1.0]])]
    biases = [np.array([[0.0, 0.0]])]
    
    grad_weights = [np.array([[0.1, 0.2]])]
    grad_biases = [np.array([[0.01, 0.02]])]
    
    # First step (t=1)
    optimizer.update(
        weights=weights,
        biases=biases,
        grad_weights=grad_weights,
        grad_biases=grad_biases,
        m_samples=1,
        lam_l2=0.0,
        active_lr=lr
    )
    
    # Verify tracking state incremented and matched allocated attributes
    assert optimizer.t == 1
    assert len(optimizer.ms_w) == 1
    assert len(optimizer.vs_w) == 1
    assert not np.any(np.isnan(weights[0]))
    assert not np.any(np.isnan(biases[0]))
    
    print("[PASSED] Adam Optimizer momentum tracking and update step verified.")


def run_optimizer_tests():
    print("=" * 60)
    print(" RUNNING OPTIMIZER UNIT TESTS ")
    print("=" * 60)
    try:
        test_sgd_optimizer()
        test_adam_optimizer()
        print("=" * 60)
        print("[SUCCESS] All optimizer tests passed cleanly!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n[FAILURE] Optimizer test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(run_optimizer_tests())