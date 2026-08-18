import sys
import os
import numpy as np

# Ensure project root is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.schedulers import StepDecay, ExponentialDecay


def test_step_decay():
    print("Running Step Decay Scheduler Unit Test...")
    initial_lr = 0.1
    drop_ratio = 0.5
    epochs_per_drop = 10
    
    scheduler = StepDecay(initial_lr=initial_lr, drop_ratio=drop_ratio, epochs_per_drop=epochs_per_drop)
    
    # Epoch 0-9: No drop -> LR remains 0.1
    lr_epoch_0 = scheduler.step(0)
    lr_epoch_9 = scheduler.step(9)
    assert np.isclose(lr_epoch_0, 0.1), f"Expected 0.1 at epoch 0, got {lr_epoch_0}"
    assert np.isclose(lr_epoch_9, 0.1), f"Expected 0.1 at epoch 9, got {lr_epoch_9}"
    
    # Epoch 10: 1st drop -> 0.1 * 0.5 = 0.05
    lr_epoch_10 = scheduler.step(10)
    assert np.isclose(lr_epoch_10, 0.05), f"Expected 0.05 at epoch 10, got {lr_epoch_10}"
    
    # Epoch 20: 2nd drop -> 0.1 * (0.5 ** 2) = 0.025
    lr_epoch_20 = scheduler.step(20)
    assert np.isclose(lr_epoch_20, 0.025), f"Expected 0.025 at epoch 20, got {lr_epoch_20}"
    
    print("[PASSED] StepDecay arithmetic and interval thresholds verified.")


def test_exponential_decay():
    print("Running Exponential Decay Scheduler Unit Test...")
    initial_lr = 0.1
    decay_rate = 0.9
    
    scheduler = ExponentialDecay(initial_lr=initial_lr, decay_rate=decay_rate)
    
    # Epoch 0: 0.1 * (0.9 ** 0) = 0.1
    lr_epoch_0 = scheduler.step(0)
    assert np.isclose(lr_epoch_0, 0.1), f"Expected 0.1 at epoch 0, got {lr_epoch_0}"
    
    # Epoch 1: 0.1 * 0.9 = 0.09
    lr_epoch_1 = scheduler.step(1)
    assert np.isclose(lr_epoch_1, 0.09), f"Expected 0.09 at epoch 1, got {lr_epoch_1}"
    
    # Epoch 5: 0.1 * (0.9 ** 5) = 0.059049
    lr_epoch_5 = scheduler.step(5)
    expected_lr_5 = 0.1 * (0.9 ** 5)
    assert np.isclose(lr_epoch_5, expected_lr_5), f"Expected {expected_lr_5} at epoch 5, got {lr_epoch_5}"
    
    print("[PASSED] ExponentialDecay power step arithmetic verified.")


def run_scheduler_tests():
    print("=" * 60)
    print(" RUNNING SCHEDULER UNIT TESTS ")
    print("=" * 60)
    try:
        test_step_decay()
        test_exponential_decay()
        print("=" * 60)
        print("[SUCCESS] All scheduler tests passed cleanly!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n[FAILURE] Scheduler test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(run_scheduler_tests())