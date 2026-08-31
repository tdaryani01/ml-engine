# testing/test_benchmark_harness.py
"""Benchmark harness regression tests (provider reuse, epoch budget)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.benchmark_cnn import reset_benchmark_data_provider


class _FakeEpochProvider:
    """Minimal stand-in for InMemoryDataProvider epoch bookkeeping."""

    def __init__(self, epochs: int = 2, num_batches: int = 3):
        self.epochs = epochs
        self.num_batches = num_batches
        self._epochs_completed = 0
        self._batch_idx = 0
        self._has_more = True
        self.backward_calls = 0

    def reset_epoch(self) -> None:
        if self._batch_idx > 0:
            self._epochs_completed += 1

        if self._epochs_completed >= self.epochs:
            self._has_more = False
            return

        self._batch_idx = 0
        self._has_more = True

    def has_more_batches(self) -> bool:
        return self._has_more and self._batch_idx < self.num_batches

    def next_batch(self):
        self._batch_idx += 1
        return np.zeros(1), np.zeros(1)

    def simulate_training_epochs(self, target_epochs: int) -> int:
        """Drain provider like a full benchmark run."""
        backward_calls = 0
        for _ in range(target_epochs):
            self.reset_epoch()
            while self.has_more_batches():
                self.next_batch()
                backward_calls += 1
        return backward_calls


def test_provider_exhausted_without_reset_skips_work() -> None:
    provider = _FakeEpochProvider(epochs=2, num_batches=2)
    first_run_bwd = provider.simulate_training_epochs(2)
    assert first_run_bwd == 4
    assert not provider.has_more_batches()

    # Bug that hit kernel sweeps: second run without reset does zero backward passes.
    second_run_bwd = 0
    provider.reset_epoch()
    while provider.has_more_batches():
        provider.next_batch()
        second_run_bwd += 1

    assert second_run_bwd == 0, "expected exhausted provider to skip batches without reset"
    print("[PASSED] exhausted provider skips work without epoch reset")


def test_reset_benchmark_data_provider_restores_epoch_budget() -> None:
    provider = _FakeEpochProvider(epochs=2, num_batches=2)
    provider.simulate_training_epochs(2)

    reset_benchmark_data_provider(provider)

    assert provider._epochs_completed == 0
    assert provider._batch_idx == 0
    assert provider._has_more is True

    second_run_bwd = 0
    provider.reset_epoch()
    while provider.has_more_batches():
        provider.next_batch()
        second_run_bwd += 1

    assert second_run_bwd == 2, f"expected 2 batches after reset, got {second_run_bwd}"
    print("[PASSED] reset_benchmark_data_provider restores training budget")


if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING BENCHMARK HARNESS REGRESSION TESTS ")
    print("=" * 60)
    test_provider_exhausted_without_reset_skips_work()
    test_reset_benchmark_data_provider_restores_epoch_budget()
    print("=" * 60)
    print("[SUCCESS] All benchmark harness regression tests passed.")
    print("=" * 60)
