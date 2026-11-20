"""How much the training loss zigzags from step to step (maze backlog A9, design 13a): the spread of the per-step training losses.

Over the steps since the last boundary (the ledger's ``step.metrics`` document drains the accumulator, as it does ``grad_stats``):
  train_loss_step_std    population standard deviation of the per-step training losses (Welford), the loss of each optimizer step as ``_run_epoch`` collects it
It rides the document's ``extra_metrics`` (same channel as the gradient statistics). A small batch zigzags more than a large one on the same data. It only READS the losses the epoch already
computed, so a run's losses are bit-equal with it on or off. ML_STEP_LOSS_STATS=0 switches it off; a run that took no step reports nothing."""
from __future__ import annotations

import math
import os
import threading
from typing import Iterable

KEY = "train_loss_step_std"


def enabled() -> bool:
    return os.environ.get("ML_STEP_LOSS_STATS", "1") != "0"


class StepLossStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        self._n, self._mean, self._m2 = 0, 0.0, 0.0

    def observe(self, loss: float) -> None:
        x = float(loss)
        if not math.isfinite(x):
            return
        with self._lock:
            self._n += 1
            d = x - self._mean
            self._mean += d / self._n
            self._m2 += d * (x - self._mean)

    def drain(self) -> dict[str, float]:
        with self._lock:
            out = {KEY: math.sqrt(max(0.0, self._m2 / self._n))} if self._n else {}
            self._reset()
            return out


STATS = StepLossStats()


def observe_epoch(batch_losses: Iterable[object]) -> None:
    """The per-step losses of one epoch (no-op when switched off). Never raises into the training step."""
    if not enabled():
        return
    try:
        for x in batch_losses:
            STATS.observe(float(x))
    except Exception:  # noqa: BLE001
        pass


def drain() -> dict[str, float]:
    return STATS.drain() if enabled() else {}
