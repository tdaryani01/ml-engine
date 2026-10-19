"""Per-boundary gradient statistics: the signal a stop or restore decision needs and the loss curve alone does not carry.

Over the steps since the last boundary (the ledger's ``step.metrics`` document drains the accumulator):
  grad_norm_mean         mean over steps of the global gradient L2 norm (all parameter tensors together)
  grad_norm_var          variance over those steps of that norm (population variance, Welford)
  update_weight_ratio    mean over steps of ||applied update|| / ||weights|| (the usual health ratio; ~1e-3 is the folk rule of thumb)
They ride the existing ``extra_metrics`` of the step document (the same channel as a family's held-out accuracy), so every consumer that carries extras
(runner snapshot, thin row, harness) carries these too; a family or path that does not observe a quantity simply does not report it. ML_GRAD_STATS=0 switches the whole thing off.
Only Python-visible quantities are observed: a step applied by the native Adam reports its gradient norm where the gradients come back to Python and no update ratio (see observe_grads)."""
from __future__ import annotations

import math
import os
import threading
from typing import Iterable

import numpy as np

KEYS = ("grad_norm_mean", "grad_norm_var", "update_weight_ratio")


def enabled() -> bool:
    return os.environ.get("ML_GRAD_STATS", "1") != "0"


def sq_norm(arrays: Iterable[object | None]) -> float:
    """Sum of squares over every non-None array, in float64."""
    total = 0.0
    for a in arrays:
        if a is None:
            continue
        v = np.asarray(a).reshape(-1)
        total += float(np.dot(v.astype(np.float64, copy=False), v.astype(np.float64, copy=False)))
    return total


class GradStats:
    """A thread-safe running accumulator drained at each boundary."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._ratio_sum = 0.0
        self._ratio_n = 0

    def observe(self, grad_sq: float | None = None, update_sq: float | None = None, weight_sq: float | None = None) -> None:
        """One optimizer step: the squared gradient norm, and, when the update was visible, its squared norm and the squared weight norm."""
        with self._lock:
            if grad_sq is not None and math.isfinite(grad_sq) and grad_sq >= 0.0:
                g = math.sqrt(grad_sq)
                self._n += 1
                d = g - self._mean
                self._mean += d / self._n
                self._m2 += d * (g - self._mean)
            if update_sq is not None and weight_sq is not None and math.isfinite(update_sq) and math.isfinite(weight_sq) and weight_sq > 0.0 and update_sq >= 0.0:
                self._ratio_sum += math.sqrt(update_sq / weight_sq)
                self._ratio_n += 1

    def drain(self) -> dict[str, float]:
        """The statistics since the last drain (only what was observed), then reset."""
        with self._lock:
            out: dict[str, float] = {}
            if self._n:
                out["grad_norm_mean"] = float(self._mean)
                out["grad_norm_var"] = float(self._m2 / self._n)
            if self._ratio_n:
                out["update_weight_ratio"] = float(self._ratio_sum / self._ratio_n)
            self._reset()
            return out


STATS = GradStats()


def observe_step(grad_arrays: Iterable[object | None], update_arrays: Iterable[object | None] | None = None, weight_arrays: Iterable[object | None] | None = None) -> None:
    """Record one step from the arrays (no-op when switched off). Pass the weights as they were BEFORE the update for an exact ratio."""
    if not enabled():
        return
    STATS.observe(sq_norm(grad_arrays), None if update_arrays is None else sq_norm(update_arrays), None if weight_arrays is None else sq_norm(weight_arrays))


def observe_grads(grad_arrays: Iterable[object | None]) -> None:
    """A step whose update was applied outside Python (the native Adam): only the gradient norm is visible."""
    if enabled():
        STATS.observe(sq_norm(grad_arrays))


def drain() -> dict[str, float]:
    return STATS.drain() if enabled() else {}
