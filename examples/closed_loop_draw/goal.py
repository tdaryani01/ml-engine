# examples/closed_loop_draw/goal.py
"""Opaque goal payload for the draw stack.

The generic trainer accepts a single ``goal: Goal``. The draw task needs two
payloads: the policy condition (``command_ids`` → Actor) and the scoring
reference (``target`` → LossEvaluator). ``DrawGoal`` carries both so the 
protocols stay single-argument.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DrawGoal:
    """Composite draw goal: actor condition + optional scoring reference."""

    command_ids: np.ndarray
    target: np.ndarray | None = None

    @property
    def batch_size(self) -> int:
        """Batch dimension convention consumed by the generic trainer."""
        return int(np.asarray(self.command_ids).reshape(-1).shape[0])

    @staticmethod
    def render(command_ids: np.ndarray) -> "DrawGoal":
        """Inference-only goal (no scoring reference)."""
        return DrawGoal(command_ids=command_ids, target=None)


__all__ = ["DrawGoal"]
