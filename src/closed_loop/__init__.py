# src/closed_loop/__init__.py
"""Reusable closed-loop trajectory training (Python-orchestrated BPTT)."""

from src.closed_loop.adapter import LinearAdapter
from src.closed_loop.conditioning import ConditioningBank
from src.closed_loop.interleaver import TokenInterleaver
from src.closed_loop.protocols import (
    ConditioningBankProto,
    Environment,
    TrajectoryLoss,
    UpstreamEncoder,
)
from src.closed_loop.trainer import ClosedLoopTrainer, RolloutResult

__all__ = [
    "ClosedLoopTrainer",
    "ConditioningBank",
    "ConditioningBankProto",
    "Environment",
    "LinearAdapter",
    "RolloutResult",
    "TokenInterleaver",
    "TrajectoryLoss",
    "UpstreamEncoder",
]
