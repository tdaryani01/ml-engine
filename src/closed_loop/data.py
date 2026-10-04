"""The data seat of a closed-loop run: what goal each training step is scored against, and a held-out one for validation."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.closed_loop.registry import register


@register("data", "stock_commands")
def stock_commands(cfg: dict[str, Any]):
    return StockCommandData(cfg)


class StockCommandData:
    """Each step trains on one stock command's target; validation uses a different stock command (held out)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        from examples.closed_loop_draw.assemble import make_target
        from examples.closed_loop_draw.commands import STOCK_COMMAND_IDS

        cl = cfg["closed_loop"]
        self.C, self.H, self.W = (int(x) for x in cl["canvas"])
        self.batch_size = int(cl["batch_size"])
        self.command_id = int(cl.get("command_id", 0))
        self.stock_ids = tuple(int(c) for c in STOCK_COMMAND_IDS)
        self._target = make_target(cfg, self.batch_size)

    def train_goal(self, step: int):
        from examples.closed_loop_draw.goal import DrawGoal

        return DrawGoal(command_ids=np.full(self.batch_size, self.command_id, dtype=np.int64), target=self._target)

    def val_goal(self, step: int):
        from examples.closed_loop_draw.commands import load_command_target
        from examples.closed_loop_draw.goal import DrawGoal

        others = [c for c in self.stock_ids if c != self.command_id]
        cid = int(others[(max(1, int(step)) - 1) % len(others)]) if others else self.command_id
        target = load_command_target(cid, batch_size=self.batch_size, height=self.H, width=self.W, channels=self.C)
        return DrawGoal(command_ids=np.full(self.batch_size, cid, dtype=np.int64), target=target)

    def next_plate(self) -> None:
        """Autopilot's rotation: the next stock command becomes the training one."""
        if self.command_id in self.stock_ids:
            self.command_id = self.stock_ids[(self.stock_ids.index(self.command_id) + 1) % len(self.stock_ids)]
        else:
            self.command_id = self.stock_ids[0]
