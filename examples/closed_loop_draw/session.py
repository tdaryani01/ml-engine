# examples/closed_loop_draw/session.py
"""A headless draw training session: the pieces a controller needs (train n trajectories, score a held-out shape with no updates, fork and restore the whole
training state, change the rate) with no window and no network. The window (interactive.py) and the Training Manager's brain controller call the same functions.

``val`` is a no-update rollout on a held-out stock shape, scored as ``src.closed_loop.fit`` scores it (the terminal loss), so a draw run's validation means
what it means for every closed-loop fit. The draw rollout has no sampling noise (``BackpropFeedback``): the same weights give the same val, so ``val(k)`` is the
mean of k identical rollouts and ``k`` buys nothing; the spread between runs comes from the seed (the initial weights), not from the rollout.

``fork``/``restore`` copy EVERYTHING that decides the next update: the weights and Adam moments of the MHSA (``capture_state``) plus the adapters', the conditioning
bank's and the CNN optimiser's moments and step counters, which ``capture_state`` does not carry. A restore followed by the same trajectories gives the same losses.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from examples.closed_loop_draw.assemble import assemble, load_config
from examples.closed_loop_draw.commands import STOCK_COMMAND_IDS, load_command_target
from examples.closed_loop_draw.goal import DrawGoal
from src.closed_loop.fit import terminal_loss
from src.closed_loop.state import capture_state, restore_state

_PLAIN = (int, float, bool, np.integer, np.floating)


@dataclass
class Fork:
    """One copy of the whole training state (opaque to callers)."""
    state: dict[str, np.ndarray]
    extra: dict[str, dict[str, Any]]


_STATE_ATTRS = {"adapter": ("W", "b", "_mW", "_vW", "_mb", "_vb", "_t"), "action_embed": ("W", "b", "_mW", "_vW", "_mb", "_vb", "_t"), "conditioning": ("embeddings", "_m", "_v", "_t")}      # the weights and Adam state of the small modules; NOT their forward caches (``_last_x``) or gradients


def _copyable(v: Any) -> bool:
    if isinstance(v, np.ndarray) or isinstance(v, _PLAIN):
        return True
    return isinstance(v, list) and all(x is None or isinstance(x, np.ndarray) for x in v)


class DrawSession:
    def __init__(self, cfg: dict[str, Any] | str | Path, *, seed: int, train_command: int, val_command: int, lr: float | None = None) -> None:
        self.cfg = load_config(cfg) if isinstance(cfg, (str, Path)) else cfg
        if train_command == val_command:
            raise ValueError("the validation shape must differ from the training shape")
        if train_command not in STOCK_COMMAND_IDS or val_command not in STOCK_COMMAND_IDS:
            raise ValueError("train_command and val_command must be stock commands")
        self.seed, self.train_command, self.val_command = int(seed), int(train_command), int(val_command)
        self.app = assemble(self.cfg, seed=self.seed)
        self.lr = float(lr if lr is not None else self.app.lr)
        self.trained = 0
        self.tick = None            # optional callback ``tick(kind)`` after each train trajectory (kind 'train') and each val rollout ('val'): the window pumps its events there; it may raise to stop the run
        self._ids = {c: np.full(self.app.batch_size, c, dtype=np.int64) for c in (self.train_command, self.val_command)}
        C, H, W = (int(x) for x in self.cfg["closed_loop"]["canvas"])
        self._target = {c: load_command_target(c, batch_size=self.app.batch_size, height=H, width=W, channels=C) for c in self._ids}

    def _goal(self, command: int) -> DrawGoal:
        return DrawGoal(command_ids=self._ids[command], target=self._target[command])

    def train(self, n: int) -> list[float]:
        """n update trajectories on the training shape at the current rate; the terminal loss of each."""
        out = []
        for _ in range(int(n)):
            out.append(terminal_loss(self.app.trainer.rollout_train(goal=self._goal(self.train_command), lr=self.lr, apply_updates=True)))
            self.trained += 1
            if self.tick is not None:
                self.tick("train")
        return out

    def val(self, k: int = 1) -> float:
        """Mean terminal loss of k no-update rollouts on the held-out shape (identical rollouts: the draw has no sampling noise)."""
        vals = []
        for _ in range(max(1, int(k))):
            vals.append(terminal_loss(self.app.trainer.rollout_train(goal=self._goal(self.val_command), lr=self.lr, apply_updates=False)))
            if self.tick is not None:
                self.tick("val")
        return float(np.mean(vals))

    def canvas(self) -> np.ndarray:
        """What the model draws now for the training shape: the last frame of an inference rollout, (1, C, H, W). Does not touch the weights or the optimiser."""
        from examples.closed_loop_draw.goal import DrawGoal as _G

        frames = self.app.trainer.rollout_eval(goal=_G.render(self._ids[self.train_command][:1])).extras["frames"]
        return frames[-1]

    def target(self) -> np.ndarray:
        return self._target[self.train_command][:1]

    def _parts(self) -> dict[str, Any]:
        a = self.app
        return {"adapter": a.adapter, "action_embed": a.action_embed, "conditioning": a.conditioning, "cnn_opt": getattr(a.cnn, "optimizer", None)}

    def fork(self) -> Fork:
        extra = {name: {k: copy.deepcopy(v) for k, v in vars(o).items() if _copyable(v) and (name not in _STATE_ATTRS or k in _STATE_ATTRS[name])} for name, o in self._parts().items() if o is not None}
        return Fork(capture_state(self.app.actor), extra)

    def restore(self, f: Fork) -> None:
        restore_state(self.app.actor, f.state)
        for name, o in self._parts().items():
            if o is None:
                continue
            for k, v in f.extra[name].items():
                cur = getattr(o, k)
                if isinstance(cur, np.ndarray):
                    cur[...] = v
                elif isinstance(cur, list):
                    for dst, src in zip(cur, v):
                        if dst is not None:
                            dst[...] = src
                else:
                    setattr(o, k, copy.deepcopy(v))

    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)

    def close(self) -> None:
        self.app.close()
