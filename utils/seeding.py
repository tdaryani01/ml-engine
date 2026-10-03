"""Run-level seeding (fit contract). ME's randomness (weight init, batch shuffles, dropout masks)
is all numpy's global RNG, so seeding it once at the start of a run makes the fit repeatable."""
from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed))
