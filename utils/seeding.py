"""Run-level seeding (fit contract). ME's randomness (weight init, batch shuffles, dropout masks)
is numpy's global RNG, so seeding it once at the start of a run makes the fit repeatable.

Data design A4 (§9 line 20): with an ``init_seed`` and/or a ``batch_seed`` the plain network's starting weights and its batch
order (shuffle and dropout masks) come from two streams of their own. With neither, nothing changes: one global stream, as
before. A ``RandomState(s)`` yields the numbers ``np.random.seed(s)`` gives the global stream, so each stream draws with the
same calls and the same numbers as the single stream did."""
from __future__ import annotations

import random

import numpy as np

_streams: dict[str, np.random.RandomState | None] = {"init": None, "batch": None}


def seed_everything(seed: int | None, init_seed: int | None = None, batch_seed: int | None = None) -> None:
    """Seed the global stream with ``seed``. Given an ``init_seed`` or a ``batch_seed``, the plain network's init draws come
    from ``RandomState(init_seed)`` and its shuffle and dropout draws from ``RandomState(batch_seed)``; the one not given
    falls back to ``seed`` (None = unseeded). Neither given = the global stream for both (the legacy path)."""
    _streams["init"] = _streams["batch"] = None
    if init_seed is not None or batch_seed is not None:
        pick = lambda s: np.random.RandomState(int(s) if s is not None else (int(seed) if seed is not None else None))  # noqa: E731
        _streams["init"], _streams["batch"] = pick(init_seed), pick(batch_seed)
    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed))


def init_rng():
    """The stream the plain network draws its starting weights from (numpy's global RNG unless an init or batch seed is set)."""
    return np.random if _streams["init"] is None else _streams["init"]


def batch_rng():
    """The stream the plain network's batch order and dropout masks are drawn from (numpy's global RNG unless one is set)."""
    return np.random if _streams["batch"] is None else _streams["batch"]
