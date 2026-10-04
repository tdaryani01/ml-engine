"""A closed-loop model's trainable state as named arrays (what a checkpoint stores), and putting it back."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np


def _parts(actor: Any) -> Any:
    return SimpleNamespace(mhsa=actor.mhsa, cnn=actor.encoder.cnn, adapter=actor.adapter,
                           action_embed=actor.action_embed, conditioning=actor.conditioning)


_SCALARS = frozenset({"opt_t"})  # stored as arrays; the wire gives them back as one-element arrays


def capture_state(actor: Any) -> dict[str, np.ndarray]:
    """Copies of every trainable tensor and optimizer moment, flattened to ``{name: array}``."""
    from examples.closed_loop_draw.draw_checkpoint import snapshot_trainable

    flat: dict[str, np.ndarray] = {}
    for key, value in snapshot_trainable(_parts(actor)).items():
        if isinstance(value, list):
            for i, arr in enumerate(value):
                flat[f"{key}#{i}"] = np.array(arr, copy=True)
        else:
            flat[key] = np.array(value, copy=True)
    return flat


def restore_state(actor: Any, flat: dict[str, np.ndarray]) -> None:
    from examples.closed_loop_draw.draw_checkpoint import restore_trainable

    grouped: dict[str, Any] = {}
    lists: dict[str, dict[int, np.ndarray]] = {}
    for name, arr in flat.items():
        if "#" in name:
            key, _, idx = name.partition("#")
            lists.setdefault(key, {})[int(idx)] = arr
        elif name in _SCALARS or arr.ndim == 0:
            grouped[name] = int(np.asarray(arr).reshape(-1)[0])
        else:
            grouped[name] = arr
    for key, items in lists.items():
        grouped[key] = [items[i] for i in sorted(items)]
    restore_trainable(_parts(actor), grouped)
