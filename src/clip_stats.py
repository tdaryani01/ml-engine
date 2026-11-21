"""How many input cells the engine's clip cut (data design A2, §9 row 2, 03ec5b3).

Written once when a provider with the clip switched on is built, and drained by the ledger's first ``step.metrics`` document into its
``extra_metrics`` (the channel of ``grad_stats``); the harness copies the numbers to the sheet. Keys, per part (train, val), counted on the standardised cells
of the non-exempt columns: ``clip_cells_<part>`` (cells beyond the bound) and ``clip_rows_multi_<part>`` (rows with two or more such cells).
The accumulator is process-wide: it is safe for one fit per process (the harness's way). Two clip-on providers built before the first step document, or a fit that writes no step document, would hand their counts to the wrong fit; if fits ever share a process, keep the counts on the provider and drain them from its own fit."""
from __future__ import annotations

import threading

_lock = threading.Lock()
_counts: dict[str, float] = {}


def set_counts(counts: dict[str, float]) -> None:
    """Replace what the next ``drain`` returns. Called only by a provider with the clip on; a provider without it does not call (it must not erase another's counts)."""
    with _lock:
        _counts.clear()
        _counts.update({str(k): float(v) for k, v in counts.items()})


def drain() -> dict[str, float]:
    with _lock:
        out = dict(_counts)
        _counts.clear()
        return out
