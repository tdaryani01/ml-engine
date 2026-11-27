"""The validation predictions of a fit, written out on request (training-manager maze design point 15: "average the predictions of the best checkpoints ... reported beside the single best model").

With ``ML_DUMP_VAL_PREDS=<path>`` set, the engine writes ``<path>`` (an npz: ``preds``, ``y``, ``cost``) at the end of every epoch's validation pass, overwriting the previous epoch's: the predictions the epoch's validation loss was computed from, the targets it was
computed against, and that loss (``model.calculate_raw_cost``). A one-epoch fit (the harness's held-out score of a checkpoint) leaves exactly that checkpoint's predictions. It only READS what the epoch already computed, so a run's losses are bit-equal with it on or off.
Off by default (the variable unset): nothing is written and the code path is one dictionary lookup."""
from __future__ import annotations

import os

import numpy as np

ENV = "ML_DUMP_VAL_PREDS"


def maybe_dump(preds, y, cost: float) -> None:
    path = os.environ.get(ENV)
    if not path:
        return
    tmp = path + ".part.npz"
    np.savez(tmp, preds=np.asarray(preds), y=np.asarray(y), cost=np.float64(cost))
    os.replace(tmp, path)
