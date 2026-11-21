"""Per-epoch recordings for the pilots (data design B7, backlog 4c46cab): what a whisperer would look at, written beside the losses and changing none of them.

Off unless the run sets ``optimization.recordings`` (default False); a run that does not set it records nothing and its documents gain no key. Plain network only (§8).
Per epoch, in the step document's ``extra_metrics``:
  rec_loss_p10/p50/p90_<part>, rec_loss_over3med_<part>    percentiles of the per-row loss and the share of rows above 3 x the median, <part> = train, val
                                                           (one evaluation pass at the end of the epoch with dropout off, the weights the epoch ended with)
  rec_dead_l<i>, rec_dead_l<i>_start                       per hidden layer, the share of units whose pre-activation is not above 0 for every row of a fixed sample
                                                           (every k-th training row, k = ceil(rows / DEAD_SAMPLE), from row 0: at most DEAD_SAMPLE rows over the whole part), at the end of the epoch and at the start (the initial weights)
  rec_grad_agreement                                       mean over the epoch's consecutive pairs of steps of the cosine of the two batch gradients (weights and biases);
                                                           0.0 on an epoch with no pair (fewer than two steps, or a zero gradient), so the key is on every epoch: read it with rec_steps
  rec_clip_fires, rec_steps                                steps whose gradient norm exceeded the clipping norm, and the steps taken
The per-step part keeps its state on the model, not in a module, and only reads the gradient the step computed."""
from __future__ import annotations

import threading

import numpy as np

DEAD_SAMPLE = 1024
_lock = threading.Lock()
_staged: dict[str, float] = {}


class StepState:
    """What the steps of one epoch leave: the last gradient (flattened), the sum of cosines, the pairs, the clip fires, the steps."""

    def __init__(self) -> None:
        self.prev: np.ndarray | None = None
        self.cos_sum = 0.0
        self.pairs = 0
        self.fires = 0
        self.steps = 0


def dead_sample(n_rows: int) -> slice:
    """Every k-th training row from row 0, k = ceil(n_rows / DEAD_SAMPLE): at most DEAD_SAMPLE rows spread over the whole part, the same rows every epoch and every run on the same rows."""
    return slice(0, None, max(1, -(-int(n_rows) // DEAD_SAMPLE)))


def reset_epoch(model) -> None:
    model._rec_steps = StepState()


def step_clipped(model) -> None:
    if model._rec_steps is not None:
        model._rec_steps.fires += 1


def step_gradient(model, grad_weights, grad_biases) -> None:
    st = model._rec_steps
    if st is None:
        return
    g = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in (*grad_weights, *(b for b in grad_biases if b is not None))])
    st.steps += 1
    if st.prev is not None:
        d = float(np.linalg.norm(g) * np.linalg.norm(st.prev))
        if d > 0.0 and np.isfinite(d):
            st.cos_sum += float(np.dot(g, st.prev)) / d
            st.pairs += 1
    st.prev = g


def row_loss_stats(rows: np.ndarray, part: str) -> dict[str, float]:
    r = np.asarray(rows, dtype=np.float64).ravel()
    if r.size == 0:
        return {}
    p10, p50, p90 = (float(x) for x in np.percentile(r, [10, 50, 90]))
    return {f"rec_loss_p10_{part}": p10, f"rec_loss_p50_{part}": p50, f"rec_loss_p90_{part}": p90, f"rec_loss_over3med_{part}": float(np.mean(r > 3.0 * p50))}


def dead_stats(shares: list[float], start: list[float] | None) -> dict[str, float]:
    out = {f"rec_dead_l{i}": float(s) for i, s in enumerate(shares)}
    if start is not None:
        out.update({f"rec_dead_l{i}_start": float(s) for i, s in enumerate(start)})
    return out


def step_stats(model) -> dict[str, float]:
    st = model._rec_steps
    if st is None:
        return {}
    out = {"rec_clip_fires": float(st.fires), "rec_steps": float(st.steps), "rec_grad_agreement": st.cos_sum / st.pairs if st.pairs else 0.0}      # 0.0 when no pair of steps exists (fewer than two steps): read it with rec_steps
    return out


def stage(values: dict[str, float]) -> None:
    """Hand the epoch's numbers to the ledger's next step document (drained there once)."""
    with _lock:
        _staged.clear()
        _staged.update(values)


def drain() -> dict[str, float]:
    with _lock:
        out = dict(_staged)
        _staged.clear()
        return out
