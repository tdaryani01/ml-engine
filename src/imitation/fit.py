"""One imitation fit: split by whole tape once, train in rounds, score held-out tapes each round, early-stop, checkpoint.

Writes the documents a supervised fit writes (``step.metrics``, ``checkpoint``, ``run.end``). Detection events are
structured records (``events``) returned with the ``run.end`` body and logged as ``imitation.<name> {json}``, so TM's
activity log can show them later without parsing prose.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable

import numpy as np

_log = logging.getLogger(__name__)


class LeakageError(RuntimeError):
    """A held-out tape is also in train: the validation numbers would be fiction, so the fit stops."""


def _emit(events: list[dict[str, Any]], name: str, **fields: Any) -> dict[str, Any]:
    rec = {"event": name, **fields}
    events.append(rec)
    level = logging.ERROR if name.endswith(".DETECTED") else logging.WARNING  # ME logs at warning: keep them visible
    _log.log(level, "imitation.%s %s", name, json.dumps(fields, sort_keys=True, default=str))
    return rec


def blob_state(blob: bytes) -> dict[str, np.ndarray]:
    """The brain's weights blob as a checkpoint state (named arrays)."""
    return {"blob": np.frombuffer(blob, dtype=np.uint8).copy()}


def blob_from_state(state: dict[str, Any]) -> bytes:
    return np.asarray(state["blob"], dtype=np.uint8).tobytes()


def _benchmark(
    blob: bytes, parent_blob: bytes | None, episodes: list[Any], train_eps: list[Any], wanted: set[str], events: list[dict[str, Any]], *, history_k: int
) -> dict[str, Any]:
    """Score the published weights on the run's frozen benchmark tapes: the one number comparable across a run's fits, and, when the
    fit restored from a checkpoint, a PAIRED comparison with it by tape (mean per-tape difference in standard errors).

    A benchmark tape that is in THIS fit's train set would make the numbers fiction, so that fails the fit."""
    from tm_brain_contracts.native_core import paired_tape_comparison, tape_losses

    in_train = sorted(wanted & {e.instance_id for e in train_eps})
    if in_train:
        _emit(events, "leakage.DETECTED", kind="benchmark_in_train", tapes=in_train[:20], n=len(in_train))
        raise LeakageError(f"{len(in_train)} benchmark tape(s) are in this fit's train set")
    bench_eps = [e for e in episodes if e.instance_id in wanted]
    child = tape_losses(blob, bench_eps, history_k=history_k)
    import hashlib

    # Which benchmark this is: scores are comparable only between fits that share it (TM freezes the tapes for a run).
    bench_id = hashlib.sha256("\n".join(sorted(wanted)).encode("utf-8")).hexdigest()[:12]
    out: dict[str, Any] = {"id": bench_id, "log_loss": (sum(child.values()) / len(child)) if child else None, "n_tapes": len(child),
                           "tapes_requested": len(wanted), "parent": None, "delta": None, "se": None, "z": None}
    if parent_blob is not None and child:
        cmp = paired_tape_comparison(tape_losses(parent_blob, bench_eps, history_k=history_k), child)
        out.update(parent=cmp["parent"], delta=cmp["delta"], se=cmp["se"], z=cmp["z"])
    _emit(events, "benchmark" if child else "benchmark.EMPTY", **out)
    return out


def fit_imitation(
    episodes: list[Any],
    ledger: Any,
    *,
    lr: float,
    steps: int | None,
    seed: int = 0,
    holdout_frac: float = 0.2,
    steps_per_round: int = 5,
    patience: int = 0,
    es_warmup: int = 3,
    checkpoint_every: int = 5,
    es_min_delta: float = 1e-4,
    train_kwargs: dict[str, Any] | None = None,
    init_blob: bytes | None = None,
    benchmark_tapes: list[str] | None = None,
    model_instance_id: str | None = None,
    should_stop: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """Return the ``run.end`` body that was written (it carries ``events``)."""
    from tm_brain_contracts.encode import HISTORY_K
    from tm_brain_contracts.native_core import train_imitation
    from tm_brain_contracts.windows import _prepare_imitation_windows, split_by_tape, window_overlap

    kw = dict(train_kwargs or {})
    events: list[dict[str, Any]] = []
    train_eps, hold_eps, rep = split_by_tape(episodes, seed=seed, holdout_frac=holdout_frac)
    _emit(events, "split", **rep)
    if rep["tape_overlap"] != 0:  # impossible by construction; checked anyway, because this is the one thing that must hold
        _emit(events, "leakage.DETECTED", kind="tape_overlap", overlap=rep["tape_overlap"])
        raise LeakageError(f"{rep['tape_overlap']} held-out tape(s) are also in train")
    if not hold_eps:
        raise LeakageError("the corpus has fewer than two tapes: there is nothing to hold out, so nothing to validate on")
    tw, hw, _n, _classes = _prepare_imitation_windows(train_eps, holdout_episodes=hold_eps, history_k=kw.get("history_k", HISTORY_K))
    n_over, n_hold = window_overlap(tw, hw)
    _emit(events, "windows", n_train=len(tw), n_holdout=len(hw), identical_windows_in_train=n_over,
          identical_pct=round(100.0 * n_over / max(1, n_hold), 2))
    if n_over:
        # Different tapes that hold the same window: not a split defect, but those held-out rows are not independent.
        _emit(events, "leakage.DETECTED", kind="identical_windows", overlap=n_over, of=n_hold)

    best_val, best_round, best_blob = float("inf"), 0, init_blob
    best_train: float | None = None
    blob, bad, reason, done, written = init_blob, 0, "success", 0, set()
    rounds = None if steps is None else max(1, -(-int(steps) // max(1, int(steps_per_round))))  # None: until early stop

    def checkpoint(version: int, val: float, *, best: bool, b: bytes) -> None:
        if version in written and not best:
            return
        ledger.push_checkpoint_state(blob_state(b), version, val_loss=val, is_local_best=best, model_instance_id=model_instance_id)
        written.add(version)

    r = 0
    while rounds is None or r < rounds:
        r += 1
        if should_stop():
            break
        # The batch sampler advances every round (it restarts from seed 0 on every call otherwise, and the fit retrains the same few batches).
        blob, m = train_imitation(train_eps, steps=int(steps_per_round), lr=float(lr), init_blob=blob, holdout_episodes=hold_eps,
                                  sampler_seed=int(seed) * 1_000_003 + r, **kw)
        done = r
        # The train loss a chart shows is the NATURAL slice (not selected for difficulty), the one comparable with validation; the all-rows
        # number sits above validation by construction (the train set is enriched with hard rehearsal/golden rows the held-out tapes lack).
        train_all = m.get("train_log_loss")
        train_loss = m.get("train_log_loss_natural") if m.get("train_log_loss_natural") is not None else train_all
        val_loss = m.get("holdout_log_loss")
        if val_loss is None:
            raise LeakageError("no held-out windows were scored")
        ledger.push_step_metrics(r, r, float(train_loss), float(val_loss))
        ledger.version = r
        _emit(events, "round", round=r, train_log_loss=train_loss, train_log_loss_all=train_all, train_by_source=m.get("train_log_loss_by_source"),
              holdout_log_loss=val_loss,
              holdout_accuracy=m.get("holdout_accuracy"), n_train=m.get("n_train"), n_holdout=m.get("n_holdout"))
        # The best is tracked from the FIRST round: warm-up only means a bad round is not counted yet (a restored checkpoint is often at its
        # best in the first rounds, and the loss swings a lot early; overwriting the best with the latest warm-up round lost it).
        improved = val_loss < best_val - es_min_delta
        if improved:
            best_val, best_round, best_blob, bad = float(val_loss), r, blob, 0
            best_train = float(train_loss)
            checkpoint(r, float(val_loss), best=True, b=blob)
        elif r > es_warmup:
            bad += 1
        if checkpoint_every and r % int(checkpoint_every) == 0:
            checkpoint(r, float(val_loss), best=False, b=blob)
        ledger.store.flush()
        if patience > 0 and bad >= patience:
            reason = "es_trip"
            break
    if reason == "success" and done and best_blob is not None:
        # One model per fit, the BEST one: the final slot holds the best weights (as an early stop leaves them), so whatever
        # publishes this fit's model reads one checkpoint and never has to pick.
        checkpoint(done, best_val if best_val != float("inf") else 0.0, best=True, b=best_blob)
    published = best_blob if best_blob is not None else blob
    bench: dict[str, Any] | None = None
    # The benchmark tapes are the ones the corpus tags as such (TM re-freezes them per fit) plus any named in the config.
    wanted = set(benchmark_tapes or []) | {e.instance_id for e in episodes if getattr(e, "mix_source", None) == "benchmark"}
    if wanted and published is not None:
        bench = _benchmark(published, init_blob, episodes, train_eps, wanted, events, history_k=kw.get("history_k", HISTORY_K))
    end = {"reason": reason, "epochs_run": done, "best_version": best_round or None,
           "best_val_loss": None if best_val == float("inf") else float(best_val), "final_version": done or None,
           # The losses OF THE PUBLISHED CHECKPOINT's round: what a chart must show for this fit (the last round's train loss is a different,
           # often diverged, model). The runner reads these.
           "published_val_loss": None if best_val == float("inf") else float(best_val), "published_train_loss": best_train,
           "split": rep, "events": events,
           "benchmark_loss": None if bench is None else bench["log_loss"], "benchmark_z": None if bench is None else bench["z"],
           "benchmark_n_tapes": None if bench is None else bench["n_tapes"], "benchmark_id": None if bench is None else bench["id"],
           "benchmark_parent_loss": None if bench is None else bench["parent"],
           "benchmark": bench}
    ledger.push_run_end(end)
    return end
