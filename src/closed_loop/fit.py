"""One closed-loop fit: train on a goal, validate on a held-out goal, early-stop on validation, write the ledger.

It writes the same documents a supervised fit does (``step.metrics`` per step, ``checkpoint`` on cadence and on a new
best, ``run.end`` at the end), so whatever tails a ledger treats both alike. Versions are relative to this fit.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from src.closed_loop.state import capture_state, restore_state

_log = logging.getLogger(__name__)


def terminal_loss(result: Any) -> float:
    """The last scored step's loss (with a terminal-only loss only the last step is non-zero)."""
    extras = getattr(result, "extras", None) or {}
    step_losses = extras.get("step_losses", []) if isinstance(extras, dict) else []
    return float(step_losses[-1]) if step_losses else float(result.total_loss)


def fit_closed_loop(
    run: Any,
    ledger: Any,
    *,
    lr: float,
    steps: int,
    patience: int = 0,
    es_warmup: int = 10,
    checkpoint_every: int = 25,
    es_min_delta: float = 1e-3,
    model_instance_id: str | None = None,
    should_stop: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """Run up to ``steps`` steps; return the ``run.end`` body that was written."""
    steps = max(1, int(steps))
    best_val, best_step, best_state = float("inf"), 0, None
    bad = 0
    reason = "success"
    done = 0
    written: set[int] = set()

    def checkpoint(version: int, val: float, *, best: bool, state: dict | None = None) -> None:
        if version in written and not best:
            return
        ledger.push_checkpoint_state(state if state is not None else capture_state(run.actor), version, val_loss=val,
                                     is_local_best=best, model_instance_id=model_instance_id)
        written.add(version)

    for i in range(1, steps + 1):
        if should_stop():
            break
        train_loss = terminal_loss(run.trainer.rollout_train(goal=run.data.train_goal(i), lr=lr))
        val_result = run.trainer.rollout_train(goal=run.data.val_goal(i), lr=lr, apply_updates=False)
        val_loss = terminal_loss(val_result)
        done = i
        held_out = {f"val_{k}": v for k, v in ((getattr(val_result, "extras", None) or {}).get("metrics") or {}).items()}
        ledger.push_step_metrics(i, i, train_loss, val_loss, extra_metrics=held_out or None)
        ledger.version = i

        if i <= es_warmup:  # warm-up: always the baseline, never a bad streak
            improved, bad = True, 0
        else:
            improved = val_loss < best_val - es_min_delta
        if improved:
            best_val, best_step, bad = val_loss, i, 0
            best_state = capture_state(run.actor)
            checkpoint(i, val_loss, best=True, state=best_state)
        else:
            bad += 1
        if checkpoint_every and i % int(checkpoint_every) == 0:
            checkpoint(i, val_loss, best=False)
        ledger.store.flush()  # make this step durable now: whatever tails the ledger sees it while the run is going
        if patience > 0 and bad >= patience:
            reason = "es_trip"
            if best_state is not None:
                restore_state(run.actor, best_state)  # leave the BEST weights, as a supervised fit does
            break
    else:
        pass
    if reason == "success" and done and done not in written:
        checkpoint(done, best_val if best_val != float("inf") else 0.0, best=False)
    end = {"reason": reason, "epochs_run": done, "best_version": best_step or None,
           "best_val_loss": None if best_val == float("inf") else float(best_val), "final_version": done or None}
    ledger.push_run_end(end)
    return end
