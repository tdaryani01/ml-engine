# src/training_engine.py
"""Single-thread training loop: train → ledger documents → consolidate → checkpoint."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

import numpy as np

from src.ledger import BatchRef, LedgerConfig, TrainingLedger, VERDICT_HEALTHY
from src.training_session import TrainStepResult, TrainingSession

try:
    profile  # type: ignore[name-defined]  # noqa: F821 — kernprof injects this at runtime
except NameError:
    def profile(func):  # noqa: D103
        return func


@dataclass
class StepInput:
    X: np.ndarray
    y: np.ndarray
    batch_ref: BatchRef
    lr: float
    val_loss: float | None = None


@dataclass
class _InflightStep:
    step: StepInput
    step_id: int
    base_version: int


class LoopState(Enum):
    """Single-thread main-loop states."""

    FINALIZE = auto()  # completion signal posted → ledger finalize
    SPIN = auto()  # inflight, spin until native callback signal
    SUBMIT = auto()  # ask CNN add_training_step (OK|BUSY)
    TICK = auto()  # ledger flush check-back only (no-op if idle)
    EXIT = auto()  # no batches left and nothing in flight


class TrainingEngine:
    """
    SQL-style main loop owner.

    Loop states: FINALIZE → SPIN → SUBMIT → TICK → EXIT.

    tick() = ledger flush check-back only; finalize if completion signal posted.
    It is NOT one batch and is a no-op when nothing needs service.

    Training loop: ask CNN add_training_step → OK|BUSY. On BUSY, TICK ledger and
    retry. While waiting for callback signal, SPIN (not a blind per-iteration wait).
    """

    def __init__(
        self,
        session: TrainingSession,
        ledger: TrainingLedger,
        config: LedgerConfig | None = None,
    ):
        self.session = session
        self.ledger = ledger
        self.config = config or LedgerConfig()
        self._steps_since_checkpoint = 0
        self._last_healthy_version = 0
        self._flush_stall_count = 0
        self._inflight: _InflightStep | None = None
        self._async_contract = False

    @property
    def version(self) -> int:
        return self.ledger.version

    def uses_async_contract(self) -> bool:
        return self._async_contract

    def has_pending(self) -> bool:
        if self._inflight is not None:
            return True
        rt = getattr(self.session.model, "_contract_runtime", None)
        if rt is None:
            return False
        return rt.native_in_flight() or rt.has_completed()

    def can_submit(self) -> bool:
        return self._inflight is None and not self._native_in_flight()

    def _ledger_needs_work(self) -> bool:
        store = self.ledger.store
        if not hasattr(store, "has_flush_pending"):
            return False
        return store.has_flush_pending() or store.queue_pending()

    def _contract_runtime(self) -> Any | None:
        return getattr(self.session.model, "_contract_runtime", None)

    def _completion_signaled(self) -> bool:
        rt = self._contract_runtime()
        return rt is not None and rt.completion_signaled()

    def _native_ready(self) -> bool:
        return self._inflight is not None and self._completion_signaled()

    def _awaiting_signal(self) -> bool:
        if self._inflight is None or self._completion_signaled():
            return False
        rt = self._contract_runtime()
        return rt is not None and rt.waiting_on_native_worker()

    def _spin_wait_for_signal(self) -> None:
        """Spin until native callback posts completion signal (not a ledger tick)."""
        while self._awaiting_signal() and not self._completion_signaled():
            pass

    def _finalize_if_ready(self) -> float | None:
        if not self._native_ready():
            return None
        self._ledger_io_tick()
        loss = self._try_finalize_inflight()
        self._ledger_io_begin_flush()
        return loss

    def _service_ledger(self) -> None:
        if not self._ledger_needs_work():
            return
        self._ledger_io_tick()
        self._ledger_io_begin_flush()

    def _resolve_loop_state(
        self,
        *,
        steps_done: int,
        steps_budget: int,
        pending: StepInput | None,
        drain_only: bool = False,
    ) -> LoopState:
        """Pick the next main-loop state from current conditions."""
        if self._native_ready():
            return LoopState.FINALIZE
        if self._awaiting_signal():
            return LoopState.SPIN
        if not drain_only:
            if steps_done >= steps_budget:
                return LoopState.EXIT if not self.has_pending() else LoopState.SPIN
            if pending is None:
                return LoopState.SUBMIT
            if not self.can_submit() or self.session.contract_busy():
                return LoopState.TICK
            return LoopState.SUBMIT
        if not self.has_pending():
            return LoopState.EXIT
        return LoopState.TICK

    @profile
    def tick(self) -> float | None:
        """
        Ledger maintenance pass — NOT a training batch.

        Runs only when completion signal is posted (finalize) or journal I/O
        needs check-back. Otherwise no-op (returns None).
        """
        loss = self._finalize_if_ready()
        if loss is not None:
            return loss
        self._service_ledger()
        return None

    def run_training_loop(
        self,
        *,
        steps_budget: int,
        next_step: Callable[[], StepInput | None],
        on_submitted: Callable[[StepInput], None] | None = None,
    ) -> list[float]:
        losses: list[float] = []
        steps_done = 0
        pending: StepInput | None = None
        state = self._resolve_loop_state(
            steps_done=steps_done, steps_budget=steps_budget, pending=pending
        )

        while state != LoopState.EXIT:
            if state == LoopState.FINALIZE:
                loss = self._finalize_if_ready()
                if loss is not None:
                    losses.append(loss)
                    steps_done += 1
            elif state == LoopState.SPIN:
                self._service_ledger()
                self._spin_wait_for_signal()
            elif state == LoopState.SUBMIT:
                if pending is None:
                    pending = next_step()
                    if pending is None:
                        state = (
                            LoopState.EXIT
                            if not self.has_pending()
                            else LoopState.SPIN
                        )
                        continue
                if self.try_submit(pending):
                    if on_submitted is not None:
                        on_submitted(pending)
                    pending = None
                else:
                    state = LoopState.TICK
                    continue
            elif state == LoopState.TICK:
                self.tick()

            state = self._resolve_loop_state(
                steps_done=steps_done, steps_budget=steps_budget, pending=pending
            )

        return losses

    def drain_pending(self) -> list[float]:
        """Finalize inflight work at epoch/fit end."""
        losses: list[float] = []
        state = self._resolve_loop_state(
            steps_done=0, steps_budget=0, pending=None, drain_only=True
        )
        if state == LoopState.EXIT:
            return losses

        while state != LoopState.EXIT:
            if state == LoopState.FINALIZE:
                loss = self._finalize_if_ready()
                if loss is not None:
                    losses.append(loss)
            elif state == LoopState.SPIN:
                self._spin_wait_for_signal()
            elif state == LoopState.TICK:
                self.tick()
                if not self._awaiting_signal() and not self._native_ready():
                    state = LoopState.EXIT
                    continue

            state = self._resolve_loop_state(
                steps_done=0, steps_budget=0, pending=None, drain_only=True
            )

        return losses

    def try_submit(self, step: StepInput) -> bool:
        """Ask CNN to accept step. Returns False if BUSY (no train attempted)."""
        if not self.can_submit():
            return False
        if self.session.contract_busy():
            return False
        step_id = self.session.reserve_step_id()
        base_version = self.ledger.version
        if not self.session.submit_contract_step(
            step.X, step.y, step.lr, step_id=step_id, apply_adam=True
        ):
            return False
        self._inflight = _InflightStep(step=step, step_id=step_id, base_version=base_version)
        self._ledger_io_begin_flush()
        return True

    @profile
    def run_step(self, step: StepInput) -> float:
        """Sync path (non-contract or tests): one blocking step."""
        if self._async_contract:
            raise RuntimeError("async contract path requires tick/submit loop, not run_step")

        self._ledger_io_tick()
        step_id = self.session.reserve_step_id()
        base_version = self.ledger.version
        result = self.session.train_step(step.X, step.y, step.lr, step_id=step_id)
        loss = self.run_step_finalize(step, result, step_id=step_id, base_version=base_version)
        self._ledger_io_begin_flush()
        return loss

    def _native_in_flight(self) -> bool:
        rt = getattr(self.session.model, "_contract_runtime", None)
        if rt is None:
            return False
        return rt.native_in_flight()

    def _try_finalize_inflight(self) -> float | None:
        if self._inflight is None:
            return None
        packed = self.session.try_reap_contract_step()
        if packed is None:
            return None
        inflight = self._inflight
        self._inflight = None
        loss, gw, gb, m, gg, gbb, weights_applied = packed
        result = self.session._pack_train_step_result(
            inflight.step_id,
            loss,
            gw,
            gb,
            m,
            gg,
            gbb,
            weights_applied=weights_applied,
        )
        return self.run_step_finalize(
            inflight.step,
            result,
            step_id=inflight.step_id,
            base_version=inflight.base_version,
        )

    def _ledger_io_tick(self) -> None:
        store = self.ledger.store
        if not hasattr(store, "try_reap_flush"):
            return
        if store.try_reap_flush():
            self._flush_stall_count = 0
            store.begin_flush()
        elif store.has_flush_pending():
            self._flush_stall_count += 1
            if self._flush_stall_count >= self.config.flush_stall_threshold:
                raise RuntimeError(
                    f"ledger flush stalled {self._flush_stall_count} steps "
                    f"(threshold={self.config.flush_stall_threshold})"
                )
        elif store.queue_pending():
            store.begin_flush()

    def _ledger_io_begin_flush(self) -> None:
        store = self.ledger.store
        if hasattr(store, "begin_flush"):
            store.begin_flush()

    @profile
    def run_step_finalize(
        self,
        step: StepInput,
        result: TrainStepResult,
        *,
        step_id: int,
        base_version: int,
    ) -> float:
        """Record step docs on the ledger (main-thread only, strict FIFO)."""
        if __debug__:
            if result.step_id != step_id:
                raise RuntimeError(
                    f"finalize step_id mismatch: expected={step_id}, result={result.step_id}"
                )
            if self.ledger.version != base_version:
                raise RuntimeError(
                    f"ledger version order violation: expected base={base_version}, "
                    f"actual={self.ledger.version}"
                )

        if not result.weights_applied:
            self.session.apply_step(result, step.lr)
        new_version = base_version + 1
        self.ledger.version = new_version
        optimizer_t = int(getattr(self.session.model.optimizer, "t", new_version))

        verdict = VERDICT_HEALTHY
        is_local_best = (
            step.val_loss is not None
            and self.config.checkpoint_on_local_best
            and step.val_loss < self.ledger._best_val_loss
        )

        self.ledger.push_step_complete(
            step_id=step_id,
            base_version=base_version,
            batch_ref=step.batch_ref,
            lr=step.lr,
            m_samples=int(step.y.shape[0]),
            result=result,
            version=new_version,
            optimizer_t=optimizer_t,
            train_loss=result.loss,
            val_loss=step.val_loss,
            verdict=verdict,
        )

        if verdict == VERDICT_HEALTHY:
            self._last_healthy_version = new_version

        self._steps_since_checkpoint += 1
        periodic = self._steps_since_checkpoint >= self.config.checkpoint_every_steps
        if periodic or is_local_best:
            self.ledger.push_checkpoint(
                self.session.model,
                version=new_version,
                val_loss=step.val_loss,
                is_local_best=is_local_best,
            )
            self._steps_since_checkpoint = 0

        return result.loss

    def run_steps(self, steps: list[StepInput]) -> list[float]:
        if not self._async_contract:
            return [self.run_step(s) for s in steps]

        idx = 0

        def _next() -> StepInput | None:
            nonlocal idx
            if idx >= len(steps):
                return None
            step = steps[idx]
            idx += 1
            return step

        return self.run_training_loop(steps_budget=len(steps), next_step=_next)

    def fork_from_last_healthy(
        self,
        new_branch_id: str,
        reason: str,
        settings_delta: dict[str, Any] | None = None,
        restore_model: bool = True,
    ) -> TrainingEngine:
        target = self._last_healthy_version
        if restore_model and target > 0:
            self.ledger.restore_checkpoint(self.session.model, target)
        if self.config.checkpoint_on_fork and target > 0:
            cp = self.ledger.store.get_checkpoint(self.ledger.branch_id, target)
            if cp is None:
                self.ledger.push_checkpoint(self.session.model, version=target)
        child_ledger = self.ledger.fork_branch(new_branch_id, target, reason, settings_delta)
        return TrainingEngine(session=self.session, ledger=child_ledger, config=self.config)

    def on_fit_start(self) -> None:
        """Ledger lifecycle: v0 checkpoint + optional contract path."""
        logging.info(
            "[TrainingEngine] Ledger active: branch=%s path=%s version=%d",
            self.ledger.branch_id,
            getattr(self.ledger.store, "root", "?"),
            self.ledger.version,
        )
        if self.ledger.version == 0:
            self.ledger.push_checkpoint(self.session.model, version=0)
        if self.config.contract_list_enabled and hasattr(self.session.model, "enable_contract_list"):
            self.session.model.enable_contract_list()
            rt = getattr(self.session.model, "_contract_runtime", None)
            if rt is not None and rt._async_enabled:
                rt.set_engine_driven(True)
                self._async_contract = True
                logging.info("[TrainingEngine] Contract-list async path enabled")
            else:
                logging.info("[TrainingEngine] Contract-list training path enabled")

    def on_fit_end(self) -> None:
        """Drain pending native work, block until journal durable, close store."""
        self.drain_pending()
        from src.contract_runtime import shutdown_contract_async

        shutdown_contract_async()
        store = self.ledger.store
        flush = getattr(store, "flush", None)
        if flush is not None:
            flush()
        close = getattr(store, "close", None)
        if close is not None:
            close()

    def save_best_checkpoint(self, val_loss: float | None) -> int:
        version = self.ledger.version
        self.ledger.push_checkpoint(
            self.session.model,
            version=version,
            val_loss=val_loss,
            is_local_best=True,
        )
        return version

    def restore_best_checkpoint(self, version: int) -> None:
        self.ledger.restore_checkpoint(self.session.model, version)

    def on_early_stopping_improved(self, epoch: int, val_loss: float | None) -> int:
        version = self.save_best_checkpoint(val_loss)
        logging.info(
            "[Early Stopping] New best epoch %d → ledger checkpoint version=%d",
            epoch,
            version,
        )
        return version

    def on_early_stopping_triggered(self, epoch: int, best_epoch: int, best_version: int) -> None:
        logging.info(
            "[Early Stopping] Validation divergence at epoch %d. "
            "Restoring best checkpoint from epoch %d.",
            epoch,
            best_epoch,
        )
        self.restore_best_checkpoint(best_version)
        logging.info(
            "[Early Stopping] Restored weights from ledger version=%s",
            best_version,
        )


def create_training_engine(
    session: TrainingSession,
    ledger_dir: str,
    *,
    branch_id: str = "main",
    model_instance_id: str = "default",
    architecture_id: str = "unknown",
    config: LedgerConfig | None = None,
) -> TrainingEngine:
    """Open ledger store and return a TrainingEngine bound to session."""
    from pathlib import Path

    from src.ledger_store import create_ledger_store

    root = Path(ledger_dir)
    root.mkdir(parents=True, exist_ok=True)
    backend = getattr(config, "store_backend", "file_streaming") if config else "file_streaming"
    store = create_ledger_store(backend, root)
    impl = type(store).__name__
    if impl == "SyncFileLedgerStore":
        logging.warning(
            "[TrainingEngine] Ledger store_backend=%r uses SyncFileLedgerStore (fsync every push). "
            "Use file_streaming for overlapped journal I/O.",
            backend,
        )
    else:
        writer = getattr(getattr(store, "_writer", None), "__class__", type(None)).__name__
        logging.info(
            "[TrainingEngine] Ledger store backend=%s impl=%s writer=%s",
            backend,
            impl,
            writer,
        )
    ledger = TrainingLedger(
        store=store,
        branch_id=branch_id,
        model_instance_id=model_instance_id,
        architecture_id=architecture_id,
    )
    return TrainingEngine(session=session, ledger=ledger, config=config)
