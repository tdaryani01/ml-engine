# src/training_engine.py
"""Single-thread training loop: train → ledger documents → consolidate → checkpoint."""
from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
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


class TrainingEngine:
    """
    Long-lived training manager: one engine, many sessions.

    Train machine: submit → arm next StepInput → ledger push/flush only if
    work pending → wait solely for native completion. Bind happens at submit
    (single shared native ctx cannot bind while a job is in flight).
    """

    def __init__(
        self,
        ledger: TrainingLedger,
        config: LedgerConfig | None = None,
        session: TrainingSession | None = None,
    ):
        self.session = session
        self.ledger = ledger
        self.config = config or LedgerConfig()
        self._steps_since_checkpoint = 0
        self._last_healthy_version = 0
        self._flush_stall_count = 0
        self._inflight: _InflightStep | None = None
        self._async_contract = False
        self._epoch_losses: list[float] = []
        self._epoch_steps_done = 0
        self._flag_step_done = False
        self._flag_capacity = True
        self._sessions_started = 0
        self._prefetch: deque[StepInput] = deque()
        self._prefetch_depth = int(getattr(config, "prefetch_depth", 4) or 4)
        self._ledger_lock = threading.RLock()
        self._deferred: deque[Callable[[], None]] = deque()

    def create_session(
        self,
        *,
        model: Any,
        data_provider: Any,
        initial_lr: float,
        scheduler: Any = None,
        predict_fn: Callable[..., Any] | None = None,
        steps_completed: int = 0,
    ) -> TrainingSession:
        """Create a new TrainingSession owned by this engine."""
        session = TrainingSession(
            model=model,
            data_provider=data_provider,
            initial_lr=initial_lr,
            scheduler=scheduler,
            predict_fn=predict_fn,
        )
        session.steps_completed = steps_completed
        return session

    def start_session(
        self,
        session: TrainingSession | None = None,
        *,
        model: Any | None = None,
        data_provider: Any | None = None,
        initial_lr: float | None = None,
        scheduler: Any = None,
        predict_fn: Callable[..., Any] | None = None,
        steps_completed: int = 0,
        **fit_kwargs: Any,
    ) -> tuple[list[float], list[float]]:
        """Create/bind a session and run fit on the calling thread."""
        if session is None:
            if model is None or data_provider is None or initial_lr is None:
                raise ValueError(
                    "start_session requires session= or model+data_provider+initial_lr"
                )
            session = self.create_session(
                model=model,
                data_provider=data_provider,
                initial_lr=initial_lr,
                scheduler=scheduler,
                predict_fn=predict_fn,
                steps_completed=steps_completed,
            )

        self.session = session
        self._sessions_started += 1
        logging.info(
            "[TrainingEngine] Session %d started (caller thread)",
            self._sessions_started,
        )
        return session.fit(engine=self, **fit_kwargs)

    def defer(self, fn: Callable[[], None]) -> None:
        """Queue work to run during useful-work pass (e.g. contract build)."""
        self._deferred.append(fn)

    def _do_useful_work(self, *, fill_next: Callable[[], bool] | None = None) -> bool:
        """Do non-wait work only: arm next / deferred / flush if pending. No sleep."""
        did = False
        if fill_next is not None:
            did = fill_next() or did
        while self._deferred:
            fn = self._deferred.popleft()
            fn()
            did = True
        if self._ledger_needs_work():
            self._service_ledger()
            did = True
        did = self._ensure_contract_ready() or did
        return did

    def _ensure_contract_ready(self) -> bool:
        """Compile/enable contract list if configured and not yet ready."""
        if not self.config.contract_list_enabled or self.session is None:
            return False
        model = self.session.model
        if not hasattr(model, "enable_contract_list"):
            return False
        if getattr(model, "_contract_runtime", None) is not None:
            return False
        model.enable_contract_list()
        return True

    def _fill_prefetch(
        self,
        next_step: Callable[[], StepInput | None],
        *,
        exhausted: bool,
        steps_budget: int,
    ) -> tuple[bool, bool]:
        """Stage next StepInput(s) while native runs. Bind happens only at submit."""
        did = False
        steps_done = self._epoch_steps_done
        in_flight = 1 if self._inflight is not None else 0
        queued = len(self._prefetch)
        while (
            not exhausted
            and queued < self._prefetch_depth
            and steps_done + in_flight + queued < steps_budget
        ):
            step = next_step()
            if step is None:
                return did, True
            step.X = np.ascontiguousarray(step.X)
            step.y = np.ascontiguousarray(step.y)
            self._prefetch.append(step)
            queued += 1
            did = True
        return did, exhausted

    def _arm_one(
        self,
        next_step: Callable[[], StepInput | None],
        *,
        exhausted: bool,
        steps_budget: int,
        pending: StepInput | None,
    ) -> tuple[StepInput | None, bool, bool]:
        """Ensure one armed step ready to submit. Returns (pending, exhausted, did)."""
        if pending is not None:
            return pending, exhausted, False
        if self._prefetch:
            return self._prefetch.popleft(), exhausted, True
        did, exhausted = self._fill_prefetch(
            next_step, exhausted=exhausted, steps_budget=steps_budget
        )
        if self._prefetch:
            return self._prefetch.popleft(), exhausted, True
        return None, exhausted, did

    def _wait_native_done(self) -> None:
        """Only wait point: native completion (after all useful work is done)."""
        if self._flag_step_done or self._native_ready():
            return
        rt = self._contract_runtime()
        if rt is None:
            return
        rt.wait_for_completion()

    @property
    def version(self) -> int:
        return self.ledger.version

    def uses_async_contract(self) -> bool:
        return self._async_contract

    def has_pending(self) -> bool:
        if self._inflight is not None or self._flag_step_done:
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

    def on_contract_step_done(self) -> None:
        """Native completion on-event — set flag only."""
        self._flag_step_done = True

    def on_capacity(self) -> None:
        """Capacity on-event — slot free, producer may submit."""
        self._flag_capacity = True

    def _finalize_if_ready(self) -> float | None:
        if not self._native_ready():
            return None
        with self._ledger_lock:
            self._ledger_io_tick()
            loss = self._try_finalize_inflight()
            if self._ledger_needs_work():
                self._ledger_io_begin_flush()
            return loss

    def _service_ledger(self) -> None:
        if not self._ledger_needs_work():
            return
        with self._ledger_lock:
            if not self._ledger_needs_work():
                return
            self._ledger_io_tick()
            if self._ledger_needs_work():
                self._ledger_io_begin_flush()

    @profile
    def tick(self) -> float | None:
        """Ledger maintenance pass — NOT a training batch."""
        if self._flag_step_done or self._native_ready():
            loss = self._finalize_if_ready()
            self._flag_step_done = False
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
        """
        Train machine: submit → arm next → push/flush if work → wait native only.

        Next StepInput is staged while native runs (single ctx cannot bind ahead).
        Bind+submit happens the instant capacity returns.
        """
        self._epoch_losses = []
        self._epoch_steps_done = 0
        self._flag_step_done = False
        self._flag_capacity = True
        self._prefetch.clear()
        pending: StepInput | None = None
        exhausted = False

        while self._epoch_steps_done < steps_budget:
            pending, exhausted, _ = self._arm_one(
                next_step,
                exhausted=exhausted,
                steps_budget=steps_budget,
                pending=pending,
            )
            if pending is None:
                break

            if not self.can_submit() or self.session.contract_busy():
                self._do_useful_work()
                self._wait_native_done()
                if self._flag_step_done or self._native_ready():
                    loss = self._finalize_if_ready()
                    self._flag_step_done = False
                    if loss is not None:
                        self._epoch_losses.append(loss)
                        self._epoch_steps_done += 1
                continue

            if not self.try_submit(pending):
                self._flag_capacity = False
                self._do_useful_work()
                self._wait_native_done()
                continue

            self._flag_capacity = False
            if on_submitted is not None:
                on_submitted(pending)
            pending = None

            # Native running: arm next + push previous docs / flush only if pending.
            pending, exhausted, _ = self._arm_one(
                next_step,
                exhausted=exhausted,
                steps_budget=steps_budget,
                pending=pending,
            )
            if self._flag_step_done or self._native_ready():
                loss = self._finalize_if_ready()
                self._flag_step_done = False
                if loss is not None:
                    self._epoch_losses.append(loss)
                    self._epoch_steps_done += 1
            self._do_useful_work()

            # Only wait: native free, with next step already armed when available.
            if self._inflight is not None and not (
                self._flag_step_done or self._native_ready()
            ):
                self._wait_native_done()

            if self._flag_step_done or self._native_ready():
                loss = self._finalize_if_ready()
                self._flag_step_done = False
                if loss is not None:
                    self._epoch_losses.append(loss)
                    self._epoch_steps_done += 1

        self.drain_pending()
        return list(self._epoch_losses)

    def drain_pending(self) -> list[float]:
        """Drain inflight: useful work first, then wait on native only."""
        while self._inflight is not None or self._flag_step_done:
            if self._flag_step_done or self._native_ready():
                loss = self._finalize_if_ready()
                self._flag_step_done = False
                if loss is not None:
                    self._epoch_losses.append(loss)
                    self._epoch_steps_done += 1
                continue
            self._do_useful_work()
            if self._flag_step_done or self._native_ready():
                continue
            self._wait_native_done()
        return list(self._epoch_losses)

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
        """Record step docs on the ledger (strict FIFO; serialized with RLock)."""
        with self._ledger_lock:
            return self._run_step_finalize_locked(
                step, result, step_id=step_id, base_version=base_version
            )

    def _run_step_finalize_locked(
        self,
        step: StepInput,
        result: TrainStepResult,
        *,
        step_id: int,
        base_version: int,
    ) -> float:
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
        return TrainingEngine(ledger=child_ledger, config=self.config, session=self.session)

    def on_fit_start(self) -> None:
        """Ledger lifecycle: v0 checkpoint + optional contract path."""
        if self.session is None:
            raise RuntimeError("on_fit_start requires an active session")
        logging.info(
            "[TrainingEngine] Ledger active: branch=%s path=%s version=%d",
            self.ledger.branch_id,
            getattr(self.ledger.store, "root", "?"),
            self.ledger.version,
        )
        if self.ledger.version == 0:
            with self._ledger_lock:
                self.ledger.push_checkpoint(self.session.model, version=0)
        if self.config.contract_list_enabled and hasattr(self.session.model, "enable_contract_list"):
            self.session.model.enable_contract_list()
            rt = getattr(self.session.model, "_contract_runtime", None)
            if rt is not None and rt._async_enabled:
                rt.set_engine_driven(True)
                rt.subscribe_completion(self.on_contract_step_done)
                rt.subscribe_capacity(self.on_capacity)
                self._flag_capacity = True
                self._async_contract = True
                logging.info("[TrainingEngine] Contract-list async path enabled (flag subscriber)")
            else:
                logging.info("[TrainingEngine] Contract-list training path enabled")

    def on_fit_end(self) -> None:
        """End of one session: drain inflight work (engine stays alive for more sessions)."""
        self.drain_pending()

    def close(self) -> None:
        """Tear down native worker and ledger store (call when engine is done)."""
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
        if self.session is None:
            raise RuntimeError("save_best_checkpoint requires an active session")
        version = self.ledger.version
        with self._ledger_lock:
            self.ledger.push_checkpoint(
                self.session.model,
                version=version,
                val_loss=val_loss,
                is_local_best=True,
            )
        return version

    def restore_best_checkpoint(self, version: int) -> None:
        if self.session is None:
            raise RuntimeError("restore_best_checkpoint requires an active session")
        with self._ledger_lock:
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
    ledger_dir: str,
    *,
    session: TrainingSession | None = None,
    branch_id: str = "main",
    model_instance_id: str = "default",
    architecture_id: str = "unknown",
    config: LedgerConfig | None = None,
) -> TrainingEngine:
    """Open ledger store and return a TrainingEngine (session optional / attached later)."""
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
    return TrainingEngine(ledger=ledger, config=config, session=session)
