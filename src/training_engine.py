# src/training_engine.py
"""Single-thread training loop: train → ledger documents → consolidate → checkpoint."""
from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from src.ledger import BatchRef, LedgerConfig, TrainingLedger, VERDICT_HEALTHY
from src.training_session import SessionStatus, TrainStepResult, TrainingSession

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


@dataclass
class _CommittedStep:
    inflight: _InflightStep
    version: int


# ---------------------------------------------------------------------------
# Async training pipeline state machine (run_training_loop).
# Each phase owns its transition; the driver is only `while state: state=...`.
# ---------------------------------------------------------------------------


@dataclass
class _TrainLoopContext:
    engine: "TrainingEngine"
    session: "TrainingSession"
    steps_budget: int
    next_step: Callable[[], StepInput | None]
    on_submitted: Callable[[StepInput], None] | None = None
    pending: StepInput | None = None
    exhausted: bool = False
    committed: _CommittedStep | None = None


class _TrainLoopState(ABC):
    """One pipeline phase; return the next phase or None when the epoch is done."""

    @abstractmethod
    def advance(self, ctx: _TrainLoopContext) -> Optional["_TrainLoopState"]:
        raise NotImplementedError


class _Bootstrap(_TrainLoopState):
    """Prime prefetch and submit the first contract step."""

    def advance(self, ctx: _TrainLoopContext) -> Optional[_TrainLoopState]:
        eng = ctx.engine
        sess = ctx.session
        sess._epoch_losses = []
        sess._epoch_steps_done = 0
        sess._flag_step_done = False
        sess._flag_capacity = True
        sess._prefetch.clear()
        ctx.pending = None
        ctx.exhausted = False
        ctx.committed = None

        ctx.pending, ctx.exhausted, _ = eng._arm_one(
            sess,
            ctx.next_step,
            exhausted=ctx.exhausted,
            steps_budget=ctx.steps_budget,
            pending=None,
        )
        if ctx.pending is None:
            return None
        if not eng.try_submit(ctx.pending, session=sess):
            raise RuntimeError("initial async contract submit was rejected")
        sess._flag_capacity = False
        if ctx.on_submitted is not None:
            ctx.on_submitted(ctx.pending)
        ctx.pending = None
        return _ARM


class _Arm(_TrainLoopState):
    """While a step is in flight, arm the next StepInput (prefetch / producer)."""

    def advance(self, ctx: _TrainLoopContext) -> Optional[_TrainLoopState]:
        if ctx.session._inflight is None:
            return None
        ctx.pending, ctx.exhausted, _ = ctx.engine._arm_one(
            ctx.session,
            ctx.next_step,
            exhausted=ctx.exhausted,
            steps_budget=ctx.steps_budget,
            pending=ctx.pending,
        )
        return _PREPARE


class _Prepare(_TrainLoopState):
    """Prepare the inactive slot for the armed next step (overlap with native)."""

    def advance(self, ctx: _TrainLoopContext) -> Optional[_TrainLoopState]:
        if ctx.pending is not None:
            ctx.engine.prepare_submit(ctx.pending, session=ctx.session)
        return _OVERLAP


class _Overlap(_TrainLoopState):
    """Main-thread useful work: deferred, ledger flush, contract ensure."""

    def advance(self, ctx: _TrainLoopContext) -> Optional[_TrainLoopState]:
        ctx.engine._do_useful_work(session=ctx.session)
        return _WAIT


class _Wait(_TrainLoopState):
    """Block only for native completion; then hand off to commit."""

    def advance(self, ctx: _TrainLoopContext) -> Optional[_TrainLoopState]:
        eng = ctx.engine
        sess = ctx.session
        eng._wait_native_done(sess)
        if not eng._native_ready(sess):
            raise RuntimeError("native wait returned without a completed contract")
        return _COMMIT


class _Commit(_TrainLoopState):
    """Advance ledger version for the finished inflight step."""

    def advance(self, ctx: _TrainLoopContext) -> Optional[_TrainLoopState]:
        eng = ctx.engine
        sess = ctx.session
        finished = sess._inflight
        sess._inflight = None
        if finished is None:
            raise RuntimeError("commit without an inflight step")
        ctx.committed = eng._commit_completed_version(finished)
        sess._flag_step_done = False
        return _SUBMIT


class _Submit(_TrainLoopState):
    """Submit the prepared next step now that capacity is free."""

    def advance(self, ctx: _TrainLoopContext) -> Optional[_TrainLoopState]:
        eng = ctx.engine
        sess = ctx.session
        if ctx.pending is not None:
            if not eng.try_submit(ctx.pending, session=sess):
                raise RuntimeError("prepared async contract submit was rejected")
            sess._flag_capacity = False
            if ctx.on_submitted is not None:
                ctx.on_submitted(ctx.pending)
            ctx.pending = None
        return _FINALIZE


class _Finalize(_TrainLoopState):
    """Reap grads, ledger step docs, optional flush; then arm again or stop."""

    def advance(self, ctx: _TrainLoopContext) -> Optional[_TrainLoopState]:
        eng = ctx.engine
        sess = ctx.session
        committed = ctx.committed
        ctx.committed = None
        if committed is None:
            raise RuntimeError("finalize without a committed step")
        loss = eng._finalize_committed(committed, session=sess)
        sess._epoch_losses.append(loss)
        sess._epoch_steps_done += 1
        if eng._ledger_needs_work():
            eng._ledger_io_begin_flush()
        return _ARM


_BOOTSTRAP = _Bootstrap()
_ARM = _Arm()
_PREPARE = _Prepare()
_OVERLAP = _Overlap()
_WAIT = _Wait()
_COMMIT = _Commit()
_SUBMIT = _Submit()
_FINALIZE = _Finalize()


class TrainingEngine:
    """
    Long-lived training manager: one engine, many sessions on a shared ledger.

    Sessions are PENDING until activated; run() round-robins ACTIVE sessions
    one epoch at a time, then drops finished sessions (resume by start_session
    with the same session_id).

    Async train pipeline (per session): Bootstrap → Arm → Prepare → Overlap →
    Wait → Commit → Submit → Finalize → Arm… until no inflight work remains.
    """

    def __init__(
        self,
        ledger: TrainingLedger,
        config: LedgerConfig | None = None,
        session: TrainingSession | None = None,
    ):
        self.sessions: list[TrainingSession] = []
        self._current_session: TrainingSession | None = None
        self.ledger = ledger
        self.config = config or LedgerConfig()
        self._flush_stall_count = 0
        self._sessions_started = 0
        self._prefetch_depth = int(getattr(config, "prefetch_depth", 4) or 4)
        self._ledger_lock = threading.RLock()
        self._deferred: deque[Callable[[], None]] = deque()
        if session is not None:
            self._register_session(session, activate=True)

    @property
    def session(self) -> TrainingSession | None:
        """Backward-compat: last/current session (prefer get_session / sessions)."""
        return self._current_session

    @session.setter
    def session(self, value: TrainingSession | None) -> None:
        self._current_session = value
        if value is not None and value not in self.sessions:
            self._register_session(value, activate=True)

    def _register_session(
        self, session: TrainingSession, *, activate: bool
    ) -> TrainingSession:
        if session in self.sessions:
            if activate:
                session.status = SessionStatus.ACTIVE
            self._current_session = session
            return session
        session.engine = self
        session.status = SessionStatus.ACTIVE if activate else SessionStatus.PENDING
        self.sessions.append(session)
        self._current_session = session
        self._sessions_started += 1
        return session

    def _require_session(self, session: TrainingSession | None = None) -> TrainingSession:
        sess = session if session is not None else self._current_session
        if sess is None:
            raise RuntimeError("TrainingEngine requires an active session")
        return sess

    def get_session(self, session_id: str) -> TrainingSession | None:
        for s in self.sessions:
            if s.session_id == session_id:
                return s
        return None

    def activate_session(self, session_id: str) -> TrainingSession:
        sess = self.get_session(session_id)
        if sess is None:
            raise KeyError(f"unknown session_id={session_id!r}")
        sess.status = SessionStatus.ACTIVE
        self._current_session = sess
        return sess

    def drop_session(self, session_id: str) -> TrainingSession | None:
        """Remove a finished (or cancelled) session from the live list."""
        sess = self.get_session(session_id)
        if sess is None:
            return None
        sess.status = SessionStatus.FINISHED
        self.sessions = [s for s in self.sessions if s.session_id != session_id]
        if self._current_session is sess:
            self._current_session = self.sessions[-1] if self.sessions else None
        logging.info("[TrainingEngine] Dropped session %s", session_id)
        return sess

    def end_session(
        self,
        session_id: str,
        *,
        finalize: bool = True,
        close_runtime: bool = True,
    ) -> tuple[list[float], list[float]]:
        """
        Stop a live session: drain inflight work, optionally write fit summary,
        close its contract runtime, and drop it from the engine list.

        Returns (train_history, val_history). Same as finish_session.
        """
        sess = self.get_session(session_id)
        if sess is None:
            raise KeyError(f"unknown session_id={session_id!r}")
        self._current_session = sess
        try:
            self.drain_pending(sess)
        except Exception:
            logging.exception(
                "[TrainingEngine] drain_pending failed while ending session %s",
                session_id,
            )
        if finalize and sess._fit_ready and not sess._fit_done:
            sess.finish_fit()
        elif not sess._fit_done:
            sess._fit_done = True
        if close_runtime:
            rt = getattr(sess.model, "_contract_runtime", None)
            if rt is not None and hasattr(rt, "close"):
                rt.close()
        hist = (list(sess.train_history), list(sess.val_history))
        sess.detach_model_ownership()
        self.drop_session(session_id)
        return hist

    def finish_session(
        self,
        session_id: str,
        *,
        finalize: bool = True,
        close_runtime: bool = True,
    ) -> tuple[list[float], list[float]]:
        """Alias for end_session."""
        return self.end_session(
            session_id, finalize=finalize, close_runtime=close_runtime
        )

    def resume_session(
        self,
        *,
        session_id: str,
        model: Any,
        data_provider: Any,
        initial_lr: float,
        scheduler: Any = None,
        predict_fn: Callable[..., Any] | None = None,
        version: int | None = None,
        replay_to_head: bool = False,
        activate: bool = True,
        **fit_kwargs: Any,
    ) -> TrainingSession:
        """
        Re-attach a prior session_id: restore weights from that session's checkpoint,
        then register again on this engine (shared ledger head unchanged).

        If replay_to_head=True, apply later step.complete docs for this session after
        the checkpoint up to session_head_version.
        """
        if self.get_session(session_id) is not None:
            raise RuntimeError(
                f"session_id={session_id!r} is already live; end_session first"
            )
        cp = self.ledger.restore_session_checkpoint(
            model, session_id, version=version
        )
        cp_ver = int(
            cp.version if cp.version is not None else cp.body.get("version", 0)
        )
        session = self.start_session(
            model=model,
            data_provider=data_provider,
            initial_lr=initial_lr,
            scheduler=scheduler,
            predict_fn=predict_fn,
            session_id=session_id,
            activate=activate,
            steps_completed=0,
            **fit_kwargs,
        )
        session._last_healthy_version = cp_ver
        if replay_to_head:
            head = self.ledger.session_head_version(session_id)
            if head > cp_ver:
                self.ledger.replay_session_apply_from_version(
                    session,
                    session_id,
                    from_version=cp_ver,
                    to_version=head,
                    default_lr=initial_lr,
                )
        logging.info(
            "[TrainingEngine] Resumed session %s from checkpoint v=%s (replay_to_head=%s)",
            session_id,
            cp_ver,
            replay_to_head,
        )
        return session

    def create_session(
        self,
        *,
        model: Any,
        data_provider: Any,
        initial_lr: float,
        scheduler: Any = None,
        predict_fn: Callable[..., Any] | None = None,
        steps_completed: int = 0,
        session_id: str | None = None,
    ) -> TrainingSession:
        """Create a new TrainingSession (not yet registered)."""
        session = TrainingSession(
            model=model,
            data_provider=data_provider,
            initial_lr=initial_lr,
            scheduler=scheduler,
            predict_fn=predict_fn,
            session_id=session_id,
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
        session_id: str | None = None,
        activate: bool = True,
        **fit_kwargs: Any,
    ) -> TrainingSession:
        """
        Register a session on this engine (does not block on fit).

        Pass activate=False to leave it PENDING until activate_session / run(activate_pending=True).
        Fit kwargs (steps, model_type, …) are stored for engine.run().
        Resume later with the same session_id so ledger docs tag correctly.
        """
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
                session_id=session_id,
            )
        elif session_id is not None:
            session.session_id = session_id
        session._fit_kwargs = dict(fit_kwargs)
        self._register_session(session, activate=activate)
        logging.info(
            "[TrainingEngine] Session %s registered status=%s (total=%d)",
            session.session_id,
            session.status.value,
            len(self.sessions),
        )
        return session

    def run(
        self, *, activate_pending: bool = False
    ) -> dict[str, tuple[list[float], list[float]]]:
        """
        Drive all ACTIVE sessions to completion (round-robin one epoch each).

        Finished sessions are dropped; results keyed by session_id.
        """
        if activate_pending:
            for s in self.sessions:
                if s.status == SessionStatus.PENDING:
                    s.status = SessionStatus.ACTIVE

        results: dict[str, tuple[list[float], list[float]]] = {}
        for s in self.sessions:
            if s.status != SessionStatus.ACTIVE:
                continue
            if not s._fit_ready and s._fit_kwargs:
                self._begin_registered_fit(s)

        while True:
            progressed = False
            for s in list(self.sessions):
                if s.status != SessionStatus.ACTIVE:
                    continue
                if s._fit_done:
                    results[s.session_id] = (list(s.train_history), list(s.val_history))
                    self.drop_session(s.session_id)
                    progressed = True
                    continue
                if not s._fit_ready:
                    continue
                self._current_session = s
                more = s.advance_fit_epoch()
                progressed = True
                if not more:
                    results[s.session_id] = (list(s.train_history), list(s.val_history))
                    self.drop_session(s.session_id)
            if not progressed:
                break
        return results

    def _begin_registered_fit(self, session: TrainingSession) -> None:
        kw = dict(session._fit_kwargs)
        required = ("steps", "source_mode", "model_type")
        missing = [k for k in required if k not in kw]
        if missing:
            raise ValueError(
                f"session {session.session_id}: start_session missing fit kwargs {missing}"
            )
        session.begin_fit(engine=self, **kw)

    def defer(self, fn: Callable[[], None]) -> None:
        """Queue work to run during useful-work pass (e.g. contract build)."""
        self._deferred.append(fn)

    def _do_useful_work(
        self,
        *,
        session: TrainingSession | None = None,
        fill_next: Callable[[], bool] | None = None,
    ) -> bool:
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
        did = self._ensure_contract_ready(session) or did
        return did

    def _ensure_contract_ready(self, session: TrainingSession | None = None) -> bool:
        """Compile/enable contract list if configured and not yet ready."""
        if not self.config.contract_list_enabled:
            return False
        sess = session if session is not None else self._current_session
        if sess is None:
            return False
        model = sess.model
        if not hasattr(model, "enable_contract_list"):
            return False
        if getattr(model, "_contract_runtime", None) is not None:
            return False
        model.enable_contract_list(native_async_submit=self.config.native_async_submit)
        return True

    def _fill_prefetch(
        self,
        session: TrainingSession,
        next_step: Callable[[], StepInput | None],
        *,
        exhausted: bool,
        steps_budget: int,
    ) -> tuple[bool, bool]:
        """Stage next StepInput(s) while native runs. Bind happens only at submit."""
        did = False
        steps_done = session._epoch_steps_done
        in_flight = 1 if session._inflight is not None else 0
        queued = len(session._prefetch)
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
            session._prefetch.append(step)
            queued += 1
            did = True
        return did, exhausted

    def _arm_one(
        self,
        session: TrainingSession,
        next_step: Callable[[], StepInput | None],
        *,
        exhausted: bool,
        steps_budget: int,
        pending: StepInput | None,
    ) -> tuple[StepInput | None, bool, bool]:
        """Ensure one armed step ready to submit. Returns (pending, exhausted, did)."""
        if pending is not None:
            return pending, exhausted, False
        if session._prefetch:
            return session._prefetch.popleft(), exhausted, True
        did, exhausted = self._fill_prefetch(
            session, next_step, exhausted=exhausted, steps_budget=steps_budget
        )
        if session._prefetch:
            return session._prefetch.popleft(), exhausted, True
        return None, exhausted, did

    def _wait_native_done(self, session: TrainingSession) -> None:
        """Only wait point: native completion (after all useful work is done)."""
        if not session._async_contract:
            return
        if session._flag_step_done or self._native_ready(session):
            return
        rt = self._contract_runtime(session)
        if rt is None:
            return
        rt.wait_for_completion()

    @property
    def version(self) -> int:
        return self.ledger.version

    def uses_async_contract(self, session: TrainingSession | None = None) -> bool:
        sess = session if session is not None else self._current_session
        return bool(sess is not None and sess._async_contract)

    def has_pending(self, session: TrainingSession | None = None) -> bool:
        sess = self._require_session(session)
        if sess._inflight is not None or sess._flag_step_done:
            return True
        rt = getattr(sess.model, "_contract_runtime", None)
        if rt is None:
            return False
        return rt.native_in_flight() or rt.has_completed()

    def can_submit(self, session: TrainingSession | None = None) -> bool:
        sess = self._require_session(session)
        return sess._inflight is None and not self._native_in_flight(sess)

    def _ledger_needs_work(self) -> bool:
        store = self.ledger.store
        if not hasattr(store, "has_flush_pending"):
            return False
        return store.has_flush_pending() or store.queue_pending()

    def _contract_runtime(self, session: TrainingSession | None = None) -> Any | None:
        sess = session if session is not None else self._current_session
        if sess is None:
            return None
        return getattr(sess.model, "_contract_runtime", None)

    def _completion_signaled(self, session: TrainingSession) -> bool:
        rt = self._contract_runtime(session)
        return rt is not None and rt.completion_signaled()

    def _native_ready(self, session: TrainingSession) -> bool:
        return session._inflight is not None and self._completion_signaled(session)

    def on_contract_step_done(self) -> None:
        """Legacy: set flag on current session only."""
        if self._current_session is not None:
            self._current_session.on_contract_step_done()

    def on_capacity(self) -> None:
        if self._current_session is not None:
            self._current_session.on_capacity()

    def _finalize_if_ready(self, session: TrainingSession | None = None) -> float | None:
        sess = self._require_session(session)
        if not self._native_ready(sess):
            return None
        with self._ledger_lock:
            self._ledger_io_tick()
            loss = self._try_finalize_inflight(sess)
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
    def tick(self, session: TrainingSession | None = None) -> float | None:
        """Ledger maintenance pass — NOT a training batch."""
        sess = self._require_session(session)
        if sess._flag_step_done or self._native_ready(sess):
            loss = self._finalize_if_ready(sess)
            sess._flag_step_done = False
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
        session: TrainingSession | None = None,
    ) -> list[float]:
        """Two-slot pipeline driven by phase state classes (arm→prep→wait→…)."""
        sess = self._require_session(session)
        ctx = _TrainLoopContext(
            engine=self,
            session=sess,
            steps_budget=steps_budget,
            next_step=next_step,
            on_submitted=on_submitted,
        )
        state: _TrainLoopState | None = _BOOTSTRAP
        while state is not None:
            state = state.advance(ctx)
        return list(sess._epoch_losses)

    def drain_pending(self, session: TrainingSession | None = None) -> list[float]:
        """Drain one externally submitted async step."""
        sess = self._require_session(session)
        if not sess._async_contract:
            return list(sess._epoch_losses)
        while sess._inflight is not None:
            self._do_useful_work(session=sess)
            self._wait_native_done(sess)
            if not self._native_ready(sess):
                continue
            finished = sess._inflight
            sess._inflight = None
            committed = self._commit_completed_version(finished)
            loss = self._finalize_committed(committed, session=sess)
            sess._flag_step_done = False
            sess._epoch_losses.append(loss)
            sess._epoch_steps_done += 1
        return list(sess._epoch_losses)

    def try_submit(
        self, step: StepInput, *, session: TrainingSession | None = None
    ) -> bool:
        """Ask model contract to accept step. Returns False if BUSY."""
        sess = self._require_session(session)
        if not self.can_submit(sess):
            return False
        if sess.contract_busy():
            return False
        prepared = sess._prepared
        if prepared is not None:
            if prepared.step is not step:
                raise RuntimeError("prepared engine step does not match submitted batch")
            step_id = prepared.step_id
            base_version = prepared.base_version
        else:
            step_id = sess.reserve_step_id()
            base_version = self.ledger.version
        if not sess.submit_contract_step(
            step.X, step.y, step.lr, step_id=step_id, apply_adam=True
        ):
            return False
        sess._inflight = _InflightStep(step=step, step_id=step_id, base_version=base_version)
        sess._prepared = None
        return True

    def prepare_submit(
        self, step: StepInput, *, session: TrainingSession | None = None
    ) -> bool:
        """Prepare a future submit in the inactive runtime slot."""
        sess = self._require_session(session)
        if sess._prepared is not None:
            return sess._prepared.step is step
        step_id = sess.reserve_step_id()
        base_version = (
            sess._inflight.base_version + 1
            if sess._inflight is not None
            else self.ledger.version
        )
        if not sess.prepare_contract_step(
            step.X, step.y, step.lr, step_id=step_id, apply_adam=True
        ):
            raise RuntimeError("inactive contract slot unavailable during prepare")
        sess._prepared = _InflightStep(step, step_id, base_version)
        return True

    def _commit_completed_version(self, inflight: _InflightStep) -> _CommittedStep:
        if self.ledger.version != inflight.base_version:
            raise RuntimeError(
                f"ledger version order violation: expected base={inflight.base_version}, "
                f"actual={self.ledger.version}"
            )
        version = inflight.base_version + 1
        self.ledger.version = version
        return _CommittedStep(inflight, version)

    def _finalize_committed(
        self, committed: _CommittedStep, *, session: TrainingSession | None = None
    ) -> float:
        sess = self._require_session(session)
        inflight = committed.inflight
        packed = sess.try_reap_contract_step()
        if packed is None:
            raise RuntimeError("native completion disappeared before finalize")
        loss, gw, gb, m, gg, gbb, weights_applied = packed
        result = sess._pack_train_step_result(
            inflight.step_id,
            loss,
            gw,
            gb,
            m,
            gg,
            gbb,
            weights_applied=weights_applied,
            gradients_owned=True,
        )
        return self.run_step_finalize(
            inflight.step,
            result,
            step_id=inflight.step_id,
            base_version=inflight.base_version,
            committed_version=committed.version,
            session=sess,
        )

    @profile
    def run_step(
        self, step: StepInput, *, session: TrainingSession | None = None
    ) -> float:
        """Sync path (non-contract or tests): one blocking step."""
        sess = self._require_session(session)
        if sess._async_contract:
            raise RuntimeError("async contract path requires tick/submit loop, not run_step")

        self._ledger_io_tick()
        step_id = sess.reserve_step_id()
        base_version = self.ledger.version
        result = sess.train_step(step.X, step.y, step.lr, step_id=step_id)
        loss = self.run_step_finalize(
            step, result, step_id=step_id, base_version=base_version, session=sess
        )
        self._ledger_io_begin_flush()
        return loss

    def _native_in_flight(self, session: TrainingSession | None = None) -> bool:
        sess = self._require_session(session)
        rt = getattr(sess.model, "_contract_runtime", None)
        if rt is None:
            return False
        return rt.native_in_flight()

    def _try_finalize_inflight(self, session: TrainingSession | None = None) -> float | None:
        sess = self._require_session(session)
        if sess._inflight is None:
            return None
        packed = sess.try_reap_contract_step()
        if packed is None:
            return None
        inflight = sess._inflight
        sess._inflight = None
        loss, gw, gb, m, gg, gbb, weights_applied = packed
        result = sess._pack_train_step_result(
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
            session=sess,
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
        committed_version: int | None = None,
        session: TrainingSession | None = None,
    ) -> float:
        """Record step docs on the ledger (strict FIFO; serialized with RLock)."""
        with self._ledger_lock:
            return self._run_step_finalize_locked(
                step,
                result,
                step_id=step_id,
                base_version=base_version,
                committed_version=committed_version,
                session=self._require_session(session),
            )

    def _run_step_finalize_locked(
        self,
        step: StepInput,
        result: TrainStepResult,
        *,
        step_id: int,
        base_version: int,
        committed_version: int | None = None,
        session: TrainingSession,
    ) -> float:
        if __debug__:
            if result.step_id != step_id:
                raise RuntimeError(
                    f"finalize step_id mismatch: expected={step_id}, result={result.step_id}"
                )
            if committed_version is not None and self.ledger.version != committed_version:
                raise RuntimeError(
                    f"ledger version order violation: expected={committed_version}, "
                    f"actual={self.ledger.version}"
                )

        if not result.weights_applied:
            session.apply_step(result, step.lr)
        if committed_version is None:
            # Allocate under lock so concurrent sessions sharing one ledger cannot
            # both snapshot the same base_version before either commits.
            base_version = self.ledger.version
            new_version = base_version + 1
            self.ledger.version = new_version
        else:
            new_version = base_version + 1
            if committed_version != new_version:
                raise RuntimeError(
                    f"committed version mismatch: expected={new_version}, "
                    f"actual={committed_version}"
                )
            if self.ledger.version != committed_version:
                raise RuntimeError(
                    f"ledger version order violation: expected={committed_version}, "
                    f"actual={self.ledger.version}"
                )
        optimizer_t = int(getattr(session.model.optimizer, "t", new_version))

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
            model_instance_id=session.session_id,
        )

        if verdict == VERDICT_HEALTHY:
            session._last_healthy_version = new_version

        session._steps_since_checkpoint += 1
        periodic = session._steps_since_checkpoint >= self.config.checkpoint_every_steps
        if periodic or is_local_best:
            self.ledger.push_checkpoint(
                session.model,
                version=new_version,
                val_loss=step.val_loss,
                is_local_best=is_local_best,
                model_instance_id=session.session_id,
            )
            session._steps_since_checkpoint = 0

        return result.loss

    def run_steps(
        self,
        steps: list[StepInput],
        *,
        session: TrainingSession | None = None,
    ) -> list[float]:
        sess = self._require_session(session)
        if not sess._async_contract:
            return [self.run_step(s, session=sess) for s in steps]

        idx = 0

        def _next() -> StepInput | None:
            nonlocal idx
            if idx >= len(steps):
                return None
            step = steps[idx]
            idx += 1
            return step

        return self.run_training_loop(
            session=sess, steps_budget=len(steps), next_step=_next
        )

    def fork_from_last_healthy(
        self,
        new_branch_id: str,
        reason: str,
        settings_delta: dict[str, Any] | None = None,
        restore_model: bool = True,
        session: TrainingSession | None = None,
    ) -> TrainingEngine:
        sess = self._require_session(session)
        target = sess._last_healthy_version
        if restore_model and target > 0:
            self.ledger.restore_checkpoint(sess.model, target)
        if self.config.checkpoint_on_fork and target > 0:
            cp = self.ledger.store.get_checkpoint(self.ledger.branch_id, target)
            if cp is None:
                self.ledger.push_checkpoint(
                    sess.model, version=target, model_instance_id=sess.session_id
                )
        child_ledger = self.ledger.fork_branch(new_branch_id, target, reason, settings_delta)
        return TrainingEngine(ledger=child_ledger, config=self.config, session=sess)

    def on_fit_start(self, session: TrainingSession | None = None) -> None:
        """Ledger lifecycle: v0 checkpoint + optional contract path."""
        sess = self._require_session(session)
        self._current_session = sess
        logging.info(
            "[TrainingEngine] Ledger active: branch=%s path=%s version=%d session=%s",
            self.ledger.branch_id,
            getattr(self.ledger.store, "root", "?"),
            self.ledger.version,
            sess.session_id,
        )
        if self.ledger.version == 0:
            with self._ledger_lock:
                self.ledger.push_checkpoint(
                    sess.model, version=0, model_instance_id=sess.session_id
                )
        if self.config.contract_list_enabled and hasattr(sess.model, "enable_contract_list"):
            sess.model.enable_contract_list(
                native_async_submit=self.config.native_async_submit
            )
            rt = getattr(sess.model, "_contract_runtime", None)
            if rt is not None and rt._async_enabled:
                rt.set_engine_driven(True)
                rt.subscribe_completion(sess.on_contract_step_done)
                rt.subscribe_capacity(sess.on_capacity)
                sess._flag_capacity = True
                sess._async_contract = True
                logging.info(
                    "[TrainingEngine] Contract-list async path enabled (session=%s)",
                    sess.session_id,
                )
            else:
                logging.info(
                    "[TrainingEngine] Contract-list training path enabled (session=%s)",
                    sess.session_id,
                )

    def on_fit_end(self, session: TrainingSession | None = None) -> None:
        """End of one session: drain inflight work (engine stays alive for more sessions)."""
        sess = self._require_session(session)
        self.drain_pending(sess)

    def close(self) -> None:
        """Tear down native workers and ledger store (call when engine is done)."""
        for sess in list(self.sessions):
            try:
                self.drain_pending(sess)
            except Exception:
                logging.exception(
                    "[TrainingEngine] drain_pending failed for session %s", sess.session_id
                )
            rt = getattr(sess.model, "_contract_runtime", None)
            if rt is not None and hasattr(rt, "close"):
                rt.close()
        from src.contract_runtime import shutdown_contract_async

        shutdown_contract_async()
        store = self.ledger.store
        flush = getattr(store, "flush", None)
        if flush is not None:
            flush()
        close = getattr(store, "close", None)
        if close is not None:
            close()

    def save_best_checkpoint(
        self, val_loss: float | None, session: TrainingSession | None = None
    ) -> int:
        sess = self._require_session(session)
        version = self.ledger.version
        with self._ledger_lock:
            self.ledger.push_checkpoint(
                sess.model,
                version=version,
                val_loss=val_loss,
                is_local_best=True,
                model_instance_id=sess.session_id,
            )
        return version

    def restore_best_checkpoint(
        self, version: int, session: TrainingSession | None = None
    ) -> None:
        sess = self._require_session(session)
        with self._ledger_lock:
            self.ledger.restore_checkpoint(sess.model, version)

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
