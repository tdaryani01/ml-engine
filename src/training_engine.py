# src/training_engine.py
"""Single-thread training loop: train → ledger documents → consolidate → checkpoint."""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from src.ledger import BatchRef, LedgerConfig, TrainingLedger, VERDICT_HEALTHY
from src.manager_heartbeat import ManagerCommand, ManagerHeartbeat, decode_checkpoint_blob
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
        manager_heartbeat: ManagerHeartbeat | None = None,
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
        self._manager_heartbeat = manager_heartbeat
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._work_poll: Callable[[], bool] | None = None
        # Optional train step when no TrainingSession (e.g. closed-loop config).
        self._external_step: Callable[[], bool] | None = None
        # Optional control hooks for config-specific start/pause/restore.
        self._on_start_resume: Callable[[Any], bool] | None = None
        self._on_pause: Callable[[Any], None] | None = None
        self._on_cancel: Callable[[Any], None] | None = None
        self._on_restore: Callable[[Any], bool] | None = None
        self._pause_gate: Callable[[], bool] | None = None
        # BL-007d: configure from job on claim; reset when lease drops.
        self._on_claim_config: Callable[[dict[str, Any]], bool] | None = None
        self._on_release_config: Callable[[], None] | None = None
        self._restore_inflight: dict[str, threading.Thread] = {}
        self._restore_ready: dict[str, Any] = {}
        self._restore_lock = threading.Lock()
        self._last_status_line: str | None = None
        self._restored_checkpoint_version: int | None = None
        # After fit completes sessions are dropped; keep the model so TM restore
        # can still apply weights while parked paused.
        self._held_model_for_restore: Any | None = None
        # Only an explicit start/resume (or claim/adopt) in *this* process
        # authorizes run — boot never auto-trains.
        self._run_authorized: bool = False
        # Redelivered pre-boot shutdown must not kill a new process.
        self._process_started_at: float = time.time()
        self._shutdown_accepted: bool = False
        # BL-006c: True while we hold a claimed work lease (cleared on release/ack).
        self._work_lease_active: bool = False
        if session is not None:
            self._register_session(session, activate=True)

    def request_stop(self) -> None:
        """Wake the idle park loop and exit ``run`` on the next check."""
        self._stop.set()
        self._emit_engine_status("stopped")

    def request_pause(self) -> None:
        self._paused.set()
        self._publish_manager_metrics("paused")
        self._emit_engine_status("paused")

    def request_resume(self) -> None:
        self._paused.clear()
        self._publish_manager_metrics("training")
        self._emit_engine_status("training")

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

    def _emit_engine_status(
        self,
        state: str,
        *,
        force: bool = False,
        note: str | None = None,
    ) -> None:
        """Print a compact run-state line (status changes only). Details stay in logs.

        Worker model is binary: running/training vs paused (idle ≡ paused, no claim).
        """
        if state == "idle":
            state = "paused"
        ckpt = int(self.ledger.version)
        loss = None
        sess = self._current_session
        if sess is not None and sess._epoch_losses:
            loss = float(sess._epoch_losses[-1])
        parts = [f"status={state}", f"checkpoint=v{ckpt}"]
        if self._restored_checkpoint_version is not None:
            parts.append(f"restored_from=v{self._restored_checkpoint_version}")
        if loss is not None:
            parts.append(f"loss={loss:.4f}")
        if note:
            parts.append(note)
        line = "[Engine] " + " ".join(parts)
        if not force and line == self._last_status_line:
            return
        self._last_status_line = line
        print(line, flush=True)
        logging.info("%s", line)

    def _publish_manager_metrics(self, state: str, session: TrainingSession | None = None) -> None:
        hb = self._manager_heartbeat
        if hb is None:
            return
        # BL-026: while a train job is leased, metrics.state is a worker *output*
        # (agent trip/progress). Engine idle/pause loops must not overwrite
        # es-onset with idle (board idle + pool busy; TM never steers).
        if bool(getattr(hb, "job_bound", False)):
            return
        # Once training advances past a restore point, stop advertising it as tip.
        if (
            self._restored_checkpoint_version is not None
            and int(self.ledger.version) > int(self._restored_checkpoint_version)
        ):
            self._restored_checkpoint_version = None
        sess = session if session is not None else self._current_session
        metrics: dict[str, Any] = {
            "state": state,
            "checkpoint_version": int(self.ledger.version),
        }
        if self._restored_checkpoint_version is not None:
            metrics["restored_from_checkpoint"] = int(
                self._restored_checkpoint_version
            )
        if sess is not None and sess._epoch_losses:
            metrics["loss"] = float(sess._epoch_losses[-1])
        hb.set_metrics(metrics)

    def set_work_poll(self, poll: Callable[[], bool] | None) -> None:
        """Optional hook: return True when external work was queued (wake idle)."""
        self._work_poll = poll

    def set_external_step(self, step: Callable[[], bool] | None) -> None:
        """Optional train tick when this process has no supervised sessions.

        Called from the main ``run()`` loop while authorized and not paused.
        Return True if a step ran (keep training cadence).
        """
        self._external_step = step

    def set_control_hooks(
        self,
        *,
        on_start_resume: Callable[[Any], bool] | None = None,
        on_pause: Callable[[Any], None] | None = None,
        on_cancel: Callable[[Any], None] | None = None,
        on_restore: Callable[[Any], bool] | None = None,
        pause_gate: Callable[[], bool] | None = None,
        on_claim_config: Callable[[dict[str, Any]], bool] | None = None,
        on_release_config: Callable[[], None] | None = None,
    ) -> None:
        """Config-specific control extras (not a second claim/HB loop).

        on_start_resume: return False to block authorize (e.g. user_paused).
        on_restore: return True if fully handled (skip ledger restore).
        pause_gate: when True, block train/claim (human Pause / site-interrupt).
        on_claim_config: apply job.config before authorize; False aborts claim.
        on_release_config: reset in-process config when lease returns to pool.
        """
        self._on_start_resume = on_start_resume
        self._on_pause = on_pause
        self._on_cancel = on_cancel
        self._on_restore = on_restore
        self._pause_gate = pause_gate
        self._on_claim_config = on_claim_config
        self._on_release_config = on_release_config

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
        if getattr(session, "model", None) is not None:
            self._held_model_for_restore = session.model
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
        if self._manager_heartbeat is not None and getattr(sess, "model", None) is not None:
            self._held_model_for_restore = sess.model
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
        if self._manager_heartbeat is not None and getattr(sess, "model", None) is not None:
            self._held_model_for_restore = sess.model
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
        logging.debug(
            "[TrainingEngine] Session %s registered status=%s (total=%d)",
            session.session_id,
            session.status.value,
            len(self.sessions),
        )
        return session

    def _ledger_model_instance_id(self, session: TrainingSession | None = None) -> str:
        """Durable TM agent id (job.model_id). Never the pool worker / session UUID."""
        mid = str(getattr(self.ledger, "model_instance_id", "") or "").strip()
        if mid and mid != "unbound":
            return mid
        if session is not None:
            return str(session.session_id)
        return mid or "unbound"

    def adopt_claimed_job(self, job: dict[str, Any]) -> None:
        """Authorize a lease already claimed by the outer pool-worker loop."""
        mid = str(job.get("model_id") or "").strip()
        if mid and getattr(self, "ledger", None) is not None:
            self.ledger.model_instance_id = mid
        self._work_lease_active = True
        self._run_authorized = True
        hb = self._manager_heartbeat
        self.request_resume()
        logging.info(
            "[TrainingEngine] adopted job=%s model=%s pool=%s",
            job.get("job_id"),
            mid or job.get("model_id"),
            getattr(hb, "pool_session_id", None) if hb is not None else None,
        )
        self._publish_manager_metrics("training")
        self._emit_engine_status("training", force=True, note="work_adopted")

    def run(
        self,
        *,
        activate_pending: bool = False,
        park_when_idle: bool | None = None,
        job_scoped: bool = False,
    ) -> dict[str, tuple[list[float], list[float]]]:
        """
        Drive ACTIVE sessions to completion (round-robin one epoch each).

        Finished sessions are dropped; results keyed by session_id.

        When a Training Manager heartbeat client is attached, the engine
        **parks paused** until ``start`` / ``resume`` (or claim/adopt)
        authorizes run in this process. Pause / restore / shutdown are
        honored while parked.

        ``job_scoped=True``: lease was claimed outside (pool worker). Do not
        re-claim; stay in the lease until released/stopped (park within the
        job — one False ``external_step`` must not ack the whole train job).
        """
        hb = self._manager_heartbeat
        if job_scoped:
            # Hold the lease: idle/pause sleep inside the job, do not exit.
            park = True
        elif hb is not None:
            # TM-managed: always park; never auto-train on boot.
            park = True
        elif park_when_idle is None:
            park = False
        else:
            park = bool(park_when_idle)

        if activate_pending:
            for s in self.sessions:
                if s.status == SessionStatus.PENDING:
                    s.status = SessionStatus.ACTIVE

        results: dict[str, tuple[list[float], list[float]]] = {}
        self._stop.clear()
        boot_state = "paused" if hb is not None and not job_scoped else "training"
        if job_scoped and self._run_authorized:
            boot_state = "training"
        self._maybe_manager_heartbeat(state=boot_state)
        self._emit_engine_status(boot_state, force=True)

        while not self._stop.is_set():
            self.drain_manager_commands(allow_restore=True)
            self._maybe_truncate_imported_journal()
            self._sync_work_lease()
            if job_scoped and not self._work_lease_active:
                break

            if self._stop.is_set():
                break

            if not self._manager_allows_training():
                if job_scoped and not self._work_lease_active:
                    break
                if not job_scoped and self._maybe_claim_tm_work():
                    continue
                state = "paused"
                self._maybe_manager_heartbeat(state=state)
                self._emit_engine_status(state)
                sleep_s = self._idle_sleep_s()
                if self._stop.wait(timeout=max(0.1, sleep_s)):
                    break
                continue

            if self._paused.is_set():
                if job_scoped and not self._work_lease_active:
                    break
                self._maybe_manager_heartbeat(state="paused")
                self._emit_engine_status("paused")
                sleep_s = self._idle_sleep_s()
                if self._stop.wait(timeout=max(0.1, sleep_s)):
                    break
                continue

            self._arm_pending_fits()
            progressed = self._drive_active_epochs(results)
            if not progressed and self._external_step is not None:
                try:
                    progressed = bool(self._external_step())
                except Exception:
                    logging.exception("[TrainingEngine] external_step failed")
                    progressed = False
            if progressed:
                self.drain_manager_commands(allow_restore=False)
                self._maybe_manager_heartbeat(state="training")
                self._emit_engine_status("training")
                continue

            if not park:
                break

            # Park within lease (or long-lived park): sleep, wake, check work.
            sleep_s = self._idle_sleep_s()
            if self._stop.wait(timeout=max(0.1, sleep_s)):
                break
            self.drain_manager_commands(allow_restore=True)
            self._sync_work_lease()
            if job_scoped and not self._work_lease_active:
                break
            if self._paused.is_set() or not self._manager_allows_training():
                continue
            if not job_scoped and self._check_for_work():
                self._maybe_manager_heartbeat(state="training")
                self._emit_engine_status("training")
                continue
            self._idle_heartbeat()
            self._emit_engine_status("paused")
        return results

    def _idle_sleep_s(self) -> float:
        hb = self._manager_heartbeat
        if hb is None:
            return 10.0
        direct = getattr(hb, "idle_sleep_s", None)
        if direct is not None:
            return float(direct)
        cfg = getattr(hb, "_cfg", None) or getattr(hb, "cfg", None)
        return float(getattr(cfg, "idle_sleep_s", 10.0) if cfg is not None else 10.0)

    def _manager_allows_training(self) -> bool:
        """Without TM, always allow. With TM: authorized and not pause_gate."""
        hb = self._manager_heartbeat
        if hb is None:
            return True
        if not self._run_authorized:
            return False
        if self._pause_gate is not None:
            try:
                if bool(self._pause_gate()):
                    return False
            except Exception:
                logging.exception("[TrainingEngine] pause_gate failed")
                return False
        return True

    def drain_manager_commands(self, *, allow_restore: bool) -> None:
        """Apply TM commands at a safe point. Never blocks on network I/O."""
        hb = self._manager_heartbeat
        if hb is None:
            return
        self._finish_ready_restores()
        self._maybe_auto_restore_from_manager(allow_restore=allow_restore)
        for cmd in hb.poll_commands():
            action = cmd.action.strip().lower()
            try:
                if action in ("start", "resume"):
                    if self._on_start_resume is not None:
                        try:
                            ok = self._on_start_resume(cmd)
                        except Exception as exc:  # noqa: BLE001
                            hb.mark_command_seen(cmd.id)
                            hb.queue_ack(cmd.id, ok=False, detail=str(exc))
                            continue
                        if ok is False:
                            hb.mark_command_seen(cmd.id)
                            hb.queue_ack(
                                cmd.id, ok=False, detail="blocked:start_resume_hook"
                            )
                            continue
                    self._run_authorized = True
                    self.request_resume()
                    hb.mark_command_seen(cmd.id)
                    hb.queue_ack(cmd.id, ok=True)
                elif action == "pause":
                    if self._on_pause is not None:
                        try:
                            self._on_pause(cmd)
                        except Exception:
                            logging.exception("[TrainingEngine] on_pause hook failed")
                    self.request_pause()
                    hb.mark_command_seen(cmd.id)
                    hb.queue_ack(cmd.id, ok=True)
                elif action == "cancel":
                    if self._on_cancel is not None:
                        try:
                            self._on_cancel(cmd)
                        except Exception:
                            logging.exception("[TrainingEngine] on_cancel hook failed")
                    self._run_authorized = False
                    was_leased = self._work_lease_active
                    self._work_lease_active = False
                    self._paused.clear()
                    if was_leased and self._on_release_config is not None:
                        try:
                            self._on_release_config()
                        except Exception:
                            logging.exception(
                                "[TrainingEngine] on_release_config on cancel failed"
                            )
                    self._publish_manager_metrics("paused")
                    self._emit_engine_status("paused", force=True, note="start_cancelled")
                    hb.mark_command_seen(cmd.id)
                    hb.queue_ack(cmd.id, ok=True)
                elif action == "shutdown":
                    # Redelivered pre-boot shutdowns: ack and ignore.
                    created = float(cmd.created_at or 0.0)
                    if created > 0.0 and created < (self._process_started_at - 2.0):
                        logging.warning(
                            "[TrainingEngine] ignoring stale shutdown id=%s "
                            "created_at=%.3f process_started=%.3f",
                            cmd.id,
                            created,
                            self._process_started_at,
                        )
                        hb.mark_command_seen(cmd.id)
                        hb.queue_ack(
                            cmd.id, ok=True, detail="ignored_stale_pre_boot_shutdown"
                        )
                        continue
                    self._shutdown_accepted = True
                    # Pause before exiting.
                    if not self._paused.is_set():
                        self._paused.set()
                        self._publish_manager_metrics("paused")
                        self._emit_engine_status("paused", note="shutdown_pause")
                    self.request_stop()
                    hb.mark_command_seen(cmd.id)
                    hb.queue_ack(cmd.id, ok=True)
                elif action == "restore":
                    if self._on_restore is not None:
                        try:
                            handled = bool(self._on_restore(cmd))
                        except Exception as exc:  # noqa: BLE001
                            hb.mark_command_seen(cmd.id)
                            hb.queue_ack(cmd.id, ok=False, detail=str(exc))
                            continue
                        if handled:
                            continue
                    # Restore is weight sync — hold locally if mid-train drain.
                    if not self._paused.is_set() and not allow_restore:
                        self._paused.set()
                    self._start_restore_async(cmd)
                else:
                    hb.mark_command_seen(cmd.id)
                    hb.queue_ack(cmd.id, ok=False, detail=f"unknown action {action}")
            except Exception as exc:  # noqa: BLE001
                logging.exception("[TrainingEngine] command %s failed", cmd.id)
                hb.mark_command_seen(cmd.id)
                hb.queue_ack(cmd.id, ok=False, detail=str(exc))

    def _maybe_auto_restore_from_manager(self, *, allow_restore: bool) -> None:
        """Boot-only: if local ledger is still v0 and TM advertises a blob, pull it.

        Never fights an explicit restore or a settled local tip — that caused
        1150↔1250 restore loops when ``active_checkpoint`` was stale across HBs.
        Skip when a config ``on_restore`` hook owns restore (e.g. closed-loop draw).
        """
        hb = self._manager_heartbeat
        if hb is None or not allow_restore:
            return
        if self._on_restore is not None:
            return
        if int(self.ledger.version) > 0:
            return
        if self._restored_checkpoint_version is not None:
            return
        with self._restore_lock:
            if self._restore_inflight or self._restore_ready:
                return
        active = hb.active_checkpoint
        if not isinstance(active, dict):
            return
        blob_key = active.get("blob_key")
        version = active.get("version")
        if not blob_key or version is None:
            return
        try:
            version_i = int(version)
        except (TypeError, ValueError):
            return
        if version_i <= 0:
            return
        fake = ManagerCommand(
            id=f"auto-restore-v{version_i}",
            action="restore",
            payload={"version": version_i, "blob_key": str(blob_key)},
            created_at=0.0,
        )
        self._start_restore_async(fake)

    def _start_restore_async(self, cmd: ManagerCommand) -> None:
        hb = self._manager_heartbeat
        if hb is None:
            return
        blob_key = cmd.payload.get("blob_key")
        if not blob_key:
            hb.mark_command_seen(cmd.id)
            if not str(cmd.id).startswith("auto-restore-"):
                hb.queue_ack(cmd.id, ok=False, detail="missing blob_key")
            return
        # One restore at a time — overlapping 1150/1250 was the ping-pong.
        with self._restore_lock:
            busy = bool(self._restore_inflight or self._restore_ready)
        if busy:
            if str(cmd.id).startswith("auto-restore-"):
                return
            # Explicit: leave unseen so a later drain can retry after current finishes.
            return
        # Silent hold only — never advertise "paused" for restore (desired stays as-is).
        self._paused.set()
        with self._restore_lock:
            if cmd.id in self._restore_inflight or cmd.id in self._restore_ready:
                return
            hb.mark_command_seen(cmd.id)

            def _worker() -> None:
                try:
                    data = hb.fetch_blob(str(blob_key))
                    if data is None:
                        raise RuntimeError(f"blob not found: {blob_key}")
                    with self._restore_lock:
                        self._restore_ready[cmd.id] = (cmd, data)
                except Exception as exc:  # noqa: BLE001
                    with self._restore_lock:
                        self._restore_ready[cmd.id] = (cmd, exc)
                finally:
                    with self._restore_lock:
                        self._restore_inflight.pop(cmd.id, None)

            t = threading.Thread(
                target=_worker, name=f"tm-restore-{cmd.id[:8]}", daemon=True
            )
            self._restore_inflight[cmd.id] = t
            t.start()

    def _recover_after_restore_hold(self) -> None:
        """Clear silent restore hold; re-align pause from authorize + pause_gate."""
        gate = False
        if self._pause_gate is not None:
            try:
                gate = bool(self._pause_gate())
            except Exception:
                gate = False
        if gate:
            self._paused.set()
            self._publish_manager_metrics("paused")
            self._emit_engine_status("paused", force=True, note="after_restore_hold")
        elif self._run_authorized:
            self._paused.clear()
            state = "training" if self._manager_allows_training() else "paused"
            self._publish_manager_metrics(state)
            self._emit_engine_status(state, force=True, note="after_restore_hold")
        else:
            self._paused.set()
            self._publish_manager_metrics("paused")
            self._emit_engine_status("paused", force=True, note="after_restore_hold")

    def _finish_ready_restores(self) -> None:
        hb = self._manager_heartbeat
        if hb is None:
            return
        with self._restore_lock:
            ready = dict(self._restore_ready)
            self._restore_ready.clear()
        for cmd_id, packed in ready.items():
            cmd, payload = packed
            auto = str(cmd_id).startswith("auto-restore-")
            if isinstance(payload, BaseException):
                if not auto:
                    hb.queue_ack(cmd_id, ok=False, detail=str(payload))
                fail = f"[Engine] restore FAILED: {payload}"
                print(fail, flush=True)
                logging.error("%s", fail)
                self._recover_after_restore_hold()
                continue
            try:
                body = decode_checkpoint_blob(payload)
                sess = self._current_session
                if sess is None and self.sessions:
                    sess = self.sessions[0]
                model = None if sess is None else getattr(sess, "model", None)
                if model is None:
                    model = self._held_model_for_restore
                if model is None:
                    raise RuntimeError(
                        "no session/model to restore into "
                        "(train once or keep a parked session before restore)"
                    )
                from src.ledger import restore_model_checkpoint

                restore_model_checkpoint(model, body)
                version = cmd.payload.get("version")
                version_i: int | None = None
                if version is not None:
                    try:
                        version_i = int(version)
                        self.ledger.version = version_i
                    except (TypeError, ValueError):
                        version_i = None
                if version_i is not None:
                    self._restored_checkpoint_version = version_i
                blob_key = cmd.payload.get("blob_key")
                report = (
                    f"[Engine] RESTORED to checkpoint v{version_i if version_i is not None else version}"
                    f" (blob={blob_key})"
                )
                print(report, flush=True)
                logging.info("%s", report)
                # Pin local active tip so a stale HB active cannot immediately
                # auto-restore a different version on the next drain.
                if version_i is not None and blob_key:
                    hb.set_active_checkpoint(
                        {"version": version_i, "blob_key": str(blob_key)}
                    )
                if not auto:
                    hb.queue_ack(cmd_id, ok=True, detail=f"restored version={version}")
                self._recover_after_restore_hold()
            except Exception as exc:  # noqa: BLE001
                logging.exception("[TrainingEngine] restore apply failed")
                fail = f"[Engine] restore FAILED: {exc}"
                print(fail, flush=True)
                if not auto:
                    hb.queue_ack(cmd_id, ok=False, detail=str(exc))
                self._recover_after_restore_hold()

    def _maybe_truncate_imported_journal(self) -> None:
        """Ask the file store to roll an already-imported journal prefix (if any)."""
        store = getattr(self.ledger, "store", None)
        if store is None:
            return
        roll = getattr(store, "maybe_drop_imported_prefix", None)
        if not callable(roll):
            return
        try:
            through = roll()
        except OSError:
            logging.exception("[TrainingEngine] journal prefix roll failed")
            return
        if through is not None:
            logging.debug(
                "[TrainingEngine] journal prefix rolled through=%s", through
            )

    def _arm_pending_fits(self) -> None:
        for s in self.sessions:
            if s.status != SessionStatus.ACTIVE:
                continue
            if not s._fit_ready and s._fit_kwargs:
                self._begin_registered_fit(s)

    def _drive_active_epochs(
        self, results: dict[str, tuple[list[float], list[float]]]
    ) -> bool:
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
        return progressed

    def _sync_work_lease(self) -> None:
        """If TM parked/cancelled our job (release_job), drop local authorization."""
        hb = self._manager_heartbeat
        if hb is None or not self._work_lease_active:
            return
        if bool(getattr(hb, "job_bound", False)):
            return
        self._work_lease_active = False
        self._run_authorized = False
        site_hold = bool(getattr(hb, "site_interrupt_hold", False))
        gate = False
        if self._pause_gate is not None:
            try:
                gate = bool(self._pause_gate())
            except Exception:
                gate = False
        if site_hold or gate:
            # BL-023 / user pause: look paused, not cleared-for-claim.
            if not self._paused.is_set():
                self._paused.set()
            self._publish_manager_metrics("paused")
            self._emit_engine_status("paused", force=True, note="work_lease_released_paused")
            if self._on_release_config is not None:
                try:
                    self._on_release_config()
                except Exception:
                    logging.exception("[TrainingEngine] on_release_config failed")
            logging.info(
                "[TrainingEngine] work lease released — staying paused (site/user hold)"
            )
            return
        if self._paused.is_set():
            self._paused.clear()
        if self._on_release_config is not None:
            try:
                self._on_release_config()
            except Exception:
                logging.exception("[TrainingEngine] on_release_config failed")
        self._publish_manager_metrics("paused")
        self._emit_engine_status("paused", force=True, note="work_lease_released")
        logging.info("[TrainingEngine] work lease released — parking paused")

    def _maybe_claim_tm_work(self) -> bool:
        """Idle → claim any queued train job (worker = capacity; BL-006c)."""
        hb = self._manager_heartbeat
        claim_fn = getattr(hb, "try_claim_work", None) if hb is not None else None
        if hb is None or claim_fn is None:
            return False
        if bool(getattr(hb, "job_bound", False)):
            return False
        if self._paused.is_set() or self._stop.is_set():
            return False
        if self._run_authorized:
            return False
        if self._pause_gate is not None and self._pause_gate():
            return False
        try:
            job = claim_fn()
        except Exception:
            logging.exception("[TrainingEngine] try_claim_work failed")
            return False
        if not job:
            return False
        if self._on_claim_config is not None:
            try:
                ok = bool(self._on_claim_config(dict(job)))
            except Exception:
                logging.exception("[TrainingEngine] on_claim_config failed")
                ok = False
            if not ok:
                fail = getattr(hb, "fail_work", None)
                if callable(fail):
                    try:
                        fail(error="claim_config_rejected")
                    except Exception:
                        logging.exception("[TrainingEngine] fail_work after config reject")
                else:
                    unbind = getattr(hb, "unbind_job", None)
                    if callable(unbind):
                        unbind()
                return False
        mid = str(job.get("model_id") or "").strip()
        if mid and getattr(self, "ledger", None) is not None:
            self.ledger.model_instance_id = mid
        self._work_lease_active = True
        self._run_authorized = True
        self.request_resume()
        logging.info(
            "[TrainingEngine] claimed job=%s model=%s kind=%s pool=%s",
            job.get("job_id"),
            mid or job.get("model_id"),
            job.get("kind"),
            getattr(hb, "pool_session_id", None),
        )
        try:
            from src.manager_heartbeat import _diag

            _diag(
                "engine_claim_authorized",
                job_id=job.get("job_id"),
                model_id=mid,
                kind=job.get("kind"),
                pool=getattr(hb, "pool_session_id", None),
                ledger_model=getattr(self.ledger, "model_instance_id", None),
                has_on_claim_config=self._on_claim_config is not None,
                has_external_step=self._external_step is not None,
            )
        except Exception:  # noqa: BLE001
            pass
        self._publish_manager_metrics("training")
        self._emit_engine_status("training", force=True, note="work_claimed")
        return True

    def _check_for_work(self) -> bool:
        """True if there is (or was just accepted) training work to drive."""
        self._sync_work_lease()
        if self._maybe_claim_tm_work():
            return True
        if self._work_poll is not None:
            try:
                if self._work_poll():
                    return True
            except Exception:
                logging.exception("[TrainingEngine] work_poll failed")
        for s in self.sessions:
            if s.status == SessionStatus.PENDING and s._fit_kwargs:
                s.status = SessionStatus.ACTIVE
                return True
            if s.status == SessionStatus.ACTIVE and (
                s._fit_ready or (not s._fit_done and s._fit_kwargs)
            ):
                return True
        return False

    def _idle_heartbeat(self) -> None:
        hb = self._manager_heartbeat
        if hb is None:
            return
        self._publish_manager_metrics("paused")
        hb.maybe_ping(force=True)

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
        self.drain_manager_commands(allow_restore=False)
        did = self._maybe_manager_heartbeat(session, state="training") or did
        return did

    def _maybe_manager_heartbeat(
        self,
        session: TrainingSession | None = None,
        *,
        state: str = "training",
    ) -> bool:
        hb = self._manager_heartbeat
        if hb is None:
            return False
        effective = "paused" if self._paused.is_set() else state
        self._publish_manager_metrics(effective, session)
        return hb.maybe_ping()

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
                self._maybe_manager_heartbeat(sess, state="training")
                return loss
        self._service_ledger()
        self._maybe_manager_heartbeat(sess, state="training")
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
        self._maybe_manager_heartbeat(sess, state="training")
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
            model_instance_id=self._ledger_model_instance_id(session),
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
                model_instance_id=self._ledger_model_instance_id(session),
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
                    sess.model,
                    version=target,
                    model_instance_id=self._ledger_model_instance_id(sess),
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
                    sess.model,
                    version=0,
                    model_instance_id=self._ledger_model_instance_id(sess),
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
                model_instance_id=self._ledger_model_instance_id(sess),
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
    manager_heartbeat: ManagerHeartbeat | None = None,
    store_kwargs: dict[str, Any] | None = None,
) -> TrainingEngine:
    """Open ledger store and return a TrainingEngine (session optional / attached later)."""
    from pathlib import Path

    from src.ledger_store import create_ledger_store

    root = Path(ledger_dir)
    root.mkdir(parents=True, exist_ok=True)
    backend = getattr(config, "store_backend", "file_streaming") if config else "file_streaming"
    store = create_ledger_store(backend, root, **(store_kwargs or {}))
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
    return TrainingEngine(
        ledger=ledger,
        config=config,
        session=session,
        manager_heartbeat=manager_heartbeat,
    )
