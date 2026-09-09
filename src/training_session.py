# src/training_session.py
"""Training session boundary: step results, grad compute vs optimizer apply, fit loop.

Isolation policy:
- One TrainingSession owns exactly one model instance (weights, arena, optimizer).
- Many sessions may run in one process, each with its own model instance
  (same architecture/config is fine; sharing one model object is not).
- Native stage caches / async mailboxes are per-model tenant — concurrent
  different models are supported once each has its own ContractRuntime tenant.
"""
from __future__ import annotations

import copy
import logging
import uuid
import weakref
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from config.constants import DataKeys, ModelType

# model id(model) -> weakref to the live TrainingSession that owns it
_model_session_owners: dict[int, weakref.ReferenceType] = {}


class SessionStatus(str, Enum):
    """Engine scheduling state. Finished sessions are dropped from the engine list."""

    PENDING = "pending"  # registered; not advanced by run() until activated
    ACTIVE = "active"  # run() advances this session
    FINISHED = "finished"  # ended; normally dropped from engine.sessions


@dataclass
class TrainStepResult:
    """Immutable-ish step output suitable for ledger handoff (Phase E)."""

    step_id: int
    loss: float
    grad_weights: List[np.ndarray]
    grad_biases: List[np.ndarray]
    m_samples: int
    grad_gammas: List[np.ndarray] | None = None
    grad_betas: List[np.ndarray] | None = None
    weights_applied: bool = False

    def copy_grads(self) -> TrainStepResult:
        """Return a copy with detached numpy arrays."""
        return TrainStepResult(
            step_id=self.step_id,
            loss=self.loss,
            grad_weights=[np.copy(g) for g in self.grad_weights],
            grad_biases=[np.copy(g) for g in self.grad_biases],
            m_samples=self.m_samples,
            grad_gammas=[np.copy(g) for g in self.grad_gammas] if self.grad_gammas else None,
            grad_betas=[np.copy(g) for g in self.grad_betas] if self.grad_betas else None,
            weights_applied=self.weights_applied,
        )

    def to_bytes(self) -> bytes:
        """E2: raw binary serialize for ledger step.result bodies."""
        from src.ledger_wire import pack_step_result

        return pack_step_result(self)

    @classmethod
    def from_bytes(cls, data: bytes) -> TrainStepResult:
        from src.ledger_wire import unpack_step_result

        return unpack_step_result(data)


class TrainingSession:
    """
    Owns one model's training loop. Computes grads in train_step; applies via apply_step.

    Pipeline hot state (inflight/prefetch/async flags) lives on the session so one
    engine can keep many sessions alive and round-robin them.
    """

    def __init__(
        self,
        model: Any,
        data_provider: Any,
        initial_lr: float = 0.01,
        scheduler: Any = None,
        predict_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        *,
        session_id: str | None = None,
        status: SessionStatus = SessionStatus.PENDING,
    ):
        mid = id(model)
        existing = _model_session_owners.get(mid)
        if existing is not None and existing() is not None:
            raise RuntimeError(
                "TrainingSession: model instance is already owned by another session. "
                "Create a separate model instance per session (shared architecture OK)."
            )
        self.model = model
        self._model_id = mid
        _model_session_owners[mid] = weakref.ref(
            self, lambda _r, _mid=mid: _model_session_owners.pop(_mid, None)
        )
        self.data_provider = data_provider
        self.initial_lr = initial_lr
        self.scheduler = scheduler
        self.predict_fn = predict_fn or model.predict
        self.step_id = 0
        self.steps_completed = 0
        self.train_history: List[float] = []
        self.val_history: List[float] = []
        self.engine: Any = None  # TrainingEngine when ledger enabled
        self.session_id = session_id or str(uuid.uuid4())
        self.status = status
        self._fit_kwargs: Dict[str, Any] = {}
        self._fit_ready = False
        self._fit_done = False
        self._fit_epoch = 0
        self._fit_es_state: Dict[str, Any] | None = None
        self._fit_is_classification = False
        self._fit_X_val = None
        self._fit_y_val = None
        # Per-session async / pipeline state (was on TrainingEngine)
        self._inflight: Any = None
        self._prepared: Any = None
        self._prefetch: deque = deque()
        self._async_contract = False
        self._epoch_losses: List[float] = []
        self._epoch_steps_done = 0
        self._flag_step_done = False
        self._flag_capacity = True
        self._steps_since_checkpoint = 0
        self._last_healthy_version = 0

    def detach_model_ownership(self) -> None:
        """Release the one-session-per-model registry entry (used by end_session)."""
        mid = getattr(self, "_model_id", None)
        if mid is None:
            return
        existing = _model_session_owners.get(mid)
        if existing is not None and existing() is self:
            _model_session_owners.pop(mid, None)

    def on_contract_step_done(self) -> None:
        self._flag_step_done = True

    def on_capacity(self) -> None:
        self._flag_capacity = True

    def reserve_step_id(self) -> int:
        """Allocate the next step_id on the main thread (strict FIFO submit order)."""
        self.step_id += 1
        return self.step_id

    
    def train_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        *,
        step_id: int | None = None,
        tick_fn: Callable[[], None] | None = None,
    ) -> TrainStepResult:
        """Forward + backward grad compute; does not mutate weights."""
        gg: list[np.ndarray] | None = None
        gbb: list[np.ndarray] | None = None
        weights_applied = False
        if getattr(self.model, "contract_list_enabled", False) and hasattr(
            self.model, "run_contract_train_step"
        ):
            loss, gw, gb, m, gg, gbb = self._contract_compute_step(
                X, y, lr, step_id=step_id, tick_fn=tick_fn
            )
            weights_applied = True
        elif hasattr(self.model, "forward_train"):
            _, cache = self.model.forward_train(X)
            loss, gw, gb, m, gg, gbb = self.model._compute_grads_from_cache(cache, y)
        else:
            loss, gw, gb, m, gg, gbb = self.model._compute_grads(X, y)

        if step_id is None:
            self.step_id += 1
            step_id = self.step_id

        return self._pack_train_step_result(
            step_id, loss, gw, gb, m, gg, gbb, weights_applied=weights_applied
        )

    def submit_contract_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        *,
        step_id: int,
        apply_adam: bool = True,
    ) -> bool:
        """Returns True if CNN accepted step, False if BUSY."""
        if not (
            getattr(self.model, "contract_list_enabled", False)
            and hasattr(self.model, "add_training_step")
        ):
            raise RuntimeError("submit_contract_step requires contract-list path")
        return (
            self.model.add_training_step(
                X, y, lr, apply_adam=apply_adam, step_token=step_id
            )
            == "OK"
        )

    def prepare_contract_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        *,
        step_id: int,
        apply_adam: bool = True,
    ) -> bool:
        """Prepare the model's inactive async slot without submitting it."""
        if not hasattr(self.model, "prepare_training_step"):
            return False
        return bool(
            self.model.prepare_training_step(
                X, y, lr, apply_adam=apply_adam, step_token=step_id
            )
        )

    def contract_busy(self) -> bool:
        if hasattr(self.model, "contract_busy"):
            return bool(self.model.contract_busy())
        return False

    def _predict_with_thread_policy(self, X: np.ndarray) -> np.ndarray:
        """Keep async contract's native worker as the only parallel OMP owner."""
        if self.engine is not None and self.engine.uses_async_contract():
            from utils.runtime import current_thread_omp_threads

            with current_thread_omp_threads(1):
                return self.predict_fn(X)
        return self.predict_fn(X)

    def try_reap_contract_step(
        self,
    ) -> tuple[float, list[np.ndarray], list[np.ndarray], int, None, None, bool] | None:
        if not hasattr(self.model, "try_reap_contract_train_step"):
            return None
        out = self.model.try_reap_contract_train_step()
        if out is None:
            return None
        loss, gw, gb, m = out
        return loss, gw, gb, m, None, None, True

    
    def _contract_compute_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        *,
        step_id: int | None = None,
        tick_fn: Callable[[], None] | None = None,
    ) -> tuple[float, list[np.ndarray], list[np.ndarray], int, None, None]:
        loss, gw, gb, m = self.model.run_contract_train_step(
            X, y, lr, apply_adam=True, step_token=step_id
        )
        return loss, gw, gb, m, None, None

    
    def _pack_train_step_result(
        self,
        step_id: int,
        loss: float,
        gw: list[np.ndarray],
        gb: list[np.ndarray],
        m: int,
        gg: list[np.ndarray] | None,
        gbb: list[np.ndarray] | None,
        *,
        weights_applied: bool = False,
        gradients_owned: bool = False,
    ) -> TrainStepResult:
        return TrainStepResult(
            step_id=step_id,
            loss=loss,
            grad_weights=list(gw) if gradients_owned else [np.copy(g) for g in gw],
            grad_biases=list(gb) if gradients_owned else [np.copy(g) for g in gb],
            m_samples=m,
            grad_gammas=[np.copy(g) for g in gg] if gg else None,
            grad_betas=[np.copy(g) for g in gbb] if gbb else None,
            weights_applied=weights_applied,
        )

    
    def apply_step(self, result: TrainStepResult, lr: float) -> None:
        """Apply optimizer update from a prior train_step result."""
        self.model._apply_grads(
            result.grad_weights,
            result.grad_biases,
            result.m_samples,
            lr,
            grad_gammas=result.grad_gammas,
            grad_betas=result.grad_betas,
        )

    def train_and_apply(self, X: np.ndarray, y: np.ndarray, lr: float) -> float:
        """Forward + backward + apply on the same hot path as legacy model.backward()."""
        if getattr(self.model, "contract_list_enabled", False) and hasattr(
            self.model, "run_contract_train_step"
        ):
            loss, gw, gb, m = self.model.run_contract_train_step(X, y, lr, apply_adam=True)
        elif hasattr(self.model, "forward_train"):
            _, cache = self.model.forward_train(X)
            loss, gw, gb, m, gg, gbb = self.model._compute_grads_from_cache(cache, y)
            self.model._apply_grads(gw, gb, m, lr, grad_gammas=gg, grad_betas=gbb)
        else:
            loss, gw, gb, m, gg, gbb = self.model._compute_grads(X, y)
            self.model._apply_grads(gw, gb, m, lr, grad_gammas=gg, grad_betas=gbb)
        self.step_id += 1
        return loss

    def _set_train_batch_caps(self, model_type: ModelType) -> None:
        if model_type != ModelType.CNN:
            return
        if not hasattr(self.model, "set_train_batch_cap"):
            return
        cap = int(getattr(self.data_provider, "batch_size", 32))
        self.model.set_train_batch_cap(cap)
        logging.debug(f"[TrainingSession] Train buffer cap set to N={cap}.")

    def begin_fit(
        self,
        steps: int,
        source_mode: Any,
        model_type: ModelType,
        early_stopping_enabled: bool = True,
        patience: int = 15,
        min_delta: float = 1e-5,
        compute_r2_score: Callable[[np.ndarray, np.ndarray], float] | None = None,
        engine: Any = None,
        max_epochs: int | None = None,
    ) -> None:
        """Prepare epoch state for engine.run() round-robin (does not train yet)."""
        self.engine = engine
        self._fit_kwargs = {
            "steps": steps,
            "source_mode": source_mode,
            "model_type": model_type,
            "early_stopping_enabled": early_stopping_enabled,
            "patience": patience,
            "min_delta": min_delta,
            "compute_r2_score": compute_r2_score,
            "max_epochs": max_epochs,
        }
        self._fit_done = False
        self._fit_epoch = 0
        self._fit_is_classification = model_type in (
            ModelType.BINARY_CLASSIFICATION,
            ModelType.MULTI_CLASS,
            ModelType.CNN,
        )
        self._fit_X_val, self._fit_y_val = self.data_provider.get_validation_set()
        if self._fit_is_classification:
            y_val_target = self._fit_y_val
            val_class_dist = (
                np.sum(y_val_target, axis=0).tolist()
                if hasattr(y_val_target, "ndim") and y_val_target.ndim > 1
                else np.unique(y_val_target, return_counts=True)[1].tolist()
            )
            logging.info(
                f"[Forensic Trace] Static Validation Set Class Distribution: {val_class_dist}"
            )
        if steps <= 0:
            logging.info(
                "[TrainingSession] Steps count set to 0. Skipping training execution loops."
            )
            self._fit_done = True
            self._fit_ready = True
            return
        self._set_train_batch_caps(model_type)
        if hasattr(self.model.optimizer, "_setup_done"):
            self.model.optimizer._setup_done = False
            self.model.optimizer.t = 0
            logging.info(
                "[TrainingSession] Patched Adam state: Tracking vectors cleared for new execution pass."
            )
        self._fit_es_state = {
            "best_val_loss": float("inf"),
            "best_epoch": 0,
            "patience_counter": 0,
            "weights": None,
            "biases": None,
            "best_version": None,
        }
        if self.engine is not None:
            self.engine.on_fit_start(self)
        self._fit_ready = True

    def advance_fit_epoch(self) -> bool:
        """
        Run one training epoch for this session.
        Returns True if more epochs remain; False when fit is complete.
        """
        if self._fit_done:
            return False
        if not self._fit_ready:
            raise RuntimeError("advance_fit_epoch requires begin_fit first")
        kw = self._fit_kwargs
        steps = int(kw["steps"])
        if steps <= 0:
            self._fit_done = True
            return False

        epoch = self._fit_epoch
        active_lr = self.scheduler.step(epoch) if self.scheduler else self.initial_lr
        epoch_train_loss, _ = self._run_epoch_training_pass(
            active_lr, steps, epoch, self._fit_is_classification
        )
        if epoch_train_loss == 0.0:
            self.finish_fit()
            return False
        self.train_history.append(epoch_train_loss)

        X_val, y_val_target = self._fit_X_val, self._fit_y_val
        val_preds = self._predict_with_thread_policy(X_val)
        current_val_loss = self.model.compute_total_loss(val_preds, y_val_target)
        current_val_raw_cost = self.model.calculate_raw_cost(val_preds, y_val_target)
        self.val_history.append(current_val_loss)

        self._evaluate_epoch_performance(
            epoch,
            epoch_train_loss,
            val_preds,
            y_val_target,
            current_val_loss,
            active_lr,
            self._fit_is_classification,
            kw.get("compute_r2_score"),
        )

        es_state = self._fit_es_state or {}
        if kw.get("early_stopping_enabled", True):
            if self._handle_early_stopping(
                epoch,
                current_val_raw_cost,
                float(kw.get("min_delta", 1e-5)),
                int(kw.get("patience", 15)),
                es_state,
                current_val_loss=current_val_loss,
            ):
                self.finish_fit()
                return False
        max_epochs = kw.get("max_epochs")
        if max_epochs is not None and epoch + 1 >= max_epochs:
            self.finish_fit()
            return False
        self._fit_epoch = epoch + 1
        return True

    def finish_fit(self) -> Tuple[List[float], List[float]]:
        """Finalize reports and engine on_fit_end; marks session fit done."""
        if self._fit_done and not self._fit_ready:
            return self.train_history, self.val_history
        kw = self._fit_kwargs
        if self._fit_X_val is not None and kw:
            self._generate_final_summary_report(
                self._fit_X_val,
                self._fit_y_val,
                kw.get("source_mode"),
                self._fit_is_classification,
                kw.get("model_type"),
                self._fit_es_state or {
                    "best_val_loss": float("inf"),
                    "best_epoch": 0,
                },
                bool(kw.get("early_stopping_enabled", True)),
                kw.get("compute_r2_score"),
            )
        if self.engine is not None:
            self.engine.on_fit_end(self)
        self._fit_done = True
        self._fit_ready = False
        return self.train_history, self.val_history

    def fit(
        self,
        steps: int,
        source_mode: Any,
        model_type: ModelType,
        early_stopping_enabled: bool = True,
        patience: int = 15,
        min_delta: float = 1e-5,
        compute_r2_score: Callable[[np.ndarray, np.ndarray], float] | None = None,
        engine: Any = None,
        max_epochs: int | None = None,
    ) -> Tuple[List[float], List[float]]:
        """Blocking epoch loop (single session). Prefer engine.start_session + run()."""
        self.begin_fit(
            steps=steps,
            source_mode=source_mode,
            model_type=model_type,
            early_stopping_enabled=early_stopping_enabled,
            patience=patience,
            min_delta=min_delta,
            compute_r2_score=compute_r2_score,
            engine=engine,
            max_epochs=max_epochs,
        )
        while self.advance_fit_epoch():
            pass
        if not self._fit_done:
            self.finish_fit()
        return self.train_history, self.val_history

    def _run_epoch_training_pass(
        self,
        active_lr: float,
        steps: int,
        epoch: int,
        is_classification: bool,
    ) -> Tuple[float, Dict[str, Any]]:
        self.data_provider.reset_epoch()
        batch_losses: List[float] = []
        batch_idx = 0
        forensic_data: Dict[str, Any] = {}

        from src.ledger import BatchRef
        from src.training_engine import StepInput

        async_contract = (
            self.engine is not None and self.engine.uses_async_contract(self)
        )

        if async_contract:
            def _next_step() -> StepInput | None:
                if not self.data_provider.has_more_batches():
                    return None
                X_b, y_b = self.data_provider.next_batch()
                if X_b.size == 0:
                    return None
                X_b_norm = self.data_provider.normalize(X_b)
                return StepInput(
                    X=X_b_norm,
                    y=y_b,
                    batch_ref=BatchRef.new(epoch=epoch, batch_idx=batch_idx),
                    lr=active_lr,
                )

            def _on_submitted(step: StepInput) -> None:
                nonlocal batch_idx, forensic_data
                if batch_idx == 0 and epoch % 10 == 0:
                    forensic_data = self._forensic_batch_zero(step.y, is_classification)
                batch_idx += 1
                self.steps_completed += 1

            batch_losses = self.engine.run_training_loop(
                session=self,
                steps_budget=steps,
                next_step=_next_step,
                on_submitted=_on_submitted,
            )
            self.steps_completed = 0
            return float(np.mean(batch_losses)) if batch_losses else 0.0, forensic_data

        while True:
            if not self.data_provider.has_more_batches():
                break

            X_b, y_b = self.data_provider.next_batch()
            if X_b.size == 0:
                break

            X_b_norm = self.data_provider.normalize(X_b)
            batch_ref = BatchRef.new(epoch=epoch, batch_idx=batch_idx)
            step = StepInput(X=X_b_norm, y=y_b, batch_ref=batch_ref, lr=active_lr)

            if self.engine is not None:
                batch_losses.append(self.engine.run_step(step, session=self))
            else:
                batch_losses.append(self.train_and_apply(X_b_norm, y_b, active_lr))

            if batch_idx == 0 and epoch % 10 == 0:
                forensic_data = self._forensic_batch_zero(y_b, is_classification)

            batch_idx += 1
            self.steps_completed += 1
            if self.steps_completed >= steps:
                self.steps_completed = 0
                break

        return float(np.mean(batch_losses)) if batch_losses else 0.0, forensic_data

    def _forensic_batch_zero(
        self, y_b: np.ndarray, is_classification: bool
    ) -> Dict[str, Any]:
        if is_classification:
            y_classes = (
                np.sum(y_b, axis=0)
                if len(y_b.shape) > 1
                else np.unique(y_b, return_counts=True)[1]
            )
        else:
            y_classes = "N/A"

        if hasattr(self.model, "activations") and len(self.model.activations) > 0:
            preds = self.model.activations[-1]
            if is_classification and hasattr(preds, "shape") and len(preds.shape) > 1:
                pred_classes = np.argmax(preds, axis=1)
                pred_spread = [int(np.sum(pred_classes == c)) for c in range(preds.shape[1])]
            else:
                pred_spread = "N/A"

            l1_act = self.model.activations[1] if len(self.model.activations) > 1 else None
            dead_pct = float(np.mean(l1_act <= 0.0) * 100) if l1_act is not None else 0.0
        else:
            pred_spread = []
            dead_pct = 0.0

        return {
            "batch_target_dist": y_classes.tolist()
            if isinstance(y_classes, np.ndarray)
            else y_classes,
            "batch_pred_spread": pred_spread,
            "dead_zone_pct": dead_pct,
        }

    def _evaluate_epoch_performance(
        self,
        epoch: int,
        train_loss: float,
        val_preds: np.ndarray,
        y_val_target: np.ndarray,
        current_val_loss: float,
        active_lr: float,
        is_classification: bool,
        compute_r2_score: Callable[[np.ndarray, np.ndarray], float] | None,
    ) -> None:
        if not logging.getLogger().isEnabledFor(logging.INFO):
            return

        if epoch % 10 == 0 or not self.data_provider.has_more_batches():
            metric_name = "Acc" if is_classification else "R²"

            if hasattr(self.data_provider, "splits") and DataKeys.X_TRAIN in self.data_provider.splits:
                X_train = self.data_provider.splits[DataKeys.X_TRAIN]
                train_preds = self._predict_with_thread_policy(X_train)
                y_train_target = self.data_provider.y_train_processed

                if is_classification:
                    train_score = np.mean(np.argmax(train_preds, axis=1) == np.argmax(y_train_target, axis=1))
                    val_score = np.mean(np.argmax(val_preds, axis=1) == np.argmax(y_val_target, axis=1))
                else:
                    r2 = compute_r2_score or (lambda a, b: 0.0)
                    train_score = r2(y_train_target, train_preds)
                    val_score = r2(y_val_target, val_preds)
            else:
                train_score = 0.0
                val_score = (
                    np.mean(np.argmax(val_preds, axis=1) == np.argmax(y_val_target, axis=1))
                    if is_classification
                    else (compute_r2_score or (lambda a, b: 0.0))(y_val_target, val_preds)
                )

            train_score_str = f"{train_score * 100:.2f}%" if is_classification else f"{train_score:.4f}"
            val_score_str = f"{val_score * 100:.2f}%" if is_classification else f"{val_score:.4f}"

            logging.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {current_val_loss:.4f} | "
                f"Train {metric_name}: {train_score_str} | "
                f"Val {metric_name}: {val_score_str} | "
                f"Active LR: {active_lr:.6f}"
            )

    def _handle_early_stopping(
        self,
        epoch: int,
        current_val_raw_cost: float,
        min_delta: float,
        patience: int,
        es_state: Dict[str, Any],
        current_val_loss: float | None = None,
    ) -> bool:
        if current_val_raw_cost < (es_state["best_val_loss"] - min_delta):
            es_state["best_val_loss"] = current_val_raw_cost
            es_state["best_epoch"] = epoch
            es_state["patience_counter"] = 0
            es_state["weights"] = copy.deepcopy(self.model.weights)
            es_state["biases"] = copy.deepcopy(self.model.biases)
            if self.engine is not None:
                es_state["best_version"] = self.engine.on_early_stopping_improved(
                    epoch, current_val_loss
                )
            return False

        es_state["patience_counter"] += 1
        if es_state["patience_counter"] >= patience:
            if self.engine is not None and es_state.get("best_version") is not None:
                self.engine.on_early_stopping_triggered(
                    epoch,
                    int(es_state["best_epoch"]),
                    int(es_state["best_version"]),
                )
            else:
                logging.info(
                    f"[Early Stopping] Validation divergence at epoch {epoch}. "
                    f"Restoring best checkpoint from epoch {es_state['best_epoch']}."
                )
                self.model.weights = es_state["weights"]
                self.model.biases = es_state["biases"]
                if hasattr(self.model, "_sync_restored_weights"):
                    self.model._sync_restored_weights()
            return True
        return False

    def _generate_final_summary_report(
        self,
        X_val_raw: np.ndarray,
        y_val_target: np.ndarray,
        source_mode: Any,
        is_classification: bool,
        model_type: ModelType,
        es_state: Dict[str, Any],
        es_enabled: bool,
        compute_r2_score: Callable[[np.ndarray, np.ndarray], float] | None,
    ) -> None:
        final_val_preds = self._predict_with_thread_policy(X_val_raw)
        val_loss = self.val_history[-1] if self.val_history else float("inf")
        r2 = compute_r2_score or (lambda a, b: 0.0)

        if hasattr(self.data_provider, "splits") and DataKeys.X_TRAIN in self.data_provider.splits:
            X_train = self.data_provider.splits[DataKeys.X_TRAIN]
            final_train_preds = self._predict_with_thread_policy(X_train)
            y_train_target = self.data_provider.y_train_processed

            train_acc = np.mean(np.argmax(final_train_preds, axis=1) == np.argmax(y_train_target, axis=1))
            val_acc = np.mean(np.argmax(final_val_preds, axis=1) == np.argmax(y_val_target, axis=1))
            train_r2 = r2(y_train_target, final_train_preds)
            val_r2 = r2(y_val_target, final_val_preds)
        else:
            train_acc, train_r2 = 0.0, 0.0
            val_acc = (
                np.mean(np.argmax(final_val_preds, axis=1) == np.argmax(y_val_target, axis=1))
                if is_classification
                else 0.0
            )
            val_r2 = r2(y_val_target, final_val_preds)

        total_weights_count = 0
        zeroed_weights_count = 0
        sparsity_tolerance = 1e-5

        if hasattr(self.model, "weights"):
            for w in self.model.weights:
                total_weights_count += w.size
                zeroed_weights_count += np.sum(np.abs(w) <= sparsity_tolerance)

        sparsity_percentage = (zeroed_weights_count / total_weights_count * 100) if total_weights_count > 0 else 0.0
        mode_str = source_mode.name.upper() if hasattr(source_mode, "name") else str(source_mode).upper()
        type_str = model_type.name if hasattr(model_type, "name") else str(model_type)

        performance_metrics_block = ""
        if is_classification:
            performance_metrics_block += f"  • Final Training Accuracy : {train_acc * 100:.2f}%\n"
            performance_metrics_block += f"  • Final Target Accuracy   : {val_acc * 100:.2f}%\n"
            performance_metrics_block += f"  • In-Sample R² Score      : {train_r2:.6f} (Train Alignment)\n"
            performance_metrics_block += f"  • Out-of-Sample R² Score  : {val_r2:.6f} (Validation Alignment)\n"
        else:
            performance_metrics_block += f"  • Final Training R² Score : {train_r2:.6f}\n"
            performance_metrics_block += f"  • Final Target R² Score   : {val_r2:.6f}\n"

        divider = "═" * 70
        sub_divider = "─" * 70

        report = (
            f"\n{divider}\n"
            f"                     TRAINING RUN EXECUTION REPORT\n"
            f"{divider}\n"
            f"  [ PIPELINE ENVIRONMENT ]\n"
            f"  • Ingestion Strategy Mode : {mode_str}\n"
            f"  • Network Task Profile    : {type_str}\n"
            f"  • Total Epochs Executed   : {len(self.val_history)} / 300\n"
            f"  • Early Stopping State    : {'ACTIVE' if es_enabled else 'DISABLED'}\n"
            f"  • Best Operational Epoch  : {es_state['best_epoch'] if es_enabled else 'N/A'}\n"
            f"{sub_divider}\n"
            f"  [ MATHEMATICAL OPTIMIZATION BENCHMARKS ]\n"
            f"  • Final Validation Loss   : {val_loss:.6f}\n"
            f"  • Best Validation Raw Cost: {es_state['best_val_loss']:.6f}\n"
            f"{performance_metrics_block}"
            f"{sub_divider}\n"
            f"  [ WEIGHT MATRIX STRUCTURAL METRICS ]\n"
            f"  • Active Synaptic Tensors : {total_weights_count:,} parameters\n"
            f"  • Dead/Sparse Connections : {zeroed_weights_count:,} elements\n"
            f"  • Total Network Sparsity  : {sparsity_percentage:.4f}% Density Suppression\n"
            f"{divider}\n"
        )
        logging.info(report)

        if hasattr(self.model, "weights") and hasattr(self.model, "biases") and len(self.model.weights) > 0:
            final_w_norm = np.linalg.norm(self.model.weights[0])
            final_biases = np.round(
                self.model.biases[-1][0] if len(self.model.biases[-1].shape) > 1 else self.model.biases[-1],
                4,
            )
            logging.info("=== FORENSIC FINAL TRACE ===")
            logging.info(f"Layer 0 Final Weight Norm: {final_w_norm:.6f}")
            logging.info(f"Final Layer Output Biases: {final_biases.tolist()}")
            logging.info("============================")
