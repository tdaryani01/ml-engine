# src/training_engine.py
"""Single-thread training loop: train → ledger documents → consolidate → checkpoint."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.ledger import LedgerConfig, TrainingLedger, VERDICT_HEALTHY
from src.training_session import TrainingSession


@dataclass
class StepInput:
    X: np.ndarray
    y: np.ndarray
    batch_ref: BatchRef
    lr: float
    val_loss: float | None = None


class TrainingEngine:
    """
    SQL-style main loop owner. Writes step documents; does not replace fit() until E gates pass.
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

    @property
    def version(self) -> int:
        return self.ledger.version

    def run_step(self, step: StepInput) -> float:
        """One batch: command → train_step → result → apply → metrics → optional checkpoint."""
        base_version = self.ledger.version
        self.ledger.push_step_command(
            step_id=self.session.step_id + 1,
            base_version=base_version,
            batch_ref=step.batch_ref,
            lr=step.lr,
            m_samples=int(step.y.shape[0]),
        )

        result = self.session.train_step(step.X, step.y, step.lr)
        self.ledger.push_step_result(result)

        self.session.apply_step(result, step.lr)
        new_version = base_version + 1
        self.ledger.version = new_version
        optimizer_t = int(getattr(self.session.model.optimizer, "t", new_version))
        self.ledger.push_step_consolidated(result.step_id, new_version, optimizer_t)

        verdict = VERDICT_HEALTHY
        is_local_best = (
            step.val_loss is not None
            and self.config.checkpoint_on_local_best
            and step.val_loss < self.ledger._best_val_loss
        )
        self.ledger.push_step_metrics(
            step_id=result.step_id,
            version=new_version,
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
        losses: list[float] = []
        for step in steps:
            losses.append(self.run_step(step))
        return losses

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
