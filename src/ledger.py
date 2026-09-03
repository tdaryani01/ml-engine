# src/ledger.py
"""Append-only training ledger: documents, pluggable store, replay."""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np

from src.training_session import TrainStepResult

# Document type constants
STEP_COMMAND = "step.command"
STEP_RESULT = "step.result"
STEP_CONSOLIDATED = "step.consolidated"
STEP_METRICS = "step.metrics"
STEP_COMPLETE = "step.complete"
CHECKPOINT = "checkpoint"
BRANCH_FORK = "branch.fork"
REWIND = "rewind"
PATH_RECORD = "path.record"

VERDICT_HEALTHY = "HEALTHY"
VERDICT_SUSPECT = "SUSPECT"
VERDICT_OVERFIT = "OVERFIT"


@dataclass
class BatchRef:
    """Canonical batch identity (Q5): UUID assigned at fetch time."""

    batch_id: str
    data_version: int = 1
    epoch: int = 0
    batch_idx: int = 0

    @classmethod
    def new(cls, epoch: int = 0, batch_idx: int = 0, data_version: int = 1) -> BatchRef:
        return cls(batch_id=str(uuid.uuid4()), data_version=data_version, epoch=epoch, batch_idx=batch_idx)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "data_version": self.data_version,
            "epoch": self.epoch,
            "batch_idx": self.batch_idx,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BatchRef:
        return cls(
            batch_id=str(d["batch_id"]),
            data_version=int(d.get("data_version", 1)),
            epoch=int(d.get("epoch", 0)),
            batch_idx=int(d.get("batch_idx", 0)),
        )


@dataclass
class LedgerDocument:
    doc_type: str
    branch_id: str
    model_instance_id: str
    architecture_id: str
    body: dict[str, Any]
    lsn: int | None = None
    version: int | None = None
    step_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lsn": self.lsn,
            "doc_type": self.doc_type,
            "branch_id": self.branch_id,
            "model_instance_id": self.model_instance_id,
            "architecture_id": self.architecture_id,
            "version": self.version,
            "step_id": self.step_id,
            "body": self.body,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LedgerDocument:
        return cls(
            lsn=d.get("lsn"),
            doc_type=str(d["doc_type"]),
            branch_id=str(d["branch_id"]),
            model_instance_id=str(d["model_instance_id"]),
            architecture_id=str(d["architecture_id"]),
            version=d.get("version"),
            step_id=d.get("step_id"),
            body=dict(d["body"]),
        )


@dataclass
class LedgerConfig:
    consolidate_every: int = 1
    checkpoint_every_steps: int = 50
    checkpoint_on_local_best: bool = True
    checkpoint_on_fork: bool = True
    keep_last_k_checkpoints: int = 20
    contract_list_enabled: bool = False  # Phase F: off until grad/benchmark parity
    store_backend: str = "file_streaming"
    flush_stall_threshold: int = 64



from src.ledger_wire import (
    _pack_body,
    pack_checkpoint_body,
    pack_document,
    pack_step_complete_body,
    pack_step_result,
    unpack_document,
)


def train_step_result_to_body(result: TrainStepResult) -> dict[str, Any]:
    """In-memory step.result body — raw arrays, no wire encoding."""
    return {"_result": result}


def train_step_result_from_body(body: dict[str, Any]) -> TrainStepResult:
    if "_result" in body:
        return body["_result"]
    raise ValueError("step.result body missing _result")


def document_to_bytes(doc: LedgerDocument) -> bytes:
    if doc.doc_type == STEP_RESULT:
        result = train_step_result_from_body(doc.body)
        body_bytes = pack_step_result(result)
    elif doc.doc_type == STEP_COMPLETE:
        body_bytes = pack_step_complete_body(doc.body)
    elif doc.doc_type == CHECKPOINT:
        body_bytes = pack_checkpoint_body(doc.body)
    else:
        body_bytes = _pack_body(doc.doc_type, doc.body)
    return pack_document(
        doc_type=doc.doc_type,
        branch_id=doc.branch_id,
        model_instance_id=doc.model_instance_id,
        architecture_id=doc.architecture_id,
        body=body_bytes,
        lsn=doc.lsn,
        version=doc.version,
        step_id=doc.step_id,
    )


def document_from_bytes(data: bytes) -> LedgerDocument:
    parsed = unpack_document(data)
    body = parsed["body"]
    if parsed["doc_type"] == STEP_RESULT:
        body = {"_result": parsed["body"]["_result"]}
    elif parsed["doc_type"] == STEP_COMPLETE:
        body = parsed["body"]
    return LedgerDocument(
        lsn=parsed["lsn"],
        doc_type=parsed["doc_type"],
        branch_id=parsed["branch_id"],
        model_instance_id=parsed["model_instance_id"],
        architecture_id=parsed["architecture_id"],
        version=parsed["version"],
        step_id=parsed["step_id"],
        body=body,
    )



def capture_model_checkpoint(model: Any, version: int, val_loss: float | None = None) -> dict[str, Any]:
    """Build checkpoint document body from a live model (raw ndarray lists)."""
    opt = model.optimizer
    body: dict[str, Any] = {
        "version": version,
        "weights": list(model.weights),
        "biases": list(model.biases),
        "gammas": None,
        "betas": None,
        "optimizer": {
            "type": type(opt).__name__,
            "t": int(getattr(opt, "t", 0)),
            "beta1": float(getattr(opt, "beta1", 0.9)),
            "beta2": float(getattr(opt, "beta2", 0.999)),
            "eps": float(getattr(opt, "eps", 1e-8)),
            "ms_w": getattr(opt, "ms_w", None),
            "vs_w": getattr(opt, "vs_w", None),
            "ms_b": getattr(opt, "ms_b", None),
            "vs_b": getattr(opt, "vs_b", None),
            "ms_g": getattr(opt, "ms_g", None),
            "vs_g": getattr(opt, "vs_g", None),
            "ms_beta": getattr(opt, "ms_beta", None),
            "vs_beta": getattr(opt, "vs_beta", None),
        },
        "val_loss": val_loss,
        "is_local_best": False,
    }
    if hasattr(model, "gammas"):
        body["gammas"] = getattr(model, "gammas", None)
    if hasattr(model, "betas"):
        body["betas"] = getattr(model, "betas", None)
    return body


def restore_model_checkpoint(model: Any, body: dict[str, Any]) -> None:
    """Hydrate model weights and Adam state from a checkpoint body."""
    weights = body["weights"] or []
    biases = body["biases"] or []
    for i, w in enumerate(weights):
        model.weights[i][...] = w
    for i, b in enumerate(biases):
        model.biases[i][...] = b
    if body.get("gammas") and hasattr(model, "gammas"):
        for i, g in enumerate(body["gammas"] or []):
            model.gammas[i][...] = g
    if body.get("betas") and hasattr(model, "betas"):
        for i, b in enumerate(body["betas"] or []):
            model.betas[i][...] = b
    if hasattr(model, "_sync_restored_weights"):
        model._sync_restored_weights()

    opt = model.optimizer
    o = body["optimizer"]
    opt.t = int(o["t"])
    if hasattr(opt, "beta1"):
        opt.beta1 = float(o.get("beta1", opt.beta1))
    if hasattr(opt, "beta2"):
        opt.beta2 = float(o.get("beta2", opt.beta2))
    if hasattr(opt, "eps"):
        opt.eps = float(o.get("eps", opt.eps))
    opt._setup_done = True
    opt.ms_w = o.get("ms_w") or [np.zeros_like(w) for w in model.weights]
    opt.vs_w = o.get("vs_w") or [np.zeros_like(w) for w in model.weights]
    opt.ms_b = o.get("ms_b") or [np.zeros_like(b) for b in model.biases]
    opt.vs_b = o.get("vs_b") or [np.zeros_like(b) for b in model.biases]
    opt.ms_g = o.get("ms_g")
    opt.vs_g = o.get("vs_g")
    opt.ms_beta = o.get("ms_beta")
    opt.vs_beta = o.get("vs_beta")


class LedgerStore(Protocol):
    def push(self, doc: LedgerDocument) -> int: ...
    def begin_flush(self) -> bool: ...
    def try_reap_flush(self) -> bool: ...
    def has_flush_pending(self) -> bool: ...
    def queue_pending(self) -> bool: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
    def get(self, lsn: int) -> LedgerDocument: ...
    def scan(self, from_lsn: int = 1, to_lsn: int | None = None) -> Iterator[LedgerDocument]: ...
    def head_lsn(self) -> int: ...
    def put_checkpoint(self, doc: LedgerDocument) -> None: ...
    def get_checkpoint(self, branch_id: str, version: int) -> LedgerDocument | None: ...


@dataclass
class TrainingLedger:
    """High-level ledger: branch head, step documents, checkpoints, replay."""

    store: LedgerStore
    branch_id: str = "main"
    model_instance_id: str = "default"
    architecture_id: str = "unknown"
    version: int = 0
    frozen: bool = False
    _best_val_loss: float = field(default=float("inf"), repr=False)

    def _envelope(self, doc_type: str, body: dict[str, Any], version: int | None = None, step_id: int | None = None) -> LedgerDocument:
        return LedgerDocument(
            doc_type=doc_type,
            branch_id=self.branch_id,
            model_instance_id=self.model_instance_id,
            architecture_id=self.architecture_id,
            body=body,
            version=version,
            step_id=step_id,
        )

    
    def push(self, doc: LedgerDocument) -> int:
        if self.frozen:
            raise RuntimeError(f"Branch {self.branch_id!r} is frozen")
        return self.store.push(doc)

    
    def push_step_command(
        self,
        step_id: int,
        base_version: int,
        batch_ref: BatchRef,
        lr: float,
        m_samples: int,
        scheduler_epoch: int = 0,
    ) -> int:
        body = {
            "step_id": step_id,
            "base_version": base_version,
            "batch_ref": batch_ref.to_dict(),
            "lr": float(lr),
            "m_samples": int(m_samples),
            "scheduler_epoch": int(scheduler_epoch),
        }
        return self.push(self._envelope(STEP_COMMAND, body, step_id=step_id))

    
    def push_step_result(self, result: TrainStepResult) -> int:
        return self.push(self._envelope(STEP_RESULT, {"_result": result}, step_id=result.step_id))

    
    def push_step_consolidated(self, step_id: int, version: int, optimizer_t: int) -> int:
        body = {"step_id": step_id, "version": version, "optimizer_t": optimizer_t}
        return self.push(self._envelope(STEP_CONSOLIDATED, body, version=version, step_id=step_id))

    
    def push_step_metrics(
        self,
        step_id: int,
        version: int,
        train_loss: float,
        val_loss: float | None = None,
        verdict: str = VERDICT_HEALTHY,
    ) -> int:
        gap = None
        if val_loss is not None:
            gap = float(val_loss - train_loss)
        is_best = False
        if val_loss is not None and val_loss < self._best_val_loss:
            self._best_val_loss = float(val_loss)
            is_best = True
        body = {
            "step_id": step_id,
            "version": version,
            "train_loss": float(train_loss),
            "val_loss": val_loss,
            "train_val_gap": gap,
            "verdict": verdict,
            "is_local_best_val": is_best,
        }
        return self.push(self._envelope(STEP_METRICS, body, version=version, step_id=step_id))

    def _metrics_body(
        self,
        step_id: int,
        version: int,
        train_loss: float,
        val_loss: float | None = None,
        verdict: str = VERDICT_HEALTHY,
    ) -> dict[str, Any]:
        gap = None
        if val_loss is not None:
            gap = float(val_loss - train_loss)
        is_best = False
        if val_loss is not None and val_loss < self._best_val_loss:
            self._best_val_loss = float(val_loss)
            is_best = True
        return {
            "step_id": step_id,
            "version": version,
            "train_loss": float(train_loss),
            "val_loss": val_loss,
            "train_val_gap": gap,
            "verdict": verdict,
            "is_local_best_val": is_best,
        }

    def push_step_complete(
        self,
        *,
        step_id: int,
        base_version: int,
        batch_ref: BatchRef,
        lr: float,
        m_samples: int,
        result: TrainStepResult,
        version: int,
        optimizer_t: int,
        train_loss: float,
        val_loss: float | None = None,
        verdict: str = VERDICT_HEALTHY,
        scheduler_epoch: int = 0,
    ) -> int:
        """One journal record per batch (command + result + consolidated + metrics)."""
        command = {
            "step_id": step_id,
            "base_version": base_version,
            "batch_ref": batch_ref.to_dict(),
            "lr": float(lr),
            "m_samples": int(m_samples),
            "scheduler_epoch": int(scheduler_epoch),
        }
        consolidated = {"step_id": step_id, "version": version, "optimizer_t": optimizer_t}
        metrics = self._metrics_body(step_id, version, train_loss, val_loss, verdict)
        body = {
            "command": command,
            "_result": result,
            "consolidated": consolidated,
            "metrics": metrics,
        }
        return self.push(
            self._envelope(STEP_COMPLETE, body, version=version, step_id=step_id)
        )

    
    def push_checkpoint(self, model: Any, version: int, val_loss: float | None = None, is_local_best: bool = False) -> int:
        body = capture_model_checkpoint(model, version, val_loss=val_loss)
        body["is_local_best"] = is_local_best
        doc = self._envelope(CHECKPOINT, body, version=version, step_id=version)
        lsn = self.push(doc)
        self.store.put_checkpoint(doc)
        return lsn

    def fork_branch(self, new_branch_id: str, parent_version: int, reason: str, settings_delta: dict[str, Any] | None = None) -> TrainingLedger:
        parent_lsn = self.store.head_lsn()
        body = {
            "parent_branch_id": self.branch_id,
            "parent_version": parent_version,
            "parent_lsn": parent_lsn,
            "new_branch_id": new_branch_id,
            "reason": reason,
            "settings_delta": settings_delta or {},
            "checkpoint_version": parent_version,
        }
        self.push(self._envelope(BRANCH_FORK, body, version=parent_version))
        cp = self.store.get_checkpoint(self.branch_id, parent_version)
        if cp is not None:
            child_cp = LedgerDocument(
                doc_type=CHECKPOINT,
                branch_id=new_branch_id,
                model_instance_id=self.model_instance_id,
                architecture_id=self.architecture_id,
                body=cp.body,
                version=parent_version,
                step_id=parent_version,
            )
            self.store.put_checkpoint(child_cp)
        self.frozen = True
        child = TrainingLedger(
            store=self.store,
            branch_id=new_branch_id,
            model_instance_id=self.model_instance_id,
            architecture_id=self.architecture_id,
            version=parent_version,
        )
        child._best_val_loss = self._best_val_loss
        return child

    def scan_branch(self, doc_type: str | None = None) -> Iterator[LedgerDocument]:
        for doc in self.store.scan():
            if doc.branch_id != self.branch_id:
                continue
            if doc_type is not None and doc.doc_type != doc_type:
                continue
            yield doc

    def replay_apply_from_version(
        self,
        session: Any,
        from_version: int,
        to_version: int,
        default_lr: float,
    ) -> None:
        """Apply stored step records for (from_version, to_version] via apply_step."""
        complete = list(self.scan_branch(STEP_COMPLETE))
        if complete:
            for doc in complete:
                result = train_step_result_from_body(doc.body)
                if result.step_id <= from_version or result.step_id > to_version:
                    continue
                lr = float(doc.body["command"]["lr"])
                session.apply_step(result, lr)
            return

        lr_by_step: dict[int, float] = {}
        for doc in self.scan_branch(STEP_COMMAND):
            lr_by_step[int(doc.body["step_id"])] = float(doc.body["lr"])

        for doc in self.scan_branch(STEP_RESULT):
            result = train_step_result_from_body(doc.body)
            if result.step_id <= from_version or result.step_id > to_version:
                continue
            lr = lr_by_step.get(result.step_id, default_lr)
            session.apply_step(result, lr)

    def restore_checkpoint(self, model: Any, version: int) -> LedgerDocument:
        doc = self.store.get_checkpoint(self.branch_id, version)
        if doc is None:
            raise KeyError(f"No checkpoint for branch={self.branch_id!r} version={version}")
        restore_model_checkpoint(model, doc.body)
        self.version = version
        return doc


# Re-export store implementations (avoid circular import at module load).
from src.ledger_store import (  # noqa: E402
    FileLedgerStore,
    LedgerStore as LedgerStoreBackend,
    StreamingFileLedgerStore,
    SyncFileLedgerStore,
    create_ledger_store,
)
