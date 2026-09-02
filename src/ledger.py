# src/ledger.py
"""Append-only training ledger: documents, pluggable store, replay."""
from __future__ import annotations

import base64
import json
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


def _encode_array(arr: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(arr)
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def _decode_array(payload: dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(payload["data"])
    return np.frombuffer(raw, dtype=np.dtype(payload["dtype"])).reshape(payload["shape"]).copy()


def _encode_array_list(arrays: list[np.ndarray] | None) -> list[dict[str, Any]] | None:
    if arrays is None:
        return None
    return [_encode_array(a) for a in arrays]


def _decode_array_list(payload: list[dict[str, Any]] | None) -> list[np.ndarray] | None:
    if payload is None:
        return None
    return [_decode_array(p) for p in payload]


def train_step_result_to_body(result: TrainStepResult) -> dict[str, Any]:
    return {
        "step_id": result.step_id,
        "loss": float(result.loss),
        "m_samples": int(result.m_samples),
        "grad_weights": _encode_array_list(result.grad_weights),
        "grad_biases": _encode_array_list(result.grad_biases),
        "grad_gammas": _encode_array_list(result.grad_gammas),
        "grad_betas": _encode_array_list(result.grad_betas),
    }


def train_step_result_from_body(body: dict[str, Any]) -> TrainStepResult:
    return TrainStepResult(
        step_id=int(body["step_id"]),
        loss=float(body["loss"]),
        m_samples=int(body["m_samples"]),
        grad_weights=_decode_array_list(body["grad_weights"]) or [],
        grad_biases=_decode_array_list(body["grad_biases"]) or [],
        grad_gammas=_decode_array_list(body.get("grad_gammas")),
        grad_betas=_decode_array_list(body.get("grad_betas")),
    )


def document_to_bytes(doc: LedgerDocument) -> bytes:
    return json.dumps(doc.to_dict(), separators=(",", ":")).encode("utf-8")


def document_from_bytes(data: bytes) -> LedgerDocument:
    return LedgerDocument.from_dict(json.loads(data.decode("utf-8")))


def capture_model_checkpoint(model: Any, version: int, val_loss: float | None = None) -> dict[str, Any]:
    """Build checkpoint document body from a live model."""
    opt = model.optimizer
    body: dict[str, Any] = {
        "version": version,
        "weights": _encode_array_list(model.weights),
        "biases": _encode_array_list(model.biases),
        "gammas": None,
        "betas": None,
        "optimizer": {
            "type": type(opt).__name__,
            "t": int(getattr(opt, "t", 0)),
            "beta1": float(getattr(opt, "beta1", 0.9)),
            "beta2": float(getattr(opt, "beta2", 0.999)),
            "eps": float(getattr(opt, "eps", 1e-8)),
            "ms_w": _encode_array_list(getattr(opt, "ms_w", None)),
            "vs_w": _encode_array_list(getattr(opt, "vs_w", None)),
            "ms_b": _encode_array_list(getattr(opt, "ms_b", None)),
            "vs_b": _encode_array_list(getattr(opt, "vs_b", None)),
            "ms_g": _encode_array_list(getattr(opt, "ms_g", None)),
            "vs_g": _encode_array_list(getattr(opt, "vs_g", None)),
            "ms_beta": _encode_array_list(getattr(opt, "ms_beta", None)),
            "vs_beta": _encode_array_list(getattr(opt, "vs_beta", None)),
        },
        "val_loss": val_loss,
        "is_local_best": False,
    }
    if hasattr(model, "gammas"):
        body["gammas"] = _encode_array_list(getattr(model, "gammas", None))
    if hasattr(model, "betas"):
        body["betas"] = _encode_array_list(getattr(model, "betas", None))
    return body


def restore_model_checkpoint(model: Any, body: dict[str, Any]) -> None:
    """Hydrate model weights and Adam state from a checkpoint body."""
    weights = _decode_array_list(body["weights"]) or []
    biases = _decode_array_list(body["biases"]) or []
    for i, w in enumerate(weights):
        model.weights[i][...] = w
    for i, b in enumerate(biases):
        model.biases[i][...] = b
    if body.get("gammas") and hasattr(model, "gammas"):
        gammas = _decode_array_list(body["gammas"]) or []
        for i, g in enumerate(gammas):
            model.gammas[i][...] = g
    if body.get("betas") and hasattr(model, "betas"):
        betas = _decode_array_list(body["betas"]) or []
        for i, b in enumerate(betas):
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
    opt.ms_w = _decode_array_list(o.get("ms_w")) or [np.zeros_like(w) for w in model.weights]
    opt.vs_w = _decode_array_list(o.get("vs_w")) or [np.zeros_like(w) for w in model.weights]
    opt.ms_b = _decode_array_list(o.get("ms_b")) or [np.zeros_like(b) for b in model.biases]
    opt.vs_b = _decode_array_list(o.get("vs_b")) or [np.zeros_like(b) for b in model.biases]
    opt.ms_g = _decode_array_list(o.get("ms_g"))
    opt.vs_g = _decode_array_list(o.get("vs_g"))
    opt.ms_beta = _decode_array_list(o.get("ms_beta"))
    opt.vs_beta = _decode_array_list(o.get("vs_beta"))


class LedgerStore(Protocol):
    def push(self, doc: LedgerDocument) -> int: ...
    def get(self, lsn: int) -> LedgerDocument: ...
    def scan(self, from_lsn: int = 1, to_lsn: int | None = None) -> Iterator[LedgerDocument]: ...
    def head_lsn(self) -> int: ...
    def put_checkpoint(self, doc: LedgerDocument) -> None: ...
    def get_checkpoint(self, branch_id: str, version: int) -> LedgerDocument | None: ...


class FileLedgerStore:
    """Append-only JSONL journal + checkpoint sidecar files."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.root / "journal.jsonl"
        self.checkpoint_dir = self.root / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._next_lsn = self._load_head_lsn() + 1

    def _load_head_lsn(self) -> int:
        if not self.journal_path.exists():
            return 0
        last = 0
        with open(self.journal_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                last = int(json.loads(line)["lsn"])
        return last

    def push(self, doc: LedgerDocument) -> int:
        lsn = self._next_lsn
        self._next_lsn += 1
        doc.lsn = lsn
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(document_to_bytes(doc).decode("utf-8") + "\n")
        return lsn

    def get(self, lsn: int) -> LedgerDocument:
        for doc in self.scan(from_lsn=lsn, to_lsn=lsn):
            return doc
        raise KeyError(f"LSN {lsn} not found")

    def scan(self, from_lsn: int = 1, to_lsn: int | None = None) -> Iterator[LedgerDocument]:
        if not self.journal_path.exists():
            return
        with open(self.journal_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = document_from_bytes(line.encode("utf-8"))
                if doc.lsn is None or doc.lsn < from_lsn:
                    continue
                if to_lsn is not None and doc.lsn > to_lsn:
                    break
                yield doc

    def head_lsn(self) -> int:
        return max(0, self._next_lsn - 1)

    def _checkpoint_path(self, branch_id: str, version: int) -> Path:
        safe_branch = branch_id.replace(os.sep, "_")
        return self.checkpoint_dir / f"{safe_branch}_v{version}.json"

    def put_checkpoint(self, doc: LedgerDocument) -> None:
        if doc.doc_type != CHECKPOINT:
            raise ValueError("put_checkpoint expects doc_type=checkpoint")
        version = int(doc.body["version"])
        path = self._checkpoint_path(doc.branch_id, version)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc.to_dict(), f, separators=(",", ":"))

    def get_checkpoint(self, branch_id: str, version: int) -> LedgerDocument | None:
        path = self._checkpoint_path(branch_id, version)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return LedgerDocument.from_dict(json.load(f))


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
        return self.push(
            self._envelope(STEP_RESULT, train_step_result_to_body(result), step_id=result.step_id)
        )

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
        """Apply stored step.result documents for (from_version, to_version] via apply_step."""
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
