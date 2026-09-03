# src/ledger_wire.py
"""Binary ledger journal/checkpoint encoding (no JSON, no base64)."""
from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

from src.training_session import TrainStepResult

MAGIC = b"MLE1"
_JOURNAL_VERSION = 1

# doc_type id -> name
_DOC_TYPE_TO_ID: dict[str, int] = {
    "step.command": 1,
    "step.result": 2,
    "step.consolidated": 3,
    "step.metrics": 4,
    "checkpoint": 5,
    "branch.fork": 6,
    "rewind": 7,
    "path.record": 8,
    "step.complete": 9,
}
_ID_TO_DOC_TYPE = {v: k for k, v in _DOC_TYPE_TO_ID.items()}


def _pack_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<H", len(b)) + b


def _unpack_str(buf: memoryview, off: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<H", buf, off)
    off += 2
    s = bytes(buf[off : off + n]).decode("utf-8")
    return s, off + n


def pack_array(arr: np.ndarray) -> bytes:
    c = np.ascontiguousarray(arr)
    dtype_b = str(c.dtype).encode("ascii")
    if len(dtype_b) > 255:
        raise ValueError(f"dtype string too long: {c.dtype!r}")
    if c.ndim > 255:
        raise ValueError(f"ndim too large: {c.ndim}")
    head = struct.pack("<B", len(dtype_b)) + dtype_b + struct.pack("<B", c.ndim)
    shape = b"".join(struct.pack("<I", int(d)) for d in c.shape)
    return head + shape + c.tobytes()


def unpack_array(buf: memoryview, off: int) -> tuple[np.ndarray, int]:
    dtype_len = buf[off]
    off += 1
    dtype = np.dtype(bytes(buf[off : off + dtype_len]).decode("ascii"))
    off += dtype_len
    ndim = buf[off]
    off += 1
    shape: list[int] = []
    for _ in range(ndim):
        (dim,) = struct.unpack_from("<I", buf, off)
        shape.append(int(dim))
        off += 4
    count = int(np.prod(shape)) if shape else 0
    arr = np.frombuffer(buf, dtype=dtype, count=count, offset=off).reshape(shape).copy()
    off += count * dtype.itemsize
    return arr, off


def _pack_array_list(arrays: list[np.ndarray] | None) -> bytes:
    if arrays is None:
        return struct.pack("<H", 0xFFFF)
    out = [struct.pack("<H", len(arrays))]
    for arr in arrays:
        blob = pack_array(arr)
        out.append(struct.pack("<I", len(blob)))
        out.append(blob)
    return b"".join(out)


def _unpack_array_list(buf: memoryview, off: int) -> tuple[list[np.ndarray] | None, int]:
    (n,) = struct.unpack_from("<H", buf, off)
    off += 2
    if n == 0xFFFF:
        return None, off
    arrays: list[np.ndarray] = []
    for _ in range(n):
        (blob_len,) = struct.unpack_from("<I", buf, off)
        off += 4
        arr, _ = unpack_array(buf, off)
        arrays.append(arr)
        off += blob_len
    return arrays, off


def pack_step_result(result: TrainStepResult) -> bytes:
    parts = [
        struct.pack("<QfI", int(result.step_id), float(result.loss), int(result.m_samples)),
        _pack_array_list(result.grad_weights),
        _pack_array_list(result.grad_biases),
        _pack_array_list(result.grad_gammas),
        _pack_array_list(result.grad_betas),
    ]
    return b"".join(parts)


def unpack_step_result(data: bytes | memoryview) -> TrainStepResult:
    buf = memoryview(data)
    off = 0
    step_id, loss, m_samples = struct.unpack_from("<QfI", buf, off)
    off += struct.calcsize("<QfI")
    gw, off = _unpack_array_list(buf, off)
    gb, off = _unpack_array_list(buf, off)
    gg, off = _unpack_array_list(buf, off)
    gbb, off = _unpack_array_list(buf, off)
    return TrainStepResult(
        step_id=int(step_id),
        loss=float(loss),
        m_samples=int(m_samples),
        grad_weights=gw or [],
        grad_biases=gb or [],
        grad_gammas=gg,
        grad_betas=gbb,
    )


def pack_step_complete_body(body: dict[str, Any]) -> bytes:
    cmd = _pack_step_command(body["command"])
    res = pack_step_result(body["_result"])
    con = _pack_step_consolidated(body["consolidated"])
    met = _pack_step_metrics(body["metrics"])
    return b"".join(
        [
            struct.pack("<I", len(cmd)),
            cmd,
            struct.pack("<I", len(res)),
            res,
            struct.pack("<I", len(con)),
            con,
            struct.pack("<I", len(met)),
            met,
        ]
    )


def unpack_step_complete_body(data: bytes | memoryview) -> dict[str, Any]:
    buf = memoryview(data)
    off = 0
    (cmd_len,) = struct.unpack_from("<I", buf, off)
    off += 4
    command, _ = _unpack_step_command(buf[off : off + cmd_len])
    off += cmd_len
    (res_len,) = struct.unpack_from("<I", buf, off)
    off += 4
    result = unpack_step_result(buf[off : off + res_len])
    off += res_len
    (con_len,) = struct.unpack_from("<I", buf, off)
    off += 4
    consolidated, _ = _unpack_step_consolidated(buf[off : off + con_len])
    off += con_len
    (met_len,) = struct.unpack_from("<I", buf, off)
    off += 4
    metrics, _ = _unpack_step_metrics(buf[off : off + met_len])
    return {
        "command": command,
        "_result": result,
        "consolidated": consolidated,
        "metrics": metrics,
    }


def _pack_step_command(body: dict[str, Any]) -> bytes:
    br = body["batch_ref"]
    return b"".join(
        [
            struct.pack(
                "<QQfdII",
                int(body["step_id"]),
                int(body["base_version"]),
                float(body["lr"]),
                int(body["m_samples"]),
                int(body.get("scheduler_epoch", 0)),
                int(br.get("data_version", 1)),
            ),
            struct.pack("<II", int(br.get("epoch", 0)), int(br.get("batch_idx", 0))),
            _pack_str(str(br["batch_id"])),
        ]
    )


def _unpack_step_command(data: memoryview) -> tuple[dict[str, Any], int]:
    off = 0
    step_id, base_version, lr, m_samples, sched, data_version = struct.unpack_from("<QQfdII", data, off)
    off += struct.calcsize("<QQfdII")
    epoch, batch_idx = struct.unpack_from("<II", data, off)
    off += 8
    batch_id, off = _unpack_str(data, off)
    body = {
        "step_id": int(step_id),
        "base_version": int(base_version),
        "lr": float(lr),
        "m_samples": int(m_samples),
        "scheduler_epoch": int(sched),
        "batch_ref": {
            "batch_id": batch_id,
            "data_version": int(data_version),
            "epoch": int(epoch),
            "batch_idx": int(batch_idx),
        },
    }
    return body, off


def _pack_step_consolidated(body: dict[str, Any]) -> bytes:
    return struct.pack("<QQQ", int(body["step_id"]), int(body["version"]), int(body["optimizer_t"]))


def _unpack_step_consolidated(data: memoryview) -> tuple[dict[str, Any], int]:
    step_id, version, optimizer_t = struct.unpack_from("<QQQ", data, 0)
    return {
        "step_id": int(step_id),
        "version": int(version),
        "optimizer_t": int(optimizer_t),
    }, struct.calcsize("<QQQ")


def _pack_step_metrics(body: dict[str, Any]) -> bytes:
    val = body.get("val_loss")
    has_val = val is not None
    gap = body.get("train_val_gap")
    parts = [
        struct.pack("<QQd", int(body["step_id"]), int(body["version"]), float(body["train_loss"])),
        struct.pack("<B", 1 if has_val else 0),
    ]
    if has_val:
        parts.append(struct.pack("<dd", float(val), float(gap if gap is not None else 0.0)))
    parts.append(_pack_str(str(body.get("verdict", "HEALTHY"))))
    parts.append(struct.pack("<B", 1 if body.get("is_local_best_val") else 0))
    return b"".join(parts)


def _unpack_step_metrics(data: memoryview) -> tuple[dict[str, Any], int]:
    off = 0
    step_id, version, train_loss = struct.unpack_from("<QQd", data, off)
    off += struct.calcsize("<QQd")
    has_val = data[off]
    off += 1
    val_loss = None
    gap = None
    if has_val:
        val_loss, gap = struct.unpack_from("<dd", data, off)
        off += 16
    verdict, off = _unpack_str(data, off)
    is_best = bool(data[off])
    off += 1
    body: dict[str, Any] = {
        "step_id": int(step_id),
        "version": int(version),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss) if val_loss is not None else None,
        "train_val_gap": float(gap) if gap is not None else None,
        "verdict": verdict,
        "is_local_best_val": is_best,
    }
    return body, off


def _pack_optimizer(o: dict[str, Any]) -> bytes:
    parts = [
        _pack_str(str(o.get("type", "Adam"))),
        struct.pack(
            "<qddd",
            int(o.get("t", 0)),
            float(o.get("beta1", 0.9)),
            float(o.get("beta2", 0.999)),
            float(o.get("eps", 1e-8)),
        ),
        _pack_array_list(o.get("ms_w")),
        _pack_array_list(o.get("vs_w")),
        _pack_array_list(o.get("ms_b")),
        _pack_array_list(o.get("vs_b")),
        _pack_array_list(o.get("ms_g")),
        _pack_array_list(o.get("vs_g")),
        _pack_array_list(o.get("ms_beta")),
        _pack_array_list(o.get("vs_beta")),
    ]
    return b"".join(parts)


def _unpack_optimizer(data: memoryview, off: int) -> tuple[dict[str, Any], int]:
    typ, off = _unpack_str(data, off)
    t, beta1, beta2, eps = struct.unpack_from("<qddd", data, off)
    off += struct.calcsize("<qddd")
    o: dict[str, Any] = {
        "type": typ,
        "t": int(t),
        "beta1": float(beta1),
        "beta2": float(beta2),
        "eps": float(eps),
    }
    for key in ("ms_w", "vs_w", "ms_b", "vs_b", "ms_g", "vs_g", "ms_beta", "vs_beta"):
        o[key], off = _unpack_array_list(data, off)
    return o, off


def pack_checkpoint_body(body: dict[str, Any]) -> bytes:
    val = body.get("val_loss")
    parts = [
        struct.pack("<Q", int(body["version"])),
        struct.pack("<B", 1 if val is not None else 0),
    ]
    if val is not None:
        parts.append(struct.pack("<d", float(val)))
    parts.append(struct.pack("<B", 1 if body.get("is_local_best") else 0))
    parts.extend(
        [
            _pack_array_list(body.get("weights")),
            _pack_array_list(body.get("biases")),
            _pack_array_list(body.get("gammas")),
            _pack_array_list(body.get("betas")),
            _pack_optimizer(body["optimizer"]),
        ]
    )
    return b"".join(parts)


def unpack_checkpoint_body(data: bytes | memoryview) -> dict[str, Any]:
    buf = memoryview(data)
    off = 0
    (version,) = struct.unpack_from("<Q", buf, off)
    off += 8
    has_val = buf[off]
    off += 1
    val_loss = None
    if has_val:
        (val_loss,) = struct.unpack_from("<d", buf, off)
        off += 8
    is_best = bool(buf[off])
    off += 1
    body: dict[str, Any] = {
        "version": int(version),
        "val_loss": float(val_loss) if val_loss is not None else None,
        "is_local_best": is_best,
    }
    body["weights"], off = _unpack_array_list(buf, off)
    body["biases"], off = _unpack_array_list(buf, off)
    body["gammas"], off = _unpack_array_list(buf, off)
    body["betas"], off = _unpack_array_list(buf, off)
    body["optimizer"], off = _unpack_optimizer(buf, off)
    return body


def _pack_body(doc_type: str, body: dict[str, Any]) -> bytes:
    if doc_type == "step.command":
        return _pack_step_command(body)
    if doc_type == "step.result":
        raise TypeError("step.result must use pack_step_result()")
    if doc_type == "step.consolidated":
        return _pack_step_consolidated(body)
    if doc_type == "step.metrics":
        return _pack_step_metrics(body)
    if doc_type == "step.complete":
        return pack_step_complete_body(body)
    if doc_type == "checkpoint":
        return pack_checkpoint_body(body)
    if doc_type == "branch.fork":
        return b"".join(
            [
                _pack_str(str(body["parent_branch_id"])),
                struct.pack(
                    "<QQQ",
                    int(body["parent_version"]),
                    int(body["parent_lsn"]),
                    int(body["checkpoint_version"]),
                ),
                _pack_str(str(body["new_branch_id"])),
                _pack_str(str(body.get("reason", ""))),
                _pack_str(json.dumps(body.get("settings_delta", {}), separators=(",", ":"))),
            ]
        )
    raise ValueError(f"unsupported doc_type for binary pack: {doc_type!r}")


def _unpack_body(doc_type: str, data: memoryview) -> dict[str, Any]:
    if doc_type == "step.command":
        body, _ = _unpack_step_command(data)
        return body
    if doc_type == "step.consolidated":
        body, _ = _unpack_step_consolidated(data)
        return body
    if doc_type == "step.metrics":
        body, _ = _unpack_step_metrics(data)
        return body
    if doc_type == "step.complete":
        return unpack_step_complete_body(data)
    if doc_type == "checkpoint":
        return unpack_checkpoint_body(data)
    if doc_type == "branch.fork":
        off = 0
        parent_branch, off = _unpack_str(data, off)
        parent_version, parent_lsn, cp_version = struct.unpack_from("<QQQ", data, off)
        off += 24
        new_branch, off = _unpack_str(data, off)
        reason, off = _unpack_str(data, off)
        settings_raw, off = _unpack_str(data, off)
        return {
            "parent_branch_id": parent_branch,
            "parent_version": int(parent_version),
            "parent_lsn": int(parent_lsn),
            "new_branch_id": new_branch,
            "reason": reason,
            "settings_delta": json.loads(settings_raw) if settings_raw else {},
            "checkpoint_version": int(cp_version),
        }
    raise ValueError(f"unsupported doc_type for binary unpack: {doc_type!r}")


def pack_document(
    *,
    doc_type: str,
    branch_id: str,
    model_instance_id: str,
    architecture_id: str,
    body: bytes,
    lsn: int | None = None,
    version: int | None = None,
    step_id: int | None = None,
) -> bytes:
    type_id = _DOC_TYPE_TO_ID.get(doc_type)
    if type_id is None:
        raise ValueError(f"unknown doc_type: {doc_type!r}")
    parts = [
        MAGIC,
        struct.pack("<B", _JOURNAL_VERSION),
        struct.pack("<Q", int(lsn or 0)),
        struct.pack("<B", type_id),
        _pack_str(branch_id),
        _pack_str(model_instance_id),
        _pack_str(architecture_id),
        struct.pack("<q", -1 if version is None else int(version)),
        struct.pack("<q", -1 if step_id is None else int(step_id)),
        struct.pack("<I", len(body)),
        body,
    ]
    return b"".join(parts)


def unpack_document(data: bytes | memoryview) -> dict[str, Any]:
    buf = memoryview(data)
    if len(buf) < 4 or bytes(buf[:4]) != MAGIC:
        raise ValueError("not a binary ledger record (bad magic)")
    off = 4
    (ver,) = struct.unpack_from("<B", buf, off)
    off += 1
    if ver != _JOURNAL_VERSION:
        raise ValueError(f"unsupported journal version: {ver}")
    (lsn,) = struct.unpack_from("<Q", buf, off)
    off += 8
    (type_id,) = struct.unpack_from("<B", buf, off)
    off += 1
    doc_type = _ID_TO_DOC_TYPE[type_id]
    branch_id, off = _unpack_str(buf, off)
    model_instance_id, off = _unpack_str(buf, off)
    architecture_id, off = _unpack_str(buf, off)
    (version_i,) = struct.unpack_from("<q", buf, off)
    off += 8
    (step_id_i,) = struct.unpack_from("<q", buf, off)
    off += 8
    (body_len,) = struct.unpack_from("<I", buf, off)
    off += 4
    body_blob = buf[off : off + body_len]
    if doc_type == "step.result":
        result = unpack_step_result(body_blob)
        body = {
            "step_id": result.step_id,
            "loss": result.loss,
            "m_samples": result.m_samples,
            "_result": result,
        }
    elif doc_type == "step.complete":
        body = unpack_step_complete_body(body_blob)
    else:
        body = _unpack_body(doc_type, body_blob)
    return {
        "lsn": int(lsn),
        "doc_type": doc_type,
        "branch_id": branch_id,
        "model_instance_id": model_instance_id,
        "architecture_id": architecture_id,
        "version": None if version_i < 0 else int(version_i),
        "step_id": None if step_id_i < 0 else int(step_id_i),
        "body": body,
    }


def frame_record(payload: bytes) -> bytes:
    return struct.pack("<I", len(payload)) + payload


def iter_framed_records(raw: bytes) -> list[bytes]:
    out: list[bytes] = []
    off = 0
    n = len(raw)
    while off + 4 <= n:
        (rec_len,) = struct.unpack_from("<I", raw, off)
        off += 4
        if off + rec_len > n:
            break
        out.append(bytes(raw[off : off + rec_len]))
        off += rec_len
    return out
