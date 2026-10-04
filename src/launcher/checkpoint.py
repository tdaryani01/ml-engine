"""Read what an ML engine checkpoint document says about itself (a pure function of its bytes; where the bytes come from is the
caller's business)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_physical_version(checkpoint_path: Path | None) -> int:
    """The cumulative (absolute) version stored in a checkpoint file; missing or unreadable means 0 (a cold start).

    Supports ML engine checkpoint documents and JSON stand-ins."""
    if checkpoint_path is None:
        return 0
    path = Path(checkpoint_path)
    if not path.is_file():
        return 0
    try:
        data = path.read_bytes()
    except OSError:
        return 0
    if not data:
        return 0
    stripped = data.lstrip()
    if stripped.startswith(b"{") or stripped.startswith(b"["):
        try:
            meta = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return 0
        if isinstance(meta, dict):
            for key in ("version", "traj", "physical_version"):
                if meta.get(key) is not None:
                    try:
                        return max(0, int(meta[key]))
                    except (TypeError, ValueError):
                        continue
        return 0
    try:
        from src.ledger import document_from_bytes

        doc = document_from_bytes(data)
        ver = getattr(doc, "version", None)
        if ver is not None:
            return max(0, int(ver))
        body = getattr(doc, "body", None)
        if isinstance(body, dict) and body.get("version") is not None:
            return max(0, int(body["version"]))
    except Exception:  # noqa: BLE001
        pass
    return 0


class CheckpointConfigError(Exception):
    """``status`` is the HTTP code an API would map it to (400 bad request, 404 missing or undecodable)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def config_from_checkpoint_bytes(raw: bytes, handle: str = "") -> dict[str, Any]:
    """``{handle, version, config, knobs}`` from a checkpoint's bytes (weights are never returned). The config is the one the
    checkpoint was written with: ML engine puts it beside the model state."""
    if not raw:
        raise CheckpointConfigError(404, "empty blob")
    body: dict | None = None
    stripped = raw.lstrip()
    if stripped.startswith(b"{") or stripped.startswith(b"["):
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict):
            body = parsed
    if body is None:
        try:
            from src.ledger import document_from_bytes

            doc_body = document_from_bytes(raw).body
            if isinstance(doc_body, dict):
                body = doc_body
        except Exception as exc:  # noqa: BLE001
            raise CheckpointConfigError(404, f"undecodable checkpoint: {exc}") from exc
    if not isinstance(body, dict):
        raise CheckpointConfigError(404, "checkpoint body not a dict")
    cfg = body.get("config") if isinstance(body.get("config"), dict) else {}
    knobs = body.get("knobs") if isinstance(body.get("knobs"), dict) else {}
    try:
        version = int(body["version"]) if body.get("version") is not None else None
    except (TypeError, ValueError):
        version = None
    return {"handle": handle, "version": version, "config": dict(cfg), "knobs": dict(knobs)}


def imitation_model_from_checkpoint_bytes(raw: bytes) -> dict[str, Any]:
    """``{blob, version, val_loss}`` from an imitation checkpoint: the brain's weights and what they scored on held-out tapes."""
    from src.ledger import document_from_bytes

    try:
        body = document_from_bytes(raw).body
        state = body["state"]
        blob = bytes(state["blob"].astype("uint8").tobytes() if hasattr(state["blob"], "astype") else bytes(state["blob"]))
    except Exception as exc:  # noqa: BLE001
        raise CheckpointConfigError(404, f"not an imitation checkpoint: {exc}") from exc
    return {"blob": blob, "version": body.get("version"), "val_loss": body.get("val_loss")}
