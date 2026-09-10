"""HTTP Training Manager ledger store — JSON-friendly docs to TM /api/ledger/docs.

store_backend: http_tm

Binary payloads (e.g. step.result ``_result``) are stripped before POST; metrics /
command envelopes remain. Local replay is not supported (scan/get empty like noop
except in-memory checkpoints for early-stopping).
"""
from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Iterator

from src.ledger import CHECKPOINT, LedgerDocument
from src.ledger_store import _HeadMeta, _QueueItem

_log = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    """Best-effort conversion for TM JSONB bodies; drop non-serializable leaves."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if k == "_result":
                continue
            try:
                out[str(k)] = _jsonable(v)
            except TypeError:
                continue
        return out
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    # numpy scalars / arrays
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _jsonable(tolist())
        except Exception:
            pass
    raise TypeError(f"not jsonable: {type(value)!r}")


class HttpTmLedgerStore:
    """Queue like noop; on flush POST a JSON doc to the Training Manager API."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        uri: str,
        timeout_s: float = 0.5,
    ):
        from src.ledger_async_writer import NoopJournalWriter

        if not uri or not str(uri).strip():
            raise ValueError("http_tm store requires a non-empty training_manager uri")
        self.root = Path(root) if root is not None else Path(".")
        self._uri = str(uri).rstrip("/")
        self._timeout_s = float(timeout_s)
        self._meta = _HeadMeta()
        self._queue: deque[tuple[int, _QueueItem]] = deque()
        self._writer = NoopJournalWriter(None)
        self._checkpoints: dict[tuple[str, int], LedgerDocument] = {}
        self._closed = False
        self._post_lock = threading.Lock()

    def push(self, doc: LedgerDocument) -> int:
        if self._closed:
            raise RuntimeError("ledger store is closed")
        lsn = self._meta.next_lsn
        self._meta.next_lsn += 1
        doc.lsn = lsn
        self._queue.append((lsn, doc))
        return lsn

    def begin_flush(self) -> bool:
        if self._closed or self._writer.has_pending() or not self._queue:
            return False
        lsn, item = self._queue.popleft()

        def _encode() -> bytes:
            return b""

        if not self._writer.submit_work(_encode, lsn):
            self._queue.appendleft((lsn, item))
            return False
        # NoopJournalWriter skips work(); post after accept so LSN still advances.
        if isinstance(item, LedgerDocument):
            threading.Thread(
                target=self._post_doc,
                args=(item,),
                name="http-tm-ledger",
                daemon=True,
            ).start()
        return True

    def try_reap_flush(self) -> bool:
        completed, lsn = self._writer.try_reap()
        if completed and lsn is not None:
            self._meta.head_lsn = lsn
            return True
        return False

    def has_flush_pending(self) -> bool:
        return self._writer.has_pending()

    def queue_pending(self) -> bool:
        return bool(self._queue)

    def poll(self, limit: int = 8) -> int:
        n = 0
        if self.try_reap_flush():
            n += 1
        if n < limit and self.begin_flush():
            n += 1
        return n

    def flush(self) -> None:
        while self._queue or self._writer.has_pending():
            if self._writer.has_pending():
                lsn = self._writer.wait_pending()
                if lsn is not None:
                    self._meta.head_lsn = lsn
            elif self._queue:
                self.begin_flush()
        self._writer.flush_os()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
        finally:
            self._writer.close()
            self._closed = True

    def get(self, lsn: int) -> LedgerDocument:
        raise KeyError(f"LSN {lsn} not found (http_tm store does not retain a local journal)")

    def scan(self, from_lsn: int = 1, to_lsn: int | None = None) -> Iterator[LedgerDocument]:
        return iter(())

    def head_lsn(self) -> int:
        return self._meta.head_lsn

    def put_checkpoint(self, doc: LedgerDocument) -> None:
        if doc.doc_type != CHECKPOINT:
            raise ValueError("put_checkpoint expects doc_type=checkpoint")
        version = int(doc.body["version"])
        self._checkpoints[(doc.branch_id, version)] = doc
        # Mirror a thin checkpoint notice to TM (no weight blobs).
        thin = LedgerDocument(
            doc_type=CHECKPOINT,
            branch_id=doc.branch_id,
            model_instance_id=doc.model_instance_id,
            architecture_id=doc.architecture_id,
            body={
                "version": version,
                "note": "checkpoint_meta",
            },
            lsn=doc.lsn,
            version=doc.version,
            step_id=doc.step_id,
        )
        self._post_doc(thin)

    def get_checkpoint(self, branch_id: str, version: int) -> LedgerDocument | None:
        return self._checkpoints.get((branch_id, version))

    def _post_doc(self, doc: LedgerDocument) -> None:
        try:
            body = _jsonable(dict(doc.body))
        except TypeError as exc:
            _log.debug("http_tm skip non-json doc_type=%s: %s", doc.doc_type, exc)
            return
        payload = {
            "instance_id": doc.model_instance_id,
            "branch_id": doc.branch_id,
            "doc_type": doc.doc_type,
            "body": body,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._uri}/api/ledger/docs",
            data=data,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with self._post_lock:
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                    if not (200 <= int(getattr(resp, "status", 200)) < 300):
                        _log.debug("http_tm POST status=%s", getattr(resp, "status", "?"))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                _log.debug("http_tm POST failed: %s", exc)
