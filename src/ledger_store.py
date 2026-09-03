# src/ledger_store.py
"""Pluggable ledger persistence: sync file, streaming file (async-style), future Redis."""
from __future__ import annotations

import json
import os
import struct
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, Union

from src.ledger import CHECKPOINT, LedgerDocument, document_from_bytes, document_to_bytes
from src.ledger_wire import frame_record, iter_framed_records

_QueueItem = Union[LedgerDocument, bytes]


class LedgerStore(Protocol):
    """Append-only document store. Implementations choose durability strategy."""

    def push(self, doc: LedgerDocument) -> int:
        """Assign LSN and enqueue/submit. Must not block on remote/disk completion."""
        ...

    def poll(self, limit: int = 8) -> int:
        """Deprecated: use begin_flush / try_reap_flush from the training loop."""
        ...

    def begin_flush(self) -> bool:
        """Encode one queued record and start async write. Returns True if submitted."""
        ...

    def try_reap_flush(self) -> bool:
        """Non-blocking: reap completed write and update head_lsn."""
        ...

    def has_flush_pending(self) -> bool:
        ...

    def queue_pending(self) -> bool:
        ...

    def flush(self) -> None:
        """Block until all prior push() records are durable."""
        ...

    def close(self) -> None:
        ...

    def get(self, lsn: int) -> LedgerDocument: ...

    def scan(self, from_lsn: int = 1, to_lsn: int | None = None) -> Iterator[LedgerDocument]: ...

    def head_lsn(self) -> int: ...

    def put_checkpoint(self, doc: LedgerDocument) -> None: ...

    def get_checkpoint(self, branch_id: str, version: int) -> LedgerDocument | None: ...


@dataclass
class _HeadMeta:
    next_lsn: int = 1
    head_lsn: int = 0


def _iter_journal_payloads(path: Path):
    if not path.exists():
        return
    with open(path, "rb") as f:
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            (rec_len,) = struct.unpack("<I", hdr)
            payload = f.read(rec_len)
            if len(payload) < rec_len:
                break
            yield payload


def _scan_head_lsn_from_journal(path: Path) -> int:
    last = 0
    for payload in _iter_journal_payloads(path):
        doc = document_from_bytes(payload)
        if doc.lsn is not None:
            last = doc.lsn
    return last


def _load_head_meta(path: Path) -> _HeadMeta:
    if not path.exists():
        return _HeadMeta()
    data = json.loads(path.read_text(encoding="utf-8"))
    next_lsn = int(data.get("next_lsn", 1))
    head_lsn = int(data.get("head_lsn", max(0, next_lsn - 1)))
    return _HeadMeta(next_lsn=next_lsn, head_lsn=head_lsn)


def _save_head_meta(path: Path, meta: _HeadMeta) -> None:
    path.write_text(
        json.dumps({"next_lsn": meta.next_lsn, "head_lsn": meta.head_lsn}, separators=(",", ":")),
        encoding="utf-8",
    )


class SyncFileLedgerStore:
    """Per-push durable append (tests). One journal handle; fsync each push."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.root / "journal.bin"
        self.meta_path = self.root / "journal.head"
        self.checkpoint_dir = self.root / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._meta = _load_head_meta(self.meta_path)
        if self._meta.next_lsn == 1 and self.journal_path.exists() and not self.meta_path.exists():
            head = _scan_head_lsn_from_journal(self.journal_path)
            self._meta = _HeadMeta(next_lsn=head + 1, head_lsn=head)
        self._fh = open(self.journal_path, "ab", buffering=0)

    def _scan_head_lsn(self) -> int:
        return _scan_head_lsn_from_journal(self.journal_path)

    def push(self, doc: LedgerDocument) -> int:
        lsn = self._meta.next_lsn
        self._meta.next_lsn += 1
        self._meta.head_lsn = lsn
        doc.lsn = lsn
        self._fh.write(frame_record(document_to_bytes(doc)))
        self._fh.flush()
        os.fsync(self._fh.fileno())
        return lsn

    def poll(self, limit: int = 8) -> int:
        return 0

    def begin_flush(self) -> bool:
        return False

    def try_reap_flush(self) -> bool:
        return False

    def has_flush_pending(self) -> bool:
        return False

    def queue_pending(self) -> bool:
        return False

    def flush(self) -> None:
        _save_head_meta(self.meta_path, self._meta)

    def close(self) -> None:
        self.flush()
        self._fh.close()

    def get(self, lsn: int) -> LedgerDocument:
        for doc in self.scan(from_lsn=lsn, to_lsn=lsn):
            return doc
        raise KeyError(f"LSN {lsn} not found")

    def scan(self, from_lsn: int = 1, to_lsn: int | None = None) -> Iterator[LedgerDocument]:
        for payload in _iter_journal_payloads(self.journal_path):
            doc = document_from_bytes(payload)
            if doc.lsn is None or doc.lsn < from_lsn:
                continue
            if to_lsn is not None and doc.lsn > to_lsn:
                break
            yield doc

    def head_lsn(self) -> int:
        return self._meta.head_lsn

    def _checkpoint_path(self, branch_id: str, version: int) -> Path:
        safe_branch = branch_id.replace(os.sep, "_")
        return self.checkpoint_dir / f"{safe_branch}_v{version}.bin"

    def put_checkpoint(self, doc: LedgerDocument) -> None:
        if doc.doc_type != CHECKPOINT:
            raise ValueError("put_checkpoint expects doc_type=checkpoint")
        version = int(doc.body["version"])
        path = self._checkpoint_path(doc.branch_id, version)
        path.write_bytes(frame_record(document_to_bytes(doc)))
        self.flush()

    def get_checkpoint(self, branch_id: str, version: int) -> LedgerDocument | None:
        path = self._checkpoint_path(branch_id, version)
        if not path.exists():
            return None
        records = iter_framed_records(path.read_bytes())
        if not records:
            return None
        return document_from_bytes(records[0])


class StreamingFileLedgerStore:
    """
    push() queues in memory; begin_flush() encodes+submits one write (no thread).
    Windows: overlapped WriteFile + Wait(0) check-back.
    Linux: sync write on submit; reap marks durable on next tick.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.root / "journal.bin"
        self.meta_path = self.root / "journal.head"
        self.checkpoint_dir = self.root / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._meta = _load_head_meta(self.meta_path)
        if self._meta.next_lsn == 1 and self.journal_path.exists() and not self.meta_path.exists():
            head = _scan_head_lsn_from_journal(self.journal_path)
            self._meta = _HeadMeta(next_lsn=head + 1, head_lsn=head)
        self._queue: deque[tuple[int, _QueueItem]] = deque()
        self._writer = _open_async_writer(self.journal_path)
        self._closed = False

    
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
            return _encode_queue_item(item)

        if self._writer.submit_work(_encode, lsn):
            return True
        self._queue.appendleft((lsn, item))
        return False

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
        """Compat shim: reap once then begin one flush."""
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
        _save_head_meta(self.meta_path, self._meta)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
        finally:
            self._writer.close()
            self._closed = True

    def get(self, lsn: int) -> LedgerDocument:
        self.flush()
        for doc in self.scan(from_lsn=lsn, to_lsn=lsn):
            return doc
        raise KeyError(f"LSN {lsn} not found")

    def scan(self, from_lsn: int = 1, to_lsn: int | None = None) -> Iterator[LedgerDocument]:
        self.flush()
        for payload in _iter_journal_payloads(self.journal_path):
            doc = document_from_bytes(payload)
            if doc.lsn is None or doc.lsn < from_lsn:
                continue
            if to_lsn is not None and doc.lsn > to_lsn:
                break
            yield doc

    def head_lsn(self) -> int:
        return self._meta.head_lsn

    def _checkpoint_path(self, branch_id: str, version: int) -> Path:
        safe_branch = branch_id.replace(os.sep, "_")
        return self.checkpoint_dir / f"{safe_branch}_v{version}.bin"

    def put_checkpoint(self, doc: LedgerDocument) -> None:
        if doc.doc_type != CHECKPOINT:
            raise ValueError("put_checkpoint expects doc_type=checkpoint")
        version = int(doc.body["version"])
        path = self._checkpoint_path(doc.branch_id, version)
        path.write_bytes(frame_record(document_to_bytes(doc)))

    def get_checkpoint(self, branch_id: str, version: int) -> LedgerDocument | None:
        path = self._checkpoint_path(branch_id, version)
        if not path.exists():
            return None
        records = iter_framed_records(path.read_bytes())
        if not records:
            return None
        return document_from_bytes(records[0])


class RedisLedgerStore:
    """Placeholder for distributed ledger backend (Redis Streams, etc.)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Redis ledger backend is not implemented. Use store_backend=file_streaming or file_sync."
        )


def _encode_queue_item(item: _QueueItem) -> bytes:
    if isinstance(item, bytes):
        return item
    return frame_record(document_to_bytes(item))


def _open_async_writer(path: Path):
    import sys

    if sys.platform == "win32":
        try:
            from src.ledger_store_win import WinOverlappedJournalWriter

            return WinOverlappedJournalWriter(path)
        except OSError:
            pass
    from src.ledger_async_writer import SyncJournalWriter

    return SyncJournalWriter(path)


# Back-compat alias — production default is streaming, not per-push sync open/close.
FileLedgerStore = StreamingFileLedgerStore


def create_ledger_store(backend: str, root: str | Path) -> LedgerStore:
    key = backend.strip().lower()
    if key in ("file_sync", "sync", "file"):
        return SyncFileLedgerStore(root)
    if key in ("file_streaming", "streaming", "async"):
        return StreamingFileLedgerStore(root)
    if key in ("redis",):
        return RedisLedgerStore(root)
    raise ValueError(f"Unknown ledger store backend: {backend!r}")
