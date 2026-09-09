# src/ledger_async_writer.py
"""Journal writers without background threads — sync (Linux) or overlapped (Windows)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


class SyncJournalWriter:
    """Encode+write on caller thread; try_reap() marks prior write done. No threads."""

    def __init__(self, path: Path):
        self._fh = open(path, "ab", buffering=0)
        self._reap_lsn: int | None = None

    def submit_work(self, work: Callable[[], bytes], lsn: int) -> bool:
        if self._reap_lsn is not None:
            return False
        self._fh.write(work())
        self._reap_lsn = lsn
        return True

    def try_reap(self) -> tuple[bool, int | None]:
        if self._reap_lsn is None:
            return False, None
        lsn = self._reap_lsn
        self._reap_lsn = None
        return True, lsn

    def has_pending(self) -> bool:
        return self._reap_lsn is not None

    def wait_pending(self) -> int | None:
        if self._reap_lsn is None:
            return None
        lsn = self._reap_lsn
        self._reap_lsn = None
        return lsn

    def flush_os(self) -> None:
        self.wait_pending()
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        try:
            self.wait_pending()
        finally:
            self._fh.close()


class NoopJournalWriter:
    """Same submit/reap contract as SyncJournalWriter; discards bytes and never calls work()."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self._reap_lsn: int | None = None

    def submit_work(self, work: Callable[[], bytes], lsn: int) -> bool:
        if self._reap_lsn is not None:
            return False
        # Intentionally skip work() — no encode, no I/O.
        self._reap_lsn = lsn
        return True

    def try_reap(self) -> tuple[bool, int | None]:
        if self._reap_lsn is None:
            return False, None
        lsn = self._reap_lsn
        self._reap_lsn = None
        return True, lsn

    def has_pending(self) -> bool:
        return self._reap_lsn is not None

    def wait_pending(self) -> int | None:
        if self._reap_lsn is None:
            return None
        lsn = self._reap_lsn
        self._reap_lsn = None
        return lsn

    def flush_os(self) -> None:
        self.wait_pending()

    def close(self) -> None:
        self.wait_pending()
