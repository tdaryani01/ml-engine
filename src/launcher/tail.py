"""Read ME's ledger journal incrementally while ME is still writing it.

The journal is a sequence of length-prefixed records (``<u32 length><payload>``). A writer that is
mid-flush leaves a partial record at the end; the tail only returns complete records and picks up
at the same byte offset on the next poll. The decoding lives in ME (``document_from_bytes``), so EE
depends on ME's document schema only through that function.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any



class LedgerTail:
    def __init__(self, ledger_dir: Path | str) -> None:
        self._journal = Path(ledger_dir) / "journal.bin"
        self._offset = 0
        self.consumed = 0  # complete records returned so far

    def poll(self) -> list[Any]:
        """New complete documents since the last poll (empty when none)."""
        if not self._journal.is_file():
            return []
        size = self._journal.stat().st_size
        if size < self._offset:
            # The journal shrank (a truncated prefix): start over rather than read garbage.
            self._offset = 0
        if size == self._offset:
            return []
        from src.ledger import document_from_bytes

        with open(self._journal, "rb") as fh:
            fh.seek(self._offset)
            raw = fh.read(size - self._offset)
        docs: list[Any] = []
        off = 0
        n = len(raw)
        while off + 4 <= n:
            (rec_len,) = struct.unpack_from("<I", raw, off)
            if off + 4 + rec_len > n:
                break  # partial trailing record: wait for the writer
            payload = raw[off + 4 : off + 4 + rec_len]
            docs.append(document_from_bytes(payload))
            off += 4 + rec_len
        self._offset += off
        self.consumed += len(docs)
        return docs
