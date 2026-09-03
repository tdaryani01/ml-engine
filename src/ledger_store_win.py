# src/ledger_store_win.py
"""Windows overlapped journal writer — submit without waiting; reap with Wait(0) only."""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from pathlib import Path
from typing import Callable

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

GENERIC_WRITE = 0x40000000
FILE_SHARE_READ = 0x00000001
OPEN_ALWAYS = 4
FILE_ATTRIBUTE_NORMAL = 0x80
FILE_FLAG_OVERLAPPED = 0x40000000
INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value
ERROR_IO_PENDING = 997
INFINITE = 0xFFFFFFFF
WAIT_OBJECT_0 = 0


class OVERLAPPED(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_ulonglong),
        ("InternalHigh", ctypes.c_ulonglong),
        ("Offset", wintypes.DWORD),
        ("OffsetHigh", wintypes.DWORD),
        ("hEvent", wintypes.HANDLE),
    ]


class WinOverlappedJournalWriter:
    def __init__(self, path: Path):
        self._path = path
        self._handle = kernel32.CreateFileW(
            str(path),
            GENERIC_WRITE,
            FILE_SHARE_READ,
            None,
            OPEN_ALWAYS,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
            None,
        )
        if self._handle == INVALID_HANDLE_VALUE:
            raise OSError(ctypes.get_last_error(), "CreateFileW failed")
        self._offset = os.path.getsize(path) if path.exists() else 0
        self._active: tuple[OVERLAPPED, int, int, ctypes.Array] | None = None
        self._event = kernel32.CreateEventW(None, True, False, None)

    def submit_work(self, work: Callable[[], bytes], lsn: int) -> bool:
        if self._active is not None:
            return False
        payload = work()
        ov = OVERLAPPED()
        ov.hEvent = self._event
        ov.Offset = self._offset & 0xFFFFFFFF
        ov.OffsetHigh = (self._offset >> 32) & 0xFFFFFFFF
        buf = (ctypes.c_char * len(payload)).from_buffer_copy(payload)
        nwritten = wintypes.DWORD()
        ok = kernel32.WriteFile(
            self._handle,
            buf,
            len(payload),
            ctypes.byref(nwritten),
            ctypes.byref(ov),
        )
        if not ok:
            err = ctypes.get_last_error()
            if err != ERROR_IO_PENDING:
                raise OSError(err, "WriteFile failed")
        self._active = (ov, lsn, len(payload), buf)
        return True

    def try_reap(self) -> tuple[bool, int | None]:
        if self._active is None:
            return False, None
        ov, lsn, _nbytes, _buf = self._active
        rc = kernel32.WaitForSingleObject(ov.hEvent, 0)
        if rc != WAIT_OBJECT_0:
            return False, None
        nwritten = wintypes.DWORD()
        if not kernel32.GetOverlappedResult(
            self._handle, ctypes.byref(ov), ctypes.byref(nwritten), False
        ):
            raise OSError(ctypes.get_last_error(), "GetOverlappedResult failed")
        self._offset += int(nwritten.value)
        kernel32.ResetEvent(ov.hEvent)
        self._active = None
        return True, lsn

    def has_pending(self) -> bool:
        return self._active is not None

    def wait_pending(self) -> int | None:
        if self._active is None:
            return None
        ov, lsn, _nbytes, _buf = self._active
        rc = kernel32.WaitForSingleObject(ov.hEvent, INFINITE)
        if rc != WAIT_OBJECT_0:
            raise OSError(ctypes.get_last_error(), "WaitForSingleObject failed")
        nwritten = wintypes.DWORD()
        if not kernel32.GetOverlappedResult(
            self._handle, ctypes.byref(ov), ctypes.byref(nwritten), False
        ):
            raise OSError(ctypes.get_last_error(), "GetOverlappedResult failed")
        self._offset += int(nwritten.value)
        kernel32.ResetEvent(ov.hEvent)
        self._active = None
        return lsn

    def flush_os(self) -> None:
        self.wait_pending()
        kernel32.FlushFileBuffers(self._handle)

    def close(self) -> None:
        try:
            self.wait_pending()
        finally:
            if self._handle != INVALID_HANDLE_VALUE:
                kernel32.CloseHandle(self._handle)
                self._handle = INVALID_HANDLE_VALUE
            if self._event:
                kernel32.CloseHandle(self._event)
                self._event = None
