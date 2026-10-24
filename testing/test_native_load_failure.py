# testing/test_native_load_failure.py
"""A native library that is present but cannot be loaded (on Windows: its OpenMP runtime, libomp140.x86_64.dll, is missing on a PC without Visual Studio) is "no native library", not a crash:
the first real fit on such a PC died with FileNotFoundError out of ctypes.CDLL, before the Numba fallback (ML_ENGINE_NATIVE_FALLBACK) was ever consulted."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import conv_dispatch as c  # noqa: E402


class _Log:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg: str, *args) -> None:
        self.warnings.append(msg % args if args else msg)

    def __getattr__(self, _name):          # debug / info / error: not under test
        return lambda *a, **k: None


def _break_the_library(monkeypatch) -> _Log:
    log = _Log()
    monkeypatch.setattr(c, "logger", log)
    monkeypatch.setattr(c, "_primitives_lib", None)
    monkeypatch.setattr(c, "_native_load_failure_seen", False, raising=False)      # raising=False: this test also runs against the code before the fix
    monkeypatch.setattr(c, "_locate_native_dll", lambda: "C:/nowhere/conv_kernels.dll")

    def cannot_load(path):
        raise OSError("[WinError 126] The specified module could not be found: " + str(path))

    monkeypatch.setattr(c.ctypes, "CDLL", cannot_load)
    return log


def test_a_library_that_cannot_load_is_no_native_library_and_says_so_once(monkeypatch) -> None:
    log = _break_the_library(monkeypatch)
    assert c._load_conv_dll() is None
    assert c._ensure_primitives_lib() is None
    assert c._load_conv_dll() is None
    assert len(log.warnings) == 1 and "could not be loaded" in log.warnings[0] and "conv_kernels.dll" in log.warnings[0]


def test_with_the_fallback_allowed_a_missed_native_call_takes_the_numba_path(monkeypatch) -> None:
    monkeypatch.setenv("ML_ENGINE_NATIVE_FALLBACK", "1")
    assert c._native_miss("conv2d_forward", "the library could not be loaded") == c.EngineBackend.IM2COL_GEMM


def test_without_the_fallback_a_missed_native_call_still_refuses_loudly(monkeypatch) -> None:
    monkeypatch.delenv("ML_ENGINE_NATIVE_FALLBACK", raising=False)
    monkeypatch.delenv("ML_ENGINE_FORCE_FALLBACK", raising=False)
    try:
        c._native_miss("conv2d_forward", "the library could not be loaded")
    except RuntimeError as exc:
        assert "ML_ENGINE_NATIVE_FALLBACK=1" in str(exc)
    else:
        raise AssertionError("the strict native-only default must still refuse")
