"""The native library holds no approximate reciprocal / reciprocal-square-root instruction.

-ffast-math turns 1/sqrt(x) into RSQRTSS (plus a Newton step), and Intel and AMD return different last bits for it: the same MHSA fit then differs between CPUs from the first step. build_native.sh compiles
mhsa_kernels.cpp with -fno-unsafe-math-optimizations so the layer norm is exact. This test fails if a rebuilt library has such an instruction again. It skips when the library is not built or objdump is missing.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

SO = Path(__file__).resolve().parent.parent / "bin" / "conv_kernels.so"
APPROX = re.compile(r"\b(v?rsqrt(ss|ps)|v?rcp(ss|ps))\b")


@pytest.mark.skipif(not SO.exists() or shutil.which("objdump") is None, reason="needs the built bin/conv_kernels.so and objdump")
def test_the_library_has_no_approximate_rsqrt_or_rcp_instruction():
    asm = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(SO)], capture_output=True, text=True, check=True).stdout
    hits = [ln.strip() for ln in asm.splitlines() if APPROX.search(ln)]
    assert not hits, f"{len(hits)} approximate rsqrt/rcp instructions, e.g. {hits[:3]}"
