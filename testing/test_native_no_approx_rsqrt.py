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
# the SSE/AVX forms and the AVX-512 14/28-bit and FP16 forms; the same pattern as the build-time check in build_native.sh
APPROX = re.compile(r"\b(v?rsqrt(14|28)?(ss|ps|sd|pd|sh|ph)|v?rcp(14|28)?(ss|ps|sd|pd|sh|ph))\b")


@pytest.mark.skipif(not SO.exists() or shutil.which("objdump") is None, reason="needs the built bin/conv_kernels.so and objdump")
def test_the_library_has_no_approximate_rsqrt_or_rcp_instruction():
    asm = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(SO)], capture_output=True, text=True, check=True).stdout
    hits = [ln.strip() for ln in asm.splitlines() if APPROX.search(ln)]
    assert not hits, f"{len(hits)} approximate rsqrt/rcp instructions, e.g. {hits[:3]}"


@pytest.mark.parametrize("line", ["vrsqrtss %xmm9,%xmm9,%xmm2", "rsqrtps %xmm1,%xmm0", "vrcpps %ymm1,%ymm0", "vrsqrt14ps %zmm1,%zmm0", "vrsqrt14ss %xmm1,%xmm1,%xmm0",
                                  "vrcp14ps %zmm1,%zmm0", "vrsqrt28ps %zmm1,%zmm0", "vrcp14sd %xmm1,%xmm1,%xmm0", "vrsqrtph %zmm1,%zmm0", "vrcpsh %xmm1,%xmm1,%xmm0"])
def test_the_pattern_catches_every_approximate_form(line):
    assert APPROX.search("   83d0e:\t" + line)         # runs everywhere (no library needed): the AVX-512 forms were missed by the first pattern


@pytest.mark.parametrize("line", ["vsqrtps %ymm1,%ymm0", "vdivps %ymm1,%ymm2,%ymm3", "vmulps %ymm1,%ymm2,%ymm3", "vfmadd231ps %ymm1,%ymm2,%ymm3"])
def test_the_pattern_leaves_the_exact_instructions_alone(line):
    assert not APPROX.search("   83d0e:\t" + line)
