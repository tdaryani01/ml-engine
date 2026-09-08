#!/usr/bin/env python3
"""Diagnostic: side-by-side forward layout/ops for native vs Torch/oneDNN.

Uses ML_ENGINE_FWD_TRACE=1 and ONEDNN_VERBOSE=1. Does not change kernels beyond
what those env flags already enable. Run:

  ML_ENGINE_FWD_TRACE=1 ONEDNN_VERBOSE=1 OMP_NUM_THREADS=4 \\
    .venv/bin/python scripts/fwd_layout_trace.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["ML_ENGINE_FWD_TRACE"] = "1"
# Ensure oneDNN verbose even if parent did not set it.
os.environ.setdefault("ONEDNN_VERBOSE", "1")

from config.constants import EngineBackend
from utils.conv_dispatch import conv2d_forward, init_engine_backend, sync_native_thread_policy


def _fmt_tensor(name: str, t: torch.Tensor) -> None:
    mf = (
        "channels_last"
        if t.is_contiguous(memory_format=torch.channels_last)
        else ("contiguous" if t.is_contiguous() else "non-contiguous")
    )
    print(
        f"[TORCH_TRACE] {name}: shape={tuple(t.shape)} stride={tuple(t.stride())} "
        f"dtype={t.dtype} {mf} ptr={t.data_ptr():#x}",
        flush=True,
    )
    flat = t.detach().float().contiguous().view(-1)
    n = min(8, flat.numel())
    vals = ", ".join(f"{float(flat[i]):.5g}" for i in range(n))
    print(f"[TORCH_TRACE]   first[{n}]={vals}", flush=True)


def run_case(tag: str, N: int, Cin: int, Cout: int, H: int, W: int, K: int, pad: int) -> None:
    oh = (H + 2 * pad - K) + 1
    ow = (W + 2 * pad - K) + 1
    rng = np.random.default_rng(0)
    x = np.ascontiguousarray(rng.standard_normal((N, Cin, H, W), dtype=np.float32))
    Wn = np.ascontiguousarray(rng.standard_normal((Cout, Cin, K, K), dtype=np.float32))
    b = np.zeros((Cout,), dtype=np.float32)
    out = np.empty((N, Cout, oh, ow), dtype=np.float32)

    print("\n" + "=" * 72, flush=True)
    print(f"CASE {tag}: N={N} {Cin}->{Cout} in={H}x{W} K={K} pad={pad} out={oh}x{ow}", flush=True)
    print("=" * 72, flush=True)

    print("\n----- NATIVE (ML_ENGINE_FWD_TRACE) -----", flush=True)
    conv2d_forward(
        x, Wn, b, out_buf=out, stride=1, pad=pad, backend=EngineBackend.NATIVE
    )
    print(
        f"[NATIVE_TRACE] out NCHW shape={out.shape} "
        f"first8={', '.join(f'{v:.5g}' for v in out.reshape(-1)[:8])}",
        flush=True,
    )

    print("\n----- TORCH / oneDNN (ONEDNN_VERBOSE + strides) -----", flush=True)
    xt = torch.from_numpy(x.copy())
    Wt = torch.from_numpy(Wn.copy())
    _fmt_tensor("x before", xt)
    _fmt_tensor("W before", Wt)
    print("[TORCH_TRACE] op: F.conv2d → oneDNN may reorder + jit:avx2 + reorder", flush=True)
    yt = F.conv2d(xt, Wt, None, stride=1, padding=pad)
    _fmt_tensor("y after", yt)
    print(
        f"[TORCH_TRACE] compare max_abs(native-torch)="
        f"{float(np.max(np.abs(out - yt.detach().numpy()))):.3e}",
        flush=True,
    )


def main() -> None:
    init_engine_backend(EngineBackend.NATIVE)
    sync_native_thread_policy(4)
    torch.set_num_threads(4)
    print(
        f"torch={torch.__version__} mkldnn={torch.backends.mkldnn.is_available()} "
        f"ONEDNN_VERBOSE={os.environ.get('ONEDNN_VERBOSE')} "
        f"ML_ENGINE_FWD_TRACE={os.environ.get('ML_ENGINE_FWD_TRACE')}",
        flush=True,
    )
    # L1 pad2 — the gap case (Cin%8==0 → OC-mimo path)
    run_case("L1_pad2", 2, 8, 16, 13, 13, 7, 2)
    # L0 pad2 — Cin=3 → OW specialists
    run_case("L0_pad2", 2, 3, 8, 28, 28, 7, 2)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
