# utils/perf_experiments.py
"""Env toggles for perf A/B — mock culprits without permanent behavior changes."""
from __future__ import annotations

import os


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def col2im_native_memset() -> bool:
    """C++ memset inside col2im_avx2 (per call). Default off."""
    return _flag("ML_ENGINE_COL2IM_MEMSET", default=False)


def step_dx_zero() -> bool:
    """arena.zero_dx_buffers once per spatial backward. Default on."""
    return _flag("ML_ENGINE_STEP_DX_ZERO", default=True)


def aligned_scratch() -> bool:
    """32B-aligned scratch arena buffers. Default on."""
    return _flag("ML_ENGINE_ALIGNED_SCRATCH", default=True)


def skip_dx_zero() -> bool:
    """Disable ALL dx zeroing (wrong grads — perf ceiling only)."""
    return _flag("ML_ENGINE_SKIP_DX_ZERO", default=False)


def perf_ab_mode() -> bool:
    """When set, do not override ML_ENGINE_* perf toggles (manual A/B)."""
    return _flag("ML_ENGINE_PERF_AB", default=False)


def apply_im2col_gemm_perf_defaults() -> None:
    """Production im2col+gemm buffer policy (one dx zero/step, no per-col2im memset)."""
    if perf_ab_mode():
        return
    os.environ["ML_ENGINE_COL2IM_MEMSET"] = "0"
    os.environ["ML_ENGINE_STEP_DX_ZERO"] = "1"
    os.environ["ML_ENGINE_ALIGNED_SCRATCH"] = "1"
    os.environ.pop("ML_ENGINE_SKIP_DX_ZERO", None)


def experiment_summary() -> str:
    return (
        f"perf_experiments: col2im_memset={col2im_native_memset()} "
        f"step_dx_zero={step_dx_zero() and not skip_dx_zero()} "
        f"aligned_scratch={aligned_scratch()} "
        f"skip_dx_zero={skip_dx_zero()}"
    )
