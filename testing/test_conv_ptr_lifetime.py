# testing/test_conv_ptr_lifetime.py
"""
ml-engine#60: a non-contiguous array handed to the native conv kernels is copied to a contiguous temporary first. The pointer to that copy used to outlive the copy (``_get_c_ptr`` returned only the address),
so the kernel read freed memory: gradients that changed with the heap layout, ASLR and MALLOC_PERTURB_, and a zero gradient on the first backward of a process.

Run standalone:  python testing/test_conv_ptr_lifetime.py      (the bit-identity check sets one native thread itself)
"""
import ctypes
import gc
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import conv_dispatch as cd  # noqa: E402


def _ref_conv_backward(dout, x, W, pad, inv_m):
    """Plain NumPy: dx[n,ci,ih,iw] = sum dout[n,co,oh,ow] W[co,ci,kh,kw] with ih = oh - pad + kh; dW = inv_m * sum dout x."""
    N, C_in, H, Wd = x.shape
    C_out, _, kh, kw = W.shape
    OH, OW = dout.shape[2], dout.shape[3]
    dx = np.zeros_like(x, dtype=np.float64)
    dW = np.zeros(W.shape, dtype=np.float64)
    for co in range(C_out):
        for ci in range(C_in):
            for a in range(kh):
                for b in range(kw):
                    for oh in range(OH):
                        ih = oh - pad + a
                        if not 0 <= ih < H:
                            continue
                        for ow in range(OW):
                            iw = ow - pad + b
                            if not 0 <= iw < Wd:
                                continue
                            dx[:, ci, ih, iw] += dout[:, co, oh, ow] * W[co, ci, a, b]
                            dW[co, ci, a, b] += inv_m * np.sum(dout[:, co, oh, ow] * x[:, ci, ih, iw])
    return dx, dW


def test_the_copy_made_for_a_non_contiguous_view_lives_as_long_as_the_pointer() -> None:
    base = np.arange(32 * 4 * 4 * 8, dtype=np.float32).reshape(32, 4, 4, 8)
    view = base[:, :, :, :4]                                  # the logical slice of a SIMD-padded gradient: not C-contiguous
    assert not view.flags["C_CONTIGUOUS"]
    want = np.ascontiguousarray(view).ravel().copy()
    ptr = cd._get_c_ptr(view)
    churn = [np.full(2048, float(i), dtype=np.float32) for i in range(400)]     # reuse any freed block of that size
    gc.collect()
    got = np.ctypeslib.as_array((ctypes.c_float * 2048).from_address(ptr.value)).copy()
    assert np.array_equal(got, want), "the pointer no longer points at the copy: use-after-free"
    assert len(churn) == 400


def test_a_contiguous_array_is_pointed_at_in_place_and_none_stays_none() -> None:
    a = np.zeros((4, 4), dtype=np.float32)
    assert cd._get_c_ptr(a).value == a.ctypes.data
    assert cd._get_c_ptr(None) is None


def test_native_conv_backward_of_a_padded_gradient_matches_numpy_and_repeats_bit_for_bit() -> None:
    cd._ensure_initialized()
    if cd._native_lib is None:
        pytest.skip("native library not built")
    if hasattr(cd._native_lib, "configure_native_threads"):
        cd._native_lib.configure_native_threads(1)         # dW is summed per OpenMP thread under dynamic scheduling: more than one thread changes the last bits of dW from call to call by design
    rng = np.random.default_rng(3)
    N, C_in, C_out, H, W_log, W_str, k, pad = 32, 4, 4, 4, 4, 8, 3, 1     # the geometry of the last conv layers of the cnn bank: 4 channels, 4x4 maps, 8-wide SIMD rows
    x = np.zeros((N, C_in, H, W_str), dtype=np.float32)
    x[..., :W_log] = rng.standard_normal((N, C_in, H, W_log)).astype(np.float32)
    dout_full = np.zeros((N, C_out, H, W_str), dtype=np.float32)
    dout_full[..., :W_log] = rng.standard_normal((N, C_out, H, W_log)).astype(np.float32)
    dout_logical = dout_full[:, :, :H, :W_log]                           # non-contiguous: exactly what Conv2D.backward passes
    W = rng.standard_normal((C_out, C_in, k, k)).astype(np.float32)
    inv_m = 1.0 / N
    ref_dx, ref_dW = _ref_conv_backward(dout_full[..., :W_log].astype(np.float64), x[..., :W_log].astype(np.float64), W.astype(np.float64), pad, inv_m)
    dx_buf = np.zeros((N, C_in, H, W_str), dtype=np.float32)       # persistent output buffers, as the engine's scratch arena and the layer's dW are
    dW_buf = np.zeros(W.shape, dtype=np.float32)
    first = None
    for i in range(12):
        junk = [np.random.default_rng(i + j).standard_normal(2048).astype(np.float32) for j in range(64)]    # allocation churn between calls
        dx_buf.fill(0.0)
        dW_buf.fill(0.0)
        dx, dW = cd.conv2d_backward_fused(dout_logical, x, W, dx_buf, dW_buf, stride=1, pad=pad, inv_m=inv_m, W_logical=W_log, backend=cd.EngineBackend.NATIVE)
        assert np.allclose(dx[..., :W_log], ref_dx, rtol=1e-4, atol=1e-5), f"call {i}: dx differs from the NumPy reference"
        assert np.allclose(dW, ref_dW, rtol=1e-4, atol=1e-5), f"call {i}: dW differs from the NumPy reference"
        assert float(np.abs(dx).sum()) > 0.0, f"call {i}: dx is all zero"
        cur = (dx.tobytes(), dW.tobytes())
        first = first or cur
        assert cur == first, f"call {i}: not bit-identical to call 0 at one thread"
        assert len(junk) == 64


def test_a_cnn_gives_the_same_gradients_every_time_on_the_same_cache() -> None:
    """The in-process form of ml-engine#60: forward once, backward four times on the same cache, then four full cycles. Four 4-channel 3x3 convolutions on 4x4 maps (SIMD-padded to 8 columns, so Conv2D.backward
    receives a non-contiguous logical slice), one native thread. Every gradient is identical and none is zero."""
    from config.constants import EngineBackend
    from src.model_factory import ModelFactory

    cd._ensure_initialized()
    if cd._native_lib is None:
        pytest.skip("native library not built")
    if hasattr(cd._native_lib, "configure_native_threads"):
        cd._native_lib.configure_native_threads(1)
    conv = {"type": "conv", "in_channels": 4, "out_channels": 4, "kernel_size": 3, "stride": 1, "pad": 1}
    cfg = {"input_shape": [4, 4, 4], "spatial_pipeline": [conv, {"type": "relu"}, dict(conv), {"type": "relu"}, dict(conv), {"type": "relu"}, {"type": "flatten"}], "dense_head": []}
    np.random.seed(7)
    model = ModelFactory.create_model(model_type="cnn", layer_sizes=[5], backend=EngineBackend.NATIVE, optimizer="adam", cnn_config=cfg)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((32, 4, 4, 4)).astype(np.float32)
    y = np.eye(5, dtype=np.float32)[rng.integers(0, 5, 32)]

    def grads(cache):
        loss, gw, gb, *_ = model._compute_grads_from_cache(cache, y)
        return repr(float(loss)), tuple(g.tobytes() for g in gw), [float(np.abs(g).sum()) for g in gw]

    _, cache = model.forward_train(X)
    same_cache = [grads(cache) for _ in range(4)]
    cycles = []
    for _ in range(4):
        _, c2 = model.forward_train(X)
        cycles.append(grads(c2))
    assert len({g[:2] for g in same_cache + cycles}) == 1, "the gradients of identical inputs differ between calls"
    assert all(v > 0.0 for v in same_cache[0][2]), "a layer's weight gradient is zero on the first backward"


def test_an_exported_omp_num_threads_sets_the_native_team_size(monkeypatch) -> None:
    """A reproducible run exports OMP_NUM_THREADS=1: the engine must then use one thread, whatever runtime.yaml says."""
    from utils.runtime import load_runtime_settings

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    default = load_runtime_settings().effective_omp_threads()
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert load_runtime_settings().effective_omp_threads() == 1
    monkeypatch.setenv("OMP_NUM_THREADS", "")
    assert load_runtime_settings().effective_omp_threads() == default          # empty = not set


if __name__ == "__main__":
    test_the_copy_made_for_a_non_contiguous_view_lives_as_long_as_the_pointer()
    test_a_contiguous_array_is_pointed_at_in_place_and_none_stays_none()
    test_native_conv_backward_of_a_padded_gradient_matches_numpy_and_repeats_bit_for_bit()
    test_a_cnn_gives_the_same_gradients_every_time_on_the_same_cache()
    print("[PASSED] conv pointer lifetime")
