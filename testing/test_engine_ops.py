# testing/test_engine_ops.py
"""
Phase-gated tests for EngineContext and ConvOps (roadmap Phase A).

Each test maps to a sub-phase ID (A1–A9). Run standalone:
    python testing/test_engine_ops.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import EngineBackend
from src.cnn_network import CNNNetwork
from src.model_factory import ModelFactory
from src.optimizers import AdamOptimizer
from src.spatial_layers import ConvBlock, Conv2D, MaxPool2D
from utils.engine_ops import (
    ConvOps,
    EngineContext,
    EngineOps,
    Im2colGemmConvOps,
    NativeConvOps,
    NumpyConvOps,
    create_engine_context,
)
from utils.conv_dispatch import conv2d_forward, init_engine_backend


def _tiny_conv_fixture(stride: int = 1, pad: int = 1, kernel: int = 3):
    rng = np.random.default_rng(42)
    h, w = 8, 8
    out_h = (h + 2 * pad - kernel) // stride + 1
    out_w = (w + 2 * pad - kernel) // stride + 1
    x = (rng.standard_normal((1, 2, h, w), dtype=np.float32) * 0.1).copy()
    w_arr = (rng.standard_normal((3, 2, kernel, kernel), dtype=np.float32) * 0.1).copy()
    b = np.zeros((1, 3), dtype=np.float32)
    out = np.zeros((1, 3, out_h, out_w), dtype=np.float32)
    return x, w_arr, b, out


# --- Phase A1: protocol exists ---
def test_a1_conv_ops_protocol():
    assert issubclass(NativeConvOps, object)
    ops = NativeConvOps()
    assert isinstance(ops, ConvOps)
    for name in (
        "conv2d_forward",
        "conv2d_backward_fused",
        "conv_block_forward",
        "conv_block_backward",
        "maxpool_forward",
        "maxpool_backward",
        "fuse_dout_transpose_and_bias",
        "relu_forward",
        "relu_backward",
    ):
        assert hasattr(ops, name), f"ConvOps missing {name}"
    print("[PASSED] A1: ConvOps protocol surface")


# --- Phase A2: native delegation ---
def test_a2_native_conv_ops_matches_im2col():
    ctx = create_engine_context(EngineBackend.NATIVE)
    x, w, b, out = _tiny_conv_fixture()
    out_ops = np.zeros_like(out)
    ctx.conv.conv2d_forward(
        x=x, W=w, bias=b, stride=1, pad=1, out_buf=out_ops
    )
    init_engine_backend(EngineBackend.NATIVE)
    out_ref = np.zeros_like(out)
    conv2d_forward(x, w, b, stride=1, pad=1, out_buf=out_ref, ctx=ctx)
    assert np.allclose(out_ops, out_ref, atol=1e-5)
    print("[PASSED] A2: NativeConvOps matches im2col dispatch")


# --- Phase A3: numpy delegation ---
def test_a3_numpy_conv_ops_runs_reference_path():
    ctx = create_engine_context(EngineBackend.NUMPY)
    x, w, b, out = _tiny_conv_fixture()
    out_ops = np.zeros_like(out)
    ctx.conv.conv2d_forward(
        x=x, W=w, bias=b, stride=1, pad=1, out_buf=out_ops
    )
    assert out_ops.shape == out.shape
    assert np.isfinite(out_ops).all()
    print("[PASSED] A3: NumpyConvOps reference path runs")


# --- Phase A4: im2col+gemm explicit class ---
def test_a4_im2col_gemm_conv_ops():
    ctx = create_engine_context(EngineBackend.IM2COL_GEMM)
    assert isinstance(ctx.conv, Im2colGemmConvOps)
    x, w, b, out = _tiny_conv_fixture()
    out_ops = np.zeros_like(out)
    ctx.conv.conv2d_forward(
        x=x, W=w, bias=b, stride=1, pad=1, out_buf=out_ops
    )
    assert np.isfinite(out_ops).all()
    print("[PASSED] A4: Im2colGemmConvOps explicit backend")


# --- Phase A5: factory + native handle ---
def test_a5_engine_context_factory():
    native_ctx = create_engine_context(EngineBackend.NATIVE)
    numpy_ctx = create_engine_context(EngineBackend.NUMPY)
    assert native_ctx.backend == EngineBackend.NATIVE
    assert numpy_ctx.backend == EngineBackend.NUMPY
    assert native_ctx.native_lib is not None
    assert numpy_ctx.native_lib is None
    assert native_ctx.conv.backend == EngineBackend.NATIVE
    assert numpy_ctx.conv.backend == EngineBackend.NUMPY
    print("[PASSED] A5: create_engine_context factory")


# --- Phase A6: ctx param overrides global ---
def test_a6_ctx_param_overrides_global():
    numpy_ctx = create_engine_context(EngineBackend.NUMPY)
    x, w, b, out = _tiny_conv_fixture()

    init_engine_backend(EngineBackend.NATIVE)
    out_ctx_numpy = np.zeros_like(out)
    conv2d_forward(x, w, b, stride=1, pad=1, out_buf=out_ctx_numpy, ctx=numpy_ctx)

    init_engine_backend(EngineBackend.NUMPY)
    out_global_numpy = np.zeros_like(out)
    conv2d_forward(x, w, b, stride=1, pad=1, out_buf=out_global_numpy)

    assert np.allclose(out_ctx_numpy, out_global_numpy, atol=1e-5)
    print("[PASSED] A6: ctx= overrides legacy global backend")


# --- Phase A7: layers use injected ctx ---
def test_a7_conv_block_uses_injected_ctx():
    ctx = create_engine_context(EngineBackend.NATIVE)
    block = ConvBlock(2, 4, kernel_size=3, engine_ctx=ctx)
    assert block._ctx is ctx
    assert block.backend == EngineBackend.NATIVE
    print("[PASSED] A7: ConvBlock stores injected EngineContext")


# --- Phase A8: model factory wires one ctx ---
def test_a8_model_factory_shares_engine_ctx():
    cnn_config = {
        "input_shape": [1, 28, 28],
        "spatial_pipeline": [
            {"type": "conv", "in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 1, "pad": 1},
            {"type": "relu"},
            {"type": "pool", "pool_size": 2, "stride": 2},
            {"type": "flatten"},
        ],
        "dense_head": [16],
    }
    model = ModelFactory.create_model(
        model_type="cnn",
        layer_sizes=[10],
        backend=EngineBackend.NATIVE,
        optimizer="adam",
        cnn_config=cnn_config,
    )
    assert isinstance(model, CNNNetwork)
    assert model.engine_ctx is not None
    conv_layers = [l for l in model.layers if isinstance(l, (ConvBlock, Conv2D, MaxPool2D))]
    assert conv_layers, "expected at least one spatial layer"
    for layer in conv_layers:
        assert layer._ctx is model.engine_ctx
    print("[PASSED] A8: ModelFactory shares one EngineContext across layers")


# --- Phase A9: no init_engine_backend in src/ ---
def test_a9_no_init_engine_backend_in_src():
    src_root = os.path.join(os.path.dirname(__file__), "..", "src")
    offenders = []
    for dirpath, _, filenames in os.walk(src_root):
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(dirpath, fname)
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            if "init_engine_backend" in text:
                offenders.append(os.path.relpath(path, src_root))
    assert offenders == [], f"init_engine_backend found in src/: {offenders}"
    print("[PASSED] A9: no init_engine_backend in src/")


# --- Model-agnostic extensibility ---
def test_model_agnostic_ops_registry():
    ctx = create_engine_context(EngineBackend.NATIVE)

    class _StubAttentionOps:
        backend = EngineBackend.NATIVE
        tag = "attention-stub"

    stub = _StubAttentionOps()
    ctx.register("attention", stub)
    assert isinstance(ctx.ops("attention"), EngineOps)
    assert ctx.ops("conv") is ctx.conv
    try:
        ctx.ops("nonexistent")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass
    print("[PASSED] extensibility: ctx.register / ctx.ops for future model types")


def test_sequential_two_contexts():
    ctx_a = create_engine_context(EngineBackend.NATIVE)
    ctx_b = create_engine_context(EngineBackend.NUMPY)
    assert ctx_a.backend != ctx_b.backend
    assert ctx_a.conv.backend == EngineBackend.NATIVE
    assert ctx_b.conv.backend == EngineBackend.NUMPY
    print("[PASSED] two sequential EngineContext instances are independent")


PHASE_TESTS = [
    test_a1_conv_ops_protocol,
    test_a2_native_conv_ops_matches_im2col,
    test_a3_numpy_conv_ops_runs_reference_path,
    test_a4_im2col_gemm_conv_ops,
    test_a5_engine_context_factory,
    test_a6_ctx_param_overrides_global,
    test_a7_conv_block_uses_injected_ctx,
    test_a8_model_factory_shares_engine_ctx,
    test_a9_no_init_engine_backend_in_src,
    test_model_agnostic_ops_registry,
    test_sequential_two_contexts,
]


if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING ENGINE OPS PHASE A TESTS ")
    print("=" * 60)
    failed = []
    for test_fn in PHASE_TESTS:
        try:
            test_fn()
        except Exception as exc:
            failed.append((test_fn.__name__, exc))
            print(f"[FAILED] {test_fn.__name__}: {exc}")
    print("=" * 60)
    if failed:
        print(f"[FAILURE] {len(failed)} test(s) failed.")
        sys.exit(1)
    print(f"[SUCCESS] All {len(PHASE_TESTS)} Phase A engine-ops tests passed.")
    print("=" * 60)
