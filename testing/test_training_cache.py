# testing/test_training_cache.py
"""Phase B gates: ForwardCache, ScratchArena, explicit forward/backward split."""
import os
import sys
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import EngineBackend
from src.model_factory import ModelFactory
from src.optimizers import AdamOptimizer
from src.spatial_layers import ConvBlock
from src.scratch_arena import ScratchArena
from src.training_cache import ForwardCache


def _tiny_cnn():
    cnn_config = {
        "input_shape": [1, 28, 28],
        "spatial_pipeline": [
            {"type": "conv", "in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 1, "pad": 1},
            {"type": "relu"},
            {"type": "pool", "pool_size": 2, "stride": 2},
            {"type": "flatten"},
        ],
        "dense_head": [8],
    }
    return ModelFactory.create_model(
        model_type="cnn",
        layer_sizes=[4],
        backend=EngineBackend.NATIVE,
        optimizer="adam",
        cnn_config=cnn_config,
    )


def test_b1_forward_cache_dataclass():
    cache = ForwardCache()
    cache.activations = [np.zeros((2, 1, 28, 28), dtype=np.float32)]
    assert cache.batch_size == 2
    print("[PASSED] B1: ForwardCache dataclass")


def test_b2_scratch_arena_allocates():
    arena = ScratchArena(EngineBackend.NATIVE)
    arena.set_train_batch_cap(32)
    scratch = arena.ensure_conv_block_train(
        0,
        out_channels=4,
        in_channels=1,
        k_h=3,
        k_w=3,
        conv_stride=1,
        conv_pad=1,
        pool_size=2,
        pool_stride=2,
        N=8,
        C=1,
        H=28,
        W_stride=32,
        W_logical=28,
        dtype=np.float32,
    )
    assert scratch.dx_buffer is not None
    assert scratch.max_n >= 8
    print("[PASSED] B2: ScratchArena buffer allocation")


def test_b5_explicit_forward_backward_split():
    model = _tiny_cnn()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 1, 28, 32), dtype=np.float32)
    y = np.eye(4, dtype=np.float32)

    with mock.patch.object(model, "_forward", wraps=model._forward) as fwd_mock:
        output, cache = model.forward_train(X)
        assert isinstance(cache, ForwardCache)
        assert fwd_mock.call_count == 1
        loss = model._backward_from_cache(cache, y, active_lr=0.01)
        assert fwd_mock.call_count == 1, "backward must not re-forward"

    assert np.isfinite(loss)
    print("[PASSED] B5: backward_from_cache does not re-forward")


def test_b6_batch_cap_on_arena():
    model = _tiny_cnn()
    model.set_train_batch_cap(16)
    assert model.scratch_arena.train_batch_cap == 16
    print("[PASSED] B6: set_train_batch_cap wires to ScratchArena")


def test_b7_convblock_no_layer_cache_fields():
    model = _tiny_cnn()
    block = next(l for l in model.layers if isinstance(l, ConvBlock))
    assert not hasattr(block, "x_cached")
    for attr in ("_out_conv_buffer", "_col_buffer", "_train_N_cap"):
        assert not hasattr(block, attr), f"ConvBlock should not own {attr}"
    print("[PASSED] B7: ConvBlock has no layer-owned step cache/buffers")


PHASE_B_TESTS = [
    test_b1_forward_cache_dataclass,
    test_b2_scratch_arena_allocates,
    test_b5_explicit_forward_backward_split,
    test_b6_batch_cap_on_arena,
    test_b7_convblock_no_layer_cache_fields,
]


if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING PHASE B TRAINING CACHE TESTS ")
    print("=" * 60)
    failed = []
    for fn in PHASE_B_TESTS:
        try:
            fn()
        except Exception as exc:
            failed.append((fn.__name__, exc))
            print(f"[FAILED] {fn.__name__}: {exc}")
    print("=" * 60)
    if failed:
        print(f"[FAILURE] {len(failed)} test(s) failed.")
        sys.exit(1)
    print(f"[SUCCESS] All {len(PHASE_B_TESTS)} Phase B tests passed.")
    print("=" * 60)
