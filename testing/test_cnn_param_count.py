# testing/test_cnn_param_count.py
"""Runtime trainable-parameter count vs geometry-derived expected count."""
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import EngineBackend
from src.model_factory import ModelFactory
from src.spatial_layers import Conv2D, ConvBlock


def _conv_out(h: int, w: int, k: int, stride: int, pad: int) -> tuple[int, int]:
    return (h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1


def _pool_out(h: int, w: int, pool: int, stride: int) -> tuple[int, int]:
    return (h - pool) // stride + 1, (w - pool) // stride + 1


def expected_trainable_params(cnn_config: dict, num_classes: int) -> int:
    """Count params from architecture only — not from a live model."""
    c, h, w = cnn_config["input_shape"]
    total = 0
    in_c = c
    for cfg in cnn_config["spatial_pipeline"]:
        l_type = str(cfg["type"]).lower()
        if l_type in ("conv", "conv_block"):
            k = cfg.get("kernel_size", 3)
            k = k if isinstance(k, int) else k[0]
            stride = cfg.get("stride", 1)
            pad = cfg.get("pad", 0)
            out_c = int(cfg["out_channels"])
            total += out_c * in_c * k * k + out_c
            h, w = _conv_out(h, w, k, stride, pad)
            in_c = out_c
            if l_type == "conv_block":
                p = cfg.get("pool_size", 2)
                ps = cfg.get("pool_stride", 2)
                h, w = _pool_out(h, w, p, ps)
        elif l_type == "pool":
            p = cfg.get("pool_size", 2)
            ps = cfg.get("stride", 2)
            h, w = _pool_out(h, w, p, ps)
    flat = in_c * h * w
    sizes = [flat] + list(cnn_config.get("dense_head", [])) + [num_classes]
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        total += fan_in * fan_out + fan_out
    return total


def runtime_trainable_params(model) -> int:
    """Count from live arrays: model lists plus conv layer objects."""
    from_lists = 0
    for w, b in zip(model.weights, model.biases):
        if w is not None:
            from_lists += int(np.asarray(w).size)
        if b is not None:
            from_lists += int(np.asarray(b).size)

    from_layers = 0
    for layer in model.layers:
        if isinstance(layer, (ConvBlock, Conv2D)):
            from_layers += int(layer.W.size) + int(layer.b.size)
    for w_idx, kind in enumerate(model.param_layers):
        if kind == "dense":
            from_layers += int(model.weights[w_idx].size) + int(model.biases[w_idx].size)

    if from_lists != from_layers:
        raise AssertionError(
            f"weight-list count {from_lists} != layer-walk count {from_layers}"
        )
    return from_lists


def _load_config_cnn() -> tuple[dict, int]:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(root, "config", "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    arch = cfg["architecture"]
    return dict(arch["cnn"]), int(arch["num_classes"])


def _make_model(cnn_config: dict, num_classes: int, backend: EngineBackend):
    np.random.seed(0)
    return ModelFactory.create_model(
        model_type="cnn",
        layer_sizes=[num_classes],
        backend=backend,
        optimizer="adam",
        cnn_config=cnn_config,
        lam_l1=0.0,
        lam_l2=0.0,
    )


def _assert_backend(cnn_config: dict, num_classes: int, backend: EngineBackend, label: str) -> None:
    expected = expected_trainable_params(cnn_config, num_classes)
    model = _make_model(cnn_config, num_classes, backend)
    actual = runtime_trainable_params(model)
    assert actual == expected, (
        f"[{backend.value}] {label}: runtime params {actual:,} != expected {expected:,}"
    )
    print(f"[PASSED] {backend.value}: {label} params={actual:,}")


def test_param_count_matches_geometry_all_backends():
    backends = (EngineBackend.NATIVE, EngineBackend.IM2COL_GEMM, EngineBackend.NUMPY)
    cfg_yaml, n_cls = _load_config_cnn()
    variants = [
        ("config.yaml", cfg_yaml, n_cls),
        (
            "dense_head=[]",
            {**cfg_yaml, "dense_head": []},
            n_cls,
        ),
        (
            "dense_head=[64]",
            {**cfg_yaml, "dense_head": [64]},
            n_cls,
        ),
    ]
    for backend in backends:
        print(f"\n--- Testing Backend: {backend.value} ---")
        for label, cnn_cfg, num_classes in variants:
            _assert_backend(cnn_cfg, num_classes, backend, label)


if __name__ == "__main__":
    test_param_count_matches_geometry_all_backends()
    print("[SUCCESS] CNN param-count checks passed for all backends")
