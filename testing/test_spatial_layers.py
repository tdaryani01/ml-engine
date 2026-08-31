# testing/test_spatial_layers.py
"""
Spatial layer helpers used by gradient-check fixtures.

Conv parity and dispatch regressions live in testing/test_native_conv.py.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Re-export layer types from the real module for tests that import this file.
from src.spatial_layers import ConvBlock, Conv2D, MaxPool2D, Flatten  # noqa: F401


def test_spatial_layers_module_importable() -> None:
    assert ConvBlock is not None
    assert Conv2D is not None
    print("[PASSED] spatial layer symbols import from src.spatial_layers")


if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING SPATIAL LAYERS SMOKE TESTS ")
    print("=" * 60)
    test_spatial_layers_module_importable()
    print("=" * 60)
    print("[SUCCESS] Spatial layers smoke tests passed.")
    print("=" * 60)
