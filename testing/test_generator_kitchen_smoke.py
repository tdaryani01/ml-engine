# Light smoke: every kitchen materialize writes a tiny artifact.
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "data" / "generators"
sys.path.insert(0, str(GEN))

MODULES = [
    ("csv/generate_binary_data.py", "bin.csv", 40),
    ("csv/generate_binary_moons_data.py", "moons.csv", 40),
    ("csv/swiss_helix.py", "helix.csv", 40),
    ("csv/mobius_twist.py", "mobius.csv", 40),
    ("csv/linked_torus.py", "torus.csv", 40),
    ("csv/trefoil_knot.py", "trefoil.csv", 40),
    ("csv/generate_multiclass_data.py", "spiral.csv", 60),
    ("csv/nested_shells.py", "shells.csv", 60),
    ("csv/supply_chain.py", "supply.csv", 60),
    ("csv/generate_regression_data.py", "reg.csv", 40),
    ("csv/generate_hard_regression_data.py", "hard.csv", 40),
    ("cnn/generate_synthetic_shapes.py", "shapes.npz", 32),
]


def _load(rel: str):
    path = GEN / rel
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_kitchen_materialize_smoke(tmp_path: Path | None = None):
    base = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    for rel, name, n in MODULES:
        mod = _load(rel)
        out = base / name
        kwargs = {"n_samples": n, "seed": 3}
        if "generate_synthetic_shapes" in rel:
            kwargs["height"] = 8
            kwargs["width"] = 8
        path = mod.materialize(str(out), **kwargs)
        assert Path(path).is_file(), rel
        assert Path(path).stat().st_size > 0

    hard = _load("cnn/generate_synthetic_shapes.py")
    hard_out = base / "shapes_hard.npz"
    path = hard.materialize(
        str(hard_out), n_samples=64, seed=3, height=8, width=8, hard=True, noise=0.1
    )
    assert Path(path).is_file()


if __name__ == "__main__":
    test_kitchen_materialize_smoke()
    print("OK generator smoke")
