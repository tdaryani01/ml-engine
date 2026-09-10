# data/generators/mhsa/generate_cue_recall.py
"""
Causal cue-recall sequences for MHSA (continuous last-token actions).

Task
----
- Token 0 embeds a random target in R^{action_dim} (remaining dims noise).
- Tokens 1..T-2 are distractors.
- Token T-1 is a query marker.
- Label y is the target, tanh-scaled into (-1, 1) to match the action head.

Two size-tagged presets (same idea as synthetic_shapes_28 vs _128):
  quick  — T=32,  D=64,  A=4   → fast iteration / wiring
  sanity — T=128, D=256, A=4   → realistic attention working set

NPZ keys: X (N,T,D) float32, y (N,A) float32, plus metadata scalars/strings.
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Presets: (name, T, D, A, n_train, n_val, seed_offset)
PRESETS = {
    "quick": {
        "T": 32,
        "D": 64,
        "A": 4,
        "n_train": 8192,
        "n_val": 1024,
        "seed": 42,
    },
    "sanity": {
        "T": 128,
        "D": 256,
        "A": 4,
        # Geometry is the stress (T×T attn, D=256); keep N modest for disk (~1–1.5 GiB).
        "n_train": 8192,
        "n_val": 1024,
        "seed": 43,
    },
}


def _default_output(preset: str) -> str:
    return os.path.join("data", "samples", "mhsa", f"cue_recall_{preset}.npz")


def generate_cue_recall(
    n: int,
    T: int,
    D: int,
    A: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if D < A + 2:
        raise ValueError(f"D={D} too small for A={A} (need room for cue + markers)")
    if T < 3:
        raise ValueError(f"T={T} must be >= 3 (cue, distractor(s), query)")

    X = rng.standard_normal((n, T, D), dtype=np.float32) * 0.25
    # Targets in (-0.9, 0.9) so tanh head is not stuck at saturation.
    y = (rng.uniform(-0.9, 0.9, size=(n, A))).astype(np.float32)

    # Token 0: write cue into leading A dims; clear marker channel.
    X[:, 0, :A] = y
    X[:, 0, A] = 1.0  # cue marker
    if A + 1 < D:
        X[:, 0, A + 1 :] *= 0.1

    # Distractors: scramble leading A so last-token baseline cannot cheat.
    if T > 2:
        X[:, 1 : T - 1, :A] = rng.standard_normal((n, T - 2, A), dtype=np.float32) * 0.25
        X[:, 1 : T - 1, A] = 0.0

    # Query token: marker only; wipe cue dims.
    X[:, T - 1, :] = rng.standard_normal((n, D), dtype=np.float32) * 0.05
    X[:, T - 1, :A] = 0.0
    X[:, T - 1, A] = -1.0  # query marker (distinct from cue)

    return X, y


def write_preset(preset: str, output_path: str | None = None) -> str:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    cfg = PRESETS[preset]
    T, D, A = int(cfg["T"]), int(cfg["D"]), int(cfg["A"])
    n_train, n_val = int(cfg["n_train"]), int(cfg["n_val"])
    seed = int(cfg["seed"])
    out = output_path or _default_output(preset)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    rng = np.random.default_rng(seed)
    X_train, y_train = generate_cue_recall(n_train, T, D, A, rng)
    X_val, y_val = generate_cue_recall(n_val, T, D, A, rng)

    np.savez_compressed(
        out,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        T=np.int32(T),
        D=np.int32(D),
        A=np.int32(A),
        task=np.array("cue_recall"),
        preset=np.array(preset),
        seed=np.int32(seed),
    )
    mb = os.path.getsize(out) / (1024 * 1024)
    logging.info(
        "[cue_recall/%s] Wrote %s | train X=%s y=%s | val X=%s | ~%.1f MiB",
        preset,
        out,
        X_train.shape,
        y_train.shape,
        X_val.shape,
        mb,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate causal cue-recall NPZs for MHSA (quick + sanity sizes)."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()) + ["both"],
        default="both",
        help="Which size-tagged NPZ to write (default: both)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for a single --preset (ignored when preset=both)",
    )
    args = parser.parse_args()

    if args.preset == "both":
        for name in ("quick", "sanity"):
            write_preset(name)
        return
    write_preset(args.preset, args.output)


if __name__ == "__main__":
    main()
