# Shared CLI for kitchen generators: --out/--n/--seed → materialize(...).
from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any


def run_materialize_cli(
    description: str,
    materialize: Callable[..., str],
    *,
    default_n: int = 1000,
    default_seed: int = 42,
    extra_args: Callable[[argparse.ArgumentParser], None] | None = None,
) -> None:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--out", required=True, help="Output path (.csv / .npz)")
    p.add_argument("--n", type=int, default=default_n, help="Sample count")
    p.add_argument("--seed", type=int, default=default_seed)
    if extra_args is not None:
        extra_args(p)
    args = p.parse_args()
    kwargs: dict[str, Any] = {
        "n_samples": int(args.n),
        "seed": int(args.seed),
    }
    # Forward optional known extras when present on the namespace.
    for key in ("noise", "preset", "height", "width", "channels", "hard"):
        if hasattr(args, key):
            kwargs[key] = getattr(args, key)
    print(f"Wrote {materialize(args.out, **kwargs)}")
