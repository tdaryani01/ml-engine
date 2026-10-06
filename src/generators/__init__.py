"""Synthetic data generators (a library). See ``src/generators/base.py`` for the concepts.

The functions below are the whole API: a caller (the Desktop's Add form, the execution engine, a script) imports them and calls them; there is no
service. ``spec`` is ``{"name": ..., "params": {...}, "seed": int}``."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.generators.base import Data, Field, Generator, generators, get, register

__all__ = ["Data", "Field", "Generator", "baseline", "build", "canonical_spec", "check", "describe", "generators", "get", "register", "write_csv"]


def describe(model_class: str | None = None) -> list[dict[str, Any]]:
    """The generators for a model class and their settings, as plain data (what an Add form shows)."""
    return [{"name": g.name, "label": g.label, "description": g.description, "classes": list(g.classes), "fields": [f.describe() for f in g.fields()]}
            for g in generators(model_class)]


def _parts(spec: Any) -> tuple[str, dict[str, Any], int]:
    if not isinstance(spec, dict) or not str(spec.get("name") or "").strip():
        raise ValueError("a generator spec is {'name': ..., 'params': {...}, 'seed': int}")
    params = spec.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("spec.params must be an object")
    try:
        seed = int(spec.get("seed") or 0)
    except (TypeError, ValueError):
        raise ValueError("spec.seed must be a whole number") from None
    return str(spec["name"]).strip(), params, seed


def check(spec: Any, shapes: dict[str, Any] | None = None, model_class: str | None = None) -> list[str]:
    """What is wrong with this spec for this model (empty = fine)."""
    try:
        name, params, _seed = _parts(spec)
        gen = get(name)
    except ValueError as exc:
        return [str(exc)]
    problems: list[str] = []
    if model_class == "binary_classification" and (shapes or {}).get("num_classes") not in (None, 2):
        problems.append(f"a binary model has 2 classes, not {(shapes or {}).get('num_classes')}")
    if model_class is not None and model_class not in gen.classes:
        problems.append(f"{name!r} does not feed {model_class!r} models (it feeds: {', '.join(gen.classes)})")
    return problems + gen.check(dict(shapes or {}), params)


def canonical_spec(spec: Any, shapes: dict[str, Any] | None = None, model_class: str | None = None) -> dict[str, Any]:
    """The spec with every default filled in, validated: the form a caller stores in a model's config (the same spec always builds the same data)."""
    problems = check(spec, shapes, model_class)
    if problems:
        raise ValueError("; ".join(problems))
    name, params, seed = _parts(spec)
    return {"name": name, "params": get(name).resolve(params), "seed": seed}


def build(spec: Any, shapes: dict[str, Any] | None = None, split: str = "train", stretch: int = 1) -> Data:
    """The data for one split of a spec. ``train`` is new for every ``stretch``; ``heldout`` is the same every time."""
    name, params, seed = _parts(spec)
    return get(name).build(dict(shapes or {}), params, seed, split, stretch)


def baseline(spec: Any, shapes: dict[str, Any] | None = None) -> dict[str, Any]:
    """What a learner that learns nothing scores on this spec's data (and the best any model can do, when known)."""
    name, params, seed = _parts(spec)
    return get(name).baseline(dict(shapes or {}), params, seed)


def write_csv(data: Data, path: Path | str) -> Path:
    """The data as the plate a supervised model reads: a header, numeric cells, the target in the last column."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = data.X.shape[1]
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([f"f{i}" for i in range(n)] + ["target"])
        for row, y in zip(data.X, data.y):
            w.writerow([f"{v:.6f}" for v in row] + [str(int(y)) if data.task == "classification" else f"{float(y):.6f}"])
    return p
