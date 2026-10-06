"""Synthetic data generators: a LIBRARY (plain functions), not a service.

A generator is a named, registered object that knows the model classes it can feed, the fields a user may set, how to check them against the
model's shapes, how to build data for a class's data contract, and what a learner that learns nothing would score on that data.

A *spec* is the only thing that moves between callers (the Desktop's Add form, the execution engine, a script, a test)::

    {"name": "tabular_blobs", "params": {"noise": 0.1}, "seed": 7}

``shapes`` are the model's structural facts (for tabular models ``num_classes``; closed-loop classes will add their sizes).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SPLITS = ("train", "heldout")


@dataclass(frozen=True)
class Field:
    """One setting a user may change. ``kind``: ``number``, ``integer`` or ``choice``."""

    name: str
    kind: str
    label: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    hint: str = ""

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "label": self.label, "default": self.default, "minimum": self.minimum,
                "maximum": self.maximum, "choices": list(self.choices), "hint": self.hint}


@dataclass
class Data:
    """What a tabular generator builds: features (rows, n_features) and the target (rows,)."""

    X: np.ndarray
    y: np.ndarray
    task: str  # "classification" | "regression"
    classes: int | None = None


class Generator:
    """Subclass, set the class attributes, implement ``_fields``, ``_make`` and ``baseline``, and ``register`` it."""

    name: str = ""
    label: str = ""
    description: str = ""
    classes: tuple[str, ...] = ()

    # ---- what a subclass provides ---------------------------------------------------------------
    def _fields(self) -> list[Field]:
        raise NotImplementedError

    def _make(self, shapes: dict[str, Any], p: dict[str, Any], rng: np.random.Generator, split: str, concept: np.random.Generator) -> Data:
        """``rng`` draws the examples of this split; ``concept`` (seeded by the spec alone) draws the TEACHER, so every split and every stretch shares one concept."""
        raise NotImplementedError

    def baseline(self, shapes: dict[str, Any], params: dict[str, Any], seed: int) -> dict[str, Any]:
        """What a learner that learns nothing scores on this data: ``{"metric", "do_nothing", "note"}``."""
        raise NotImplementedError

    # ---- shared behaviour ------------------------------------------------------------------------
    def fields(self) -> list[Field]:
        return [Field("rows", "integer", "Rows", 1000, 50, 200_000, hint="How many examples to generate.")] + self._fields()

    def resolve(self, params: dict[str, Any] | None) -> dict[str, Any]:
        """The params with defaults filled in (unknown names are ignored here and reported by ``check``)."""
        given = dict(params or {})
        out: dict[str, Any] = {}
        for f in self.fields():
            v = given.get(f.name, f.default)
            out[f.name] = int(v) if f.kind == "integer" else (float(v) if f.kind == "number" else v)
        return out

    def check(self, shapes: dict[str, Any], params: dict[str, Any] | None) -> list[str]:
        problems: list[str] = []
        known = {f.name: f for f in self.fields()}
        for k in (params or {}):
            if k not in known:
                problems.append(f"unknown setting {k!r} (this generator has: {', '.join(known)})")
        for f in self.fields():
            raw = (params or {}).get(f.name, f.default)
            try:
                v = float(raw) if f.kind in ("number", "integer") else raw
            except (TypeError, ValueError):
                problems.append(f"{f.label}: not a number")
                continue
            if f.kind == "integer" and float(raw) != int(float(raw)):
                problems.append(f"{f.label}: must be a whole number")
            if f.kind in ("number", "integer"):
                if f.minimum is not None and v < f.minimum:
                    problems.append(f"{f.label}: at least {f.minimum:g}")
                if f.maximum is not None and v > f.maximum:
                    problems.append(f"{f.label}: at most {f.maximum:g}")
            elif f.kind == "choice" and raw not in f.choices:
                problems.append(f"{f.label}: one of {', '.join(f.choices)}")
        return problems + self._check_shapes(shapes, self.resolve(params))

    def _check_shapes(self, shapes: dict[str, Any], p: dict[str, Any]) -> list[str]:
        return []

    def build(self, shapes: dict[str, Any], params: dict[str, Any] | None, seed: int, split: str = "train", stretch: int = 1) -> Data:
        """Data for one split. ``train`` is drawn from (seed, stretch): every fit of a chain gets NEW data; ``heldout`` from (seed) alone, so it is
        the same every time and fits stay comparable."""
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        problems = self.check(shapes, params)
        if problems:
            raise ValueError("; ".join(problems))
        key = [int(seed), 0] if split == "heldout" else [int(seed), 1, max(0, int(stretch) - 1)]
        return self._make(shapes, self.resolve(params), np.random.default_rng(key), split, np.random.default_rng([int(seed), 2]))


_REGISTRY: dict[str, Generator] = {}


def register(gen: Generator) -> Generator:
    if not gen.name or not gen.classes:
        raise ValueError("a generator needs a name and the model classes it feeds")
    _REGISTRY[gen.name] = gen
    return gen


def get(name: str) -> Generator:
    _load_builtin()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"no generator named {name!r}; available: {', '.join(sorted(_REGISTRY)) or 'none'}") from None


def generators(model_class: str | None = None) -> list[Generator]:
    """The generators that can feed ``model_class`` (all of them when None)."""
    _load_builtin()
    return sorted((g for g in _REGISTRY.values() if model_class is None or model_class in g.classes), key=lambda g: g.name)


def _load_builtin() -> None:
    from src.generators import tabular as _t  # noqa: F401  (registers the built-in generators on import)
