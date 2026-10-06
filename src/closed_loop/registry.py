"""Named options per seat for a closed-loop model. The user picks a family and its options by name (TM's assembly stamps
``assembly.modules = {seat: option}``); ML engine builds the run from them. No user code runs: only registered options."""
from __future__ import annotations

from typing import Any, Callable

SEATS = ("encoder", "policy", "env", "loss")
OPTIONAL_SEATS = ("data",)  # defaults to the built-in option when the assembly does not name one

_FACTORIES: dict[tuple[str, str], Callable[..., Any]] = {}
_LOSS_TYPES: dict[str, tuple[str, ...]] = {}  # loss option name -> the schema ``loss_type`` values it implements


def register(seat: str, name: str, *, loss_types: tuple[str, ...] = ()) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """``loss_types`` (loss seat only): the schema ``loss_type`` names this option implements."""
    if seat not in SEATS + OPTIONAL_SEATS:
        raise ValueError(f"unknown seat {seat!r}; seats are {SEATS + OPTIONAL_SEATS}")
    if loss_types and seat != "loss":
        raise ValueError("loss_types only applies to the loss seat")

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        _FACTORIES[(seat, name)] = fn
        if loss_types:
            _LOSS_TYPES[name] = tuple(str(t).strip().lower() for t in loss_types)
        return fn

    return deco


def options(seat: str) -> list[str]:
    _load_builtin_options()
    return sorted(n for (s, n) in _FACTORIES if s == seat)


def loss_options_for(loss_type: str) -> list[str]:
    """The loss options that implement a schema ``loss_type``."""
    _load_builtin_options()
    want = str(loss_type).strip().lower()
    return sorted(n for n, kinds in _LOSS_TYPES.items() if want in kinds)


def loss_types_of(name: str) -> tuple[str, ...]:
    """The ``loss_type`` values a loss option implements (empty: it does not say)."""
    _load_builtin_options()
    return _LOSS_TYPES.get(name, ())


def resolve(seat: str, name: str) -> Callable[..., Any]:
    _load_builtin_options()
    try:
        return _FACTORIES[(seat, name)]
    except KeyError:
        raise ValueError(f"no {seat} option named {name!r}; available: {options(seat) or 'none'}") from None


def _load_builtin_options() -> None:
    from src.closed_loop import data as _d, demonstration_options as _i, options as _, reach_options as _r, navigate_options as _n  # noqa: F401  (registers the built-in options on import)


__all__ = ["OPTIONAL_SEATS", "SEATS", "options", "register", "resolve"]
