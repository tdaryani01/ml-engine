"""Named options per seat for a closed-loop model. The user picks a family and its options by name (TM's assembly stamps
``assembly.modules = {seat: option}``); ML engine builds the run from them. No user code runs: only registered options."""
from __future__ import annotations

from typing import Any, Callable

SEATS = ("encoder", "policy", "env", "loss")
OPTIONAL_SEATS = ("data",)  # defaults to the built-in option when the assembly does not name one

_FACTORIES: dict[tuple[str, str], Callable[..., Any]] = {}


def register(seat: str, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if seat not in SEATS + OPTIONAL_SEATS:
        raise ValueError(f"unknown seat {seat!r}; seats are {SEATS + OPTIONAL_SEATS}")

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        _FACTORIES[(seat, name)] = fn
        return fn

    return deco


def options(seat: str) -> list[str]:
    _load_builtin_options()
    return sorted(n for (s, n) in _FACTORIES if s == seat)


def resolve(seat: str, name: str) -> Callable[..., Any]:
    _load_builtin_options()
    try:
        return _FACTORIES[(seat, name)]
    except KeyError:
        raise ValueError(f"no {seat} option named {name!r}; available: {options(seat) or 'none'}") from None


def _load_builtin_options() -> None:
    from src.closed_loop import data as _d, options as _  # noqa: F401  (registers the built-in options on import)


__all__ = ["OPTIONAL_SEATS", "SEATS", "options", "register", "resolve"]
