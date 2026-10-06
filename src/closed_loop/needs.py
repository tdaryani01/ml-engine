"""What an option needs from the machine it runs on (a data file, a folder, a number, a secret).

An option declares its needs by name; the Desktop reads them for the model's chosen options from its own ML engine (the one
that will run the model) and asks the user at Add time. The answers stay on that machine. A need is

    {"name": str, "kind": "file" | "folder" | "number" | "secret", "label": str, "hint": str, "required": bool, "default": Any,
     "config_key": "closed_loop.some_key" | None}

``config_key`` says where the user's answer goes in the run config (dotted path); the Desktop sets it there, over anything TM sent.

``assembly_needs`` never raises: it returns what it found, so a page can show every problem at once.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.closed_loop import registry

KINDS = ("file", "folder", "number", "secret")

_NEEDS: dict[tuple[str, str], list[dict[str, Any]]] = {}


def _clean(raw: dict[str, Any]) -> dict[str, Any]:
    name = str(raw.get("name") or "").strip()
    kind = str(raw.get("kind") or "").strip().lower()
    if not name:
        raise ValueError("a need must have a name")
    if kind not in KINDS:
        raise ValueError(f"need {name!r}: kind must be one of {KINDS}, not {kind!r}")
    return {"name": name, "kind": kind, "label": str(raw.get("label") or name), "hint": str(raw.get("hint") or ""),
            "required": bool(raw.get("required", True)), "default": raw.get("default"),
            "config_key": (str(raw["config_key"]).strip() or None) if raw.get("config_key") else None}


def declare_needs(seat: str, name: str, needs: list[dict[str, Any]]) -> None:
    """Record what the option ``name`` of ``seat`` needs. (Call next to the option's ``register``.)"""
    if seat not in registry.SEATS + registry.OPTIONAL_SEATS:
        raise ValueError(f"unknown seat {seat!r}")
    _NEEDS[(seat, name)] = [_clean(n) for n in needs]


def apply_answers(config: dict[str, Any], needs: list[dict[str, Any]], answers: dict[str, Any]) -> dict[str, Any]:
    """A copy of a run config with the user's answers set at each need's ``config_key``. The user's answer always wins; for a file,
    folder or secret a value TM sent at that key is removed when the user gave none (TM never names a path or holds a secret)."""
    import copy

    out = copy.deepcopy(config)
    for n in needs:
        key = n.get("config_key")
        if not key:
            continue
        parts = key.split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {}) if isinstance(node.get(part, {}), dict) else node.setdefault(part, {})
        raw = answers.get(n["name"])
        value = "" if raw is None else str(raw).strip()
        if value:
            if n["kind"] in ("file", "folder"):
                value = str(Path(value).expanduser())  # a "~" is the user's shell shorthand: the run needs the real path
            node[parts[-1]] = float(value) if n["kind"] == "number" else value
        elif n["kind"] in ("file", "folder", "secret"):
            node.pop(parts[-1], None)
    return out


def assembly_needs(modules: dict[str, str]) -> dict[str, Any]:
    """The needs of the chosen options, merged by name, plus what is wrong: ``{"needs", "missing", "conflicts"}``.

    ``missing``: options this ML engine does not have (a package built for another version). ``conflicts``: two options that
    declare the same need name with a different kind. A need is required when any option requires it."""
    registry._load_builtin_options()
    merged: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    conflicts: list[str] = []
    for seat, name in modules.items():
        if (seat, name) not in registry._FACTORIES:
            have = registry.options(seat)
            missing.append(f"this ML engine has no {seat} option named {name!r}" + (f" (it has: {', '.join(have)})" if have else ""))
            continue
        for need in _NEEDS.get((seat, name), []):
            have = merged.get(need["name"])
            if have is None:
                merged[need["name"]] = dict(need)
            elif have["kind"] != need["kind"]:
                conflicts.append(f"need {need['name']!r} is a {have['kind']} for one option and a {need['kind']} for {seat} option {name!r}")
            else:
                have["required"] = have["required"] or need["required"]
    return {"needs": list(merged.values()), "missing": missing, "conflicts": conflicts}


def check_answers(needs: list[dict[str, Any]], answers: dict[str, Any]) -> list[str]:
    """What is wrong with the user's answers (empty when they are fine): required ones given, files readable, folders absolute,
    numbers numeric. A secret is only checked for being present."""
    problems: list[str] = []
    for n in needs:
        raw = answers.get(n["name"])
        value = "" if raw is None else str(raw).strip()
        if not value:
            if n["required"]:
                problems.append(f"{n['label']}: required")
            continue
        if n["kind"] == "file":
            path = Path(value).expanduser()
            if not path.is_absolute():
                problems.append(f"{n['label']}: give the full path")
            elif not path.is_file():
                problems.append(f"{n['label']}: no such file: {path}")
            elif not os.access(path, os.R_OK):
                problems.append(f"{n['label']}: cannot read: {path}")
        elif n["kind"] == "folder":
            if not Path(value).expanduser().is_absolute():
                problems.append(f"{n['label']}: give the full path")
        elif n["kind"] == "number":
            try:
                float(value)
            except ValueError:
                problems.append(f"{n['label']}: not a number")
    return problems


__all__ = ["KINDS", "apply_answers", "assembly_needs", "check_answers", "declare_needs"]
