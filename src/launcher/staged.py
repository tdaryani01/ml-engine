"""Staged configs that tell the supervisor to run an ME-owned fit.

Two kinds (JSON or YAML; ``/stage/config`` stores YAML):

* ``me_pipeline``: ``{"kind": "me_pipeline", "config": <TM run config>, "plate_ref": optional}``. TM's own
  supervised run config (ME payload shape) is handed to ME as-is; the PLATE is read from a configured folder
  (``EXEC_PLATE_ROOT``), never from a staged ``data_handle`` (TM's loop stages a generic fallback corpus on every
  fit; that must not win).
* ``me_supervised``: ``{"kind": "me_supervised", "spec": {...}}``, a compact spec (gym, live checks); data may
  arrive as an uploaded ``data_handle`` or a plate reference.

Anything else is NOT a fit EE can run: the toy MLP placeholder is gone, so the supervisor fails the fit loudly.
``EXEC_ME_PYTHON`` / ``EXEC_ME_EXTRA_PYTHONPATH`` configure how ME is launched (ME normally has its own env).
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from src.launcher.payload import FamilyFitSpec, SupervisedFitSpec

KIND_SUPERVISED = "me_supervised"
KIND_PIPELINE = "me_pipeline"
KIND_RUN = "me_run"  # a model of a family ML engine assembles from named options
STAGED_KINDS = {KIND_SUPERVISED, KIND_PIPELINE, KIND_RUN}
DEFAULT_PLATE_ROOT = str(Path.home() / ".local" / "share" / "tm-desktop" / "plates")  # the user's own folder


def plate_root() -> Path:
    return Path(os.environ.get("TM_DESKTOP_PLATES") or os.environ.get("EXEC_PLATE_ROOT") or DEFAULT_PLATE_ROOT).expanduser().resolve()


def resolve_plate(ref: str | Path) -> Path:
    """A plate path under the configured plate root. Relative refs resolve against it; anything that
    resolves outside the root (absolute elsewhere, ``..``, symlink escape) is refused."""
    root = plate_root()
    raw = Path(str(ref))
    path = (raw if raw.is_absolute() else root / raw).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"plate {str(ref)!r} is outside the plate root {root}")
    if not path.is_file():
        raise ValueError(f"plate not found under the plate root: {path}")
    return path


def _located(location: str | Path) -> Path:
    """The data file the user chose in the Desktop (trusted: only the Desktop writes ``data_location``)."""
    path = Path(str(location)).expanduser()
    if not path.is_file():
        raise ValueError(f"the data file chosen for this model is not there: {path}")
    return path.resolve()


def load_staged(config_path: Path | str | None) -> dict[str, Any] | None:
    if config_path is None:
        return None
    try:
        text = Path(config_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    raw: Any = None
    s = text.lstrip()
    try:
        raw = json.loads(s) if s.startswith("{") else yaml.safe_load(text)
    except (ValueError, yaml.YAMLError):
        return None
    if isinstance(raw, dict) and raw.get("kind") in STAGED_KINDS:
        return raw
    return None


def spec_from_staged(
    config_path: Path | str | None,
    data_path: Path | str | None,
    *,
    es_trip_after: int | None = None,
    checkpoint_every: int | None = None,
) -> SupervisedFitSpec | None:
    """The spec for a staged ME config, or ``None`` when the config names no ME run."""
    raw = load_staged(config_path)
    if raw is None:
        return None
    over: dict[str, Any] = {}
    if es_trip_after:  # TM's Autopilot patience knob overrides the staged one (as it does today)
        over["patience"] = int(es_trip_after)
    if checkpoint_every:
        over["checkpoint_every"] = int(checkpoint_every)

    if raw["kind"] == KIND_RUN:
        fit = dict(raw.get("fit") or {})
        if es_trip_after:
            fit["patience"] = int(es_trip_after)
        if checkpoint_every:
            fit["checkpoint_every"] = int(checkpoint_every)
        return FamilyFitSpec(model_id=str(raw.get("model_id") or "ee-fit"), config=dict(raw.get("config") or {}), fit=fit,
                             checkpoint_every=int(fit.get("checkpoint_every") or 25))

    if raw["kind"] == KIND_PIPELINE:
        cfg = dict(raw.get("config") or {})
        ing = dict(cfg.get("ingestion") or {})
        if raw.get("data_location"):  # chosen by the user in the Desktop; TM never names a path
            plate = _located(raw["data_location"])
        else:
            ref = raw.get("plate_ref") or ing.get("data_file_path")
            if not ref:
                raise ValueError("me_pipeline config names no plate (plate_ref or ingestion.data_file_path)")
            plate = resolve_plate(ref)
        arch = dict(cfg.get("architecture") or {})
        opt = dict(cfg.get("optimization") or {})
        spec = SupervisedFitSpec(
            model_type=str(arch.get("model_type")),
            data_path=str(plate),
            num_classes=int(arch.get("num_classes", 1)),
            epochs=int(opt.get("epochs_full_dataset", 1)),
            seed=(int(opt["seed"]) if opt.get("seed") is not None else None),
            patience=None,
            pipeline=cfg,
            model_id=str(raw.get("model_id") or "ee-fit"),
        )
        return replace(spec, **over) if over else spec

    sp: dict[str, Any] = dict(raw.get("spec") or {})
    if raw.get("data_location"):
        sp["data_path"] = str(_located(raw["data_location"]))
    elif data_path is not None:
        sp["data_path"] = str(data_path)  # an uploaded plate
    elif raw.get("plate_ref"):
        sp["data_path"] = str(resolve_plate(raw["plate_ref"]))
    else:
        raise ValueError("me_supervised config has no data (data_handle or plate_ref)")
    if "hidden_layers" in sp:
        sp["hidden_layers"] = tuple(int(h) for h in sp["hidden_layers"])
    if sp.get("feature_names") is not None:
        sp["feature_names"] = tuple(str(n) for n in sp["feature_names"])
    spec = SupervisedFitSpec(**sp)  # unknown/invalid keys fail loudly
    return replace(spec, **over) if over else spec


def me_launch_options() -> tuple[str | None, dict[str, str]]:
    """(python executable for ME, extra env for the ME subprocess)."""
    extra: dict[str, str] = {}
    extra_path = os.environ.get("EXEC_ME_EXTRA_PYTHONPATH", "").strip()
    if extra_path:
        extra["PYTHONPATH"] = extra_path
    return (os.environ.get("EXEC_ME_PYTHON", "").strip() or None), extra


_load_staged = load_staged  # (older private name)
