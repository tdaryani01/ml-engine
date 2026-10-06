"""Preflight for a staged ME config (BL-EX-025 ``check``): everything EE would otherwise discover mid-run.

Reads the plate, describes it (a manifest: counts and a hash, never rows), checks it against the model's config,
and asks ME's strict parser to accept the payload. Nothing here trains.
"""

from __future__ import annotations

import csv
import hashlib
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from src.launcher.payload import FamilyFitSpec, build_pipeline_payload
from src.launcher.staged import spec_from_staged

MIN_ROWS_WARN = 50


def scan_plate(path: Path) -> tuple[list[str], int, Counter, list[str]]:
    """Header, row count, label counts (last column) and any non-numeric cell problems (first few)."""
    problems: list[str] = []
    labels: Counter = Counter()
    rows = 0
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            rows += 1
            if len(row) != len(header):
                if len(problems) < 3:
                    problems.append(f"line {line_no} has {len(row)} columns, expected {len(header)}")
                continue
            for cell in row:
                try:
                    float(cell)
                except ValueError:
                    if len(problems) < 3:
                        problems.append(f"line {line_no} has a non-numeric value")
                    break
            labels[row[-1].strip()] += 1
    return header, rows, labels, problems


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_corpus(spec: FamilyFitSpec) -> dict[str, Any]:
    """An imitation fit reads a frozen corpus: it must load, and hold at least two tapes (one to hold out)."""
    from src.imitation.corpus import CorpusError, load_corpus

    if not spec.data_path:  # the corpus is frozen when the fit starts, so Start's check has none to read yet
        return {"ok": True, "errors": [], "warnings": ["the corpus is checked when the fit starts"], "manifest": {}}
    try:
        eps = load_corpus(spec.data_path)
    except CorpusError as exc:
        return {"ok": False, "errors": [str(exc)], "warnings": [], "manifest": {}}
    tapes = {e.instance_id for e in eps}
    manifest = {"rows": len(eps), "tapes": len(tapes), "sha256": _sha256(Path(spec.data_path)), "source": Path(spec.data_path).name}
    errors = [] if len(tapes) >= 2 else ["the corpus has one tape: there is nothing to hold out for validation"]
    return {"ok": not errors, "errors": errors, "warnings": [], "manifest": manifest}


def _check_arrays(spec: Any, plate: Path) -> dict[str, Any]:
    """An ``.npz`` for an image model (``X`` (N, C, H, W), ``y`` (N,)) or a sequence model (``X`` (N, T, D), ``y`` (N, A))."""
    import numpy as np

    errors: list[str] = []
    manifest: dict[str, Any] = {}
    try:
        with np.load(plate) as z:
            keys = set(z.files)
            if not {"X", "y"} <= keys:
                errors.append(f"{plate.name} needs the arrays 'X' and 'y'; it has {sorted(keys)}")
            else:
                x, y = z["X"], z["y"]
                want = {"cnn": (4, "images (N, channels, height, width)"), "mhsa": (3, "sequences (N, steps, d_model)")}.get(spec.model_type)
                if want is None:
                    errors.append(f"a {spec.model_type} model reads a CSV plate, not an .npz")
                elif x.ndim != want[0]:
                    errors.append(f"X must be {want[1]}; it is {tuple(x.shape)}")
                if len(x) != len(y):
                    errors.append(f"X has {len(x)} rows but y has {len(y)}")
                manifest = {"rows": int(len(x)), "x_shape": list(x.shape[1:]), "sha256": _sha256(plate), "source": plate.name, "model_type": spec.model_type}
    except (OSError, ValueError) as exc:
        errors.append(f"{plate.name} is not a readable .npz ({type(exc).__name__})")
    return {"ok": not errors, "errors": errors, "warnings": [], "manifest": manifest}


def check_config(config_path: Path | str, data_path: Path | str | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    manifest: dict[str, Any] = {}
    try:
        spec = spec_from_staged(config_path, data_path)
    except (ValueError, OSError) as exc:
        return {"ok": False, "errors": [str(exc)], "warnings": [], "manifest": {}}
    if spec is None:
        return {"ok": False, "errors": ["this is not an ML engine run config"], "warnings": [], "manifest": {}}
    if isinstance(spec, FamilyFitSpec) and str(((spec.config.get("assembly") or {}).get("family_id")) or "").lower() == "imitation":
        return _check_corpus(spec)
    if isinstance(spec, FamilyFitSpec):  # no data file to scan: ML engine builds the run from named options and validates them
        return {"ok": True, "errors": [], "warnings": [], "manifest": {}}

    plate = Path(spec.data_path)
    if plate.suffix.lower() == ".npz":  # images and sequences are arrays, not a CSV plate
        return _check_arrays(spec, plate)
    header, rows, labels, problems = scan_plate(plate)
    manifest = {
        "rows": rows, "columns": header, "label_counts": dict(labels), "sha256": _sha256(plate),
        "source": plate.name, "model_type": spec.model_type,
    }
    errors.extend(problems)
    if len(header) < 2:
        errors.append("the plate needs at least one feature column and a target column")
    if rows == 0:
        errors.append("the plate has no rows")
    elif rows < MIN_ROWS_WARN:
        warnings.append(f"only {rows} rows: validation will be very noisy")

    distinct = len(labels)
    if spec.model_type == "binary_classification" and rows:
        if not set(labels) <= {"0", "1", "0.0", "1.0"}:
            errors.append(f"binary classification needs 0/1 labels; found {sorted(labels)[:6]}")
        elif distinct < 2:
            errors.append("binary classification needs both classes present; the plate has one")
    elif spec.model_type == "multi_class" and rows:
        if distinct > spec.num_classes:
            errors.append(f"the plate has {distinct} distinct labels but the model has num_classes={spec.num_classes}")
        elif distinct < spec.num_classes:
            warnings.append(f"the plate has {distinct} distinct labels but the model expects {spec.num_classes}")
    elif spec.model_type == "regression" and rows and distinct <= 2:
        warnings.append(
            f"this is a regression model but the target has only {distinct} distinct value(s): "
            "it looks like a classification target"
        )
    if labels and rows and max(labels.values()) / rows > 0.9 and spec.model_type != "regression":
        warnings.append("one class is more than 90% of the rows")

    if not errors:
        try:
            from config.config_loader import parse_tm_production_config  # type: ignore[import-not-found]

            with tempfile.TemporaryDirectory() as tmp:
                parse_tm_production_config(build_pipeline_payload(spec, work_dir=Path(tmp)), profile="pipeline")
        except Exception as exc:  # noqa: BLE001 - ME's own parser is the judge
            errors.append(f"ML engine rejects this configuration: {exc}")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "manifest": manifest}


_scan = scan_plate  # (older private name)
