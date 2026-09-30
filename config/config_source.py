# config/config_source.py
"""Config provenance: local YAML vs Training-Manager-dispatched payload.

``ml-engine`` has two configuration authorities:

* **LOCAL** — ``config/config.yaml`` hydrated into the ``config.schema``
  dataclasses. Used for standalone runs, C++ debugging and profiling.
* **TRAINING_MANAGER** — a payload Training Manager hands to a direct-execution
  run: a file named by ``ML_ENGINE_TM_PAYLOAD`` at process start, or the
  ``config`` of a Start/Resume command. When TM supplies the config, the payload
  is the single source of truth and the local YAML must not silently fill gaps.

Detection is *explicit* — never inferred from file presence alone. Signals, in
priority order:

1. ``ML_ENGINE_CONFIG_SOURCE`` env override (set by a TM launcher / wrapper).
2. ``ML_ENGINE_TM_CONFIG_STRICT=1`` (force TM authority) /
   ``ML_ENGINE_TM_CONFIG_COMPAT=1`` (force legacy merge) — the compatibility
   window knob for a staged rollout.
3. ``ML_ENGINE_TM_PAYLOAD`` (path to the TM payload file) or
   ``ML_ENGINE_JOB_ID`` set by the TM launcher for this process.
4. Payload markers: an explicit ``config_source``, a ``schema_template`` /
   ``config_version``, or a non-empty ``architecture`` block (TM dictating
   topology ⇒ it must dictate it completely).
"""
from __future__ import annotations

import os
from enum import Enum
from typing import Any, Mapping


# Environment variable names (public: launchers and tests set these).
CONFIG_SOURCE_ENV = "ML_ENGINE_CONFIG_SOURCE"
JOB_ID_ENV = "ML_ENGINE_JOB_ID"
# Path to a JSON/YAML file holding the TM payload for a direct-execution run:
# either a bare config mapping or ``{"model_id": ..., "config": {...}}``.
TM_PAYLOAD_ENV = "ML_ENGINE_TM_PAYLOAD"
TM_CONFIG_STRICT_ENV = "ML_ENGINE_TM_CONFIG_STRICT"
TM_CONFIG_COMPAT_ENV = "ML_ENGINE_TM_CONFIG_COMPAT"

# Payload keys that mark a TM-authoritative (versioned / templated) payload.
_AUTHORITY_MARKERS = ("schema_template", "config_source", "config_version")

_TM_ALIASES = frozenset({"training_manager", "training-manager", "tm", "manager"})


class ConfigSource(str, Enum):
    """Which authority owns this run's configuration."""

    LOCAL = "local"
    TRAINING_MANAGER = "training_manager"

    @classmethod
    def coerce(cls, value: Any) -> "ConfigSource":
        """Parse a str/enum into ``ConfigSource`` (accepts common TM aliases)."""
        if isinstance(value, cls):
            return value
        raw = str(value or "").strip().lower()
        if raw == cls.LOCAL.value or raw in ("yaml", "local_yaml"):
            return cls.LOCAL
        if raw in _TM_ALIASES:
            return cls.TRAINING_MANAGER
        raise ValueError(
            f"Unknown config source {value!r}; expected one of "
            f"{[c.value for c in cls]} or TM aliases {sorted(_TM_ALIASES)}"
        )


def _truthy_env(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in ("1", "true", "yes", "on")


def detect_config_source() -> ConfigSource:
    """Resolve provenance from the process environment only (import-safe)."""
    raw = str(os.getenv(CONFIG_SOURCE_ENV, "")).strip()
    if raw:
        try:
            return ConfigSource.coerce(raw)
        except ValueError:
            # Unrecognised value → fail closed to the standalone behaviour.
            return ConfigSource.LOCAL
    if _truthy_env(TM_CONFIG_STRICT_ENV):
        return ConfigSource.TRAINING_MANAGER
    if _truthy_env(TM_CONFIG_COMPAT_ENV):
        return ConfigSource.LOCAL
    if str(os.getenv(TM_PAYLOAD_ENV, "")).strip():
        return ConfigSource.TRAINING_MANAGER
    if str(os.getenv(JOB_ID_ENV, "")).strip():
        return ConfigSource.TRAINING_MANAGER
    return ConfigSource.LOCAL


def load_tm_payload(path: str | None = None) -> dict[str, Any]:
    """Read the TM payload file for a direct-execution run.

    ``path`` defaults to ``$ML_ENGINE_TM_PAYLOAD``. JSON is tried first, then
    YAML. Raises ``FileNotFoundError`` / ``ValueError`` rather than returning an
    empty config: a TM-sourced run with no payload must not fall back to YAML.
    """
    import json

    target = path if path is not None else str(os.getenv(TM_PAYLOAD_ENV, "")).strip()
    if not target:
        raise ValueError(
            f"TM-sourced run but no payload: set {TM_PAYLOAD_ENV} to the TM "
            "payload file (JSON or YAML)"
        )
    with open(target, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        data = json.loads(text)
    except ValueError:
        import yaml

        data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise ValueError(
            f"TM payload {target!r} must be a mapping, got {type(data).__name__}"
        )
    return dict(data)


def tm_payload_model_id(payload: Mapping[str, Any] | None) -> str | None:
    """TM's durable model id for this run (``model_id`` on the payload or its config)."""
    if not isinstance(payload, Mapping):
        return None
    for source in (payload, payload.get("config")):
        if isinstance(source, Mapping):
            raw = str(source.get("model_id") or "").strip()
            if raw:
                return raw
    return None


def tm_payload_declares_authority(payload: Mapping[str, Any] | None) -> bool:
    """True when a TM ``config`` asserts authority over topology.

    A bare ``config_version`` / ``schema_template`` marker, an explicit
    ``config_source``, or any non-empty ``architecture`` block all mean "TM is
    dictating the model" — and therefore must dictate it completely.
    """
    if not isinstance(payload, Mapping):
        return False
    if any(key in payload for key in _AUTHORITY_MARKERS):
        return True
    source_marker = str(payload.get("config_source") or "").strip().lower()
    if source_marker in _TM_ALIASES:
        return True
    architecture = payload.get("architecture")
    if isinstance(architecture, Mapping):
        if "schema_template" in architecture:
            return True
        if architecture:
            return True
    return False


def resolve_config_source(
    *,
    payload: Mapping[str, Any] | None = None,
    job: Mapping[str, Any] | None = None,
) -> ConfigSource:
    """Resolve provenance for a TM payload (env override > payload > local)."""
    if _truthy_env(TM_CONFIG_STRICT_ENV):
        return ConfigSource.TRAINING_MANAGER
    if _truthy_env(TM_CONFIG_COMPAT_ENV):
        return ConfigSource.LOCAL

    env_source = str(os.getenv(CONFIG_SOURCE_ENV, "")).strip()
    if env_source:
        try:
            return ConfigSource.coerce(env_source)
        except ValueError:
            return ConfigSource.LOCAL

    if isinstance(job, Mapping):
        job_source = str(job.get("config_source") or "").strip().lower()
        if job_source in _TM_ALIASES:
            return ConfigSource.TRAINING_MANAGER
        if job_source == ConfigSource.LOCAL.value:
            return ConfigSource.LOCAL

    if tm_payload_declares_authority(payload):
        return ConfigSource.TRAINING_MANAGER
    return ConfigSource.LOCAL
