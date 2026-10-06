# config/config_loader.py
import copy
import dataclasses
import os
from typing import Any, Mapping

import yaml

from config.schema import *
from config.constants import ModelType, IngestionMode, LRHierarchy, EngineBackend


class TMConfigError(ValueError):
    """Raised when a Training-Manager payload is missing required config."""


# Sections ``parse_production_config`` reads unconditionally. A TM payload that
# asserts topology authority must supply all of them (no silent YAML fallback).
_REQUIRED_TM_SECTIONS = (
    "meta",
    "ingestion",
    "architecture",
    "optimization",
    "regularization",
    "transformations",
    "persistence",
    "diagnostics",
)

# Architecture keys with no usable default — must be explicit when TM dictates.
_REQUIRED_MHSA_KEYS = ("d_model", "num_heads", "max_seq_len", "action_dim")
_REQUIRED_CNN_KEYS = ("input_shape", "spatial_pipeline")

# Closed-loop dispatch payload: the app-layer topology lives under ``closed_loop``.
_REQUIRED_CLOSED_LOOP_SECTIONS = ("closed_loop", "optimization")
_REQUIRED_CLOSED_LOOP_KEYS = ("max_steps", "batch_size")  # an option that needs more (a canvas size, say) checks its own keys

# Non-config control keys that may ride along in a TM payload.
_TM_CONTROL_KEYS = (
    "model_id",
    "job_id",
    "resume_checkpoint",
    "source",
    "autopilot",
    "gym",
    "config_version",
    "config_source",
)


def deep_merge(base: Mapping[str, Any] | None, overlay: Mapping[str, Any] | None) -> dict[str, Any]:
    """Recursively merge overlay onto a deep copy of base (dict values only)."""
    out: dict[str, Any] = copy.deepcopy(dict(base or {}))
    for key, value in dict(overlay or {}).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = copy.deepcopy(value)
    return out



def parse_production_config(raw: Mapping[str, Any]) -> PipelineConfig:
    """Parse a YAML-shaped dict into typed PipelineConfig."""
    raw = dict(raw)

    # 1. Parse Ingestion Blocks & Convert String to IngestionMode Enum
    ingestion_raw = dict(raw["ingestion"])
    splits_obj = SplitConfig(**ingestion_raw.pop("splits"))

    raw_source_str = str(ingestion_raw.pop("source_mode", "csv")).strip().upper()
    try:
        source_mode_enum = IngestionMode[raw_source_str]
    except KeyError:
        raise ValueError(f"[Config Error] Unknown source_mode string in YAML: '{raw_source_str}'")

    ingestion_obj = IngestionConfig(
        source_mode=source_mode_enum,
        splits=splits_obj,
        **ingestion_raw
    )

    # 2. Parse Architecture Blocks & Convert String to ModelType and EngineBackend Enums
    architecture_raw = dict(raw["architecture"])
    # ``schema_template`` may sit at the top level or inside ``architecture``.
    template_raw = architecture_raw.pop("schema_template", None)
    if template_raw is None:
        template_raw = raw.get("schema_template")
    schema_template_obj = SchemaTemplate.from_mapping(template_raw)
    raw_model_str = str(architecture_raw.pop("model_type", "multi_class")).strip().upper()
    try:
        model_type_enum = ModelType[raw_model_str]
    except KeyError:
        raise ValueError(f"[Config Error] Unknown model_type string in YAML: '{raw_model_str}'")

    raw_backend_str = str(
        architecture_raw.pop(
            "backend",
            os.getenv("ENGINE_BACKEND", "native"),
        )
    ).strip().lower()

    # Map possible backend string variants to EngineBackend
    backend_lookup = {
        "native": EngineBackend.NATIVE,
        "im2col+gemm": EngineBackend.IM2COL_GEMM,
        "im2col_gemm": EngineBackend.IM2COL_GEMM,
        "gemm": EngineBackend.IM2COL_GEMM,
        "numpy": EngineBackend.NUMPY
    }

    try:
        backend_enum = backend_lookup[raw_backend_str]
    except KeyError:
        raise ValueError(f"[Config Error] Unknown backend string in YAML/Environment: '{raw_backend_str}'")

    architecture_obj = ArchitectureConfig(
        model_type=model_type_enum,
        backend=backend_enum,
        schema_template=schema_template_obj,
        **architecture_raw
    )
    if schema_template_obj is not None:
        # TM-authoritative dims overlay any locally configured geometry.
        architecture_obj = schema_template_obj.apply_to_architecture_config(architecture_obj)

    # 3. Parse Optimization Blocks & Convert String to LRHierarchy Enum
    optimization_raw = dict(raw["optimization"])
    raw_sched_str = str(optimization_raw.pop("lr_scheduler", "none")).strip().upper()
    try:
        scheduler_enum = LRHierarchy[raw_sched_str]
    except KeyError:
        raise ValueError(f"[Config Error] Unknown lr_scheduler string in YAML: '{raw_sched_str}'")

    optimization_obj = OptimizationConfig(
        lr_scheduler=scheduler_enum,
        **optimization_raw
    )

    # 4. Parse Remaining Nested Primitives
    fourier_obj = FourierConfig(**raw["transformations"]["fourier_expansion"])
    transform_obj = TransformationsConfig(fourier_expansion=fourier_obj)
    ledger_obj = LedgerSettings(**raw.get("ledger", {}))
    tm_raw = dict(raw.get("training_manager") or {})
    training_manager_obj = TrainingManagerSettings(**tm_raw)

    return PipelineConfig(
        meta=MetaConfig(**raw["meta"]),
        ingestion=ingestion_obj,
        architecture=architecture_obj,
        optimization=optimization_obj,
        regularization=RegularizationConfig(**raw["regularization"]),
        transformations=transform_obj,
        persistence=PersistenceConfig(**raw["persistence"]),
        diagnostics=DiagnosticsConfig(**raw["diagnostics"]),
        ledger=ledger_obj,
        training_manager=training_manager_obj,
    )


def load_production_config(file_path="config/config.yaml") -> PipelineConfig:
    """Loads and parses production configuration from a structured YAML file."""
    with open(file_path, "r") as f:
        raw = yaml.safe_load(f)
    return parse_production_config(raw)


def load_job_pipeline_config(
    boot_file_path: str,
    job_config: Mapping[str, Any] | None,
) -> PipelineConfig:
    """Compat merge: overlay a TM config onto boot YAML (not strict).

    Used only when TM authority is off (``ML_ENGINE_TM_CONFIG_COMPAT=1``); keeps
    this host's boot ``training_manager`` identity. TM-sourced runs use
    ``parse_tm_production_config`` instead.
    """
    with open(boot_file_path, "r") as f:
        boot = yaml.safe_load(f) or {}
    tm = copy.deepcopy(boot.get("training_manager") or {})
    overlay = {
        k: v
        for k, v in dict(job_config or {}).items()
        if k
        not in (
            "training_manager",
            "resume_checkpoint",
            "source",
            "autopilot",
            "gym",
            "config_version",
        )
    }
    merged = deep_merge(boot, overlay)
    merged["training_manager"] = tm
    return parse_production_config(merged)


def load_host_identity(boot_file_path: str) -> dict[str, Any] | None:
    """Read *only* the local ``training_manager`` identity block from boot YAML.

    Host identity (TM URI, instance id, M2M auth) is a host-local concern and is
    legitimately retained even when a TM payload owns the rest of the config.
    Returns ``None`` when the file or block is absent.
    """
    identity, _ = load_boot_host_context(boot_file_path)
    return identity


def load_boot_host_context(
    boot_file_path: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Read *only* the host-local bits a TM-sourced run may inherit from boot.

    Returns ``(training_manager_identity, output_dir)``. A TM payload owns model
    topology; the boot YAML legitimately supplies only this host's TM identity
    (URI / instance id / M2M auth) and the default output directory.
    """
    try:
        with open(boot_file_path, "r") as f:
            boot = yaml.safe_load(f) or {}
    except (FileNotFoundError, OSError):
        return None, None
    if not isinstance(boot, Mapping):
        return None, None
    tm = boot.get("training_manager")
    identity = copy.deepcopy(dict(tm)) if isinstance(tm, Mapping) else None
    meta = boot.get("meta")
    output_dir = None
    if isinstance(meta, Mapping):
        raw_out = meta.get("output_dir")
        output_dir = str(raw_out) if raw_out else None
    return identity, output_dir


def _unwrap_job_config(job_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Accept a bare config mapping or a whole ``{model_id, config: {...}}`` payload."""
    raw = copy.deepcopy(dict(job_config or {}))
    if isinstance(raw.get("config"), Mapping) and not any(
        section in raw for section in _REQUIRED_TM_SECTIONS
    ):
        raw = copy.deepcopy(dict(raw["config"]))
    return raw


def _extract_schema_template(raw: Mapping[str, Any]) -> SchemaTemplate | None:
    template_raw = raw.get("schema_template")
    if template_raw is None:
        architecture = raw.get("architecture")
        if isinstance(architecture, Mapping):
            template_raw = architecture.get("schema_template")
    return SchemaTemplate.from_mapping(template_raw)


def _identity_to_mapping(host_identity: Any) -> dict[str, Any] | None:
    """Normalize a host identity block (dict or ``TrainingManagerSettings``)."""
    if host_identity is None:
        return None
    if isinstance(host_identity, Mapping):
        return copy.deepcopy(dict(host_identity))
    if isinstance(host_identity, TrainingManagerSettings):
        return {
            "enabled": host_identity.enabled,
            "uri": host_identity.uri,
            "instance_id": host_identity.instance_id,
            "kind": host_identity.kind,
            "label": host_identity.label,
            "advertise_url": host_identity.advertise_url,
            "capabilities": list(host_identity.capabilities),
            "interval_s": host_identity.interval_s,
            "timeout_s": host_identity.timeout_s,
            "idle_sleep_s": host_identity.idle_sleep_s,
            "park_when_idle": host_identity.park_when_idle,
            "authorize_on_boot": host_identity.authorize_on_boot,
            "m2m": copy.deepcopy(host_identity.m2m),
        }
    raise TypeError(
        "host_identity must be a mapping or TrainingManagerSettings, "
        f"got {type(host_identity).__name__}"
    )


def _with_tm_start_policy(
    identity: dict[str, Any] | None, payload: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Host identity + TM's start policy.

    The ``training_manager`` block is host-local (URI, instance id, auth), with
    one exception TM owns: ``authorize_on_boot`` — whether this run trains on
    boot or waits for a Start/Resume.
    """
    tm_block = payload.get("training_manager")
    if not isinstance(tm_block, Mapping) or "authorize_on_boot" not in tm_block:
        return identity
    out = dict(identity or {})
    out["authorize_on_boot"] = bool(tm_block["authorize_on_boot"])
    return out


def _merge_geometry_block(
    block_raw: Mapping[str, Any] | None,
    template_mapping: Mapping[str, Any],
    *,
    block_name: str,
    required_keys: tuple[str, ...],
    template: SchemaTemplate | None,
) -> dict[str, Any]:
    """Fill a model geometry block from ``schema_template`` then require keys."""
    if not isinstance(block_raw, Mapping):
        if template is None:
            raise TMConfigError(
                f"TM payload architecture.{block_name} is required "
                f"(or provide schema_template {block_name} dims)"
            )
        block: dict[str, Any] = {}
    else:
        block = dict(block_raw)
    if template is not None:
        merged = dict(template_mapping)
        merged.update({k: v for k, v in block.items() if v is not None})
        block = merged
    missing = [key for key in required_keys if block.get(key) in (None, "", [], {})]
    if missing:
        raise TMConfigError(
            f"TM payload {block_name} config missing required key(s): "
            + ", ".join(missing)
        )
    return block


def parse_tm_production_config(
    job_config: Mapping[str, Any] | None,
    *,
    host_identity: Any = None,
    output_dir: str | None = None,
    profile: str = "pipeline",
    require_complete: bool = True,
) -> PipelineConfig | dict[str, Any]:
    """Parse a Training-Manager payload *without* deep-merging boot YAML.

    TM is the single source of truth: every required section and the model
    geometry must be present in the payload or an explicit ``schema_template`` —
    missing keys raise ``TMConfigError`` instead of silently falling back to the
    local ``config.yaml``.

    ``profile``:
      * ``"pipeline"`` → supervised run; returns a typed ``PipelineConfig``.
      * ``"closed_loop"`` → app-layer closed-loop run; returns the validated
        payload mapping (``closed_loop`` + ``optimization`` required).

    ``host_identity`` (this host's boot ``training_manager`` block) and
    ``output_dir`` are retained because they are host-local concerns, not model
    topology. Payload control keys (``model_id``, ``job_id``, ``source``, …)
    are stripped before parsing; read ``model_id`` with
    ``config.config_source.tm_payload_model_id``.
    """
    if profile == "closed_loop":
        return _parse_tm_closed_loop_config(
            job_config,
            host_identity=host_identity,
            output_dir=output_dir,
            require_complete=require_complete,
        )
    if profile != "pipeline":
        raise ValueError(f"Unknown TM config profile {profile!r}")

    raw = _unwrap_job_config(job_config)
    for key in _TM_CONTROL_KEYS:
        raw.pop(key, None)

    schema_template_obj = _extract_schema_template(raw)

    if require_complete:
        missing_sections = [
            section
            for section in _REQUIRED_TM_SECTIONS
            if not isinstance(raw.get(section), Mapping)
        ]
        if missing_sections:
            raise TMConfigError(
                "TM payload is not authoritative — missing required section(s): "
                + ", ".join(missing_sections)
            )

        if raw["optimization"].get("num_threads") in (None, ""):
            # The thread budget is TM-authoritative too (no local default).
            raise TMConfigError(
                "TM payload optimization.num_threads is required"
            )

        architecture_raw = dict(raw["architecture"])
        model_raw = architecture_raw.get("model_type")
        if model_raw in (None, ""):
            raise TMConfigError("TM payload architecture.model_type is required")
        model_str = str(model_raw).strip().upper()

        if model_str == "MHSA":
            architecture_raw["mhsa"] = _merge_geometry_block(
                architecture_raw.get("mhsa"),
                schema_template_obj.to_mhsa_mapping() if schema_template_obj else {},
                block_name="mhsa",
                required_keys=_REQUIRED_MHSA_KEYS,
                template=schema_template_obj,
            )
        elif model_str == "CNN":
            architecture_raw["cnn"] = _merge_geometry_block(
                architecture_raw.get("cnn"),
                schema_template_obj.to_cnn_mapping() if schema_template_obj else {},
                block_name="cnn",
                required_keys=_REQUIRED_CNN_KEYS,
                template=schema_template_obj,
            )
        raw["architecture"] = architecture_raw

    public_config = copy.deepcopy(raw)  # what TM sent, before host-local identity is added
    identity = _with_tm_start_policy(_identity_to_mapping(host_identity), raw)
    if identity is not None:
        raw["training_manager"] = identity

    parsed = parse_production_config(raw)
    # A caller may hand in the config to store (``ledger.run_config``: what the user's TM sent, before any adaptation).
    stored = parsed.ledger.run_config or public_config
    return dataclasses.replace(parsed, ledger=dataclasses.replace(parsed.ledger, run_config=stored))


def _parse_tm_closed_loop_config(
    job_config: Mapping[str, Any] | None,
    *,
    host_identity: Any,
    output_dir: str | None,
    require_complete: bool,
) -> dict[str, Any]:
    """Strict closed-loop app payload: the TM config is the sole topology source.

    Boot YAML contributes only the host identity and default output directory
    (passed in by the caller). Missing ``closed_loop`` topology raises
    ``TMConfigError``; if a ``schema_template`` is present its MHSA dims are
    validated so the model builder receives a complete geometry.
    """
    cfg = copy.deepcopy(dict(_unwrap_job_config(job_config)))
    for key in _TM_CONTROL_KEYS:
        cfg.pop(key, None)

    template = _extract_schema_template(cfg)

    if require_complete:
        missing_sections = [
            section
            for section in _REQUIRED_CLOSED_LOOP_SECTIONS
            if not isinstance(cfg.get(section), Mapping)
        ]
        if missing_sections:
            raise TMConfigError(
                "TM closed-loop payload missing required section(s): "
                + ", ".join(missing_sections)
            )
        closed_loop = cfg["closed_loop"]
        missing_keys = [
            key
            for key in _REQUIRED_CLOSED_LOOP_KEYS
            if closed_loop.get(key) in (None, "", [], {})
        ]
        if missing_keys:
            raise TMConfigError(
                "TM closed-loop payload closed_loop missing required key(s): "
                + ", ".join(missing_keys)
            )
        if template is not None:
            geometry = dict(template.to_mhsa_mapping())
            # The policy's dims may be authored in the payload's own mhsa block (as for the supervised profile);
            # the schema only has to supply what that block does not.
            geometry.update({k: v for k, v in dict(cfg.get("mhsa") or {}).items() if v is not None})
            missing_geo = [
                key for key in ("d_model", "num_heads") if geometry.get(key) is None
            ]
            if missing_geo:
                raise TMConfigError(
                    "TM schema_template missing required MHSA dim(s): "
                    + ", ".join(missing_geo)
                )

    identity = _with_tm_start_policy(_identity_to_mapping(host_identity), cfg)
    if identity is not None:
        cfg["training_manager"] = identity
    meta = dict(cfg.get("meta") or {})
    if not meta.get("output_dir") and output_dir:
        meta["output_dir"] = output_dir
    cfg["meta"] = meta
    return cfg

