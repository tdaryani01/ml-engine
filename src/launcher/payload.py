"""What EE hands ME for one supervised fit: a strict TM pipeline payload and a boot YAML.

ME's black-box entry (``run_pipeline.py``) reads the payload named by ``ML_ENGINE_TM_PAYLOAD``
(strict: every required section must be present) and a boot YAML named by
``ML_ENGINE_BOOT_YAML``. The boot YAML here switches off ME's own heartbeat to TM (EE is the
TM-facing side) and ``park_when_idle`` so ME runs one fit and EXITS.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CHECKPOINT_EVERY = 25  # dashboards want a checkpoint every 25 steps

_SUPERVISED = frozenset({"binary_classification", "multi_class", "regression", "mhsa", "cnn"})


_TM_ONLY = ("diet", "data_mix", "ledger", "meta", "training_manager")  # TM's own sections: not ML engine's business


@dataclass(frozen=True)
class FamilyFitSpec:
    """One fit of a model that ML engine assembles from named options (a closed-loop family): TM's own run config
    and the fit parameters. ML engine owns the loop, the early stop and the checkpoint."""

    model_id: str
    config: dict[str, Any]
    fit: dict[str, Any] = field(default_factory=dict)
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    num_threads: int = 2
    data_path: str | None = None  # a frozen corpus the engine was handed (imitation family); None for the rollout families


@dataclass(frozen=True)
class SupervisedFitSpec:
    """One supervised fit, family-agnostic: model, data and the knobs ME applies."""

    model_type: str
    data_path: str
    num_classes: int = 1
    hidden_layers: tuple[int, ...] = (16, 8)
    feature_names: tuple[str, ...] | None = None  # CSV only; ME needs explicit names
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 40
    patience: int | None = 3  # None disables ME's early stopping
    min_delta: float = 1e-4
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    train_split: float = 0.7
    val_split: float = 0.15
    p_dropout: float = 0.0
    lam_l1: float = 0.0
    lam_l2: float = 0.0
    use_batch_norm: bool = False
    optimizer: str = "adam"
    backend: str = "numpy"
    num_threads: int = 2
    seed: int | None = None  # seeds ME's RNG once per run: same seed, same fit
    mhsa: dict[str, Any] = field(default_factory=dict)  # mhsa model geometry
    cnn: dict[str, Any] = field(default_factory=dict)  # cnn model geometry
    model_id: str = "ee-fit"
    # A TM-authored run config (ME payload shape). When set it is the BASE of the payload; EE then applies
    # only the overrides it owns (output dir, ledger, plate, restore, seed, request patience/cadence).
    pipeline: dict[str, Any] | None = None
    # Autopilot's stretch number: each stretch shuffles differently, so a retry from the same weights is a new run, not
    # a copy of the last one.
    stretch_index: int = 1

    def __post_init__(self) -> None:
        if self.model_type not in _SUPERVISED:
            raise ValueError(f"unsupported model_type {self.model_type!r}; expected one of {sorted(_SUPERVISED)}")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")


# Sections ME's strict TM-payload parser knows. TM's run config also carries TM-only keys
# (assembly, diet, data_mix, ...): they are not ME's business and are dropped.
_ME_SECTIONS = (
    "meta", "ingestion", "architecture", "optimization", "regularization", "transformations",
    "persistence", "diagnostics", "ledger", "schema_template", "config_version",
)


def csv_feature_names(path: str | Path) -> list[str]:
    """Feature columns of a CSV: every column but the last (ME's loader treats the last as the target)."""
    with open(path, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    return header[:-1]


def _known_fields() -> dict[str, set[str]]:
    """The keys ME's strict config dataclasses accept, read from ME itself so EE never drifts from it."""
    import dataclasses

    from config import schema

    names = {
        "meta": schema.MetaConfig,
        "ingestion": schema.IngestionConfig,
        "architecture": schema.ArchitectureConfig,
        "optimization": schema.OptimizationConfig,
        "regularization": schema.RegularizationConfig,
    }
    return {k: {f.name for f in dataclasses.fields(v)} for k, v in names.items()}


def _from_pipeline(spec: SupervisedFitSpec, *, work_dir: Path, restore: Path | None) -> dict[str, Any]:
    base = copy.deepcopy(spec.pipeline or {})
    cfg: dict[str, Any] = {k: base[k] for k in _ME_SECTIONS if k in base}
    # ME's parser is strict: a key it does not know is a TypeError. TM's run config carries knobs of its
    # own; keep what ME understands. ``weight_decay`` is an L2 penalty, which ME calls ``lam_l2``.
    wd = (cfg.get("optimization") or {}).get("weight_decay")
    known = _known_fields()
    for section, fields_ in known.items():
        if isinstance(cfg.get(section), dict):
            cfg[section] = {k: v for k, v in cfg[section].items() if k in fields_}
    reg = {"lam_l1": 0.0, "lam_l2": 0.0, "sparsity_tolerance": 1e-5, **(cfg.get("regularization") or {})}
    if wd and not reg.get("lam_l2"):
        reg["lam_l2"] = float(wd)
    cfg["regularization"] = reg
    meta = cfg.setdefault("meta", {})
    meta.setdefault("pipeline_name", "ee_fit")
    meta.setdefault("stage", "dev")
    meta["suppress_logging"] = True
    meta["logging_level"] = "warning"
    meta["output_dir"] = str(Path(work_dir) / "out")
    ing = cfg.setdefault("ingestion", {})
    ing["source_mode"] = "csv"
    ing["data_file_path"] = str(spec.data_path)
    ing.setdefault("splits", {"train": spec.train_split, "val": spec.val_split})
    for k, v in (("drain_on_empty", False), ("val_queue_name", ""), ("amqp_url", ""), ("queue_name", "")):
        ing.setdefault(k, v)
    names = ing.get("feature_names")
    if str(spec.data_path).lower().endswith(".csv") and not (isinstance(names, list) and names):
        ing["feature_names"] = csv_feature_names(spec.data_path)  # "auto" is not understood by ME's CSV loader
    arch = cfg.setdefault("architecture", {})
    if arch.get("model_type") == "binary_classification":
        # ME's binary model is a logistic head with ONE output unit and one target column. TM's wizard
        # writes num_classes=2 ("two classes"); with two outputs the fit dies on a shape mismatch.
        arch["num_classes"] = 1
    arch.setdefault("backend", spec.backend)
    arch.setdefault("p_dropout", 0.0)
    arch.setdefault("use_batch_norm", False)
    arch.setdefault("bn_momentum", 0.9)
    arch.setdefault("hidden_layers", [])  # ML engine's parser requires it, even for cnn and mhsa, which have none of their own
    opt = cfg.setdefault("optimization", {})
    for k, v in (("optimizer", "adam"), ("steps_streaming", 1), ("lr_scheduler", "none"), ("scheduler_decay_rate", 0.98),
                 ("scheduler_epochs_per_drop", 10), ("scheduler_drop_ratio", 0.5), ("gradient_clipping_max_norm", 5.0),
                 ("early_stopping_enabled", False), ("patience", 10), ("min_delta", 1e-4)):
        opt.setdefault(k, v)
    opt.setdefault("epochs_full_dataset", spec.epochs)
    opt["num_threads"] = int(spec.num_threads)  # the Desktop owns the thread budget, whatever the model config says
    if spec.patience is not None:  # the request's patience (TM's Autopilot knob) overrides the config's
        opt["early_stopping_enabled"] = True
        opt["patience"] = int(spec.patience)
    if spec.seed is not None:
        opt["seed"] = int(spec.seed)
    if int(spec.stretch_index) > 1:  # Autopilot: stretch N shuffles with seed + N - 1, so a retry is a new run
        opt["seed"] = int(opt.get("seed") or 0) + int(spec.stretch_index) - 1
    cfg.setdefault("transformations", {"fourier_expansion": {"enabled": False, "num_frequencies": 4}})
    cfg["persistence"] = {"load_saved_model": False, "model_asset_path": "unused.npz"}
    cfg["diagnostics"] = {"enabled": False, "metric_to_plot": "loss", "save_raw_logs": False, "figure_width": 8,
                          "figure_height": 6, "plot_style": "default", "output_format": "png"}
    # The ledger is EE's to place and read, whatever TM wrote there.
    cfg["ledger"] = {
        "enabled": True, "path": "ledger", "store_backend": "file_streaming", "branch_id": "main",
        "checkpoint_every_steps": int(spec.checkpoint_every), "checkpoint_on_local_best": True,
        "contract_list_enabled": arch.get("model_type") == "mhsa", "native_async_submit": False,  # the mhsa model trains only through its contract list
        # The checkpoint stores the config TM sent (not this adapted copy), so Live and restore hand TM back its own config.
        "run_config": copy.deepcopy(base),
        **({"restore_checkpoint_path": str(restore)} if restore is not None else {}),
    }
    cfg["model_id"] = spec.model_id
    return cfg


def build_pipeline_payload(
    spec: SupervisedFitSpec,
    *,
    work_dir: Path,
    restore_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """The strict TM payload ME parses (``parse_tm_production_config``, pipeline profile)."""
    if isinstance(spec, FamilyFitSpec):
        return _from_family(spec, work_dir=work_dir, restore=restore_checkpoint_path)
    if spec.pipeline is not None:
        return _from_pipeline(spec, work_dir=work_dir, restore=restore_checkpoint_path)
    arch: dict[str, Any] = {
        "model_type": spec.model_type,
        "backend": spec.backend,
        "num_classes": int(spec.num_classes),
        "hidden_layers": [int(h) for h in spec.hidden_layers] if spec.model_type not in ("mhsa", "cnn") else [],
        "p_dropout": float(spec.p_dropout),
        "use_batch_norm": bool(spec.use_batch_norm),
        "bn_momentum": 0.9,
    }
    if spec.model_type == "mhsa":
        arch["mhsa"] = dict(spec.mhsa)
    if spec.model_type == "cnn":
        arch["cnn"] = dict(spec.cnn)
    ingestion: dict[str, Any] = {
        "source_mode": "csv",
        "data_file_path": str(spec.data_path),
        "feature_names": list(spec.feature_names) if spec.feature_names else "auto",
        "splits": {"train": float(spec.train_split), "val": float(spec.val_split)},
        "drain_on_empty": False,
        "val_queue_name": "",
        "amqp_url": "",
        "queue_name": "",
    }
    ledger: dict[str, Any] = {
        "enabled": True,
        "path": "ledger",
        "store_backend": "file_streaming",
        "branch_id": "main",
        "checkpoint_every_steps": int(spec.checkpoint_every),
        "checkpoint_on_local_best": True,
        "contract_list_enabled": spec.model_type == "mhsa",  # the mhsa model trains only through its contract list
        "native_async_submit": False,
    }
    if restore_checkpoint_path is not None:
        ledger["restore_checkpoint_path"] = str(restore_checkpoint_path)
    return {
        "model_id": spec.model_id,
        "meta": {
            "pipeline_name": "ee_fit",
            "stage": "dev",
            "suppress_logging": True,
            "logging_level": "warning",
            "output_dir": str(Path(work_dir) / "out"),
        },
        "ingestion": ingestion,
        "architecture": arch,
        "optimization": {
            "optimizer": spec.optimizer,
            "epochs_full_dataset": int(spec.epochs),
            "steps_streaming": 1,
            "batch_size": int(spec.batch_size),
            "learning_rate": float(spec.learning_rate),
            "lr_scheduler": "none",
            "scheduler_decay_rate": 0.98,
            "scheduler_epochs_per_drop": 10,
            "scheduler_drop_ratio": 0.5,
            "early_stopping_enabled": spec.patience is not None,
            "patience": int(spec.patience if spec.patience is not None else 10),
            "min_delta": float(spec.min_delta),
            "gradient_clipping_max_norm": 5.0,
            "num_threads": int(spec.num_threads),
            **({"seed": int(spec.seed)} if spec.seed is not None else {}),
        },
        "regularization": {"lam_l1": float(spec.lam_l1), "lam_l2": float(spec.lam_l2), "sparsity_tolerance": 1e-5},
        "transformations": {"fourier_expansion": {"enabled": False, "num_frequencies": 4}},
        "persistence": {"load_saved_model": False, "model_asset_path": "unused.npz"},
        "diagnostics": {
            "enabled": False,
            "metric_to_plot": "loss",
            "save_raw_logs": False,
            "figure_width": 8,
            "figure_height": 6,
            "plot_style": "default",
            "output_format": "png",
        },
        "ledger": ledger,
    }


def _from_family(spec: FamilyFitSpec, *, work_dir: Path, restore: Path | None) -> dict[str, Any]:
    base = copy.deepcopy(spec.config)
    cfg: dict[str, Any] = {k: v for k, v in base.items() if k not in _TM_ONLY}
    cfg["optimization"] = {**(cfg.get("optimization") or {}), "num_threads": int(spec.num_threads)}
    keys = ("run_budget", "resume_from", "patience", "es_warmup", "checkpoint_every", "lr", "es_min_delta", "rotate_on_es", "stretch_index", "seed")
    cfg["fit"] = {k: spec.fit[k] for k in keys if spec.fit.get(k) is not None}
    cfg["meta"] = {"pipeline_name": "ee_fit", "stage": "dev", "suppress_logging": True, "logging_level": "warning",
                   "output_dir": str(Path(work_dir) / "out")}
    # The checkpoint stores the config TM sent, so Live and restore hand TM back its own config.
    cfg["ledger"] = {"path": "ledger", "store_backend": "file_streaming", "branch_id": "main", "run_config": base,
                     **({"restore_checkpoint_path": str(restore)} if restore is not None else {})}
    cfg["model_id"] = spec.model_id
    if spec.data_path:
        cfg["imitation"] = {**(cfg.get("imitation") or {}), "corpus_path": str(spec.data_path)}
    return cfg


def build_boot_yaml(work_dir: Path) -> str:
    """ME's boot YAML for an EE-launched fit: no heartbeat to TM, exit when the fit ends."""
    return (
        "meta:\n"
        '  pipeline_name: "ee_boot"\n'
        '  stage: "dev"\n'
        "  suppress_logging: true\n"
        '  logging_level: "warning"\n'
        f'  output_dir: "{Path(work_dir) / "out"}"\n'
        "training_manager:\n"
        "  enabled: false\n"
        "  park_when_idle: false\n"
    )
