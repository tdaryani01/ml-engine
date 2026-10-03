# config/schema.py
from dataclasses import dataclass, field, replace
from typing import List, Optional, Dict, Any, Mapping
from config.constants import ModelType, IngestionMode, LRHierarchy, EngineBackend

@dataclass(frozen=True)
class MetaConfig:
    """Defines metadata settings for pipeline execution and logging."""
    pipeline_name: str
    stage: str
    suppress_logging: bool
    logging_level: str
    output_dir: str

@dataclass(frozen=True)
class SplitConfig:
    """Defines dataset split ratios for training and validation."""
    train: float
    val: float

@dataclass(frozen=True)
class IngestionConfig:
    """Defines parameters for data ingestion sources, paths, and streaming properties."""
    source_mode: IngestionMode  # Enforced Enum Type!
    data_file_path: str
    feature_names: List[str]
    splits: SplitConfig
    drain_on_empty: bool
    val_queue_name: str
    amqp_url: Optional[str] = None
    queue_name: Optional[str] = None

@dataclass(frozen=True)
class CNNConfig:
    """Defines structural specs for CNN spatial pipelines and dense head transitions."""
    input_shape: List[int]                     # e.g., [Channels, Height, Width] -> [3, 28, 28]
    spatial_pipeline: List[Dict[str, Any]]     # Sequential layer specs: conv, pool, flatten, relu
    dense_head: List[int] = field(default_factory=list)  # Intermediate dense layer dimensions

@dataclass(frozen=True)
class MHSAConfig:
    """Causal MHSA geometry (active when model_type is MHSA)."""
    d_model: int
    num_heads: int
    max_seq_len: int
    action_dim: int
    ffn_mult: int = 4
    num_layers: int = 1
    use_pos_encoding: bool = True
    use_input_proj: bool = False
    action_mode: str = "continuous"  # continuous | discrete
    # Raw token width. None → d_model (no projection). When it differs from
    # d_model, MHSANetwork allocates W_in [input_dim, d_model].
    input_dim: Optional[int] = None


_SCHEMA_TEMPLATE_MHSA_FIELDS = (
    "d_model",
    "num_heads",
    "max_seq_len",
    "action_dim",
    "ffn_mult",
    "num_layers",
    "use_pos_encoding",
    "use_input_proj",
    "action_mode",
    "input_dim",
)

_SCHEMA_TEMPLATE_CNN_FIELDS = ("input_shape", "spatial_pipeline", "dense_head")

_SCHEMA_TEMPLATE_FIELDS = (
    _SCHEMA_TEMPLATE_MHSA_FIELDS
    + _SCHEMA_TEMPLATE_CNN_FIELDS
    + ("batch_size", "seq_len", "dtype")
)


@dataclass(frozen=True)
class SchemaTemplate:
    """Training-Manager-authoritative dynamic tensor dimensions.

    A ``schema_template`` carried in a TM dispatch payload is the single source of
    truth for the geometry that decides native allocations: model width, head
    count, sequence cap, FFN width and the dynamic batch/sequence axes. It
    *overlays* ``MHSAConfig`` / ``CNNConfig`` — only fields that are not ``None``
    win — so a payload can either populate a missing model block or override
    individual dims without restating the whole block.

    Every field is optional: a template may carry only the axes relevant to the
    dispatched model type, and ``apply_to_architecture_config`` leaves unset
    fields at their configured value.
    """

    # --- MHSA geometry -------------------------------------------------------
    d_model: Optional[int] = None
    num_heads: Optional[int] = None
    max_seq_len: Optional[int] = None
    action_dim: Optional[int] = None
    ffn_mult: Optional[int] = None
    num_layers: Optional[int] = None
    use_pos_encoding: Optional[bool] = None
    use_input_proj: Optional[bool] = None
    action_mode: Optional[str] = None
    input_dim: Optional[int] = None
    # --- CNN geometry --------------------------------------------------------
    input_shape: Optional[List[int]] = None
    spatial_pipeline: Optional[List[Dict[str, Any]]] = None
    dense_head: Optional[List[int]] = None
    # --- Dynamic runtime axes ------------------------------------------------
    # Informational: recorded so callers can size allocation caches / log intent.
    # Native workspace sizing still keys off the concrete X.ndarray shape at bind.
    batch_size: Optional[int] = None
    seq_len: Optional[int] = None
    dtype: Optional[str] = None

    @classmethod
    def from_mapping(cls, raw: Any) -> "SchemaTemplate | None":
        """Coerce a payload/YAML ``schema_template`` mapping into the dataclass.

        Unknown keys are ignored (forward compatibility with newer TM payloads);
        the strictness that matters — required dims for a model type — is enforced
        by the TM parser, not here.
        """
        if raw is None:
            return None
        if isinstance(raw, SchemaTemplate):
            return raw
        if not isinstance(raw, Mapping):
            raise ValueError("schema_template must be a mapping")
        values = {
            name: raw[name]
            for name in _SCHEMA_TEMPLATE_FIELDS
            if name in raw and raw[name] is not None
        }
        return cls(**values)

    def to_mhsa_mapping(self) -> Dict[str, Any]:
        """Non-``None`` MHSA fields as a plain mapping (fills a missing block)."""
        return {
            name: getattr(self, name)
            for name in _SCHEMA_TEMPLATE_MHSA_FIELDS
            if getattr(self, name) is not None
        }

    def to_cnn_mapping(self) -> Dict[str, Any]:
        """Non-``None`` CNN fields as a plain mapping (fills a missing block)."""
        return {
            name: getattr(self, name)
            for name in _SCHEMA_TEMPLATE_CNN_FIELDS
            if getattr(self, name) is not None
        }

    def apply_to_architecture_config(
        self, architecture: "ArchitectureConfig"
    ) -> "ArchitectureConfig":
        """Overlay this template onto an ``ArchitectureConfig`` (frozen → replace).

        ``architecture.mhsa`` / ``architecture.cnn`` may be either the typed
        dataclass (``MHSAConfig`` / ``CNNConfig``) or the raw mapping produced by
        YAML hydration — both forms are supported and preserved, so standalone
        loading stays byte-identical.
        """
        mhsa = getattr(architecture, "mhsa", None)
        if architecture.model_type == ModelType.MHSA and mhsa is not None:
            if isinstance(mhsa, Mapping):
                merged = dict(mhsa)
                for name in _SCHEMA_TEMPLATE_MHSA_FIELDS:
                    value = getattr(self, name)
                    if value is not None:
                        merged[name] = value
                return replace(architecture, mhsa=merged)
            if isinstance(mhsa, MHSAConfig):
                # replace() carries every MHSAConfig field; only non-None
                # template fields override (a hand-built constructor here
                # silently dropped any field it forgot to list).
                overrides = {
                    name: getattr(self, name)
                    for name in _SCHEMA_TEMPLATE_MHSA_FIELDS
                    if getattr(self, name) is not None
                }
                new_mhsa = replace(mhsa, **overrides)
                return replace(architecture, mhsa=new_mhsa)
            return architecture

        cnn = getattr(architecture, "cnn", None)
        if architecture.model_type == ModelType.CNN and cnn is not None:
            if isinstance(cnn, Mapping):
                merged = dict(cnn)
                for name in _SCHEMA_TEMPLATE_CNN_FIELDS:
                    value = getattr(self, name)
                    if value is not None:
                        merged[name] = value
                return replace(architecture, cnn=merged)
            if isinstance(cnn, CNNConfig):
                new_cnn = CNNConfig(
                    input_shape=(
                        self.input_shape if self.input_shape is not None else cnn.input_shape
                    ),
                    spatial_pipeline=(
                        self.spatial_pipeline
                        if self.spatial_pipeline is not None
                        else cnn.spatial_pipeline
                    ),
                    dense_head=(
                        self.dense_head if self.dense_head is not None else cnn.dense_head
                    ),
                )
                return replace(architecture, cnn=new_cnn)
            return architecture
        return architecture


@dataclass(frozen=True)
class ArchitectureConfig:
    """Defines structural topology settings for the neural network model."""
    model_type: ModelType                      # Enforced Enum Type!
    num_classes: int
    hidden_layers: List[int]
    backend: EngineBackend = EngineBackend.NATIVE  # Enforced Enum Type!
    p_dropout: float = 0.0
    use_batch_norm: bool = True
    bn_momentum: float = 0.9
    cnn: Optional[CNNConfig] = None            # Populated when model_type is CNN
    mhsa: Optional[MHSAConfig] = None          # Populated when model_type is MHSA
    # TM-authoritative dynamic dims (None = local YAML only).
    schema_template: Optional[SchemaTemplate] = None

@dataclass(frozen=True)
class OptimizationConfig:
    """Defines hyperparameter and optimization settings for training loops."""
    optimizer: str
    epochs_full_dataset: int
    steps_streaming: int
    batch_size: int
    learning_rate: float
    lr_scheduler: LRHierarchy                  # Enforced Enum Type!
    scheduler_drop_ratio: float
    scheduler_epochs_per_drop: int
    scheduler_decay_rate: float
    early_stopping_enabled: bool
    patience: int
    min_delta: float
    gradient_clipping_max_norm: float
    num_threads: int
    # Fit contract: seeds numpy's global RNG once at run start (None = unseeded, as before).
    seed: int | None = None


@dataclass(frozen=True)
class RegularizationConfig:
    """Defines penalty coefficients and tolerances for regularization."""
    lam_l1: float
    lam_l2: float
    sparsity_tolerance: float

@dataclass(frozen=True)
class FourierConfig:
    """Defines configuration parameters for Fourier feature expansions."""
    enabled: bool = False
    num_frequencies: int = 4

@dataclass(frozen=True)
class TransformationsConfig:
    """Aggregates data transformation configurations."""
    fourier_expansion: FourierConfig

@dataclass(frozen=True)
class PersistenceConfig:
    """Defines asset paths and flags for saving and loading model states."""
    load_saved_model: bool
    model_asset_path: str

@dataclass(frozen=True)
class LedgerSettings:
    """Phase E: append-only training document log (optional during fit)."""
    enabled: bool = False
    path: str = "training_ledger"
    branch_id: str = "main"
    checkpoint_every_steps: int = 50
    checkpoint_on_local_best: bool = True
    contract_list_enabled: bool = False  # Phase F: CNN contract-list path (off until proven)
    # Independent of ledger I/O: native async submit for contract steps.
    # Keep false on Linux for strict OMP thread caps unless measured.
    native_async_submit: bool = False
    # Pluggable persistence. Default = file_streaming (current SyncJournalWriter path).
    # noop = tests / Docker (engine+contract, no journal I/O). Future: cache, queue, redis.
    store_backend: str = "file_streaming"
    # Fit contract: start this fit from the weights in a ledger checkpoint document (a file
    # holding ``document_to_bytes(checkpoint)``). Adam restarts (begin_fit clears it).
    restore_checkpoint_path: str | None = None

@dataclass(frozen=True)
class TrainingManagerSettings:
    """Optional heartbeats to sibling training-manager control plane.

    ``instance_id`` identifies this host/process to TM. Leave empty to get a
    random numeric id at process start. The durable ledger id is TM's
    ``model_id`` from the TM payload when one is supplied; ``instance_id`` is
    only the fallback for standalone runs.
    """
    enabled: bool = False
    uri: str = "http://127.0.0.1:8000"
    instance_id: str = ""
    kind: str = "engine"
    label: str | None = None
    advertise_url: str = "http://127.0.0.1:0"
    capabilities: List[str] = field(
        default_factory=lambda: [
            "train_step",
            "ledger",
            "start",
            "pause",
            "resume",
            "restore",
            "shutdown",
            "cancel",
        ]
    )
    interval_s: float = 10.0
    timeout_s: float = 0.5
    # Idle park: sleep this long between wake/check/idle-heartbeat cycles.
    # Manager marks the instance down after ~60s without a heartbeat.
    idle_sleep_s: float = 10.0
    # After training drains, keep the process alive in the idle park loop.
    park_when_idle: bool = True
    # Start training on boot (True) or park until TM sends Start/Resume (False).
    # Host default here; a TM payload may override it via
    # ``training_manager.authorize_on_boot``.
    authorize_on_boot: bool = True
    # BL-014g Authentik M2M (token_url / client_id / username / password). Env TM_M2M_* also works.
    m2m: Optional[Dict[str, str]] = None

@dataclass(frozen=True)
class DiagnosticsConfig:
    """Defines plotting and output properties for pipeline diagnostics."""
    enabled: bool
    metric_to_plot: str
    save_raw_logs: bool
    figure_width: int
    figure_height: int
    plot_style: str
    output_format: str

@dataclass(frozen=True)
class PipelineConfig:
    """Root configuration data class containing all modular pipeline schemas."""
    meta: MetaConfig
    ingestion: IngestionConfig
    architecture: ArchitectureConfig
    optimization: OptimizationConfig
    regularization: RegularizationConfig
    transformations: TransformationsConfig
    persistence: PersistenceConfig
    diagnostics: DiagnosticsConfig
    ledger: LedgerSettings = field(default_factory=LedgerSettings)
    training_manager: TrainingManagerSettings = field(
        default_factory=TrainingManagerSettings
    )