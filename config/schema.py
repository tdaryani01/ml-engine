# config/schema.py
from dataclasses import dataclass
from typing import List, Optional
from config.constants import ModelType, IngestionMode, LRHierarchy

@dataclass(frozen=True)
class MetaConfig:
    """Defines metadata settings for pipeline execution and logging[cite: 9]."""
    pipeline_name: str
    stage: str
    suppress_logging: bool
    logging_level: str
    output_dir: str

@dataclass(frozen=True)
class SplitConfig:
    """Defines dataset split ratios for training and validation[cite: 9]."""
    train: float
    val: float

@dataclass(frozen=True)
class IngestionConfig:
    """Defines parameters for data ingestion sources, paths, and streaming properties[cite: 9]."""
    source_mode: IngestionMode  # Enforced Enum Type!
    data_file_path: str
    feature_names: List[str]
    splits: SplitConfig
    drain_on_empty: bool
    val_queue_name: str
    amqp_url: Optional[str] = None
    queue_name: Optional[str] = None

@dataclass(frozen=True)
class ArchitectureConfig:
    """Defines structural topology settings for the neural network model[cite: 9]."""
    model_type: ModelType      # Enforced Enum Type!
    num_classes: int
    hidden_layers: List[int]
    p_dropout: float = 0.0
    use_batch_norm: bool = True
    bn_momentum: float = 0.9

@dataclass(frozen=True)
class OptimizationConfig:
    """Defines hyperparameter and optimization settings for training loops[cite: 9]."""
    optimizer: str
    epochs_full_dataset: int
    steps_streaming: int
    batch_size: int
    learning_rate: float
    lr_scheduler: LRHierarchy   # Enforced Enum Type!
    scheduler_drop_ratio: float
    scheduler_epochs_per_drop: int
    scheduler_decay_rate: float
    early_stopping_enabled: bool
    patience: int
    min_delta: float
    gradient_clipping_max_norm: float

@dataclass(frozen=True)
class RegularizationConfig:
    """Defines penalty coefficients and tolerances for regularization[cite: 9]."""
    lam_l1: float
    lam_l2: float
    sparsity_tolerance: float

@dataclass(frozen=True)
class FourierConfig:
    """Defines configuration parameters for Fourier feature expansions[cite: 9]."""
    enabled: bool = False
    num_frequencies: int = 4

@dataclass(frozen=True)
class TransformationsConfig:
    """Aggregates data transformation configurations[cite: 9]."""
    fourier_expansion: FourierConfig

@dataclass(frozen=True)
class PersistenceConfig:
    """Defines asset paths and flags for saving and loading model states[cite: 9]."""
    load_saved_model: bool
    model_asset_path: str

@dataclass(frozen=True)
class DiagnosticsConfig:
    """Defines plotting and output properties for pipeline diagnostics[cite: 9]."""
    enabled: bool
    metric_to_plot: str
    save_raw_logs: bool
    figure_width: int          # Configurable plotting dimensions
    figure_height: int         # Configurable plotting dimensions
    plot_style: str            # e.g., 'seaborn-v0_8-darkgrid', 'ggplot', 'default'
    output_format: str         # e.g., 'png', 'pdf', 'svg'

@dataclass(frozen=True)
class PipelineConfig:
    """Root configuration data class containing all modular pipeline schemas[cite: 9]."""
    meta: MetaConfig
    ingestion: IngestionConfig
    architecture: ArchitectureConfig
    optimization: OptimizationConfig
    regularization: RegularizationConfig
    transformations: TransformationsConfig
    persistence: PersistenceConfig
    diagnostics: DiagnosticsConfig