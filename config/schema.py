# config/schema.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from config.constants import ModelType, IngestionMode, LRHierarchy

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
class ArchitectureConfig:
    """Defines structural topology settings for the neural network model."""
    model_type: ModelType                      # Enforced Enum Type!
    num_classes: int
    hidden_layers: List[int]
    p_dropout: float = 0.0
    use_batch_norm: bool = True
    bn_momentum: float = 0.9
    cnn: Optional[CNNConfig] = None            # Populated when model_type is CNN

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