# config/schema.py
from dataclasses import dataclass
from typing import List, Optional
from config.constants import ModelType, IngestionMode, LRHierarchy

@dataclass(frozen=True)
class MetaConfig:
    pipeline_name: str
    stage: str
    suppress_logging: bool
    logging_level: str
    output_dir: str

@dataclass(frozen=True)
class SplitConfig:
    train: float
    val: float

@dataclass(frozen=True)
class IngestionConfig:
    source_mode: IngestionMode  # Enforced Enum Type!
    data_file_path: str
    feature_names: List[str]
    splits: SplitConfig
    amqp_url: Optional[str] = None
    queue_name: Optional[str] = None

@dataclass(frozen=True)
class ArchitectureConfig:
    model_type: ModelType      # Enforced Enum Type!
    hidden_layers: List[int]
    p_dropout: float = 0.0
    use_batch_norm: bool = True
    bn_momentum: float = 0.9

@dataclass(frozen=True)
class OptimizationConfig:
    optimizer: str
    epochs: int
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
    lam_l1: float
    lam_l2: float
    sparsity_tolerance: float

@dataclass(frozen=True)
class FourierConfig:
    enabled: bool = False
    num_frequencies: int = 4

@dataclass(frozen=True)
class TransformationsConfig:
    fourier_expansion: FourierConfig

@dataclass(frozen=True)
class PersistenceConfig:
    load_saved_model: bool
    model_asset_path: str

@dataclass(frozen=True)
class DiagnosticsConfig:
    enabled: bool
    metric_to_plot: str

@dataclass(frozen=True)
class PipelineConfig:
    meta: MetaConfig
    ingestion: IngestionConfig
    architecture: ArchitectureConfig
    optimization: OptimizationConfig
    regularization: RegularizationConfig
    transformations: TransformationsConfig
    persistence: PersistenceConfig
    diagnostics: DiagnosticsConfig