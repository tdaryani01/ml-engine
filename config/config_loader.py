# config/config_loader.py
import yaml
from config.schema import *
from config.constants import ModelType, IngestionMode, LRHierarchy

def load_production_config(file_path="config/config.yaml") -> PipelineConfig:
    with open(file_path, "r") as f:
        raw = yaml.safe_load(f)
        
    # 1. Parse Ingestion Blocks & Convert String to IngestionMode Enum
    ingestion_raw = raw["ingestion"].copy()
    splits_obj = SplitConfig(**ingestion_raw.pop("splits"))
    
    raw_source_str = ingestion_raw.pop("source_mode", "csv").strip().upper()
    try:
        source_mode_enum = IngestionMode[raw_source_str]
    except KeyError:
        raise ValueError(f"[Config Error] Unknown source_mode string in YAML: '{raw_source_str}'")
        
    ingestion_obj = IngestionConfig(
        source_mode=source_mode_enum,
        splits=splits_obj,
        **ingestion_raw
    )
    
    # 2. Parse Architecture Blocks & Convert String to ModelType Enum
    architecture_raw = raw["architecture"].copy()
    raw_model_str = architecture_raw.pop("model_type", "multi_class").strip().upper()
    try:
        model_type_enum = ModelType[raw_model_str]
    except KeyError:
        raise ValueError(f"[Config Error] Unknown model_type string in YAML: '{raw_model_str}'")
        
    architecture_obj = ArchitectureConfig(
        model_type=model_type_enum,
        **architecture_raw
    )
    
    # 3. Parse Optimization Blocks & Convert String to LRHierarchy Enum
    optimization_raw = raw["optimization"].copy()
    raw_sched_str = optimization_raw.pop("lr_scheduler", "none").strip().upper()
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
    
    return PipelineConfig(
        meta=MetaConfig(**raw["meta"]),
        ingestion=ingestion_obj,
        architecture=architecture_obj,
        optimization=optimization_obj,
        regularization=RegularizationConfig(**raw["regularization"]),
        transformations=transform_obj,
        persistence=PersistenceConfig(**raw["persistence"]),
        diagnostics=DiagnosticsConfig(**raw["diagnostics"])  # Explicit unpacking of new parameters
    )