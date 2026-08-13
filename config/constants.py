# config/constants.py
from enum import Enum, auto

class ModelType(Enum):
    """Defines the supported machine learning task types for the model architecture[cite: 9]."""
    BINARY_CLASSIFICATION = auto()
    MULTI_CLASS = auto()
    REGRESSION = auto()


class IngestionMode(Enum):
    """Defines the data ingestion source options for the pipeline[cite: 9]."""
    CSV = auto()
    STREAM = auto()


class LRHierarchy(Enum):
    """Defines the available learning rate scheduling strategies[cite: 9]."""
    NONE = auto()
    STEP = auto()
    EXPONENTIAL = auto()


class ConfigSections:
    """Defines string identifiers for configuration sections[cite: 9]."""
    META = "meta"
    INGESTION = "ingestion"
    ARCHITECTURE = "architecture"
    OPTIMIZATION = "optimization"
    REGULARIZATION = "regularization"
    TRANSFORMATIONS = "transformations"
    PERSISTENCE = "persistence"
    DIAGNOSTICS = "diagnostics"


class DataKeys:
    """Defines standard dictionary keys for dataset splits[cite: 9]."""
    X_TRAIN = "X_train"
    Y_TRAIN = "y_train"
    X_VAL = "X_val"
    Y_VAL = "y_val"
    X_TEST = "X_test"
    Y_TEST = "y_test"