# config/constants.py
from enum import Enum, auto

class ModelType(Enum):
    """Defines the supported machine learning task types for the model architecture."""
    BINARY_CLASSIFICATION = auto()
    MULTI_CLASS = auto()
    REGRESSION = auto()
    CNN = auto()  # <-- Added for Convolutional Networks


class IngestionMode(Enum):
    """Defines the data ingestion source options for the pipeline."""
    CSV = auto()
    STREAM = auto()


class LRHierarchy(Enum):
    """Defines the available learning rate scheduling strategies."""
    NONE = auto()
    STEP = auto()
    EXPONENTIAL = auto()


class ConfigSections:
    """Defines string identifiers for configuration sections."""
    META = "meta"
    INGESTION = "ingestion"
    ARCHITECTURE = "architecture"
    OPTIMIZATION = "optimization"
    REGULARIZATION = "regularization"
    TRANSFORMATIONS = "transformations"
    PERSISTENCE = "persistence"
    DIAGNOSTICS = "diagnostics"


class DataKeys:
    """Defines standard dictionary keys for dataset splits."""
    X_TRAIN = "X_train"
    Y_TRAIN = "y_train"
    X_VAL = "X_val"
    Y_VAL = "y_val"
    X_TEST = "X_test"
    Y_TEST = "y_test"