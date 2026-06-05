# config/constants.py
from enum import Enum, auto

class ModelType(Enum):
    BINARY_CLASSIFICATION = auto()
    MULTI_CLASS = auto()
    REGRESSION = auto()


class IngestionMode(Enum):
    CSV = auto()
    STREAM = auto()


class LRHierarchy(Enum):
    NONE = auto()
    STEP = auto()
    EXPONENTIAL = auto()


class ConfigSections:
    META = "meta"
    INGESTION = "ingestion"
    ARCHITECTURE = "architecture"
    OPTIMIZATION = "optimization"
    REGULARIZATION = "regularization"
    TRANSFORMATIONS = "transformations"
    PERSISTENCE = "persistence"
    DIAGNOSTICS = "diagnostics"


class DataKeys:
    X_TRAIN = "X_train"
    Y_TRAIN = "y_train"
    X_VAL = "X_val"
    Y_VAL = "y_val"
    X_TEST = "X_test"
    Y_TEST = "y_test"