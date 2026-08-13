# src/model_factory.py
import logging
from src.models import BinaryClassificationNetwork, RegressionNetwork, MultiClassNetwork
from src.optimizers import AdamOptimizer, SGDOptimizer

class ModelFactory:
    """Centralized factory pattern implementation that maps configuration types to specific neural network subclasses."""
    _registry = {
        "binary_classification": BinaryClassificationNetwork,
        "regression": RegressionNetwork,
        "multi_class": MultiClassNetwork
    }

    @classmethod
    def create_model(cls, model_type, layer_sizes, **kwargs):
        """Instantiates and returns the requested neural network model with resolved optimizers and arguments[cite: 3]."""
        normalized_type = str(model_type).strip().lower()
        
        if normalized_type not in cls._registry:
            raise KeyError(
                f"Requested model type '{model_type}' is not registered in the system. "
                f"Available structures: {list(cls._registry.keys())}"
            )
            
        # Create a clean copy of kwargs to prevent side-effects[cite: 3]
        factory_kwargs = kwargs.copy()
        
        # Extract the learning rate to isolate it from network initialization kwargs[cite: 3]
        lr_val = factory_kwargs.pop("lr", 0.001) 
        
        # Resolve decoupled optimizer injection[cite: 3]
        if "optimizer" in factory_kwargs and "optimizer_instance" not in factory_kwargs:
            opt_name = str(factory_kwargs.pop("optimizer")).strip().lower()
            
            if opt_name == "adam":
                factory_kwargs["optimizer_instance"] = AdamOptimizer(lr=lr_val)
            elif opt_name == "sgd":
                factory_kwargs["optimizer_instance"] = SGDOptimizer(lr=lr_val)
            else:
                raise ValueError(f"[Factory] Unknown optimizer strategy string configured: {opt_name}")
        
        # Ensure a default dropout fallback is explicitly defined[cite: 3]
        if "p_dropout" not in factory_kwargs:
            factory_kwargs["p_dropout"] = 0.0

        model_class = cls._registry[normalized_type]
        logging.info(f"[Factory] Initializing model block structure '{normalized_type}' via dynamic registry pass.")
        
        return model_class(layer_sizes=layer_sizes, **factory_kwargs)