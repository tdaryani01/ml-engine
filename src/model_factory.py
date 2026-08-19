# src/model_factory.py
import logging
import numpy as np
from src.models import BinaryClassificationNetwork, RegressionNetwork, MultiClassNetwork
from src.cnn_network import CNNNetwork
from src.spatial_layers import Conv2D, MaxPool2D, Flatten
from src.optimizers import AdamOptimizer, SGDOptimizer

class ModelFactory:
    """Centralized factory pattern implementation that maps configuration types to specific neural network subclasses."""
    _registry = {
        "binary_classification": BinaryClassificationNetwork,
        "regression": RegressionNetwork,
        "multi_class": MultiClassNetwork,
        "cnn": CNNNetwork
    }

    @classmethod
    def _compute_flattened_dim(cls, input_shape: list, spatial_pipeline: list) -> int:
        """
        Runs a dry forward pass on dummy spatial tensors to infer
        the exact flattened feature dimension feeding into the dense head.
        """
        dummy_x = np.zeros((1, *input_shape))
        temp_x = dummy_x

        for layer_cfg in spatial_pipeline:
            l_type = layer_cfg["type"].lower()
            if l_type == "conv":
                temp_x = Conv2D(
                    in_channels=layer_cfg["in_channels"],
                    out_channels=layer_cfg["out_channels"],
                    kernel_size=layer_cfg.get("kernel_size", 3),
                    stride=layer_cfg.get("stride", 1),
                    pad=layer_cfg.get("pad", 0)
                ).forward(temp_x)
            elif l_type == "pool":
                temp_x = MaxPool2D(
                    pool_size=layer_cfg.get("pool_size", 2),
                    stride=layer_cfg.get("stride", 2)
                ).forward(temp_x)
            elif l_type == "flatten":
                temp_x = Flatten().forward(temp_x)

        return temp_x.shape[1]

    @classmethod
    def create_model(cls, model_type, layer_sizes, **kwargs):
        """Instantiates and returns the requested neural network model with resolved optimizers and arguments."""
        normalized_type = str(model_type).strip().lower()
        
        if normalized_type not in cls._registry:
            raise KeyError(
                f"Requested model type '{model_type}' is not registered in the system. "
                f"Available structures: {list(cls._registry.keys())}"
            )
            
        # Create a clean copy of kwargs to prevent side-effects
        factory_kwargs = kwargs.copy()
        
        # Extract the learning rate to isolate it from network initialization kwargs
        lr_val = factory_kwargs.pop("lr", 0.001) 
        
        # Resolve decoupled optimizer injection
        if "optimizer" in factory_kwargs and "optimizer_instance" not in factory_kwargs:
            opt_name = str(factory_kwargs.pop("optimizer")).strip().lower()
            
            if opt_name == "adam":
                factory_kwargs["optimizer_instance"] = AdamOptimizer(lr=lr_val)
            elif opt_name == "sgd":
                factory_kwargs["optimizer_instance"] = SGDOptimizer(lr=lr_val)
            else:
                raise ValueError(f"[Factory] Unknown optimizer strategy string configured: {opt_name}")
        
        # Ensure a default dropout fallback is explicitly defined
        if "p_dropout" not in factory_kwargs:
            factory_kwargs["p_dropout"] = 0.0

        model_class = cls._registry[normalized_type]
        logging.info(f"[Factory] Initializing model block structure '{normalized_type}' via dynamic registry pass.")

        # --- CNN INSTANTIATION ROUTE ---
        if normalized_type == "cnn":
            cnn_config = factory_kwargs.pop("cnn_config", None)
            if cnn_config is None:
                raise ValueError("[Factory] 'cnn_config' dictionary is required when initializing a CNN model.")

            # Handle both dataclass and dictionary cnn_config representations
            if hasattr(cnn_config, "input_shape"):
                input_shape = cnn_config.input_shape
                spatial_pipeline = cnn_config.spatial_pipeline
                dense_head = cnn_config.dense_head
            else:
                input_shape = cnn_config.get("input_shape", [3, 28, 28])
                spatial_pipeline = cnn_config.get("spatial_pipeline", [])
                dense_head = cnn_config.get("dense_head", [])

            # Compute input size for Dense head based on spatial transformations
            flattened_dim = cls._compute_flattened_dim(input_shape, spatial_pipeline)
            output_dim = layer_sizes[-1]
            resolved_dense_sizes = [flattened_dim] + list(dense_head) + [output_dim]

            # Strip BaseNeuralNetwork-specific kwargs not used directly in CNNNetwork signature
            factory_kwargs.pop("use_batch_norm", None)
            factory_kwargs.pop("bn_momentum", None)

            return model_class(
                conv_configs=spatial_pipeline,
                dense_sizes=resolved_dense_sizes,
                **factory_kwargs
            )

        # --- MLP/STANDARD INSTANTIATION ROUTE ---
        # Remove cnn_config if passed to standard MLP networks
        factory_kwargs.pop("cnn_config", None)
        return model_class(layer_sizes=layer_sizes, **factory_kwargs)