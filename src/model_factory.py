# src/model_factory.py
import logging
import numpy as np
from config.constants import EngineBackend
from utils.engine_ops import create_engine_context
from src.models import BinaryClassificationNetwork, RegressionNetwork, MultiClassNetwork
from src.cnn_network import CNNNetwork
from src.mhsa_network import MHSANetwork
from src.spatial_layers import Conv2D, MaxPool2D, Flatten
from src.optimizers import AdamOptimizer, SGDOptimizer

class ModelFactory:
    """Centralized factory pattern implementation that maps configuration types to specific neural network subclasses."""
    _registry = {
        "binary_classification": BinaryClassificationNetwork,
        "regression": RegressionNetwork,
        "multi_class": MultiClassNetwork,
        "cnn": CNNNetwork,
        "mhsa": MHSANetwork,
    }

    @classmethod
    def _compute_flattened_dim(cls, input_shape: list, spatial_pipeline: list, backend: EngineBackend = EngineBackend.NATIVE) -> int:
        """
        Calculates the exact flattened logical feature dimension feeding into the dense head
        without executing buffer allocations or kernel passes.
        """
        c, h, w = input_shape[0], input_shape[1], input_shape[2]

        for layer_cfg in spatial_pipeline:
            l_type = layer_cfg["type"].lower()
            if l_type in ("conv", "conv_block"):
                k_size = layer_cfg.get("kernel_size", 3)
                k_h = k_size if isinstance(k_size, int) else k_size[0]
                k_w = k_size if isinstance(k_size, int) else k_size[1]
                stride = layer_cfg.get("stride", 1)
                pad = layer_cfg.get("pad", 0)
                out_channels = layer_cfg["out_channels"]

                h = (h + 2 * pad - k_h) // stride + 1
                w = (w + 2 * pad - k_w) // stride + 1
                c = out_channels

                if l_type == "conv_block":
                    p_size = layer_cfg.get("pool_size", 2)
                    p_stride = layer_cfg.get("pool_stride", 2)
                    h = (h - p_size) // p_stride + 1
                    w = (w - p_size) // p_stride + 1

            elif l_type == "pool":
                p_size = layer_cfg.get("pool_size", 2)
                p_stride = layer_cfg.get("stride", 2)
                h = (h - p_size) // p_stride + 1
                w = (w - p_size) // p_stride + 1

        return int(c * h * w)

    @classmethod
    def create_model(cls, model_type, layer_sizes, backend: EngineBackend = EngineBackend.NATIVE, **kwargs):
        """Instantiates and returns the requested neural network model with resolved optimizers and arguments."""
        normalized_type = str(model_type.value if hasattr(model_type, "value") else model_type).strip().lower()
        
        if normalized_type not in cls._registry:
            raise KeyError(
                f"Requested model type '{model_type}' is not registered in the system. "
                f"Available structures: {list(cls._registry.keys())}"
            )
            
        # Create a clean copy of kwargs to prevent side-effects
        factory_kwargs = kwargs.copy()
        
        # Extract the learning rate to isolate it from network initialization kwargs
        lr_val = factory_kwargs.pop("lr", 0.001) 
        
        # Resolve backend enum injection
        backend_val = backend
        if isinstance(backend_val, str):
            backend_lookup = {
                "native": EngineBackend.NATIVE,
                "im2col+gemm": EngineBackend.IM2COL_GEMM,
                "im2col_gemm": EngineBackend.IM2COL_GEMM,
                "gemm": EngineBackend.IM2COL_GEMM,
                "numpy": EngineBackend.NUMPY
            }
            backend_val = backend_lookup.get(backend_val.strip().lower(), EngineBackend.NATIVE)

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
        logging.info(f"[Factory] Initializing model block structure '{normalized_type}' (Backend: {backend_val.value}) via dynamic registry pass.")

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
            flattened_dim = cls._compute_flattened_dim(input_shape, spatial_pipeline, backend=backend_val)
            output_dim = layer_sizes[-1]
            resolved_dense_sizes = [flattened_dim] + list(dense_head) + [output_dim]

            # Strip BaseNeuralNetwork-specific kwargs not used directly in CNNNetwork signature
            factory_kwargs.pop("use_batch_norm", None)
            factory_kwargs.pop("bn_momentum", None)

            engine_ctx = create_engine_context(backend_val)

            return model_class(
                conv_configs=spatial_pipeline,
                dense_sizes=resolved_dense_sizes,
                backend=backend_val,
                engine_ctx=engine_ctx,
                input_logical_w=int(input_shape[2]),
                **factory_kwargs
            )

        # --- MHSA INSTANTIATION ROUTE ---
        if normalized_type == "mhsa":
            mhsa_config = factory_kwargs.pop("mhsa_config", None)
            if mhsa_config is None:
                raise ValueError(
                    "[Factory] 'mhsa_config' dictionary is required when initializing an MHSA model."
                )
            if hasattr(mhsa_config, "d_model"):
                d_model = mhsa_config.d_model
                num_heads = mhsa_config.num_heads
                max_seq_len = mhsa_config.max_seq_len
                action_dim = mhsa_config.action_dim
                ffn_mult = getattr(mhsa_config, "ffn_mult", 4)
                num_layers = getattr(mhsa_config, "num_layers", 1)
                use_pos_encoding = bool(getattr(mhsa_config, "use_pos_encoding", True))
            else:
                d_model = mhsa_config["d_model"]
                num_heads = mhsa_config["num_heads"]
                max_seq_len = mhsa_config["max_seq_len"]
                action_dim = mhsa_config["action_dim"]
                ffn_mult = mhsa_config.get("ffn_mult", 4)
                num_layers = mhsa_config.get("num_layers", 1)
                use_pos_encoding = bool(mhsa_config.get("use_pos_encoding", True))

            factory_kwargs.pop("use_batch_norm", None)
            factory_kwargs.pop("bn_momentum", None)
            factory_kwargs.pop("cnn_config", None)
            engine_ctx = create_engine_context(backend_val)
            return model_class(
                d_model=int(d_model),
                num_heads=int(num_heads),
                max_seq_len=int(max_seq_len),
                action_dim=int(action_dim),
                ffn_mult=int(ffn_mult),
                num_layers=int(num_layers),
                use_pos_encoding=use_pos_encoding,
                backend=backend_val,
                engine_ctx=engine_ctx,
                **factory_kwargs,
            )

        # --- MLP/STANDARD INSTANTIATION ROUTE ---
        factory_kwargs.pop("cnn_config", None)
        factory_kwargs.pop("mhsa_config", None)
        return model_class(layer_sizes=layer_sizes, **factory_kwargs)