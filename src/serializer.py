# src/serializer.py
import os
import numpy as np
from src.model_factory import ModelFactory

class ModelSerializer:
    """Handles serialization and deserialization of trained neural network assets, normalization stats, and metadata."""
    
    @staticmethod
    def save_model(model, config, data_provider=None, file_path="deployed_model.npz"):
        """Serializes current trained parameters, normalization stats, and structural metadata to a compressed asset."""
        kind = ModelSerializer._model_kind(model)
        if kind != "dense":
            return ModelSerializer._save_cnn_or_mhsa(model, kind, file_path)

        
        if isinstance(config, dict):
            model_type_str = str(config["architecture"]["model_type"])
            optimizer_str   = str(config["optimization"]["optimizer"])
            learning_rate   = float(config["optimization"]["learning_rate"])
            use_batch_norm  = bool(config["architecture"]["use_batch_norm"])
            bn_momentum     = float(config["architecture"]["bn_momentum"])
        else:
            model_type_str = str(config.architecture.model_type)
            optimizer_str   = str(config.optimization.optimizer)
            learning_rate   = float(config.optimization.learning_rate)
            use_batch_norm  = bool(config.architecture.use_batch_norm)
            bn_momentum     = float(config.architecture.bn_momentum)
            
        print(f"[Saver] Serializing active {model_type_str} configuration...")
        
        payload = {
            "model_type": model_type_str,
            "layer_sizes": np.array([model.weights[0].shape[0]] + [w.shape[1] for w in model.weights]),
            "optimizer_type": optimizer_str,
            "lr": learning_rate,
            "lam_l1": float(model.lam_l1),
            "lam_l2": float(model.lam_l2),
            "p_dropout": float(getattr(model, "p_dropout", 0.0)),
            "use_batch_norm": use_batch_norm,
            "bn_momentum": bn_momentum
        }
        
        # Save data normalization statistics if available
        if data_provider is not None:
            if hasattr(data_provider, "mean") and data_provider.mean is not None:
                payload["mean"] = data_provider.mean
            if hasattr(data_provider, "std") and data_provider.std is not None:
                payload["std"] = data_provider.std

        # A regression model trains on standardised targets (RegressionNetwork.preprocess_targets): without this scaling a saved model cannot be read back in the target's units (src/export_pytorch.py).
        for key in ("y_mean", "y_std"):
            if getattr(model, key, None) is not None:
                payload[key] = np.asarray(getattr(model, key))

        for i, (w, b) in enumerate(zip(model.weights, model.biases)):
            payload[f"w_{i}"] = w
            payload[f"b_{i}"] = b
            
        if hasattr(model, "gammas") and model.gammas is not None:
            for i in range(len(model.gammas)):
                payload[f"gamma_{i}"] = model.gammas[i]
                payload[f"beta_{i}"] = model.betas[i]
                payload[f"rmean_{i}"] = model.running_means[i]
                payload[f"rvar_{i}"] = model.running_vars[i]
            
        np.savez_compressed(file_path, **payload)
        print(f"[Saver] Production asset successfully saved to: {file_path}")

    @staticmethod
    def load_model(file_path="deployed_model.npz"):
        """Reconstitutes a fully trained polymorphic network architecture from a saved asset."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target serialization file asset not found at '{file_path}'")
            
        archive = np.load(file_path)
        
        # Normalize the model type string to strip enum namespaces and force lowercase
        raw_model_type = str(archive["model_type"]).strip()
        if "ModelType." in raw_model_type:
            raw_model_type = raw_model_type.split("ModelType.")[-1]
        model_type_clean = raw_model_type.lower()
        if model_type_clean in ("cnn", "mhsa"):
            raise ValueError(
                f"'{file_path}' holds a {model_type_clean} model saved for export (src/export_pytorch.py): "
                "reloading it into the engine is not supported yet; train from the ledger checkpoints instead."
            )
        
        use_bn = bool(archive["use_batch_norm"]) if "use_batch_norm" in archive else False
        bn_mom = float(archive["bn_momentum"]) if "bn_momentum" in archive else 0.9
        
        # Instantiate base framework signature directly through factory root
        model = ModelFactory.create_model(
            model_type=model_type_clean,
            layer_sizes=tuple(archive["layer_sizes"]),
            lr=float(archive["lr"]),
            optimizer=str(archive["optimizer_type"]),
            lam_l1=float(archive["lam_l1"]),
            lam_l2=float(archive["lam_l2"]),
            p_dropout=float(archive["p_dropout"]),
            use_batch_norm=use_bn,
            bn_momentum=bn_mom
        )
        
        for key in ("y_mean", "y_std"):
            if key in archive:
                setattr(model, key, archive[key])

        for i in range(len(model.weights)):
            model.weights[i] = archive[f"w_{i}"]
            model.biases[i] = archive[f"b_{i}"]
            
        if use_bn and hasattr(model, "gammas") and model.gammas is not None:
            for i in range(len(model.gammas)):
                model.gammas[i] = archive[f"gamma_{i}"]
                model.betas[i] = archive[f"beta_{i}"]
                model.running_means[i] = archive[f"rmean_{i}"]
                model.running_vars[i] = archive[f"rvar_{i}"]
            
        print(f"[Saver] Reconstituted {model_type_clean} model instance from {file_path} successfully.")
        return model

    @staticmethod
    def _model_kind(model) -> str:
        """'cnn' (conv layers + dense head), 'mhsa' (stacked Pre-LN attention blocks) or 'dense'."""
        if hasattr(model, "param_layers") and hasattr(model, "layers"):
            return "cnn"
        if hasattr(model, "d_model") and hasattr(model, "num_heads"):
            return "mhsa"
        return "dense"

    @staticmethod
    def _describe_cnn_layer(layer) -> dict:
        """One entry of the CNN's layer list, in forward order; conv layers take their parameters from model.weights in order."""
        name = type(layer).__name__
        if isinstance(layer, str):
            return {"type": layer}      # "relu"
        if name == "ConvBlock":
            return {"type": "conv_block", "in": int(layer.in_channels), "out": int(layer.out_channels), "kernel": [int(layer.k_h), int(layer.k_w)],
                    "stride": int(layer.conv_stride), "pad": int(layer.conv_pad), "pool_size": int(layer.pool_size), "pool_stride": int(layer.pool_stride)}
        if name == "Conv2D":
            return {"type": "conv", "in": int(layer.in_channels), "out": int(layer.out_channels), "kernel": [int(layer.k_h), int(layer.k_w)],
                    "stride": int(layer.stride), "pad": int(layer.pad)}
        if name == "MaxPool2D":
            return {"type": "pool", "pool_size": int(layer.pool_size), "stride": int(layer.stride)}
        if name == "Flatten":
            return {"type": "flatten"}
        raise ValueError(f"cannot describe CNN layer {name}")

    @staticmethod
    def _save_cnn_or_mhsa(model, kind: str, file_path: str) -> None:
        """A CNN or MHSA model for export to PyTorch: parameters plus the structure that is not in the weights' shapes. Inputs are used as the loader gives them (no mean/std)."""
        import json

        payload = {"model_type": kind, "format": 2}
        if kind == "cnn":
            payload["task_type"] = str(model.task_type)
            payload["layers_json"] = json.dumps([ModelSerializer._describe_cnn_layer(layer) for layer in model.layers])
        else:
            for key in ("d_model", "num_heads", "max_seq_len", "action_dim", "ffn_mult", "num_layers", "input_dim"):
                payload[key] = int(getattr(model, key))
            payload["use_pos_encoding"] = bool(model.use_pos_encoding)
            payload["action_mode"] = str(model.action_mode)
            payload["action_temperature"] = float(model.action_temperature)
            for li in range(model.num_layers):
                payload[f"ln1_gamma_{li}"], payload[f"ln1_beta_{li}"] = model.ln1_gamma[li], model.ln1_beta[li]
                payload[f"ln2_gamma_{li}"], payload[f"ln2_beta_{li}"] = model.ln2_gamma[li], model.ln2_beta[li]
            if model.pos_embed is not None:
                payload["pos_embed"] = model.pos_embed
            if model.W_in is not None:
                payload["W_in"], payload["b_in"] = model.W_in, model.b_in
        for i, (w, b) in enumerate(zip(model.weights, model.biases)):
            payload[f"w_{i}"], payload[f"b_{i}"] = w, b
        np.savez_compressed(file_path, **payload)
        print(f"[Saver] {kind} model saved for export to: {file_path}")