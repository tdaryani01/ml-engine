# src/serializer.py
import os
import numpy as np
from src.model_factory import ModelFactory

class ModelSerializer:
    """Handles serialization and deserialization of trained neural network assets, normalization stats, and metadata[cite: 7]."""
    
    @staticmethod
    def save_model(model, config, data_provider=None, file_path="deployed_model.npz"):
        """Serializes current trained parameters, normalization stats, and structural metadata to a compressed asset[cite: 7]."""
        
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
            
        print(f"[Saver] Serializing active {model_type_str} configuration[cite: 7]...")
        
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
        
        # Save data normalization statistics if available[cite: 7]
        if data_provider is not None:
            if hasattr(data_provider, "mean") and data_provider.mean is not None:
                payload["mean"] = data_provider.mean
            if hasattr(data_provider, "std") and data_provider.std is not None:
                payload["std"] = data_provider.std

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
        print(f"[Saver] Production asset successfully saved to: {file_path}[cite: 7]")

    @staticmethod
    def load_model(file_path="deployed_model.npz"):
        """Reconstitutes a fully trained polymorphic network architecture from a saved asset[cite: 7]."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target serialization file asset not found at '{file_path}'[cite: 7]")
            
        archive = np.load(file_path)
        
        # Normalize the model type string to strip enum namespaces and force lowercase[cite: 7]
        raw_model_type = str(archive["model_type"]).strip()
        if "ModelType." in raw_model_type:
            raw_model_type = raw_model_type.split("ModelType.")[-1]
        model_type_clean = raw_model_type.lower()
        
        use_bn = bool(archive["use_batch_norm"]) if "use_batch_norm" in archive else False
        bn_mom = float(archive["bn_momentum"]) if "bn_momentum" in archive else 0.9
        
        # Instantiate base framework signature directly through factory root[cite: 7]
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
        
        for i in range(len(model.weights)):
            model.weights[i] = archive[f"w_{i}"]
            model.biases[i] = archive[f"b_{i}"]
            
        if use_bn and hasattr(model, "gammas") and model.gammas is not None:
            for i in range(len(model.gammas)):
                model.gammas[i] = archive[f"gamma_{i}"]
                model.betas[i] = archive[f"beta_{i}"]
                model.running_means[i] = archive[f"rmean_{i}"]
                model.running_vars[i] = archive[f"rvar_{i}"]
            
        print(f"[Saver] Reconstituted {model_type_clean} model instance from {file_path} successfully[cite: 7].")
        return model