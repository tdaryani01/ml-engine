# models/serializer.py
import os
import numpy as np
from models.model_factory import ModelFactory

class ModelSerializer:
    @staticmethod
    def save_model(model, config, file_path="deployed_model.npz"):
        """Serializes current trained parameters and structural metadata to a compressed asset."""
        print(f"[Saver] Serializing active {config['architecture']['model_type']} configuration...")
        
        payload = {
            "model_type": config["architecture"]["model_type"],
            "layer_sizes": np.array([model.weights[0].shape[0]] + [w.shape[1] for w in model.weights]),
            "optimizer_type": config["training"]["optimizer"],
            "lr": config["training"]["learning_rate"],
            "lam_l1": model.lam_l1,
            "lam_l2": model.lam_l2,
            "p_dropout": getattr(model, "p_dropout", 0.0)
        }
        
        # Pack operational weight matrices dynamically using layer indices
        for i, (w, b) in enumerate(zip(model.weights, model.biases)):
            payload[f"w_{i}"] = w
            payload[f"b_{i}"] = b
            
        # --- FIXED: Serialize BatchNorm Layer State Components ---
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
        
        # Instantiate base framework signature directly through factory root
        model = ModelFactory.create_model(
            model_type=str(archive["model_type"]),
            layer_sizes=tuple(archive["layer_sizes"]),
            lr=float(archive["lr"]),
            optimizer=str(archive["optimizer_type"]),
            lam_l1=float(archive["lam_l1"]),
            lam_l2=float(archive["lam_l2"]),
            p_dropout=float(archive["p_dropout"])
        )
        
        # Re-inject parameter values into internal layer components
        for i in range(len(model.weights)):
            model.weights[i] = archive[f"w_{i}"]
            model.biases[i] = archive[f"b_{i}"]
            
        # --- FIXED: Re-inject BatchNorm State Vectors ---
        for i in range(len(model.gammas)):
            model.gammas[i] = archive[f"gamma_{i}"]
            model.betas[i] = archive[f"beta_{i}"]
            model.running_means[i] = archive[f"rmean_{i}"]
            model.running_vars[i] = archive[f"rvar_{i}"]
            
        print(f"[Saver] Reconstituted {archive['model_type']} model instance from {file_path} successfully.")
        return model