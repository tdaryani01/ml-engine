import sys
import os
import tempfile
import numpy as np

# Ensure project root is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_factory import ModelFactory
from src.serializer import ModelSerializer


def test_serializer_roundtrip():
    print("Running ModelSerializer Save/Load Roundtrip Unit Test...")
    
    # 1. Create a mock configuration dictionary
    config = {
        "architecture": {
            "model_type": "binary_classification",
            "use_batch_norm": True,
            "bn_momentum": 0.9,
            "p_dropout": 0.0
        },
        "optimization": {
            "optimizer": "adam",
            "learning_rate": 0.01
        }
    }
    
    # 2. Instantiate and populate a concrete model instance
    original_model = ModelFactory.create_model(
        model_type="binary_classification",
        layer_sizes=(4, 8, 1),
        lr=0.01,
        optimizer="adam",
        lam_l1=0.01,
        lam_l2=0.01,
        p_dropout=0.0,
        use_batch_norm=True,
        bn_momentum=0.9
    )
    
    # Generate deterministic mock input data
    np.random.seed(42)
    X_sample = np.random.randn(5, 4)
    
    # Get forward prediction before serialization
    pred_original = original_model._forward(X_sample, training=False)
    
    # 3. Save to a temporary .npz file and reconstitute
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_asset_path = os.path.join(tmp_dir, "test_checkpoint.npz")
        
        ModelSerializer.save_model(
            model=original_model,
            config=config,
            data_provider=None,
            file_path=temp_asset_path
        )
        
        assert os.path.exists(temp_asset_path), "Checkpoint .npz file was not written to disk."
        
        reconstituted_model = ModelSerializer.load_model(file_path=temp_asset_path)
        
        # 4. Verify parameter tensor equality
        for i in range(len(original_model.weights)):
            np.testing.assert_allclose(
                original_model.weights[i],
                reconstituted_model.weights[i],
                err_msg=f"Weight mismatch at layer {i}"
            )
            np.testing.assert_allclose(
                original_model.biases[i],
                reconstituted_model.biases[i],
                err_msg=f"Bias mismatch at layer {i}"
            )
            
        if original_model.use_batch_norm:
            for i in range(len(original_model.gammas)):
                np.testing.assert_allclose(
                    original_model.gammas[i],
                    reconstituted_model.gammas[i],
                    err_msg=f"Gamma mismatch at layer {i}"
                )
                np.testing.assert_allclose(
                    original_model.betas[i],
                    reconstituted_model.betas[i],
                    err_msg=f"Beta mismatch at layer {i}"
                )
                np.testing.assert_allclose(
                    original_model.running_means[i],
                    reconstituted_model.running_means[i],
                    err_msg=f"Running mean mismatch at layer {i}"
                )
                np.testing.assert_allclose(
                    original_model.running_vars[i],
                    reconstituted_model.running_vars[i],
                    err_msg=f"Running variance mismatch at layer {i}"
                )
        
        # 5. Verify functional forward prediction equivalence
        pred_reconstituted = reconstituted_model._forward(X_sample, training=False)
        np.testing.assert_allclose(
            pred_original,
            pred_reconstituted,
            rtol=1e-6,
            err_msg="Prediction mismatch between original and reconstituted model."
        )
        
    print("[PASSED] ModelSerializer save/load state parity verified.")


def run_serializer_tests():
    print("=" * 60)
    print(" RUNNING SERIALIZER UNIT TESTS ")
    print("=" * 60)
    try:
        test_serializer_roundtrip()
        print("=" * 60)
        print("[SUCCESS] All serializer tests passed cleanly!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n[FAILURE] Serializer test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(run_serializer_tests())