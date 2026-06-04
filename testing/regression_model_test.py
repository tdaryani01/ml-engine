# test_model_regression.py
import unittest
import os
import numpy as np
import pandas as pd
from config.config_loader import load_production_config
from data.data_loader import load_pipeline_splits
from models.controller import ModelController

class TestModelArchitectureRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Runs once before testing begins to ensure all test fixtures exist."""
        # Define mock paths or use your active workspace data file paths
        cls.regression_data_path = r".\data\regression_data.csv"
        
        # NOTE: Update these paths to point to your real class-type test files
        cls.binary_data_path = r".\data\binary_data.csv"
        cls.multiclass_data_path = r".\data\multiclass_data.csv"

    def get_base_test_config(self, task_type):
        """Hydrates a clean baseline configuration dictionary instance for the test harness."""
        cfg = load_production_config()
        
        # Override structural parameters to create a rapid execution runtime profile
        cfg["environment"]["stage"] = "testing"
        cfg["environment"]["load_saved_model"] = False
        cfg["pipeline"]["suppress_logging"] = True
        
        cfg["architecture"]["model_type"] = task_type
        cfg["architecture"]["hidden_layers"] = [32, 16] # Fast structural test sizing
        cfg["architecture"]["p_dropout"] = 0.0
        
        cfg["training"]["epochs"] = 5  # Quick burn-in execution pass
        cfg["training"]["batch_size"] = 16
        cfg["training"]["learning_rate"] = 0.01
        cfg["training"]["early_stopping_enabled"] = False
        cfg["training"]["fourier_expansion"]["enabled"] = False
        
        return cfg

    def test_regression_class_pipeline(self):
        """Validates continuous target mapping pipelines using your continuous dataset."""
        if not os.path.exists(self.regression_data_path):
            self.skipTest(f"Regression data missing at {self.regression_data_path}")

        cfg = self.get_base_test_config("regression")
        cfg["data"]["feature_names"] = ["Time", "Radius", "Angle"]
        
        # 1. Pipeline Splitting Sanity Check
        splits = load_pipeline_splits(cfg["data"], self.regression_data_path)
        self.assertIn("X_train", splits)
        self.assertEqual(splits["X_train"].shape[1], 3)

        # 2. Setup and Execution Ingestion Integrity
        controller = ModelController(config=cfg)
        controller.setup_model(splits)
        
        train_hist, val_hist = controller.fit(splits)
        
        # Verify optimization pass occurred and recorded loss boundaries
        self.assertEqual(len(train_hist), 5)
        self.assertTrue(np.isfinite(train_hist[-1]), "Loss encountered non-finite NaN/Inf bounds.")
        
        # 3. Output Dimensions and Shape Assertions
        val_preds = controller.predict(splits["X_val"])
        self.assertEqual(val_preds.shape, splits["y_val"].shape, "Prediction matrix shape mismatch.")

    def test_binary_class_pipeline(self):
        """Validates logistic sigmoidal classification routes across binary datasets."""
        if not os.path.exists(self.binary_data_path):
            # Fallback: Generate an on-the-fly matrix mock if explicit file isn't present yet
            X_mock = np.random.uniform(0, 1, size=(100, 3))
            y_mock = np.random.randint(0, 2, size=(100, 1))
            df = pd.DataFrame(np.hstack([X_mock, y_mock]), columns=["Time", "Radius", "Angle", "Outcome"])
            os.makedirs(os.path.dirname(self.binary_data_path), exist_ok=True)
            df.to_csv(self.binary_data_path, index=False)

        cfg = self.get_base_test_config("binary_classification")
        cfg["data"]["feature_names"] = ["Time", "Radius", "Angle"]
        
        splits = load_pipeline_splits(cfg["data"], self.binary_data_path)
        controller = ModelController(config=cfg)
        controller.setup_model(splits)
        
        controller.fit(splits)
        val_preds = controller.predict(splits["X_val"])
        
        # Assert binary output properties sit safely between logistic boundary restrictions
        self.assertTrue(np.all(val_preds >= 0.0) and np.all(val_preds <= 1.0), 
                        "Binary activations breached sigmoid boundaries.")

    def test_multiclass_class_pipeline(self):
        """Validates categorical softmax logit paths across multi-class configurations."""
        if not os.path.exists(self.multiclass_data_path):
            # Fallback: Generate an on-the-fly matrix mock for a 3-class target space
            X_mock = np.random.uniform(0, 1, size=(100, 3))
            y_mock = np.random.randint(0, 3, size=(100, 1))
            df = pd.DataFrame(np.hstack([X_mock, y_mock]), columns=["Time", "Radius", "Angle", "Outcome"])
            os.makedirs(os.path.dirname(self.multiclass_data_path), exist_ok=True)
            df.to_csv(self.multiclass_data_path, index=False)

        cfg = self.get_base_test_config("multi_class")
        cfg["data"]["feature_names"] = ["Time", "Radius", "Angle"]
        
        splits = load_pipeline_splits(cfg["data"], self.multiclass_data_path)
        controller = ModelController(config=cfg)
        controller.setup_model(splits)
        
        controller.fit(splits)
        val_preds = controller.predict(splits["X_val"])
        
        # Expecting an (N, K) class probability distribution layout matrix
        self.assertEqual(len(val_preds.shape), 2, "Multiclass predictions must yield a 2D matrix profile.")
        # Verify row probability sum resolves close to 1.0 (Softmax verification)
        row_sums = np.sum(val_preds, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5, 
                                   err_msg="Softmax probability distribution failed unity checks.")

if __name__ == "__main__":
    unittest.main()