# test_model_regression.py
import unittest
import os
import numpy as np
import pandas as pd
from data.data_loader import load_pipeline_splits
from data.csv_provider import CSVDataProvider
from models.controller import ModelController
from config.constants import ModelType, IngestionMode, LRHierarchy
from config.schema import (
    PipelineConfig, MetaConfig, IngestionConfig, ArchitectureConfig,
    OptimizationConfig, RegularizationConfig, TransformationsConfig,
    FourierConfig, PersistenceConfig, DiagnosticsConfig, SplitConfig
)

class TestModelArchitectureRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Runs once before testing begins to ensure all test fixtures exist."""
        cls.regression_data_path = r".\data\regression_data.csv"
        cls.binary_data_path = r".\data\binary_data.csv"
        cls.multiclass_data_path = r".\data\multiclass_data.csv"

    def get_base_test_config(self, task_type: ModelType) -> PipelineConfig:
        """Hydrates a clean baseline strongly-typed PipelineConfig instance for the test harness."""
        return PipelineConfig(
            meta=MetaConfig(
                pipeline_name="regression_test_run",
                stage="testing",
                suppress_logging=True,
                logging_level="INFO",
                output_dir="test_diagnostics"
            ),
            ingestion=IngestionConfig(
                source_mode=IngestionMode.CSV,
                data_file_path="",  # Populated dynamically by specific test cases
                feature_names=["Time", "Radius", "Angle"],
                splits=SplitConfig(train=0.70, val=0.30)
            ),
            architecture=ArchitectureConfig(
                model_type=task_type,
                hidden_layers=[32, 16],
                p_dropout=0.0,
                use_batch_norm=False,
                bn_momentum=0.9
            ),
            optimization=OptimizationConfig(
                optimizer="adam",
                epochs=5,
                batch_size=16,
                learning_rate=0.01,
                lr_scheduler=LRHierarchy.NONE,
                scheduler_drop_ratio=0.5,
                scheduler_epochs_per_drop=10,
                scheduler_decay_rate=0.98,
                early_stopping_enabled=False,
                patience=5,
                min_delta=1e-5,
                gradient_clipping_max_norm=5.0
            ),
            regularization=RegularizationConfig(
                lam_l1=0.0,
                lam_l2=0.0,
                sparsity_tolerance=1e-5
            ),
            transformations=TransformationsConfig(
                fourier_expansion=FourierConfig(enabled=False, num_frequencies=4)
            ),
            persistence=PersistenceConfig(
                load_saved_model=False,
                model_asset_path=""
            ),
            diagnostics=DiagnosticsConfig(
                enabled=False,
                metric_to_plot="loss"
            )
        )

    def test_regression_class_pipeline(self):
        """Validates continuous target mapping pipelines using your continuous dataset."""
        if not os.path.exists(self.regression_data_path):
            self.skipTest(f"Regression data missing at {self.regression_data_path}")

        cfg = self.get_base_test_config(ModelType.REGRESSION)
        
        # 1. Pipeline Splitting Sanity Check via explicit primitive parameters
        splits = load_pipeline_splits(
            data_file_path=self.regression_data_path,
            feature_names=cfg.ingestion.feature_names
        )
        self.assertIn("X_train", splits)
        self.assertEqual(splits["X_train"].shape[1], 3)

        # 2. Controller Setup and Architecture Compilation
        controller = ModelController(
            learning_rate=cfg.optimization.learning_rate,
            lr_scheduler_type=cfg.optimization.lr_scheduler
        )
        controller.initialize_network_from_dimensions(
            input_dim=len(cfg.ingestion.feature_names),
            output_dim=1,  # Continuous regression tracking head target scalar
            model_type=cfg.architecture.model_type,
            hidden_layers=cfg.architecture.hidden_layers,
            optimizer_name=cfg.optimization.optimizer,
            lam_l1=cfg.regularization.lam_l1,
            lam_l2=cfg.regularization.lam_l2,
            p_dropout=cfg.architecture.p_dropout,
            use_batch_norm=cfg.architecture.use_batch_norm,
            bn_momentum=cfg.architecture.bn_momentum,
            max_norm=cfg.optimization.gradient_clipping_max_norm
        )

        # 3. Polymorphic Data Strategy Execution Pass
        data_provider = CSVDataProvider(
            data_file_path=self.regression_data_path,
            feature_names=cfg.ingestion.feature_names,
            batch_size=cfg.optimization.batch_size,
            model_instance=controller.model
        )

        train_hist, val_hist = controller.fit_via_provider(
            data_provider=data_provider,
            epochs=cfg.optimization.epochs,
            batch_size=cfg.optimization.batch_size,
            source_mode=cfg.ingestion.source_mode,
            model_type=cfg.architecture.model_type,
            early_stopping_enabled=cfg.optimization.early_stopping_enabled
        )

        self.assertEqual(len(train_hist), 5)
        self.assertTrue(np.isfinite(train_hist[-1]), "Loss encountered non-finite NaN/Inf bounds.")
        
        # 4. Dimension Verification
        X_val, y_val = data_provider.get_validation_set()
        val_preds = controller.predict(X_val)
        self.assertEqual(val_preds.shape, y_val.shape, "Prediction matrix shape mismatch.")

    def test_binary_class_pipeline(self):
        """Validates logistic sigmoidal classification routes across binary datasets."""
        if not os.path.exists(self.binary_data_path):
            X_mock = np.random.uniform(0, 1, size=(100, 3))
            y_mock = np.random.randint(0, 2, size=(100, 1))
            df = pd.DataFrame(np.hstack([X_mock, y_mock]), columns=["Time", "Radius", "Angle", "Outcome"])
            os.makedirs(os.path.dirname(self.binary_data_path), exist_ok=True)
            df.to_csv(self.binary_data_path, index=False)

        cfg = self.get_base_test_config(ModelType.BINARY_CLASSIFICATION)
        
        controller = ModelController(
            learning_rate=cfg.optimization.learning_rate,
            lr_scheduler_type=cfg.optimization.lr_scheduler
        )
        controller.initialize_network_from_dimensions(
            input_dim=len(cfg.ingestion.feature_names),
            output_dim=1,  # Binary classification uses a single sigmoidal logit channel
            model_type=cfg.architecture.model_type,
            hidden_layers=cfg.architecture.hidden_layers,
            optimizer_name=cfg.optimization.optimizer,
            lam_l1=cfg.regularization.lam_l1,
            lam_l2=cfg.regularization.lam_l2,
            p_dropout=cfg.architecture.p_dropout,
            use_batch_norm=cfg.architecture.use_batch_norm,
            bn_momentum=cfg.architecture.bn_momentum,
            max_norm=cfg.optimization.gradient_clipping_max_norm
        )

        data_provider = CSVDataProvider(
            data_file_path=self.binary_data_path,
            feature_names=cfg.ingestion.feature_names,
            batch_size=cfg.optimization.batch_size,
            model_instance=controller.model
        )

        controller.fit_via_provider(
            data_provider=data_provider,
            epochs=cfg.optimization.epochs,
            batch_size=cfg.optimization.batch_size,
            source_mode=cfg.ingestion.source_mode,
            model_type=cfg.architecture.model_type,
            early_stopping_enabled=cfg.optimization.early_stopping_enabled
        )

        X_val, _ = data_provider.get_validation_set()
        val_preds = controller.predict(X_val)
        
        self.assertTrue(np.all(val_preds >= 0.0) and np.all(val_preds <= 1.0), 
                        "Binary activations breached sigmoid boundaries.")

    def test_multiclass_class_pipeline(self):
        """Validates categorical softmax logit paths across multi-class configurations."""
        if not os.path.exists(self.multiclass_data_path):
            X_mock = np.random.uniform(0, 1, size=(100, 3))
            y_mock = np.random.randint(0, 3, size=(100, 1))
            df = pd.DataFrame(np.hstack([X_mock, y_mock]), columns=["Time", "Radius", "Angle", "Outcome"])
            os.makedirs(os.path.dirname(self.multiclass_data_path), exist_ok=True)
            df.to_csv(self.multiclass_data_path, index=False)

        cfg = self.get_base_test_config(ModelType.MULTI_CLASS)
        
        controller = ModelController(
            learning_rate=cfg.optimization.learning_rate,
            lr_scheduler_type=cfg.optimization.lr_scheduler
        )
        controller.initialize_network_from_dimensions(
            input_dim=len(cfg.ingestion.feature_names),
            output_dim=3,  # Multi-class footprint mapped matching your output space
            model_type=cfg.architecture.model_type,
            hidden_layers=cfg.architecture.hidden_layers,
            optimizer_name=cfg.optimization.optimizer,
            lam_l1=cfg.regularization.lam_l1,
            lam_l2=cfg.regularization.lam_l2,
            p_dropout=cfg.architecture.p_dropout,
            use_batch_norm=cfg.architecture.use_batch_norm,
            bn_momentum=cfg.architecture.bn_momentum,
            max_norm=cfg.optimization.gradient_clipping_max_norm
        )

        data_provider = CSVDataProvider(
            data_file_path=self.multiclass_data_path,
            feature_names=cfg.ingestion.feature_names,
            batch_size=cfg.optimization.batch_size,
            model_instance=controller.model
        )

        controller.fit_via_provider(
            data_provider=data_provider,
            epochs=cfg.optimization.epochs,
            batch_size=cfg.optimization.batch_size,
            source_mode=cfg.ingestion.source_mode,
            model_type=cfg.architecture.model_type,
            early_stopping_enabled=cfg.optimization.early_stopping_enabled
        )

        X_val, _ = data_provider.get_validation_set()
        val_preds = controller.predict(X_val)
        
        self.assertEqual(len(val_preds.shape), 2, "Multiclass predictions must yield a 2D matrix profile.")
        row_sums = np.sum(val_preds, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5, 
                                   err_msg="Softmax probability distribution failed unity checks.")

if __name__ == "__main__":
    unittest.main()