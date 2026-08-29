# testing/test_pipeline_integration.py
import unittest
import os
import sys
import tempfile
import numpy as np
import pandas as pd

# Ensure project root is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.base_loader import BaseDataLoader
from src.data.in_memory_provider import InMemoryDataProvider
from src.controller import ModelController
from config.constants import ModelType, IngestionMode, LRHierarchy, DataKeys, EngineBackend
from config.schema import (
    PipelineConfig, MetaConfig, IngestionConfig, ArchitectureConfig,
    OptimizationConfig, RegularizationConfig, TransformationsConfig,
    FourierConfig, PersistenceConfig, DiagnosticsConfig, SplitConfig
)


class TestPipelineIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Creates an isolated temporary directory with synthetic CSV datasets."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.features = ["Time", "Radius", "Angle"]
        
        np.random.seed(42)
        n_samples = 120
        X_base = np.random.uniform(-1.0, 1.0, size=(n_samples, 3))
        
        # 1. Synthetic Continuous Regression Data
        y_reg = (X_base[:, 0] * 2.0 + X_base[:, 1] * -1.5 + np.random.normal(0, 0.1, size=n_samples)).reshape(-1, 1)
        cls.regression_data_path = os.path.join(cls.temp_dir.name, "regression_data.csv")
        df_reg = pd.DataFrame(np.hstack([X_base, y_reg]), columns=cls.features + ["Target"])
        df_reg.to_csv(cls.regression_data_path, index=False)
        
        # 2. Synthetic Binary Classification Data (0 or 1)
        y_bin = (X_base[:, 0] + X_base[:, 1] > 0).astype(int).reshape(-1, 1)
        cls.binary_data_path = os.path.join(cls.temp_dir.name, "binary_data.csv")
        df_bin = pd.DataFrame(np.hstack([X_base, y_bin]), columns=cls.features + ["Outcome"])
        df_bin.to_csv(cls.binary_data_path, index=False)
        
        # 3. Synthetic Multi-Class Data (0, 1, or 2)
        y_multi = np.random.randint(0, 3, size=(n_samples, 1))
        cls.multiclass_data_path = os.path.join(cls.temp_dir.name, "multiclass_data.csv")
        df_multi = pd.DataFrame(np.hstack([X_base, y_multi]), columns=cls.features + ["Outcome"])
        df_multi.to_csv(cls.multiclass_data_path, index=False)

    @classmethod
    def tearDownClass(cls):
        """Cleans up the temporary directory to leave zero untracked files."""
        cls.temp_dir.cleanup()

    def get_base_test_config(self, task_type: ModelType, data_file_path: str, num_classes: int = 1, backend: EngineBackend = EngineBackend.NATIVE) -> PipelineConfig:
        """Hydrates a clean baseline strongly-typed PipelineConfig instance matching schema parameters."""
        return PipelineConfig(
            meta=MetaConfig(
                pipeline_name="integration_test_run",
                stage="testing",
                suppress_logging=True,
                logging_level="ERROR",
                output_dir="test_diagnostics"
            ),
            ingestion=IngestionConfig(
                source_mode=IngestionMode.CSV,
                data_file_path=data_file_path,  
                feature_names=self.features,
                splits=SplitConfig(train=0.70, val=0.30),
                drain_on_empty=False,
                val_queue_name=""
            ),
            architecture=ArchitectureConfig(
                model_type=task_type,
                num_classes=num_classes,
                hidden_layers=[16, 8],
                backend=backend,
                p_dropout=0.0,
                use_batch_norm=False,
                bn_momentum=0.9
            ),
            optimization=OptimizationConfig(
                optimizer="adam",
                epochs_full_dataset=3,
                steps_streaming=100,
                batch_size=16,
                learning_rate=0.01,
                lr_scheduler=LRHierarchy.NONE,
                scheduler_drop_ratio=0.5,
                scheduler_epochs_per_drop=10,
                scheduler_decay_rate=0.98,
                early_stopping_enabled=False,
                patience=5,
                min_delta=1e-5,
                gradient_clipping_max_norm=5.0,
                num_threads=4
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
                metric_to_plot="loss",
                save_raw_logs=False,       
                figure_width=8,            
                figure_height=4,           
                plot_style="default",      
                output_format="png"        
            )
        )

    def test_regression_class_pipeline(self):
        """Validates continuous target mapping pipelines across all compute backends."""
        for backend in [EngineBackend.NATIVE, EngineBackend.IM2COL_GEMM, EngineBackend.NUMPY]:
            with self.subTest(backend=backend.value):
                cfg = self.get_base_test_config(ModelType.REGRESSION, data_file_path=self.regression_data_path, num_classes=1, backend=backend)

                loader = BaseDataLoader.create_loader(cfg)
                data_provider = InMemoryDataProvider(
                    loader=loader,
                    batch_size=cfg.optimization.batch_size,
                    epochs=cfg.optimization.epochs_full_dataset,
                    normalize_features=True
                )

                self.assertIn(DataKeys.X_TRAIN, data_provider.splits)
                self.assertEqual(data_provider.splits[DataKeys.X_TRAIN].shape[1], 3)

                controller = ModelController(
                    learning_rate=cfg.optimization.learning_rate,
                    lr_scheduler_type=cfg.optimization.lr_scheduler,
                    data_provider=data_provider
                )
                controller.initialize_network_from_dimensions(
                    input_dim=len(cfg.ingestion.feature_names),
                    output_dim=1,  
                    model_type=cfg.architecture.model_type,
                    hidden_layers=cfg.architecture.hidden_layers,
                    optimizer_name=cfg.optimization.optimizer,
                    lam_l1=cfg.regularization.lam_l1,
                    lam_l2=cfg.regularization.lam_l2,
                    p_dropout=cfg.architecture.p_dropout,
                    use_batch_norm=cfg.architecture.use_batch_norm,
                    bn_momentum=cfg.architecture.bn_momentum,
                    max_norm=cfg.optimization.gradient_clipping_max_norm,
                    backend=cfg.architecture.backend
                )

                train_hist, val_hist = controller.fit(
                    steps=cfg.optimization.steps_streaming,
                    source_mode=cfg.ingestion.source_mode,
                    model_type=cfg.architecture.model_type,
                    early_stopping_enabled=cfg.optimization.early_stopping_enabled,
                    patience=cfg.optimization.patience,
                    min_delta=cfg.optimization.min_delta
                )

                self.assertGreater(len(train_hist), 0)
                self.assertTrue(np.isfinite(train_hist[-1]), f"[{backend.value}] Loss encountered non-finite NaN/Inf bounds.")
                
                X_val, y_val = data_provider.get_validation_set()
                val_preds = controller.predict(X_val)
                self.assertEqual(val_preds.shape, y_val.shape, f"[{backend.value}] Prediction matrix shape mismatch.")

    def test_binary_class_pipeline(self):
        """Validates logistic sigmoidal classification routes across all compute backends."""
        for backend in [EngineBackend.NATIVE, EngineBackend.IM2COL_GEMM, EngineBackend.NUMPY]:
            with self.subTest(backend=backend.value):
                cfg = self.get_base_test_config(ModelType.BINARY_CLASSIFICATION, data_file_path=self.binary_data_path, num_classes=1, backend=backend)

                loader = BaseDataLoader.create_loader(cfg)
                data_provider = InMemoryDataProvider(
                    loader=loader,
                    batch_size=cfg.optimization.batch_size,
                    epochs=cfg.optimization.epochs_full_dataset,
                    normalize_features=True
                )

                controller = ModelController(
                    learning_rate=cfg.optimization.learning_rate,
                    lr_scheduler_type=cfg.optimization.lr_scheduler,
                    data_provider=data_provider
                )
                controller.initialize_network_from_dimensions(
                    input_dim=len(cfg.ingestion.feature_names),
                    output_dim=1,  
                    model_type=cfg.architecture.model_type,
                    hidden_layers=cfg.architecture.hidden_layers,
                    optimizer_name=cfg.optimization.optimizer,
                    lam_l1=cfg.regularization.lam_l1,
                    lam_l2=cfg.regularization.lam_l2,
                    p_dropout=cfg.architecture.p_dropout,
                    use_batch_norm=cfg.architecture.use_batch_norm,
                    bn_momentum=cfg.architecture.bn_momentum,
                    max_norm=cfg.optimization.gradient_clipping_max_norm,
                    backend=cfg.architecture.backend
                )

                controller.fit(
                    steps=cfg.optimization.steps_streaming,
                    source_mode=cfg.ingestion.source_mode,
                    model_type=cfg.architecture.model_type,
                    early_stopping_enabled=cfg.optimization.early_stopping_enabled,
                    patience=cfg.optimization.patience,
                    min_delta=cfg.optimization.min_delta
                )

                X_val, _ = data_provider.get_validation_set()
                val_preds = controller.predict(X_val)
                
                self.assertTrue(np.all(val_preds >= 0.0) and np.all(val_preds <= 1.0), 
                                f"[{backend.value}] Binary activations breached sigmoid boundaries.")

    def test_multiclass_class_pipeline(self):
        """Validates categorical softmax logit paths across all compute backends."""
        for backend in [EngineBackend.NATIVE, EngineBackend.IM2COL_GEMM, EngineBackend.NUMPY]:
            with self.subTest(backend=backend.value):
                cfg = self.get_base_test_config(ModelType.MULTI_CLASS, data_file_path=self.multiclass_data_path, num_classes=3, backend=backend)

                loader = BaseDataLoader.create_loader(cfg)
                data_provider = InMemoryDataProvider(
                    loader=loader,
                    batch_size=cfg.optimization.batch_size,
                    epochs=cfg.optimization.epochs_full_dataset,
                    normalize_features=True
                )

                controller = ModelController(
                    learning_rate=cfg.optimization.learning_rate,
                    lr_scheduler_type=cfg.optimization.lr_scheduler,
                    data_provider=data_provider
                )
                controller.initialize_network_from_dimensions(
                    input_dim=len(cfg.ingestion.feature_names),
                    output_dim=3,  
                    model_type=cfg.architecture.model_type,
                    hidden_layers=cfg.architecture.hidden_layers,
                    optimizer_name=cfg.optimization.optimizer,
                    lam_l1=cfg.regularization.lam_l1,
                    lam_l2=cfg.regularization.lam_l2,
                    p_dropout=cfg.architecture.p_dropout,
                    use_batch_norm=cfg.architecture.use_batch_norm,
                    bn_momentum=cfg.architecture.bn_momentum,
                    max_norm=cfg.optimization.gradient_clipping_max_norm,
                    backend=cfg.architecture.backend
                )

                controller.fit(
                    steps=cfg.optimization.steps_streaming,
                    source_mode=cfg.ingestion.source_mode,
                    model_type=cfg.architecture.model_type,
                    early_stopping_enabled=cfg.optimization.early_stopping_enabled,
                    patience=cfg.optimization.patience,
                    min_delta=cfg.optimization.min_delta
                )

                X_val, _ = data_provider.get_validation_set()
                val_preds = controller.predict(X_val)
                
                self.assertEqual(len(val_preds.shape), 2, f"[{backend.value}] Multiclass predictions must yield a 2D matrix profile.")
                row_sums = np.sum(val_preds, axis=1)
                np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5, 
                                           err_msg=f"[{backend.value}] Softmax probability distribution failed unity checks.")


if __name__ == "__main__":
    unittest.main()