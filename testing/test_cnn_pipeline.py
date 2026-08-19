# testing/test_cnn_pipeline.py
import numpy as np
from src.controller import ModelController
from config.constants import ModelType, LRHierarchy, IngestionMode

class DummyImageProvider:
    """Mock image provider emitting 4D batches."""
    def __init__(self, n_samples=16, c=3, h=14, w=14, num_classes=3, batch_size=4):
        self.X = np.random.randn(n_samples, c, h, w).astype(np.float32)
        labels = np.random.randint(0, num_classes, size=n_samples)
        self.y = np.zeros((n_samples, num_classes), dtype=np.float32)
        self.y[np.arange(n_samples), labels] = 1.0
        
        self.batch_size = batch_size
        self.cursor = 0
        self.splits = {"X_train": self.X}
        self.y_train_processed = self.y

    def get_validation_set(self):
        return self.X[:4], self.y[:4]

    def reset_epoch(self):
        self.cursor = 0

    def has_more_batches(self):
        return self.cursor < len(self.X)

    def next_batch(self):
        end = min(self.cursor + self.batch_size, len(self.X))
        x_b = self.X[self.cursor:end]
        y_b = self.y[self.cursor:end]
        self.cursor = end
        return x_b, y_b

    def normalize(self, x):
        return x


def test_full_cnn_training_cycle():
    cnn_cfg = {
        "input_shape": [3, 14, 14],
        "spatial_pipeline": [
            {"type": "conv", "in_channels": 3, "out_channels": 4, "kernel_size": 3, "stride": 1, "pad": 0},
            {"type": "relu"},
            {"type": "pool", "pool_size": 2, "stride": 2},
            {"type": "flatten"}
        ],
        "dense_head": [16]
    }

    controller = ModelController(
        learning_rate=0.005,
        lr_scheduler_type=LRHierarchy.NONE
    )

    controller.initialize_network_from_dimensions(
        input_dim=3 * 14 * 14,
        output_dim=3,
        model_type=ModelType.CNN,
        hidden_layers=[],
        optimizer_name="adam",
        lam_l1=1e-5,
        lam_l2=1e-4,
        p_dropout=0.0,
        use_batch_norm=False,
        cnn_config=cnn_cfg
    )

    provider = DummyImageProvider()

    train_hist, val_hist = controller.fit(
        data_provider=provider,
        steps=10,
        source_mode=IngestionMode.CSV,
        model_type=ModelType.CNN,
        early_stopping_enabled=True,  # Active early stopping
        patience=3,                   # Terminate quickly after 3 stagnant validation checks
        min_delta=1e-4
    )

    assert len(train_hist) > 0, "Train loss history empty"
    assert len(val_hist) > 0, "Val loss history empty"
    assert not np.isnan(train_hist[-1]), "NaN detected in training loss"
    assert not np.isnan(val_hist[-1]), "NaN detected in validation loss"
    print(f"[PASSED] Full CNN Training Cycle verified (Final Val Loss: {val_hist[-1]:.4f}).")


if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING CNN END-TO-END PIPELINE INTEGRATION TEST ")
    print("=" * 60)
    test_full_cnn_training_cycle()
    print("=" * 60)
    print("[SUCCESS] CNN Pipeline integration test passed cleanly!")
    print("=" * 60)