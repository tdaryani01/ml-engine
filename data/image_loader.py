# src/image_loader.py
import os
import csv
import gzip
import urllib.request
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# src/image_loader.py (Update ImageBatchProvider class)

class ImageBatchProvider:
    """
    Conforms to ModelController's DataProvider interface for feeding 4D image mini-batches,
    exposing metadata attributes for NeuralNetworkDiagnostics.
    """
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray, batch_size: int = 32):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size

        self.cursor = 0
        self.splits = {"X_train": self.X_train, "X_val": self.X_val}
        self.y_train_processed = self.y_train

        # Tabular/Diagnostics Compatibility Metadata
        self.mean = np.zeros(X_train.shape[1:], dtype=np.float32)
        self.std = np.ones(X_train.shape[1:], dtype=np.float32)
        self.feature_names = [f"channel_dim_{i}" for i in range(X_train.shape[1])]
        self.classes_ = np.arange(y_train.shape[1])
        self.label_map = {i: f"Class_{i}" for i in range(y_train.shape[1])}

    def get_validation_set(self):
        return self.X_val, self.y_val

    def reset_epoch(self):
        self.cursor = 0
        indices = np.arange(len(self.X_train))
        np.random.shuffle(indices)
        self.X_train = self.X_train[indices]
        self.y_train = self.y_train[indices]
        self.splits["X_train"] = self.X_train
        self.y_train_processed = self.y_train

    def has_more_batches(self) -> bool:
        return self.cursor < len(self.X_train)

    def next_batch(self) -> tuple:
        end = min(self.cursor + self.batch_size, len(self.X_train))
        X_b = self.X_train[self.cursor:end]
        y_b = self.y_train[self.cursor:end]
        self.cursor = end
        return X_b, y_b

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return x


class StandardImageLoader:
    """
    Ingests and caches standard image classification datasets (MNIST, Fashion-MNIST),
    as well as custom CSV image datasets, returning normalized 4D spatial tensors: (N, Channels, Height, Width).
    """
    DATASET_URLS = {
        "mnist": {
            "base_url": "https://storage.googleapis.com/cvdf-datasets/mnist/",
            "train_img": "train-images-idx3-ubyte.gz",
            "train_lbl": "train-labels-idx1-ubyte.gz",
            "test_img": "t10k-images-idx3-ubyte.gz",
            "test_lbl": "t10k-labels-idx1-ubyte.gz",
        },
        "fashion_mnist": {
            "base_url": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
            "train_img": "train-images-idx3-ubyte.gz",
            "train_lbl": "train-labels-idx1-ubyte.gz",
            "test_img": "t10k-images-idx3-ubyte.gz",
            "test_lbl": "t10k-labels-idx1-ubyte.gz",
        }
    }

    def __init__(self, cache_dir: str = os.path.join("data", "cache")):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _download_and_extract(self, base_url: str, filename: str) -> bytes:
        local_gz_path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(local_gz_path):
            url = f"{base_url}{filename}"
            logging.info(f"[Image Loader] Downloading {filename} from {url}...")
            urllib.request.urlretrieve(url, local_gz_path)

        with gzip.open(local_gz_path, "rb") as f:
            return f.read()

    def load_dataset(self, name: str = "mnist", max_train_samples: int = None,
                     max_test_samples: int = None) -> tuple:
        """
        Loads and returns (X_train, y_train, X_test, y_test).
        X is formatted as (N, 1, 28, 28) with values in [0.0, 1.0].
        y is one-hot encoded as (N, 10).
        """
        dataset_key = name.lower().replace("-", "_")
        if dataset_key not in self.DATASET_URLS:
            raise ValueError(f"Unknown dataset '{name}'. Available: {list(self.DATASET_URLS.keys())}")

        cfg = self.DATASET_URLS[dataset_key]

        train_img_bytes = self._download_and_extract(cfg["base_url"], cfg["train_img"])
        train_lbl_bytes = self._download_and_extract(cfg["base_url"], cfg["train_lbl"])
        test_img_bytes = self._download_and_extract(cfg["base_url"], cfg["test_img"])
        test_lbl_bytes = self._download_and_extract(cfg["base_url"], cfg["test_lbl"])

        # Parse IDX image buffers (Skip 16-byte magic header)
        X_train = np.frombuffer(train_img_bytes, dtype=np.uint8, offset=16).reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
        y_train_raw = np.frombuffer(train_lbl_bytes, dtype=np.uint8, offset=8)

        X_test = np.frombuffer(test_img_bytes, dtype=np.uint8, offset=16).reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
        y_test_raw = np.frombuffer(test_lbl_bytes, dtype=np.uint8, offset=8)

        if max_train_samples:
            X_train = X_train[:max_train_samples]
            y_train_raw = y_train_raw[:max_train_samples]

        if max_test_samples:
            X_test = X_test[:max_test_samples]
            y_test_raw = y_test_raw[:max_test_samples]

        # One-hot encode targets
        num_classes = 10
        y_train = np.zeros((len(y_train_raw), num_classes), dtype=np.float32)
        y_train[np.arange(len(y_train_raw)), y_train_raw] = 1.0

        y_test = np.zeros((len(y_test_raw), num_classes), dtype=np.float32)
        y_test[np.arange(len(y_test_raw)), y_test_raw] = 1.0

        logging.info(f"[Image Loader] Successfully hydrated {name.upper()} dataset:")
        logging.info(f"  • Train Set : X={X_train.shape}, y={y_train.shape}")
        logging.info(f"  • Test Set  : X={X_test.shape}, y={y_test.shape}")

        return X_train, y_train, X_test, y_test

    @staticmethod
    def load_csv_image_dataset(csv_path: str, input_shape: list, num_classes: int,
                               val_split: float = 0.15, batch_size: int = 32) -> ImageBatchProvider:
        """
        Ingests a flattened CSV image file, reshapes rows to (N, C, H, W),
        one-hot encodes targets, and wraps them inside an ImageBatchProvider.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[Image Loader Error] CSV dataset file not found at: {csv_path}")

        channels, height, width = input_shape
        expected_features = channels * height * width

        logging.info(f"[Image Loader] Parsing CSV image dataset from: {csv_path}")
        logging.info(f"[Image Loader] Target Tensor Shape: ({channels}, {height}, {width}) | Expected Features: {expected_features}")

        pixels = []
        labels = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                if not row:
                    continue
                pixels.append([float(val) for val in row[:-1]])
                labels.append(int(float(row[-1])))

        X_raw = np.array(pixels, dtype=np.float32)
        y_raw = np.array(labels, dtype=np.int32)
        N = X_raw.shape[0]

        if X_raw.shape[1] != expected_features:
            raise ValueError(
                f"[Image Loader Error] Feature dimension mismatch! CSV has {X_raw.shape[1]} features, "
                f"but input_shape {input_shape} requires {expected_features}."
            )

        # Reshape to 4D spatial tensors (N, C, H, W)
        X = X_raw.reshape(N, channels, height, width)

        # One-hot encode targets
        y = np.zeros((N, num_classes), dtype=np.float32)
        y[np.arange(N), y_raw] = 1.0

        # Partition Train / Validation
        val_count = int(N * val_split)
        indices = np.arange(N)
        np.random.shuffle(indices)

        val_idx, train_idx = indices[:val_count], indices[val_count:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        logging.info(f"[Image Loader] Split complete: Train={X_train.shape[0]} samples, Val={X_val.shape[0]} samples")

        return ImageBatchProvider(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            batch_size=batch_size
        )