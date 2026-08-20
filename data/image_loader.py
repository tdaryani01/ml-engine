# ==========================================
# FILE 2: data/image_loader.py
# ==========================================
import os
import csv
import gzip
import urllib.request
import logging
from typing import Tuple, List, Optional
import numpy as np

from data.base_loader import BaseDataLoader

logger = logging.getLogger(__name__)


class ImageCSVLoader(BaseDataLoader):
    """
    Parses flattened CSV image datasets into 4D spatial tensors (N, Channels, Height, Width)
    and one-hot encoded label matrices.
    """

    def __init__(
        self,
        csv_path: str,
        input_shape: List[int],
        num_classes: int,
        val_split: float = 0.15,
        random_state: int = 42
    ):
        self.csv_path = csv_path
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.val_split = val_split
        self.random_state = random_state

    def load_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"[Image Loader Error] CSV dataset file not found at: {self.csv_path}")

        channels, height, width = self.input_shape
        expected_features = channels * height * width

        logger.info(f"[Image Loader] Ingesting CSV image dataset from: {self.csv_path}")
        logger.info(f"[Image Loader] Expected Shape: ({channels}, {height}, {width}) | Total Features: {expected_features}")

        pixels = []
        labels = []

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if not row:
                    continue
                pixels.append([float(val) for val in row[:-1]])
                labels.append(int(float(row[-1])))

        X_raw = np.array(pixels, dtype=np.float32)
        y_raw = np.array(labels, dtype=np.int32)
        n_samples = X_raw.shape[0]

        if X_raw.shape[1] != expected_features:
            raise ValueError(
                f"[Image Loader Error] Feature dimension mismatch! CSV has {X_raw.shape[1]} features, "
                f"but input_shape requires {expected_features}."
            )

        # Reshape to 4D spatial tensors: (N, C, H, W)
        X = X_raw.reshape(n_samples, channels, height, width)

        # One-hot encode targets
        y = np.zeros((n_samples, self.num_classes), dtype=np.float32)
        y[np.arange(n_samples), y_raw] = 1.0

        # Partition Train / Validation
        # np.random.seed(self.random_state)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        val_count = int(n_samples * self.val_split)
        val_idx, train_idx = indices[:val_count], indices[val_count:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        logger.info(f"[Image Loader] Dataset loaded: Train={X_train.shape[0]}, Val={X_val.shape[0]}")
        return X_train, y_train, X_val, y_val


class BenchmarkImageLoader(BaseDataLoader):
    """
    Downloads and caches standard benchmark datasets (MNIST, Fashion-MNIST),
    returning normalized 4D spatial tensors and one-hot encoded targets.
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

    def __init__(
        self,
        name: str = "mnist",
        cache_dir: str = os.path.join("data", "cache"),
        max_train_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None
    ):
        self.name = name.lower().replace("-", "_")
        self.cache_dir = cache_dir
        self.max_train_samples = max_train_samples
        self.max_test_samples = max_test_samples
        os.makedirs(self.cache_dir, exist_ok=True)

    def _download_and_extract(self, base_url: str, filename: str) -> bytes:
        local_path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(local_path):
            url = f"{base_url}{filename}"
            logger.info(f"[Benchmark Loader] Downloading {filename} from {url}...")
            urllib.request.urlretrieve(url, local_path)

        with gzip.open(local_path, "rb") as f:
            return f.read()

    def load_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.name not in self.DATASET_URLS:
            raise ValueError(f"Unknown benchmark dataset '{self.name}'. Available: {list(self.DATASET_URLS.keys())}")

        cfg = self.DATASET_URLS[self.name]

        train_img_bytes = self._download_and_extract(cfg["base_url"], cfg["train_img"])
        train_lbl_bytes = self._download_and_extract(cfg["base_url"], cfg["train_lbl"])
        test_img_bytes = self._download_and_extract(cfg["base_url"], cfg["test_img"])
        test_lbl_bytes = self._download_and_extract(cfg["base_url"], cfg["test_lbl"])

        # Parse IDX binary buffers (16-byte header offset for images, 8-byte for labels)
        X_train = np.frombuffer(train_img_bytes, dtype=np.uint8, offset=16).reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
        y_train_raw = np.frombuffer(train_lbl_bytes, dtype=np.uint8, offset=8)

        X_test = np.frombuffer(test_img_bytes, dtype=np.uint8, offset=16).reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
        y_test_raw = np.frombuffer(test_lbl_bytes, dtype=np.uint8, offset=8)

        if self.max_train_samples:
            X_train = X_train[:self.max_train_samples]
            y_train_raw = y_train_raw[:self.max_train_samples]

        if self.max_test_samples:
            X_test = X_test[:self.max_test_samples]
            y_test_raw = y_test_raw[:self.max_test_samples]

        num_classes = 10
        y_train = np.zeros((len(y_train_raw), num_classes), dtype=np.float32)
        y_train[np.arange(len(y_train_raw)), y_train_raw] = 1.0

        y_test = np.zeros((len(y_test_raw), num_classes), dtype=np.float32)
        y_test[np.arange(len(y_test_raw)), y_test_raw] = 1.0

        logger.info(f"[Benchmark Loader] Loaded {self.name.upper()}: Train={X_train.shape}, Val={X_test.shape}")
        return X_train, y_train, X_test, y_test