# ==========================================
# FILE: data/image_loader.py
# ==========================================
import os
import csv
import gzip
import urllib.request
import logging
from typing import Tuple, List, Optional
import numpy as np

from src.data.base_loader import BaseDataLoader

logger = logging.getLogger(__name__)


def _pad_simd_width(X: np.ndarray, target_align: int = 8) -> np.ndarray:
    """
    Pads the innermost spatial dimension (W) of a 4D tensor (N, C, H, W)
    to the next multiple of `target_align` (default: 8 for AVX2).
    Leaves the active data untouched and zero-fills the trailing margins.
    """
    N, C, H, W = X.shape
    W_aligned = (W + target_align - 1) & ~(target_align - 1)
    if W == W_aligned:
        return np.ascontiguousarray(X, dtype=np.float32)

    X_padded = np.zeros((N, C, H, W_aligned), dtype=np.float32)
    X_padded[:, :, :, :W] = X
    return X_padded


class ImageCSVLoader(BaseDataLoader):
    """
    Parses flattened CSV *or* NPZ image datasets into 4D spatial tensors
    (N, Channels, Height, Width_aligned) and one-hot encoded label matrices.

    NPZ schema (preferred for >=64²): keys `X` (N,C,H,W float32) and `y` (N,) int.
    CSV schema: flattened pixels + trailing `target` column.
    """

    def __init__(
        self,
        csv_path: str,
        input_shape: List[int],
        num_classes: int,
        val_split: float = 0.15,
        train_split: Optional[float] = None,
        random_state: int = 42
    ):
        self.csv_path = csv_path
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.val_split = val_split
        # If train_split is set, use train+val counts; leftover (e.g. test) is dropped.
        # If None, all non-val samples go to train (legacy behavior).
        self.train_split = train_split
        self.random_state = random_state

    def _load_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"[Image Loader Error] Dataset file not found at: {self.csv_path}")

        channels, height, width = self.input_shape
        expected_features = channels * height * width
        path_lower = self.csv_path.lower()

        if path_lower.endswith(".npz"):
            logger.info(f"[Image Loader] Ingesting NPZ image dataset from: {self.csv_path}")
            with np.load(self.csv_path) as blob:
                if "X" not in blob or "y" not in blob:
                    raise ValueError("[Image Loader Error] NPZ must contain arrays 'X' and 'y'.")
                X = np.asarray(blob["X"], dtype=np.float32)
                y_raw = np.asarray(blob["y"], dtype=np.int32).reshape(-1)
            if X.ndim != 4:
                raise ValueError(f"[Image Loader Error] NPZ X must be NCHW, got shape {X.shape}")
            if tuple(X.shape[1:]) != (channels, height, width):
                raise ValueError(
                    f"[Image Loader Error] NPZ spatial shape {X.shape[1:]} != input_shape "
                    f"{(channels, height, width)}"
                )
            if X.shape[0] != y_raw.shape[0]:
                raise ValueError("[Image Loader Error] NPZ X/y length mismatch.")
            return X, y_raw

        logger.info(f"[Image Loader] Ingesting CSV image dataset from: {self.csv_path}")
        logger.info(
            f"[Image Loader] Expected Shape: ({channels}, {height}, {width}) | "
            f"Total Features: {expected_features}"
        )

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
        if X_raw.shape[1] != expected_features:
            raise ValueError(
                f"[Image Loader Error] Feature dimension mismatch! CSV has {X_raw.shape[1]} features, "
                f"but input_shape requires {expected_features}."
            )
        X = X_raw.reshape(X_raw.shape[0], channels, height, width)
        return X, y_raw

    def load_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        channels, height, width = self.input_shape
        X, y_raw = self._load_arrays()
        n_samples = X.shape[0]

        # Pad width to SIMD boundary
        X = _pad_simd_width(X, target_align=8)

        # One-hot encode targets
        y = np.zeros((n_samples, self.num_classes), dtype=np.float32)
        if y_raw.min() < 0 or y_raw.max() >= self.num_classes:
            raise ValueError(
                f"[Image Loader Error] Label out of range for num_classes={self.num_classes}: "
                f"[{y_raw.min()}, {y_raw.max()}]"
            )
        y[np.arange(n_samples), y_raw] = 1.0

        # Partition Train / Validation
        rng = np.random.default_rng(self.random_state)
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        val_count = int(n_samples * self.val_split)
        if self.train_split is not None:
            train_count = int(n_samples * self.train_split)
            train_idx = indices[:train_count]
            val_idx = indices[train_count : train_count + val_count]
        else:
            val_idx, train_idx = indices[:val_count], indices[val_count:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        logger.info(
            f"[Image Loader] Dataset loaded ({channels}x{height}x{width}): "
            f"Train={X_train.shape}, Val={X_val.shape}"
        )
        return X_train, y_train, X_val, y_val


class BenchmarkImageLoader(BaseDataLoader):
    """
    Downloads and caches standard benchmark datasets (MNIST, Fashion-MNIST),
    returning SIMD-aligned 4D spatial tensors and one-hot encoded targets.
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
        X_train_raw = np.frombuffer(train_img_bytes, dtype=np.uint8, offset=16).reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
        y_train_raw = np.frombuffer(train_lbl_bytes, dtype=np.uint8, offset=8)

        X_test_raw = np.frombuffer(test_img_bytes, dtype=np.uint8, offset=16).reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
        y_test_raw = np.frombuffer(test_lbl_bytes, dtype=np.uint8, offset=8)

        if self.max_train_samples:
            X_train_raw = X_train_raw[:self.max_train_samples]
            y_train_raw = y_train_raw[:self.max_train_samples]

        if self.max_test_samples:
            X_test_raw = X_test_raw[:self.max_test_samples]
            y_test_raw = y_test_raw[:self.max_test_samples]

        # Apply SIMD width alignment (28 -> 32)
        X_train = _pad_simd_width(X_train_raw, target_align=8)
        X_test  = _pad_simd_width(X_test_raw, target_align=8)

        num_classes = 10
        y_train = np.zeros((len(y_train_raw), num_classes), dtype=np.float32)
        y_train[np.arange(len(y_train_raw)), y_train_raw] = 1.0

        y_test = np.zeros((len(y_test_raw), num_classes), dtype=np.float32)
        y_test[np.arange(len(y_test_raw)), y_test_raw] = 1.0

        logger.info(f"[Benchmark Loader] Loaded {self.name.upper()}: Train={X_train.shape}, Val={X_test.shape}")
        return X_train, y_train, X_test, y_test