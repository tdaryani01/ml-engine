# data/stream_provider.py
import logging
import time
import numpy as np
from typing import Tuple, List
from multiprocessing.connection import Client
from data.base_provider import BaseDataProvider
from config.constants import DataKeys

class IPCStreamDataProvider(BaseDataProvider):
    def __init__(self, ipc_pipe_path: str, feature_names: List[str], batch_size: int, 
                 steps_per_epoch: int = 100, num_classes: int = 3):
        
        # Standardize configuration name into a valid local Windows named pipe handle URI
        pipe_clean = ipc_pipe_path.replace(".pipe", "").split("\\")[-1].split("/")[-1]
        self.address = f"\\\\.\\pipe\\{pipe_clean}"

        self.feature_names = feature_names
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.num_classes = num_classes
        self.feature_count = len(feature_names)
        
        self.splits = {
            DataKeys.X_VAL: np.zeros((0, self.feature_count), dtype=np.float32),
            DataKeys.Y_VAL: np.zeros((0, self.num_classes), dtype=np.float32),
            DataKeys.X_TRAIN: np.zeros((0, self.feature_count), dtype=np.float32)
        }
        self.y_train_processed = np.zeros((0, self.num_classes), dtype=np.float32)
        
        self._batches_served_this_epoch = 0
        self._epoch_open = True
        self._conn = None
        
        self._bootstrap_ipc_connection()

    def _bootstrap_ipc_connection(self) -> None:
        """Connects to the standalone receiver process using high-level multiprocessing client handles."""
        logging.info(f"[IPC Provider] Connecting to streaming pipe: {self.address}...")
        
        while True:
            try:
                self._conn = Client(self.address, authkey=b'supply_chain_secret')
                break
            except Exception:
                logging.info("[IPC Provider] Named pipe not available yet. Waiting for receiver...")
                time.sleep(2)
                
        logging.info("[IPC Provider] Connected to ingestion stream pipe successfully.")
        self._seed_validation_cache()

    def _read_next_shard_from_pipe(self, timeout: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        """Natively deserializes the full structured tuple object with an anti-freeze timeout."""
        try:
            # Check if data is actually available within the given timeout window
            if timeout > 0 and not self._conn.poll(timeout):
                raise TimeoutError(f"[IPC Provider] Stream timed out! No data received within {timeout} seconds.")
            elif timeout == 0 and not self._conn.poll(0):
                raise LookupError("No data waiting in the pipe buffer right now.")
                
            X_shard, y_shard = self._conn.recv()
            return X_shard, y_shard
        except LookupError:
            # Reraise immediately for non-blocking checks to catch
            raise
        except Exception as e:
            logging.error(f"[IPC Provider] Connection error or timeout: {e}")
            raise EOFError("[IPC Provider] Pipe disconnected or stream terminated.")

    def _seed_validation_cache(self) -> None:
        """Pulls initial frames off the pipe safely without letting an empty stream freeze execution."""
        logging.info("[IPC Provider] Seeding validation cache from stream...")
        
        shards_collected = 0
        # Try to gather 2 shards if they are already sitting in the pipe queue
        for _ in range(2):
            try:
                # Use timeout=0 for a strict non-blocking check
                X_v, y_v = self._read_next_shard_from_pipe(timeout=0)
                self.splits[DataKeys.X_VAL] = np.vstack([self.splits[DataKeys.X_VAL], X_v])
                self.splits[DataKeys.Y_VAL] = np.vstack([self.splits[DataKeys.Y_VAL], y_v])
                shards_collected += 1
            except LookupError:
                break
        
        # Fallback stub: If RabbitMQ didn't have data cached yet, keep the engine from stalling out
        if shards_collected == 0:
            logging.warning("[IPC Provider] Pipe buffer empty on boot. Seeding fallback validation cache.")
            self.splits[DataKeys.X_VAL] = np.zeros((self.batch_size, self.feature_count), dtype=np.float32)
            self.splits[DataKeys.Y_VAL] = np.zeros((self.batch_size, self.num_classes), dtype=np.float32)
            self.splits[DataKeys.Y_VAL][:, 0] = 1.0
        
        self.splits[DataKeys.X_TRAIN] = np.zeros((self.batch_size, self.feature_count), dtype=np.float32)
        self.y_train_processed = np.zeros((self.batch_size, self.num_classes), dtype=np.float32)
        self.y_train_processed[:, 0] = 1.0
        logging.info(f"[IPC Provider] Validation cache initialized with {len(self.splits[DataKeys.X_VAL])} rows.")

    def normalize(self, data_matrix: np.ndarray) -> np.ndarray:
        return data_matrix

    def has_more_batches(self) -> bool:
        return self._epoch_open

    def next_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._batches_served_this_epoch >= self.steps_per_epoch:
            self._epoch_open = False
            return np.array([]), np.array([])

        # Training loop steps use the standard blocking 5.0 second timeout check
        X_batch, y_batch = self._read_next_shard_from_pipe(timeout=5.0)

        if len(X_batch) > self.batch_size:
            X_batch = X_batch[:self.batch_size]
            y_batch = y_batch[:self.batch_size]

        self.splits[DataKeys.X_TRAIN] = X_batch
        self.y_train_processed = y_batch
        self._batches_served_this_epoch += 1
        return X_batch, y_batch

    def get_validation_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.splits[DataKeys.X_VAL], self.splits[DataKeys.Y_VAL]

    def reset_epoch(self) -> None:
        self._batches_served_this_epoch = 0
        self._epoch_open = True

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            logging.info("[IPC Provider] High-level pipe link closed cleanly.")