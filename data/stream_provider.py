# data/stream_provider.py
import logging
from typing import Tuple, List, Optional
import numpy as np
import pika
import time
from data.base_provider import BaseDataProvider
from config.constants import DataKeys

class StreamDataProvider(BaseDataProvider):
    def __init__(self, amqp_url: str, queue_name: str, feature_names: List[str], batch_size: int, steps_per_epoch: int = 100, val_split_size: int = 100, num_classes: int = 3):
        """Accepts explicit parameters and handles dynamic stream split buffer arrays with tracking history symmetry."""
        self.amqp_url = amqp_url
        self.queue_name = queue_name
        self.feature_names = feature_names
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.val_split_size = val_split_size
        self.num_classes = num_classes
        
        self._connection = None
        self._channel = None
        
        # Dynamic structures to fulfill controller matrix standardization checks
        self.splits = {
            DataKeys.X_VAL: np.zeros((0, len(self.feature_names)), dtype=np.float32),
            DataKeys.Y_VAL: np.zeros((0, self.num_classes), dtype=np.float32),
            DataKeys.X_TRAIN: np.zeros((0, len(self.feature_names)), dtype=np.float32)
        }
        
        # Slicing Buffers and Milestone Trackers
        self._X_train_buffer = np.zeros((0, len(self.feature_names)), dtype=np.float32)
        self._y_train_buffer = np.zeros((0, self.num_classes), dtype=np.float32)
        
        # Match targets array footprint to the rolling X_TRAIN evaluation matrix
        self.y_train_processed = np.zeros((0, self.num_classes), dtype=np.float32)
        
        self._batches_served_this_epoch = 0
        self._epoch_open = True
        
        self._bootstrap_broker_connection()

    def _bootstrap_broker_connection(self) -> None:
        """Establishes persistent network connections and declares target queues."""
        try:
            params = pika.URLParameters(self.amqp_url)
            self._connection = pika.BlockingConnection(params)
            self._channel = self._connection.channel()
            self._channel.queue_declare(queue=self.queue_name, durable=True)
            logging.info(f"[Stream Provider] Successfully connected to queue: {self.queue_name}")
        except Exception as e:
            logging.error(f"[Stream Provider] Wire connection failure: {e}")
            raise

    def _to_one_hot(self, labels: np.ndarray) -> np.ndarray:
        """Converts raw 1-column target indices into multi-column one-hot probabilities."""
        flat_labels = labels.ravel().astype(np.int32)
        flat_labels = np.clip(flat_labels, 0, self.num_classes - 1)
        one_hot = np.zeros((len(flat_labels), self.num_classes), dtype=np.float32)
        one_hot[np.arange(len(flat_labels)), flat_labels] = 1.0
        return one_hot

    def has_more_batches(self) -> bool:
        """Tells the controller loop to stop once the step milestone budget is fulfilled."""
        return self._epoch_open

    def next_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieves a precisely sized mini-batch out of the training buffer."""
        if self._batches_served_this_epoch >= self.steps_per_epoch:
            self._epoch_open = False
            return np.array([]), np.array([])

        while len(self._X_train_buffer) < self.batch_size:
            method_frame, properties, body = self._channel.basic_get(queue=self.queue_name, auto_ack=True)
            
            if method_frame is None:
                time.sleep(0)
                continue

            shape_str = properties.headers.get("shape", "")
            shard_shape = tuple(map(int, shape_str.split(",")))
            shard_matrix = np.frombuffer(body, dtype=np.float32).reshape(shard_shape)
            
            X_shard = shard_matrix[:, :-1]
            y_shard = self._to_one_hot(shard_matrix[:, -1:])

            # Inline Stream Split Assignment
            current_val_count = len(self.splits[DataKeys.X_VAL])
            if current_val_count < self.val_split_size:
                needed_rows = self.val_split_size - current_val_count
                
                self.splits[DataKeys.X_VAL] = np.vstack([self.splits[DataKeys.X_VAL], X_shard[:needed_rows]])
                self.splits[DataKeys.Y_VAL] = np.vstack([self.splits[DataKeys.Y_VAL], y_shard[:needed_rows]])
                
                if len(X_shard) > needed_rows:
                    self._X_train_buffer = np.vstack([self._X_train_buffer, X_shard[needed_rows:]])
                    self._y_train_buffer = np.vstack([self._y_train_buffer, y_shard[needed_rows:]])
            else:
                self._X_train_buffer = np.vstack([self._X_train_buffer, X_shard])
                self._y_train_buffer = np.vstack([self._y_train_buffer, y_shard])

        X_batch = self._X_train_buffer[:self.batch_size]
        y_batch = self._y_train_buffer[:self.batch_size]

        self._X_train_buffer = self._X_train_buffer[self.batch_size:]
        self._y_train_buffer = self._y_train_buffer[self.batch_size:]

        # Maintain perfect row counts across both evaluation splits arrays
        if len(self.splits[DataKeys.X_TRAIN]) < 1000:
            self.splits[DataKeys.X_TRAIN] = np.vstack([self.splits[DataKeys.X_TRAIN], X_batch])
            self.y_train_processed = np.vstack([self.y_train_processed, y_batch])
        else:
            self.splits[DataKeys.X_TRAIN] = np.vstack([self.splits[DataKeys.X_TRAIN][self.batch_size:], X_batch])
            self.y_train_processed = np.vstack([self.y_train_processed[self.batch_size:], y_batch])
        
        self._batches_served_this_epoch += 1
        return X_batch, y_batch

    def get_validation_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the validation matrix built dynamically from the early stream splits buffer."""
        if len(self.splits[DataKeys.X_VAL]) < self.val_split_size:
            logging.info(f"[Stream Provider] Seeding validation split array ({self.val_split_size} rows) from live wire frames...")
            
            while len(self.splits[DataKeys.X_VAL]) < self.val_split_size:
                method_frame, properties, body = self._channel.basic_get(queue=self.queue_name, auto_ack=True)
                if method_frame is None:
                    time.sleep(0.01)
                    continue
                
                shape_str = properties.headers.get("shape", "")
                shard_shape = tuple(map(int, shape_str.split(",")))
                shard_matrix = np.frombuffer(body, dtype=np.float32).reshape(shard_shape)
                
                needed_rows = self.val_split_size - len(self.splits[DataKeys.X_VAL])
                
                X_shard = shard_matrix[:needed_rows, :-1]
                y_shard = self._to_one_hot(shard_matrix[:needed_rows, -1:])
                
                self.splits[DataKeys.X_VAL] = np.vstack([self.splits[DataKeys.X_VAL], X_shard])
                self.splits[DataKeys.Y_VAL] = np.vstack([self.splits[DataKeys.Y_VAL], y_shard])
                
                if len(shard_matrix) > needed_rows:
                    self._X_train_buffer = np.vstack([self._X_train_buffer, shard_matrix[needed_rows:, :-1]])
                    self._y_train_buffer = np.vstack([self._y_train_buffer, self._to_one_hot(shard_matrix[needed_rows:, -1:])])

            # 🚨 Added: Pre-seed an initial batch block into the training metric splits.
            # This allows the controller to establish a safe baseline architecture step on validation pass zero.
            if len(self.splits[DataKeys.X_TRAIN]) == 0:
                fake_x = np.zeros((self.batch_size, len(self.feature_names)), dtype=np.float32)
                fake_y = np.zeros((self.batch_size, self.num_classes), dtype=np.float32)
                fake_y[:, 0] = 1.0
                self.splits[DataKeys.X_TRAIN] = fake_x
                self.y_train_processed = fake_y

        return self.splits[DataKeys.X_VAL], self.splits[DataKeys.Y_VAL]

    def reset_epoch(self) -> None:
        """🚨 Updated: Clears historical metric accumulation arrays to fix evaluation drift."""
        self._batches_served_this_epoch = 0
        self._epoch_open = True
        
        # Clear the evaluation caches so the controller computes performance 
        # metrics using only fresh data from the upcoming epoch interval
        self.splits[DataKeys.X_TRAIN] = np.zeros((0, len(self.feature_names)), dtype=np.float32)
        self.y_train_processed = np.zeros((0, self.num_classes), dtype=np.float32)
        