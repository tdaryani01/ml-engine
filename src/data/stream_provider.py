# data/stream_provider.py
import logging
from typing import Tuple, List, Optional
import numpy as np
import pika
import time
from src.data.base_provider import BaseDataProvider
from config.constants import DataKeys

class StreamDataProvider(BaseDataProvider):
    """Data provider for streaming records from an AMQP (RabbitMQ) message broker with real-time normalization and buffering."""
    def __init__(self, amqp_url: str, queue_name: str, val_queue_name: str, 
                 feature_names: List[str], batch_size: int, steps_per_epoch: int = 100, 
                 val_split_size: int = 100, num_classes: int = 3, 
                 drain_on_empty: bool = False): # Added explicit configuration rule toggle
        
        self.amqp_url = amqp_url
        self.queue_name = queue_name
        self.val_queue_name = val_queue_name  
        self.feature_names = feature_names
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.val_split_size = val_split_size
        self.num_classes = num_classes
        self.drain_on_empty = drain_on_empty # Hydrated state tracker
        
        self._connection = None
        self._channel = None
        
        # Incremental Standardization Parameters
        self.mean = np.zeros(len(self.feature_names), dtype=np.float32)
        self.M2 = np.zeros(len(self.feature_names), dtype=np.float32)
        self.std = np.ones(len(self.feature_names), dtype=np.float32)
        self.total_count = 0
        
        self.splits = {
            DataKeys.X_VAL: np.zeros((0, len(self.feature_names)), dtype=np.float32),
            DataKeys.Y_VAL: np.zeros((0, self.num_classes), dtype=np.float32),
            DataKeys.X_TRAIN: np.zeros((0, len(self.feature_names)), dtype=np.float32)
        }
        
        self._X_train_buffer = np.zeros((0, len(self.feature_names)), dtype=np.float32)
        self._y_train_buffer = np.zeros((0, self.num_classes), dtype=np.float32)
        self.y_train_processed = np.zeros((0, self.num_classes), dtype=np.float32)
        
        self._batches_served_this_epoch = 0
        self._epoch_open = True
        
        self._bootstrap_broker_connection()

    def _bootstrap_broker_connection(self) -> None:
        """Establishes connection and declares durable queues on the AMQP broker."""
        try:
            params = pika.URLParameters(self.amqp_url)
            self._connection = pika.BlockingConnection(params)
            self._channel = self._connection.channel()
            
            self._channel.queue_declare(queue=self.queue_name, durable=True)
            self._channel.queue_declare(queue=self.val_queue_name, durable=True)
            
            logging.info(f"[Stream Provider] Successfully bound to train queue: {self.queue_name} and val queue: {self.val_queue_name}")
        except Exception as e:
            logging.error(f"[Stream Provider] Wire connection failure: {e}")
            raise

    def update_running_statistics(self, X_batch: np.ndarray) -> None:
        """Updates Welford's running mean and variance statistics for online feature normalization."""
        batch_count = X_batch.shape[0]
        if batch_count == 0:
            return
            
        self.total_count += batch_count
        delta_old = X_batch - self.mean
        self.mean += np.sum(delta_old, axis=0) / self.total_count
        
        delta_new = X_batch - self.mean
        self.M2 += np.sum(delta_old * delta_new, axis=0)
        
        variance = self.M2 / self.total_count
        self.std = np.sqrt(variance) + 1e-24

    def normalize(self, data_matrix: np.ndarray) -> np.ndarray:
        """Applies online z-score normalization to incoming data matrices."""
        return (data_matrix - self.mean) / self.std

    def _to_one_hot(self, labels: np.ndarray) -> np.ndarray:
        """Converts raw label arrays into categorical one-hot encoded vectors."""
        flat_labels = labels.ravel().astype(np.int32)
        flat_labels = np.clip(flat_labels, 0, self.num_classes - 1)
        one_hot = np.zeros((len(flat_labels), self.num_classes), dtype=np.float32)
        one_hot[np.arange(len(flat_labels)), flat_labels] = 1.0
        return one_hot

    def has_more_batches(self) -> bool:
        """Determines if additional batches remain in the current training epoch."""
        return self._epoch_open

    def next_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fetches messages from the AMQP broker, buffers them, and yields the next training mini-batch."""
        if self._batches_served_this_epoch >= self.steps_per_epoch or not self._epoch_open:
            self._epoch_open = False
            return np.array([]), np.array([])

        dry_polls_counter = 0
        max_dry_polls_before_drain = 20  
        
        while len(self._X_train_buffer) < self.batch_size:
            method_frame, properties, body = self._channel.basic_get(queue=self.queue_name, auto_ack=True)
            
            if method_frame is None:
                # Configurable Decision Node:
                # If drain_on_empty is enabled, allow tracking loop to drop out after a dry timeout window.
                if self.drain_on_empty and len(self._X_train_buffer) > 0:
                    dry_polls_counter += 1
                    if dry_polls_counter >= max_dry_polls_before_drain:
                        logging.warning(f"[Stream Provider] Stream dry condition triggered. Draining partial batch.")
                        break
                
                # If drain_on_empty is FALSE (Production Mode), reset counters and block indefinitely
                dry_polls_counter = 0
                time.sleep(0.05)  
                continue

            dry_polls_counter = 0
            
            shape_str = properties.headers.get("shape", "")
            shard_shape = tuple(map(int, shape_str.split(",")))
            shard_matrix = np.frombuffer(body, dtype=np.float32).reshape(shard_shape)
            
            X_shard = shard_matrix[:, :-1]
            y_shard = self._to_one_hot(shard_matrix[:, -1:])

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

        # Dynamic Slicing Engine Evaluator
        current_available_rows = len(self._X_train_buffer)
        if current_available_rows == 0:
            self._epoch_open = False
            return np.array([]), np.array([])
            
        current_slice_size = min(self.batch_size, current_available_rows)

        X_batch = self._X_train_buffer[:current_slice_size]
        y_batch = self._y_train_buffer[:current_slice_size]

        self._X_train_buffer = self._X_train_buffer[current_slice_size:]
        self._y_train_buffer = self._y_train_buffer[current_slice_size:]

        self.update_running_statistics(X_batch)

        if len(self.splits[DataKeys.X_TRAIN]) < 1000:
            self.splits[DataKeys.X_TRAIN] = np.vstack([self.splits[DataKeys.X_TRAIN], X_batch])
            self.y_train_processed = np.vstack([self.y_train_processed, y_batch])
        else:
            self.splits[DataKeys.X_TRAIN] = np.vstack([self.splits[DataKeys.X_TRAIN][current_slice_size:], X_batch])
            self.y_train_processed = np.vstack([self.y_train_processed[current_slice_size:], y_batch])
        
        self._batches_served_this_epoch += 1
        
        # Shutdown epoch marker if a partial batch had to be squeezed out
        if current_slice_size < self.batch_size:
            self._epoch_open = False
            
        return X_batch, y_batch

    def get_validation_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieves or seeds the validation dataset split from the validation queue."""
        if len(self.splits[DataKeys.X_VAL]) < self.val_split_size:
            logging.info(f"[Stream Provider] Seeding validation split array ({self.val_split_size} rows)...")
            
            while len(self.splits[DataKeys.X_VAL]) < self.val_split_size:
                method_frame, properties, body = self._channel.basic_get(queue=self.val_queue_name, auto_ack=True)
                if method_frame is None:
                    time.sleep(0.01)
                    continue
                
                shape_str = properties.headers.get("shape", "")
                shard_shape = tuple(map(int, shape_str.split(",")))
                shard_matrix = np.frombuffer(body, dtype=np.float32).reshape(shard_shape)
                
                if self.val_split_size < 1.0:
                    target_rows = shard_matrix.shape[0]
                else:
                    target_rows = int(self.val_split_size)
                
                needed_rows = int(target_rows - len(self.splits[DataKeys.X_VAL]))
                
                X_shard = shard_matrix[:needed_rows, :-1]
                y_shard = self._to_one_hot(shard_matrix[:needed_rows, -1:])
                
                self.splits[DataKeys.X_VAL] = np.vstack([self.splits[DataKeys.X_VAL], X_shard])
                self.splits[DataKeys.Y_VAL] = np.vstack([self.splits[DataKeys.Y_VAL], y_shard])
                
                if len(shard_matrix) > needed_rows:
                    self._X_train_buffer = np.vstack([self._X_train_buffer, shard_matrix[needed_rows:, :-1]])
                    self._y_train_buffer = np.vstack([self._y_train_buffer, self._to_one_hot(shard_matrix[needed_rows:, -1:])])

            if len(self.splits[DataKeys.X_TRAIN]) == 0:
                fake_x = np.zeros((self.batch_size, len(self.feature_names)), dtype=np.float32)
                fake_y = np.zeros((self.batch_size, self.num_classes), dtype=np.float32)
                fake_y[:, 0] = 1.0
                self.splits[DataKeys.X_TRAIN] = fake_x
                self.y_train_processed = fake_y

        return self.splits[DataKeys.X_VAL], self.splits[DataKeys.Y_VAL]

    def reset_epoch(self) -> None:
        """Resets epoch counters, buffers, and state markers for a new streaming epoch."""
        self._batches_served_this_epoch = 0
        self._epoch_open = True
        self._X_train_buffer = np.zeros((0, len(self.feature_names)), dtype=np.float32)
        self._y_train_buffer = np.zeros((0, self.num_classes), dtype=np.float32)
        self.splits[DataKeys.X_TRAIN] = np.zeros((0, len(self.feature_names)), dtype=np.float32)
        self.y_train_processed = np.zeros((0, self.num_classes), dtype=np.float32)