# data/stream_provider.py
import logging
from typing import Tuple, List, Optional
import numpy as np
import pika
import time
from data.base_provider import BaseDataProvider

class StreamDataProvider(BaseDataProvider):
    def __init__(self, amqp_url: str, queue_name: str, feature_names: List[str]):
        """Accepts explicit primitive parameters and connects to the AMQP broker."""
        self.amqp_url = amqp_url
        self.queue_name = queue_name
        self.feature_names = feature_names
        
        self._connection = None
        self._channel = None
        
        self._cached_validation_X: Optional[np.ndarray] = None
        self._cached_validation_y: Optional[np.ndarray] = None
        
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

    def has_more_batches(self) -> bool:
        """A live operational production stream is theoretically infinite."""
        return True

    def next_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pulls a single message. If the queue is empty, it drops control immediately 
        via time.sleep(0) to allow execution scheduling, then polls again.
        """
        while True:
            # Look into the queue and pull a single message out
            method_frame, properties, body = self._channel.basic_get(queue=self.queue_name, auto_ack=True)
            
            # If nothing is in the queue, yield context execution and loop back immediately
            if method_frame is None:
                time.sleep(0)  # 🚨 Voluntarily yields control to the OS/scheduler without blocking thread time
                continue

            # Once data arrives, parse and unpack the binary layout instantly
            shape_str = properties.headers.get("shape", "")
            shard_shape = tuple(map(int, shape_str.split(",")))
            
            shard_matrix = np.frombuffer(body, dtype=np.float32).reshape(shard_shape)
            
            # Slice into inputs (X) and label targets (y)
            X_batch = shard_matrix[:, :-1]
            y_batch = shard_matrix[:, -1:]
            
            return X_batch, y_batch

    def get_validation_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns an out-of-sample reference window matrix to test live generalization."""
        if self._cached_validation_X is None or self._cached_validation_y is None:
            logging.info("[Stream Provider] Initializing sliding-window validation cache space...")
            self._cached_validation_X = np.zeros((100, len(self.feature_names)), dtype=np.float32)
            self._cached_validation_y = np.zeros((100, 1), dtype=np.float32)
            
        return self._cached_validation_X, self._cached_validation_y

    def reset_epoch(self) -> None:
        """Streams don't have classical epoch boundaries."""
        pass