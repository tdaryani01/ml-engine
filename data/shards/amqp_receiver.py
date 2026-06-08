# data/producers/amqp_receiver.py
import time
import logging
import pika
import numpy as np
from multiprocessing.connection import Listener

class IngestionReceiverEngine:
    def __init__(self, amqp_url: str, queue_name: str, pipe_name: str = "supply_chain_ipc", num_classes: int = 3):
        self.amqp_url = amqp_url
        self.queue_name = queue_name
        
        # Internal Windows pipe syntax handled natively by Listener
        self.address = f"\\\\.\\pipe\\{pipe_name}"
        self.num_classes = num_classes
        
        self.mean = None
        self.M2 = None
        self.total_count = 0
        
        self._connection = None
        self._channel = None
        self._listener = None
        self._conn = None

    def _initialize_ipc_pipe(self) -> None:
        """Initializes a high-level Windows Named Pipe Server."""
        logging.info(f"[Receiver] Initializing high-level Named Pipe at: {self.address}")
        self._listener = Listener(self.address, authkey=b'supply_chain_secret')
        
        logging.info("[Receiver] Waiting for separate ML pipeline process to connect...")
        # Blocks cleanly until run_pipeline.py opens its client connection
        self._conn = self._listener.accept()
        logging.info("[Receiver] Handshake complete! Training script process attached.")

    def _bootstrap_amqp(self) -> None:
        params = pika.URLParameters(self.amqp_url)
        self._connection = pika.BlockingConnection(params)
        self._channel = self._connection.channel()
        self._channel.queue_declare(queue=self.queue_name, durable=True)
        self._channel.basic_qos(prefetch_count=10)
        logging.info(f"[Receiver] Securely bound to AMQP Queue: '{self.queue_name}'")

    def _to_one_hot(self, labels: np.ndarray) -> np.ndarray:
        flat_labels = labels.ravel().astype(np.int32)
        flat_labels = np.clip(flat_labels, 0, self.num_classes - 1)
        one_hot = np.zeros((len(flat_labels), self.num_classes), dtype=np.float32)
        one_hot[np.arange(len(flat_labels)), flat_labels] = 1.0
        return one_hot

    def _update_welford_statistics(self, X_batch: np.ndarray) -> None:
        batch_count = X_batch.shape[0]
        if batch_count == 0: return
        if self.mean is None:
            self.mean = np.zeros(X_batch.shape[1], dtype=np.float32)
            self.M2 = np.zeros(X_batch.shape[1], dtype=np.float32)
            
        self.total_count += batch_count
        delta_old = X_batch - self.mean
        self.mean += np.sum(delta_old, axis=0) / self.total_count
        delta_new = X_batch - self.mean
        self.M2 += np.sum(delta_old * delta_new, axis=0)

    def _get_std(self) -> np.ndarray:
        if self.total_count <= 1:
            return np.ones_like(self.mean)
        return np.sqrt(self.M2 / self.total_count) + 1e-24

    def run_engine_loop(self) -> None:
        self._initialize_ipc_pipe()
        self._bootstrap_amqp()
        
        logging.info("[Receiver] Commencing continuous live streaming...")

        def on_message_callback(ch, method, properties, body):
            shape_str = properties.headers.get("shape", "")
            shard_shape = tuple(map(int, shape_str.split(",")))
            shard_matrix = np.frombuffer(body, dtype=np.float32).reshape(shard_shape)
            
            X_shard = shard_matrix[:, :-1]
            y_shard = self._to_one_hot(shard_matrix[:, -1:])
            
            self._update_welford_statistics(X_shard)
            X_shard_norm = (X_shard - self.mean) / self._get_std()
            
            try:
                # 🚨 NO MANUAL BYTE CONVERSION. Just send the clean NumPy tuple.
                self._conn.send((X_shard_norm, y_shard))
            except Exception as e:
                logging.error(f"[Receiver] Pipe send failed. Engine disconnected: {e}")
                self.shutdown()
                import sys; sys.exit(0)

        self._channel.basic_consume(queue=self.queue_name, on_message_callback=on_message_callback, auto_ack=True)
        
        try:
            self._channel.start_consuming()
        except KeyboardInterrupt:
            logging.info("\nReceiver execution terminated manually.")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        logging.info("[Receiver] Tearing down handles...")
        if self._conn: self._conn.close()
        if self._listener: self._listener.close()
        if self._connection and self._connection.is_open: self._connection.close()
        logging.info("[Receiver] Closed cleanly.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
    receiver = IngestionReceiverEngine(
        amqp_url="amqp://guest:guest@localhost:5672/%2f",
        queue_name="supply_chain.incoming_batches",
        pipe_name="supply_chain_ipc"
    )
    receiver.run_engine_loop()