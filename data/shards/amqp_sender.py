# data/producers/amqp_sender.py
import os
import time
import logging
import pika
import numpy as np

def stream_shards_to_queue(shard_dir: str, queue_name: str, amqp_url: str, epochs: int = 1) -> None:
    """
    Scans directory for .npy shards, serializes matrices to raw bytes,
    and publishes them directly into the AMQP broker queue, repeating for specified epochs.
    """
    # 1. Gather and sort target binary shard frames sequentially
    if not os.path.exists(shard_dir):
        raise FileNotFoundError(f"Target shard index path missing: {shard_dir}")
        
    shard_files = sorted([f for f in os.listdir(shard_dir) if f.endswith(".npy")])
    if not shard_files:
        logging.warning(f"No binary primitive .npy files detected in {shard_dir}")
        return

    logging.info(f"Connecting to AMQP Broker: {amqp_url}")
    
    # 2. Establish connection topology boundaries
    params = pika.URLParameters(amqp_url)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    
    # Ensure queue exists and is durable
    channel.queue_declare(queue=queue_name, durable=True)
    
    logging.info(f"Target egress channel locked. Sending {len(shard_files)} shards per epoch across {epochs} epochs to queue: '{queue_name}'")

    try:
        # 🚨 Added: Outer Epoch Loop Sequence
        for current_epoch in range(epochs):
            logging.info(f"=== Commencing Streaming Epoch {current_epoch + 1}/{epochs} ===")
            
            for idx, filename in enumerate(shard_files):
                file_path = os.path.join(shard_dir, filename)
                
                # Load optimized numpy payload back into active memory
                matrix_payload = np.load(file_path)
                
                # Serialize the array straight into its underlying raw byte footprint
                byte_stream = matrix_payload.tobytes()
                
                # Send shape details via AMQP property headers so receiver knows array dimensions
                shape_string = ",".join(map(str, matrix_payload.shape))
                properties = pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent on disk inside the broker
                    headers={"shape": shape_string, "shard_index": idx, "epoch": current_epoch}
                )
                
                # Publish to default exchange targeting your specific queue
                channel.basic_publish(
                    exchange="",
                    routing_key=queue_name,
                    body=byte_stream,
                    properties=properties
                )
                logging.info(f"[Egress Stream] [Epoch {current_epoch + 1}] Successfully pushed {filename} | Shape: ({shape_string})")
                
                # Light throttle pace to mimic dynamic production pipeline flow
                time.sleep(0)
                
    except KeyboardInterrupt:
        logging.info("\nSender execution interrupted manually by user. Closing streams.")
    finally:
        connection.close()
        logging.info("AMQP Connection channel closed cleanly.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
    
    # Pointing directly to your active configuration workspace parameters
    stream_shards_to_queue(
        shard_dir=r"data\samples\shards\supply_chain",
        queue_name="supply_chain.incoming_batches",
        amqp_url="amqp://guest:guest@localhost:5672/%2f",
        epochs=200  # 🚨 Added: Loops the shard sequence 10 times consecutively
    )