# data/producers/amqp_sender.py
import os
import time
import logging
import pika
import numpy as np
import random

def stream_shards_to_queue(shard_dir: str, queue_name: str, val_queue_name: str, amqp_url: str, epochs: int = 1) -> None:
    """
    Scans directory for prefixed .npy shards and routes them to independent queues:
    - Files starting with 'val_' go to the validation queue (sent once on Epoch 0).
    - Files starting with 'train_' go to the training queue (rows are shuffled inside every epoch).
    """
    if not os.path.exists(shard_dir):
        raise FileNotFoundError(f"Target shard index path missing: {shard_dir}")
        
    all_files = os.listdir(shard_dir)
    
    train_shards = sorted([f for f in all_files if f.startswith("train_") and f.endswith(".npy")])
    val_shards = sorted([f for f in all_files if f.startswith("val_") and f.endswith(".npy")])
    
    if not train_shards and not val_shards:
        logging.warning(f"No prefixed binary primitive .npy files detected in {shard_dir}")
        return

    logging.info(f"Detected {len(train_shards)} training shards and {len(val_shards)} validation shards.")
    logging.info(f"Connecting to AMQP Broker: {amqp_url}")
    
    params = pika.URLParameters(amqp_url)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    
    channel.queue_declare(queue=queue_name, durable=True)
    channel.queue_declare(queue=val_queue_name, durable=True)

    try:
        for current_epoch in range(epochs):
            logging.info(f"\n=== Commencing Streaming Epoch {current_epoch + 1}/{epochs} ===")
            
            # 1. Process Validation Shards ONLY on the very first epoch pass
            if current_epoch == 0:
                if val_shards:
                    logging.info(f"[Validation Egress] Seeding validation channel with {len(val_shards)} prefixed shards...")
                    for filename in val_shards:
                        file_path = os.path.join(shard_dir, filename)
                        matrix_payload = np.load(file_path)
                        byte_stream = matrix_payload.tobytes()
                        shape_string = ",".join(map(str, matrix_payload.shape))
                        
                        properties = pika.BasicProperties(
                            delivery_mode=2,
                            headers={"shape": shape_string, "shard_index": "val", "epoch": current_epoch}
                        )
                        channel.basic_publish(exchange="", routing_key=val_queue_name, body=byte_stream, properties=properties)
                        logging.info(f"[Validation Egress] Pushed {filename} to queue: '{val_queue_name}'")
                else:
                    logging.warning("[Validation Egress] No files starting with 'val_' were found to seed the validation queue.")

            # 2. Shuffle BOTH the file order AND the internal data rows for this epoch
            epoch_train_files = train_shards.copy()
            random.shuffle(epoch_train_files)
            logging.info(f"[Training Egress] Shuffled file queue order for epoch optimization pass.")
            
            for idx, filename in enumerate(epoch_train_files):
                file_path = os.path.join(shard_dir, filename)
                matrix_payload = np.load(file_path)
                
                # 🚨 FIXED: Scramble the individual data rows inside this matrix block on every single epoch pass!
                shuffled_indices = np.random.permutation(matrix_payload.shape[0])
                matrix_payload = matrix_payload[shuffled_indices]
                
                byte_stream = matrix_payload.tobytes()
                shape_string = ",".join(map(str, matrix_payload.shape))
                
                properties = pika.BasicProperties(
                    delivery_mode=2,
                    headers={"shape": shape_string, "shard_index": f"train_{idx}", "epoch": current_epoch}
                )
                
                channel.basic_publish(
                    exchange="",
                    routing_key=queue_name,
                    body=byte_stream,
                    properties=properties
                )
                logging.info(f"[Training Egress] [Epoch {current_epoch + 1}] Shuffled and pushed rows from {filename} to queue")
                
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        logging.info("\nSender execution interrupted manually by user. Closing streams.")
    finally:
        connection.close()
        logging.info("AMQP Connection channel closed cleanly.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
    
    stream_shards_to_queue(
        shard_dir=r"data\samples\shards\supply_chain",
        queue_name="supply_chain.incoming_batches",
        val_queue_name="supply_chain.validation_batches",
        amqp_url="amqp://guest:guest@localhost:5672/%2f",
        epochs=600  
    )