# utils/shard_creator.py
import os
import logging
import pandas as pd
import numpy as np

def shardadize_csv(csv_path: str, feature_names: list, target_name: str, 
                   rows_per_shard: int = 1000, output_dir: str = "data/shards",
                   val_split_ratio: float = 0.15) -> None:
    """
    Splits a raw CSV file into highly optimized, standard sequential .npy binary shards.
    Prefixes a percentage of shards with 'val_' and the rest with 'train_' for clean queue routing.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Commencing shardadization of {csv_path} -> {output_dir}")

    # First, calculate total chunks if possible to establish a clean validation boundary
    # Reading count rows only to keep footprint tiny
    try:
        total_rows = sum(1 for _ in open(csv_path, 'r', encoding='utf-8')) - 1 # subtract header
        total_chunks = int(np.ceil(total_rows / rows_per_shard))
        val_chunk_count = max(int(total_chunks * val_split_ratio), 1)
        logging.info(f"Total rows detected: {total_rows} (~{total_chunks} shards). Allocating {val_chunk_count} shards for validation.")
    except Exception:
        # Fallback if file streaming constraints prevent line counting
        val_chunk_count = 2
        logging.warning(f"Could not pre-calculate row constraints. Defaulting to {val_chunk_count} validation shards.")

    # Read in chunks to keep memory usage locked down
    chunk_iterator = pd.read_csv(csv_path, chunksize=rows_per_shard)
    
    train_idx = 0
    val_idx = 0

    for shard_idx, chunk in enumerate(chunk_iterator):
        if chunk.empty:
            break
            
        # 1. Isolate matrices cleanly
        X_matrix = chunk[feature_names].to_numpy(dtype=np.float32)
        y_matrix = chunk[[target_name]].to_numpy(dtype=np.float32)
        
        # 2. Package into a unified payload array
        shard_payload = np.hstack([X_matrix, y_matrix])
        
        # 3. Separate by prefix based on the pre-calculated allocation counts
        # We hold out the first N shards strictly for validation tagging
        if shard_idx < val_chunk_count:
            shard_file_name = f"val_shard_{val_idx:04d}.npy"
            val_idx += 1
        else:
            shard_file_name = f"train_shard_{train_idx:04d}.npy"
            train_idx += 1
            
        shard_file_path = os.path.join(output_dir, shard_file_name)
        np.save(shard_file_path, shard_payload)
        logging.info(f"Successfully generated tagged binary shard: {shard_file_path} (Shape: {shard_payload.shape})")

    logging.info(f"Shardadization complete! Generated {val_idx} validation shards and {train_idx} training shards.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
    
    features = [
        "Supplier_Lead_Time",
        "Logistics_Delay_Index",
        "Resource_Reserve_Percent",
        "Labor_Capacity_Utilization",
        "Macro_Inflation_Rate"
    ]
    
    shardadize_csv(
        csv_path=r"data\samples\csv\supply_chain.csv", 
        feature_names=features,
        target_name="Outcome",
        output_dir=r"data\samples\shards\supply_chain",
        val_split_ratio=0.15 # 🚨 Holds out a clean 15% ratio split of generated shards
    )