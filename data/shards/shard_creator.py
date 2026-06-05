# utils/shard_creator.py
import os
import logging
import pandas as pd
import numpy as np

def shardadize_csv(csv_path: str, feature_names: list, target_name: str, 
                   rows_per_shard: int = 1000, output_dir: str = "data/shards") -> None:
    """
    Splits a raw CSV file into highly optimized, standard sequential .npy binary shards.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Commencing shardadization of {csv_path} -> {output_dir}")

    # Read in chunks to keep memory usage locked down
    chunk_iterator = pd.read_csv(csv_path, chunksize=rows_per_shard)
    
    for shard_idx, chunk in enumerate(chunk_iterator):
        if chunk.empty:
            break
            
        # 1. Isolate matrices cleanly
        X_matrix = chunk[feature_names].to_numpy(dtype=np.float32)
        y_matrix = chunk[[target_name]].to_numpy(dtype=np.float32)
        
        # 2. Package into a unified payload array
        # Format: Column 0 to N-1 = Features | Last Column = Target
        shard_payload = np.hstack([X_matrix, y_matrix])
        
        # 3. Save as standard self-describing .npy file
        shard_file_name = f"shard_{shard_idx:04d}.npy"
        shard_file_path = os.path.join(output_dir, shard_file_name)
        
        np.save(shard_file_path, shard_payload)
        logging.info(f"Successfully generated binary shard: {shard_file_path} (Shape: {shard_payload.shape})")

    logging.info("Shardadization processing complete!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
    
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
        target_name="Outcome", # 🚨 Ensure this matches your single prediction column name!
        output_dir=r"data\samples\shards\supply_chain" # 🚨 The output path belongs here!
    )