# data/iterator.py
import numpy as np
import logging

class DatasetIterator:
    """Iterates over datasets in randomized or sequential mini-batch chunks."""
    def __init__(self, X, y, batch_size, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.shape[0]

    def __iter__(self):
        """Yields mini-batches of features and targets, optionally shuffling indices."""
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)
            
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = self.X[batch_indices]
            y_batch = self.y[batch_indices]
            
            # --- ITERATOR TRACE: INTERCEPT FIRST MINI-BATCH BOUNDARIES ---
            # if start_idx == 0:
            #     logging.info("=" * 60)
            #     logging.info("   DATASTREAM ITERATOR PROFILE (BATCH 0)")
            #     logging.info("=" * 60)
            #     logging.info(f"[ITERATOR TRACE] Source X Shape: {self.X.shape} | Source y Shape: {self.y.shape}")
            #     logging.info(f"[ITERATOR TRACE] Slice Range: {start_idx} to {end_idx}")
            #     logging.info(f"[ITERATOR TRACE] Sliced X_batch Shape: {X_batch.shape} | y_batch Shape: {y_batch.shape}")
            #     logging.info(f"[ITERATOR TRACE] Sliced y_batch Memory Layout Data Type: {y_batch.dtype}")
            #     logging.info(f"[ITERATOR TRACE] Sliced y_batch Content Snapshot: {y_batch.ravel()[:5]}")
            #     logging.info("=" * 60)
                
            yield X_batch, y_batch

    def __len__(self):
        """Returns the total number of batches in the iterator."""
        return int(np.ceil(self.num_samples / self.batch_size))