# numba_test.py
import numba
import numpy as np

@numba.njit(parallel=True)
def dummy_kernel(arr):
    for i in numba.prange(len(arr)):
        arr[i] += 1

# Trigger compilation and initialize the threading runtime
dummy_kernel(np.zeros(10))

print("Active Threading Layer:", numba.threading_layer())