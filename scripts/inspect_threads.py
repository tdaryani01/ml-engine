# inspect_threads.py
import numpy as np
import numba
import json

try:
    import threadpoolctl
    has_threadpoolctl = True
except ImportError:
    has_threadpoolctl = False

# 1. Warm up NumPy / BLAS GEMM pool
a = np.random.randn(500, 500).astype(np.float32)
_ = np.dot(a, a)

# 2. Warm up Numba OpenMP pool
@numba.njit(parallel=True)
def _warmup(x):
    for i in numba.prange(len(x)):
        x[i] += 1.0

b = np.zeros(100, dtype=np.float32)
_warmup(b)

print("=" * 80)
print("              ACTIVE NATIVE THREAD POOLS IN PROCESS")
print("=" * 80)
print(f"Logical CPU Cores Detected : {numba.config.NUMBA_DEFAULT_NUM_THREADS}")
print(f"Numba Active Thread Count  : {numba.get_num_threads()}")
print(f"Numba Threading Layer      : {numba.threading_layer()}")
print("-" * 80)

if has_threadpoolctl:
    info = threadpoolctl.threadpool_info()
    print("Underlying BLAS / OpenMP Runtime Controller Map:")
    print(json.dumps(info, indent=2))
else:
    print("NumPy BLAS Backend Configuration:")
    np.show_config()

print("=" * 80)