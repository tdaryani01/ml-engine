// OpenMP thread policy for conv_kernels.dll (shared LLVM OMP pool when USE_OPENMP=1 OpenBLAS).
#include "omp_config.h"
#include "blas_dynamic.h"

#include <algorithm>
#include <omp.h>

namespace {

int g_omp_threads = 1;
int g_im2col_parallel_cap = 1;

}  // namespace

extern "C" {

ML_ENGINE_EXPORT int32_t configure_native_threads(int32_t omp_threads) {
    if (omp_threads < 1) {
        omp_threads = 1;
    }
    g_omp_threads = omp_threads;
    g_im2col_parallel_cap = omp_threads;
    omp_set_dynamic(0);
    omp_set_num_threads(g_omp_threads);
    // USE_OPENMP=1 OpenBLAS shares this LLVM OMP runtime (OMP_MAX_ACTIVE_LEVELS=1 phases).
    ml_sync_openblas_threads(g_omp_threads);
    return 0;
}

ML_ENGINE_EXPORT int32_t configure_openblas_threads(int32_t blas_threads) {
    if (blas_threads < 1) {
        blas_threads = 1;
    }
    ml_sync_openblas_threads(blas_threads);
    return 0;
}

ML_ENGINE_EXPORT void ml_omp_before_parallel(void) {
    if (g_omp_threads > 1) {
        omp_set_dynamic(0);
        omp_set_num_threads(g_omp_threads);
        ml_sync_openblas_threads(g_omp_threads);
    }
}

ML_ENGINE_EXPORT int32_t get_omp_threads(void) {
    return g_omp_threads;
}

ML_ENGINE_EXPORT int32_t configure_im2col_parallel_cap(int32_t omp_threads) {
    if (omp_threads < 1) {
        omp_threads = 1;
    }
    g_im2col_parallel_cap = omp_threads;
    return 0;
}

ML_ENGINE_EXPORT int32_t get_im2col_parallel_cap(void) {
    return g_im2col_parallel_cap;
}

}  // extern "C"
