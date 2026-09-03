#pragma once
#include <cstdint>

#ifdef _WIN32
#define ML_ENGINE_EXPORT __declspec(dllexport)
#else
#define ML_ENGINE_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Pin LLVM OpenMP for conv_kernels + USE_OPENMP=1 OpenBLAS (one shared pool).
ML_ENGINE_EXPORT int32_t configure_native_threads(int32_t omp_threads);

ML_ENGINE_EXPORT int32_t get_omp_threads(void);

ML_ENGINE_EXPORT int32_t configure_im2col_parallel_cap(int32_t omp_threads);
ML_ENGINE_EXPORT int32_t get_im2col_parallel_cap(void);

ML_ENGINE_EXPORT int32_t configure_openblas_threads(int32_t blas_threads);

// Re-apply omp_set_num_threads before each parallel phase (env/other libs may reset runtime to 1).
ML_ENGINE_EXPORT void ml_omp_before_parallel(void);

// OpenMP schedule policy:
//   conv_fallback.cpp (native hot path) — schedule(dynamic, 8) on all parallel fors.
//   im2col / gemm helpers / dispatcher grids — schedule(static) OK (uniform grids).
// Never leave #pragma omp for without an explicit schedule.

#define ML_OMP_IF_GT1 if(get_omp_threads() > 1)

#ifdef __cplusplus
}
#endif
