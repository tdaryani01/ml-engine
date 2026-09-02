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

// Load OpenBLAS DLL (optional path; NULL = search env / module dir). Returns 0 on success.
ML_ENGINE_EXPORT int32_t init_openblas_runtime(const char* dll_path);

// Inject SGEMM from Python (scipy.linalg.cython_blas capsule). Preferred on Windows wheels.
ML_ENGINE_EXPORT int32_t init_blas_sgemm_ptr(void* sgemm_fn);

// Row-major SGEMM wrappers matching utils/im2col_fast.py layouts.
void blas_gemm_forward(const float* col, const float* W_fwd, float* out_gemm,
                       int64_t m_dim, int64_t n_dim, int64_t k_dim);

void blas_gemm_param_grad(const float* dout_trans, const float* col, float* dW_flat,
                          int64_t m_dim, int64_t n_dim, int64_t k_dim, float inv_m);

void blas_gemm_backward_input(const float* dout_trans, const float* W_2d, float* dcol,
                              int64_t m_dim, int64_t n_dim, int64_t k_dim);

ML_ENGINE_EXPORT int32_t blas_runtime_ready();

// 1 when bin/libopenblas.dll was built with USE_OPENMP=1 (shared LLVM OMP with conv_kernels).
ML_ENGINE_EXPORT int32_t blas_uses_unified_omp(void);

// Called from configure_native_threads / configure_openblas_threads.
void ml_sync_openblas_threads(int32_t omp_threads);

#ifdef __cplusplus
}
#endif
