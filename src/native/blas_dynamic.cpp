// Runtime OpenBLAS loader. Prefer bin/libopenblas.dll built with USE_OPENMP=1.
#include "blas_dynamic.h"
#include "im2col_telemetry.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

using SgemmFn = void (*)(
    const char* transa, const char* transb,
    const int* m, const int* n, const int* k,
    const float* alpha,
    const float* a, const int* lda,
    const float* b, const int* ldb,
    const float* beta,
    float* c, const int* ldc);

using OpenblasSetNumThreadsFn = void (*)(int);
using OpenblasSetNumThreadsLocalFn = int (*)(int);
using HostOmpSetNumThreadsFn = void (*)(int);

SgemmFn g_sgemm = nullptr;
OpenblasSetNumThreadsFn g_openblas_set_num_threads = nullptr;
OpenblasSetNumThreadsLocalFn g_openblas_set_num_threads_local = nullptr;
HostOmpSetNumThreadsFn g_blas_host_omp_set_num_threads = nullptr;
int32_t g_blas_threads = 1;
bool g_tried_load = false;
bool g_ready = false;
bool g_unified_omp = false;

#ifdef _WIN32
HMODULE g_blas_module = nullptr;
#else
void* g_blas_module = nullptr;
#endif

void* resolve_symbol(const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(g_blas_module, name));
#else
    return dlsym(g_blas_module, name);
#endif
}

void unload_blas_module() {
    if (!g_blas_module) {
        return;
    }
#ifdef _WIN32
    FreeLibrary(g_blas_module);
#else
    dlclose(g_blas_module);
#endif
    g_blas_module = nullptr;
}

bool bind_sgemm_symbols() {
    g_sgemm = reinterpret_cast<SgemmFn>(resolve_symbol("sgemm_"));
    if (!g_sgemm) {
        g_sgemm = reinterpret_cast<SgemmFn>(resolve_symbol("SGEMM"));
    }
    if (!g_sgemm) {
        return false;
    }
    g_openblas_set_num_threads = reinterpret_cast<OpenblasSetNumThreadsFn>(
        resolve_symbol("openblas_set_num_threads"));
    g_openblas_set_num_threads_local = reinterpret_cast<OpenblasSetNumThreadsLocalFn>(
        resolve_symbol("openblas_set_num_threads_local"));
    g_unified_omp = (g_openblas_set_num_threads != nullptr);
    return true;
}

void bind_blas_host_omp_runtime() {
    if (g_blas_host_omp_set_num_threads) {
        return;
    }
#ifdef _WIN32
    // CMake-built OpenBLAS on MSVC links VCOMP140, not libomp (conv_kernels uses libomp).
    const char* vcomp_names[] = {"vcomp140.dll", "vcomp140d.dll"};
    for (const char* name : vcomp_names) {
        HMODULE mod = GetModuleHandleA(name);
        if (!mod) {
            mod = LoadLibraryA(name);
        }
        if (!mod) {
            continue;
        }
        auto fn = reinterpret_cast<HostOmpSetNumThreadsFn>(
            GetProcAddress(mod, "omp_set_num_threads"));
        if (fn) {
            g_blas_host_omp_set_num_threads = fn;
            return;
        }
    }
#endif
}

void prep_openblas_for_sgemm() {
    if (!g_ready || g_blas_threads < 1) {
        return;
    }
    const int nt = static_cast<int>(g_blas_threads);
    bind_blas_host_omp_runtime();
    if (g_blas_host_omp_set_num_threads) {
        g_blas_host_omp_set_num_threads(nt);
    } else if (g_unified_omp) {
        // libomp-built OpenBLAS shares LLVM OMP with conv_kernels.
        omp_set_dynamic(0);
        omp_set_num_threads(nt);
    }
    if (g_openblas_set_num_threads) {
        g_openblas_set_num_threads(nt);
    }
    if (g_openblas_set_num_threads_local) {
        g_openblas_set_num_threads_local(nt);
    }
}

bool try_load_module(const char* path, bool expect_unified) {
    unload_blas_module();
#ifdef _WIN32
    g_blas_module = LoadLibraryA(path);
#else
    g_blas_module = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
#endif
    if (!g_blas_module) {
        return false;
    }
    if (!bind_sgemm_symbols()) {
        unload_blas_module();
        return false;
    }
    if (expect_unified && !g_unified_omp) {
        std::fprintf(stderr,
            "[blas] warning: %s loaded but openblas_set_num_threads missing "
            "(rebuild with USE_OPENMP=1)\n", path);
    }
    return true;
}

void sync_loaded_openblas_from_env() {
    if (!g_ready || !g_unified_omp) {
        return;
    }
    const char* nt_env = std::getenv("OPENBLAS_NUM_THREADS");
    if (!nt_env || nt_env[0] == '\0') {
        nt_env = std::getenv("OMP_NUM_THREADS");
    }
    int nt = 1;
    if (nt_env && nt_env[0] != '\0') {
        nt = std::atoi(nt_env);
    }
    if (nt < 1) {
        nt = 1;
    }
    ml_sync_openblas_threads(nt);
}

void mark_openblas_ready() {
    g_ready = true;
    bind_blas_host_omp_runtime();
    sync_loaded_openblas_from_env();
}

void call_sgemm(char transa, char transb, int m, int n, int k,
                float alpha, const float* a, int lda,
                const float* b, int ldb, float beta, float* c, int ldc) {
    if (!g_sgemm) {
        return;
    }
    // OpenBLAS USE_OPENMP: num_cpu_avail() → 1 when omp_in_parallel() (libomp path).
    if (g_unified_omp && !g_blas_host_omp_set_num_threads && omp_in_parallel()) {
        std::fprintf(stderr,
            "[blas] warning: sgemm called inside libomp parallel region; GEMM will be serial\n");
    }
    prep_openblas_for_sgemm();
    g_sgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

}  // namespace

extern "C" {

ML_ENGINE_EXPORT int32_t init_openblas_runtime(const char* dll_path) {
    if (g_ready) {
        return 0;
    }
    if (g_tried_load && !g_ready) {
        return -1;
    }
    g_tried_load = true;

    if (dll_path && dll_path[0] != '\0') {
        if (try_load_module(dll_path, true)) {
            mark_openblas_ready();
            return 0;
        }
    }

    const char* env_path = std::getenv("ML_ENGINE_OPENBLAS_DLL");
    if (env_path && env_path[0] != '\0' && try_load_module(env_path, true)) {
        mark_openblas_ready();
        return 0;
    }

#ifdef _WIN32
    char module_path[MAX_PATH];
    HMODULE self = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(&init_openblas_runtime), &self)) {
        if (GetModuleFileNameA(self, module_path, MAX_PATH) > 0) {
            char* slash = std::strrchr(module_path, '\\');
            if (slash) {
                *(slash + 1) = '\0';
                const char* candidates[] = {
                    "libopenblas.dll",
                    "openblas.dll",
                };
                for (const char* name : candidates) {
                    char full[MAX_PATH];
                    std::snprintf(full, sizeof(full), "%s%s", module_path, name);
                    if (try_load_module(full, true)) {
                        mark_openblas_ready();
                        return 0;
                    }
                }
            }
        }
    }
#else
    const char* local_candidates[] = {
        "libopenblas.so",
        "libopenblas.so.0",
        "openblas.so",
    };
    char module_path[4096];
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&init_openblas_runtime), &info) && info.dli_fname) {
        std::snprintf(module_path, sizeof(module_path), "%s", info.dli_fname);
        char* slash = std::strrchr(module_path, '/');
        if (slash) {
            *(slash + 1) = '\0';
            for (const char* name : local_candidates) {
                char full[4096];
                std::snprintf(full, sizeof(full), "%s%s", module_path, name);
                if (try_load_module(full, true)) {
                    mark_openblas_ready();
                    return 0;
                }
            }
        }
    }
#endif

    return -1;
}

ML_ENGINE_EXPORT int32_t init_blas_sgemm_ptr(void* sgemm_fn) {
    if (!sgemm_fn) {
        return -1;
    }
    unload_blas_module();
    g_sgemm = reinterpret_cast<SgemmFn>(sgemm_fn);
    g_openblas_set_num_threads = nullptr;
    g_openblas_set_num_threads_local = nullptr;
    g_unified_omp = false;
    g_ready = true;
    g_tried_load = true;
    return 0;
}

ML_ENGINE_EXPORT int32_t blas_runtime_ready() {
    return g_ready ? 1 : 0;
}

ML_ENGINE_EXPORT int32_t blas_uses_unified_omp() {
    return (g_ready && g_unified_omp) ? 1 : 0;
}

// 1 when bin/libopenblas.dll uses VCOMP140 (MSVC OpenMP), not libomp.
ML_ENGINE_EXPORT int32_t blas_uses_vcomp_omp() {
    bind_blas_host_omp_runtime();
    return g_blas_host_omp_set_num_threads ? 1 : 0;
}

ML_ENGINE_EXPORT int32_t blas_gemm_bench_layer1_loop(int32_t iters) {
    if (!g_ready || iters <= 0) {
        return -1;
    }
    const int64_t N = 32;
    const int64_t C_in = 3;
    const int64_t H = 28;
    const int64_t C_out = 8;
    const int64_t k = 7;
    const int64_t pad = 2;
    const int64_t stride = 1;
    const int64_t out_h = (H + 2 * pad - k) / stride + 1;
    const int64_t out_w = out_h;
    const int64_t m_dim = N * out_h * out_w;
    const int64_t k_dim = C_in * k * k;

    static thread_local std::vector<float> col;
    static thread_local std::vector<float> w;
    static thread_local std::vector<float> out;
    const size_t col_elems = static_cast<size_t>(m_dim * k_dim);
    const size_t w_elems = static_cast<size_t>(C_out * k_dim);
    const size_t out_elems = static_cast<size_t>(m_dim * C_out);
    if (col.size() < col_elems) {
        col.assign(col_elems, 0.1f);
        w.assign(w_elems, 0.01f);
        out.assign(out_elems, 0.0f);
    }

    for (int32_t i = 0; i < iters; ++i) {
        call_sgemm('N', 'N',
                   static_cast<int>(C_out),
                   static_cast<int>(m_dim),
                   static_cast<int>(k_dim),
                   1.0f, w.data(), static_cast<int>(C_out),
                   col.data(), static_cast<int>(k_dim),
                   0.0f, out.data(), static_cast<int>(C_out));
    }
    return 0;
}

void ml_sync_openblas_threads(int32_t omp_threads) {
    if (omp_threads > 0) {
        g_blas_threads = omp_threads;
    }
    prep_openblas_for_sgemm();
}

void blas_gemm_forward(const float* col, const float* W_fwd, float* out_gemm,
                       int64_t m_dim, int64_t n_dim, int64_t k_dim) {
    const int mi = static_cast<int>(n_dim);
    const int ni = static_cast<int>(m_dim);
    const int ki = static_cast<int>(k_dim);
    call_sgemm('N', 'N', mi, ni, ki, 1.0f, W_fwd, mi, col, ki, 0.0f, out_gemm, mi);
    im2col_telemetry::record_gemm_fwd();
}

void blas_gemm_param_grad(const float* dout_trans, const float* col, float* dW_flat,
                          int64_t m_dim, int64_t n_dim, int64_t k_dim, float inv_m) {
    const int mi = static_cast<int>(k_dim);
    const int ni = static_cast<int>(n_dim);
    const int ki = static_cast<int>(m_dim);
    call_sgemm('N', 'T', mi, ni, ki, inv_m, col, mi, dout_trans, ni, 0.0f, dW_flat, mi);
    im2col_telemetry::record_gemm_bwd_w();
}

void blas_gemm_backward_input(const float* dout_trans, const float* W_2d, float* dcol,
                              int64_t m_dim, int64_t n_dim, int64_t k_dim) {
    const int mi = static_cast<int>(k_dim);
    const int ni = static_cast<int>(m_dim);
    const int ki = static_cast<int>(n_dim);
    call_sgemm('N', 'N', mi, ni, ki, 1.0f, W_2d, mi, dout_trans, ki, 0.0f, dcol, mi);
    im2col_telemetry::record_gemm_bwd_x();
}

}  // extern "C"
