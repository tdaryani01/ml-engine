#pragma once
#include <cstdint>
#include <chrono>
#include <immintrin.h>

// Horizontal sum helper for AVX2 vectors
static inline float reduce_add_avx2(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 vsum  = _mm_add_ps(vlow, vhigh);
    __m128 vshuf = _mm_movehl_ps(vsum, vsum);
    vsum = _mm_add_ps(vsum, vshuf);
    vshuf = _mm_shuffle_ps(vsum, vsum, 0x55);
    vsum = _mm_add_ss(vsum, vshuf);
    return _mm_cvtss_f32(vsum);
}

struct KernelTelemetry {
    uint64_t fwd_1x1;
    uint64_t bwd_1x1;
    uint64_t fwd_3x3;
    uint64_t bwd_3x3;
    uint64_t bwd_3x3_dx;
    uint64_t bwd_3x3_dw;
    uint64_t fwd_5x5;
    uint64_t bwd_5x5;
    uint64_t bwd_5x5_dx;
    uint64_t fwd_fallback;
    uint64_t bwd_fallback;

    uint64_t time_fwd_1x1_ns;
    uint64_t time_bwd_1x1_ns;
    uint64_t time_fwd_3x3_ns;
    uint64_t time_bwd_3x3_ns;
    uint64_t time_bwd_3x3_dx_ns;
    uint64_t time_bwd_3x3_dw_ns;
    uint64_t time_fwd_5x5_ns;
    uint64_t time_bwd_5x5_ns;
    uint64_t time_bwd_5x5_dx_ns;
    uint64_t time_fwd_fallback_ns;
    uint64_t time_bwd_fallback_ns;
};

#ifdef ENABLE_ENGINE_DIAGNOSTICS
extern KernelTelemetry g_diag;

#define DIAG_INC(field) (g_diag.field++)

struct ScopeTimer {
    uint64_t& target_ns;
    std::chrono::high_resolution_clock::time_point start;
    ScopeTimer(uint64_t& target) 
        : target_ns(target), start(std::chrono::high_resolution_clock::now()) {}
    ~ScopeTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        target_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
};

#define TIME_SCOPE(field) ScopeTimer timer_##field(g_diag.field)
#else
#define DIAG_INC(field) ((void)0)
#define TIME_SCOPE(field) ((void)0)
#endif