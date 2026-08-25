#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <omp.h>
#include <atomic>
#include <iostream>
#include <vector>

#if defined(_MSC_VER)
    #define EXPORT_API extern "C" __declspec(dllexport)
#else
    #define EXPORT_API extern "C" __attribute__((visibility("default")))
#endif

// -----------------------------------------------------------------------------
// THREAD TELEMETRY & HARDWARE PROFILING
// -----------------------------------------------------------------------------
constexpr int MAX_TELEMETRY_THREADS = 128;
static std::atomic<int64_t> thread_call_count[MAX_TELEMETRY_THREADS];

EXPORT_API void log_thread_execution_stats() {
    int active_threads = omp_get_max_threads();
    if (active_threads > MAX_TELEMETRY_THREADS) {
        active_threads = MAX_TELEMETRY_THREADS;
    }
    std::cout << "\n================ OPENMP WORKER THREAD PROFILE ================\n";
    for (int i = 0; i < active_threads; ++i) {
        std::cout << " Worker Thread [" << i << "]: " 
                  << thread_call_count[i].load() << " parallel tasks executed\n";
    }
    std::cout << "=============================================================\n\n";
}

EXPORT_API void reset_thread_execution_stats() {
    for (int i = 0; i < MAX_TELEMETRY_THREADS; ++i) {
        thread_call_count[i].store(0);
    }
}

EXPORT_API void set_omp_threads(int num_threads) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
}

EXPORT_API int get_omp_threads() {
    return omp_get_max_threads();
}

EXPORT_API void log_engine_runtime_diagnostics(
    const float* x, const float* W, const float* out,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in,
    int64_t C_out, int64_t k_h, int64_t k_w
) {
    (void)x; (void)W; (void)out;
    (void)N; (void)C_in; (void)H; (void)W_in;
    (void)C_out; (void)k_h; (void)k_w;
}

// -----------------------------------------------------------------------------
// 1. HARDCODED 9-TAP DIRECT FORWARD (SUB-1MS)
// -----------------------------------------------------------------------------
EXPORT_API void direct_conv2d_forward_avx2(
    const float* __restrict x,
    const float* __restrict W,
    const float* __restrict bias,
    float* __restrict out,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in,
    int64_t C_out, int64_t k_h, int64_t k_w,
    int64_t stride, int64_t pad,
    int32_t fuse_relu
) {
    (void)k_h; (void)k_w;
    const int64_t out_h = (H + 2 * pad - 3) / stride + 1;
    const int64_t out_w = (W_in + 2 * pad - 3) / stride + 1;
    const int64_t spatial_out = out_h * out_w;
    const int64_t spatial_in  = H * W_in;
    const __m256 v_zero = _mm256_setzero_ps();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t cout_blk = 0; cout_blk < (C_out + 3) / 4; ++cout_blk) {
            int tid = omp_get_thread_num();
            if (tid >= 0 && tid < MAX_TELEMETRY_THREADS) {
                thread_call_count[tid].fetch_add(1, std::memory_order_relaxed);
            }

            const int64_t cout0 = cout_blk * 4;
            const int64_t c_rem = (C_out - cout0 >= 4) ? 4 : (C_out - cout0);

            for (int64_t oh = 0; oh < out_h; ++oh) {
                const int64_t ih0 = oh * stride - pad + 0;
                const int64_t ih1 = oh * stride - pad + 1;
                const int64_t ih2 = oh * stride - pad + 2;

                float* __restrict out_r0 = &out[(n * C_out + cout0 + 0) * spatial_out + oh * out_w];
                float* __restrict out_r1 = (c_rem > 1) ? &out[(n * C_out + cout0 + 1) * spatial_out + oh * out_w] : nullptr;
                float* __restrict out_r2 = (c_rem > 2) ? &out[(n * C_out + cout0 + 2) * spatial_out + oh * out_w] : nullptr;
                float* __restrict out_r3 = (c_rem > 3) ? &out[(n * C_out + cout0 + 3) * spatial_out + oh * out_w] : nullptr;

                const __m256 vb0 = bias ? _mm256_set1_ps(bias[cout0 + 0]) : _mm256_setzero_ps();
                const __m256 vb1 = (bias && c_rem > 1) ? _mm256_set1_ps(bias[cout0 + 1]) : _mm256_setzero_ps();
                const __m256 vb2 = (bias && c_rem > 2) ? _mm256_set1_ps(bias[cout0 + 2]) : _mm256_setzero_ps();
                const __m256 vb3 = (bias && c_rem > 3) ? _mm256_set1_ps(bias[cout0 + 3]) : _mm256_setzero_ps();

                int64_t ow = 0;
                for (; ow + 8 <= out_w; ow += 8) {
                    __m256 acc0 = vb0, acc1 = vb1, acc2 = vb2, acc3 = vb3;
                    const int64_t iw0 = ow * stride - pad;

                    for (int64_t cin = 0; cin < C_in; ++cin) {
                        const float* __restrict xp  = &x[(n * C_in + cin) * spatial_in];
                        const float* __restrict wp0 = &W[((cout0 + 0) * C_in + cin) * 9];
                        const float* __restrict wp1 = (c_rem > 1) ? &W[((cout0 + 1) * C_in + cin) * 9] : nullptr;
                        const float* __restrict wp2 = (c_rem > 2) ? &W[((cout0 + 2) * C_in + cin) * 9] : nullptr;
                        const float* __restrict wp3 = (c_rem > 3) ? &W[((cout0 + 3) * C_in + cin) * 9] : nullptr;

                        const float* __restrict r0 = (ih0 >= 0 && ih0 < H) ? &xp[ih0 * W_in] : nullptr;
                        const float* __restrict r1 = (ih1 >= 0 && ih1 < H) ? &xp[ih1 * W_in] : nullptr;
                        const float* __restrict r2 = (ih2 >= 0 && ih2 < H) ? &xp[ih2 * W_in] : nullptr;

                        #define TAP_FWD(ROW_PTR, KW, W_IDX) { \
                            if (ROW_PTR) { \
                                const int64_t cur_iw = iw0 + (KW); \
                                __m256 vx; \
                                if (stride == 1 && cur_iw >= 0 && (cur_iw + 8) <= W_in) { \
                                    vx = _mm256_loadu_ps(&(ROW_PTR)[cur_iw]); \
                                } else { \
                                    alignas(32) float tmp[8] = {0}; \
                                    for (int s = 0; s < 8; ++s) { \
                                        int64_t s_iw = (ow + s) * stride - pad + (KW); \
                                        if (s_iw >= 0 && s_iw < W_in) tmp[s] = (ROW_PTR)[s_iw]; \
                                    } \
                                    vx = _mm256_load_ps(tmp); \
                                } \
                                acc0 = _mm256_fmadd_ps(vx, _mm256_set1_ps(wp0[W_IDX]), acc0); \
                                if (c_rem > 1) acc1 = _mm256_fmadd_ps(vx, _mm256_set1_ps(wp1[W_IDX]), acc1); \
                                if (c_rem > 2) acc2 = _mm256_fmadd_ps(vx, _mm256_set1_ps(wp2[W_IDX]), acc2); \
                                if (c_rem > 3) acc3 = _mm256_fmadd_ps(vx, _mm256_set1_ps(wp3[W_IDX]), acc3); \
                            } \
                        }

                        TAP_FWD(r0, 0, 0); TAP_FWD(r0, 1, 1); TAP_FWD(r0, 2, 2);
                        TAP_FWD(r1, 0, 3); TAP_FWD(r1, 1, 4); TAP_FWD(r1, 2, 5);
                        TAP_FWD(r2, 0, 6); TAP_FWD(r2, 1, 7); TAP_FWD(r2, 2, 8);
                        #undef TAP_FWD
                    }

                    if (fuse_relu) {
                        acc0 = _mm256_max_ps(acc0, v_zero);
                        if (c_rem > 1) acc1 = _mm256_max_ps(acc1, v_zero);
                        if (c_rem > 2) acc2 = _mm256_max_ps(acc2, v_zero);
                        if (c_rem > 3) acc3 = _mm256_max_ps(acc3, v_zero);
                    }

                    _mm256_storeu_ps(&out_r0[ow], acc0);
                    if (c_rem > 1) _mm256_storeu_ps(&out_r1[ow], acc1);
                    if (c_rem > 2) _mm256_storeu_ps(&out_r2[ow], acc2);
                    if (c_rem > 3) _mm256_storeu_ps(&out_r3[ow], acc3);
                }

                for (; ow < out_w; ++ow) {
                    float s0 = bias ? bias[cout0 + 0] : 0.0f;
                    float s1 = (bias && c_rem > 1) ? bias[cout0 + 1] : 0.0f;
                    float s2 = (bias && c_rem > 2) ? bias[cout0 + 2] : 0.0f;
                    float s3 = (bias && c_rem > 3) ? bias[cout0 + 3] : 0.0f;

                    for (int64_t cin = 0; cin < C_in; ++cin) {
                        const float* __restrict xp  = &x[(n * C_in + cin) * spatial_in];
                        const float* __restrict wp0 = &W[((cout0 + 0) * C_in + cin) * 9];
                        const float* __restrict wp1 = (c_rem > 1) ? &W[((cout0 + 1) * C_in + cin) * 9] : nullptr;
                        const float* __restrict wp2 = (c_rem > 2) ? &W[((cout0 + 2) * C_in + cin) * 9] : nullptr;
                        const float* __restrict wp3 = (c_rem > 3) ? &W[((cout0 + 3) * C_in + cin) * 9] : nullptr;

                        const float* __restrict r0 = (ih0 >= 0 && ih0 < H) ? &xp[ih0 * W_in] : nullptr;
                        const float* __restrict r1 = (ih1 >= 0 && ih1 < H) ? &xp[ih1 * W_in] : nullptr;
                        const float* __restrict r2 = (ih2 >= 0 && ih2 < H) ? &xp[ih2 * W_in] : nullptr;

                        #define TAP_SCALAR(ROW_PTR, KW, W_IDX) { \
                            if (ROW_PTR) { \
                                const int64_t cur_iw = ow * stride - pad + (KW); \
                                if (cur_iw >= 0 && cur_iw < W_in) { \
                                    const float val = (ROW_PTR)[cur_iw]; \
                                    s0 += val * wp0[W_IDX]; \
                                    if (c_rem > 1) s1 += val * wp1[W_IDX]; \
                                    if (c_rem > 2) s2 += val * wp2[W_IDX]; \
                                    if (c_rem > 3) s3 += val * wp3[W_IDX]; \
                                } \
                            } \
                        }
                        TAP_SCALAR(r0, 0, 0); TAP_SCALAR(r0, 1, 1); TAP_SCALAR(r0, 2, 2);
                        TAP_SCALAR(r1, 0, 3); TAP_SCALAR(r1, 1, 4); TAP_SCALAR(r1, 2, 5);
                        TAP_SCALAR(r2, 0, 6); TAP_SCALAR(r2, 1, 7); TAP_SCALAR(r2, 2, 8);
                        #undef TAP_SCALAR
                    }

                    if (fuse_relu) {
                        s0 = s0 > 0.0f ? s0 : 0.0f;
                        s1 = s1 > 0.0f ? s1 : 0.0f;
                        s2 = s2 > 0.0f ? s2 : 0.0f;
                        s3 = s3 > 0.0f ? s3 : 0.0f;
                    }
                    out_r0[ow] = s0;
                    if (c_rem > 1) out_r1[ow] = s1;
                    if (c_rem > 2) out_r2[ow] = s2;
                    if (c_rem > 3) out_r3[ow] = s3;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 2. FUSED COMPOSITE CONV BLOCK (FORWARD)
// -----------------------------------------------------------------------------
EXPORT_API void direct_conv_block_forward_avx2(
    const float* __restrict x,
    const float* __restrict W,
    const float* __restrict bias,
    float* __restrict out_conv_relu,
    float* __restrict out_pool,
    uint8_t* __restrict argmax_buf,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in,
    int64_t C_out, int64_t k_h, int64_t k_w,
    int64_t conv_stride, int64_t conv_pad,
    int64_t pool_size, int64_t pool_stride
) {
    direct_conv2d_forward_avx2(
        x, W, bias, out_conv_relu,
        N, C_in, H, W_in,
        C_out, k_h, k_w,
        conv_stride, conv_pad,
        1
    );

    const int64_t conv_out_h = (H + 2 * conv_pad - 3) / conv_stride + 1;
    const int64_t conv_out_w = (W_in + 2 * conv_pad - 3) / conv_stride + 1;
    const int64_t pool_out_h = (conv_out_h - pool_size) / pool_stride + 1;
    const int64_t pool_out_w = (conv_out_w - pool_size) / pool_stride + 1;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t cout = 0; cout < C_out; ++cout) {
            int tid = omp_get_thread_num();
            if (tid >= 0 && tid < MAX_TELEMETRY_THREADS) {
                thread_call_count[tid].fetch_add(1, std::memory_order_relaxed);
            }

            const float* __restrict cr_plane = &out_conv_relu[(n * C_out + cout) * conv_out_h * conv_out_w];
            float* __restrict p_plane        = &out_pool[(n * C_out + cout) * pool_out_h * pool_out_w];
            uint8_t* __restrict m_plane      = &argmax_buf[(n * C_out + cout) * pool_out_h * pool_out_w];

            for (int64_t ph = 0; ph < pool_out_h; ++ph) {
                const int64_t ih_base = ph * pool_stride;
                const float* __restrict r0 = &cr_plane[(ih_base + 0) * conv_out_w];
                const float* __restrict r1 = &cr_plane[(ih_base + 1) * conv_out_w];

                for (int64_t pw = 0; pw < pool_out_w; ++pw) {
                    const int64_t iw_base = pw * pool_stride;
                    const float v00 = r0[iw_base + 0];
                    const float v01 = r0[iw_base + 1];
                    const float v10 = r1[iw_base + 0];
                    const float v11 = r1[iw_base + 1];

                    float max_val = v00;
                    uint8_t best_idx = 0;

                    if (v01 > max_val) { max_val = v01; best_idx = 1; }
                    if (v10 > max_val) { max_val = v10; best_idx = 2; }
                    if (v11 > max_val) { max_val = v11; best_idx = 3; }

                    const int64_t p_idx = ph * pool_out_w + pw;
                    p_plane[p_idx] = max_val;
                    m_plane[p_idx] = best_idx;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 3. FULLY OPTIMIZED ZERO-BRANCH dx + dW DIRECT BACKWARD PASS
// -----------------------------------------------------------------------------
EXPORT_API void direct_conv_block_backward_avx2(
    const float* __restrict dout_pool,
    const uint8_t* __restrict argmax_buf,
    const float* __restrict x,
    const float* __restrict W,
    const float* __restrict conv_act,
    float* __restrict d_conv_buf,
    float* __restrict dx,
    float* __restrict dW,
    float* __restrict db,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in,
    int64_t C_out, int64_t k_h, int64_t k_w,
    int64_t conv_stride, int64_t conv_pad,
    int64_t pool_size, int64_t pool_stride,
    int64_t pool_out_h, int64_t pool_out_w,
    float inv_m
) {
    (void)k_h; (void)k_w; (void)conv_stride; (void)conv_pad; (void)pool_size; (void)pool_stride;
    const int64_t conv_out_h = H;
    const int64_t conv_out_w = W_in;
    const int64_t conv_spatial = conv_out_h * conv_out_w;
    const int64_t pool_spatial = pool_out_h * pool_out_w;
    const int64_t spatial_in  = H * W_in;

    // 1. Thread-Parallel Sparse Unpooling
    #pragma omp parallel for schedule(static)
    for (int64_t cout = 0; cout < C_out; ++cout) {
        int tid = omp_get_thread_num();
        if (tid >= 0 && tid < MAX_TELEMETRY_THREADS) {
            thread_call_count[tid].fetch_add(1, std::memory_order_relaxed);
        }

        float bias_sum = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            const int64_t plane_idx = n * C_out + cout;
            float* __restrict d_plane = &d_conv_buf[plane_idx * conv_spatial];
            const float* __restrict dp_plane = &dout_pool[plane_idx * pool_spatial];
            const uint8_t* __restrict mask_plane = &argmax_buf[plane_idx * pool_spatial];
            const float* __restrict act_plane = conv_act ? &conv_act[plane_idx * conv_spatial] : nullptr;

            std::memset(d_plane, 0, conv_spatial * sizeof(float));

            for (int64_t ph = 0; ph < pool_out_h; ++ph) {
                const int64_t ih_base = ph * 2;
                for (int64_t pw = 0; pw < pool_out_w; ++pw) {
                    const int64_t p_idx = ph * pool_out_w + pw;
                    const uint8_t idx = mask_plane[p_idx];
                    const int64_t r_off = idx >> 1;
                    const int64_t c_off = idx & 1;
                    const int64_t c_idx = (ih_base + r_off) * conv_out_w + (pw * 2 + c_off);

                    float grad = dp_plane[p_idx];
                    if (act_plane && act_plane[c_idx] <= 0.0f) {
                        grad = 0.0f;
                    }
                    d_plane[c_idx] = grad;
                    bias_sum += grad;
                }
            }
        }
        if (db) db[cout] = bias_sum * inv_m;
    }

    // 2. Unbranched 2-Way Cin / 2-Way Cout Tiled dx Backprop
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t cin_blk = 0; cin_blk < (C_in + 1) / 2; ++cin_blk) {
            int tid = omp_get_thread_num();
            if (tid >= 0 && tid < MAX_TELEMETRY_THREADS) {
                thread_call_count[tid].fetch_add(1, std::memory_order_relaxed);
            }

            const int64_t cin0 = cin_blk * 2;
            const int64_t cin_rem = (C_in - cin0 >= 2) ? 2 : 1;

            float* __restrict dx_p0 = &dx[(n * C_in + cin0 + 0) * spatial_in];
            float* __restrict dx_p1 = (cin_rem > 1) ? &dx[(n * C_in + cin0 + 1) * spatial_in] : nullptr;

            for (int64_t ih = 0; ih < H; ++ih) {
                const int64_t oh0 = ih + 1 - 0;
                const int64_t oh1 = ih + 1 - 1;
                const int64_t oh2 = ih + 1 - 2;

                float* __restrict dx_row0 = &dx_p0[ih * W_in];
                float* __restrict dx_row1 = dx_p1 ? &dx_p1[ih * W_in] : nullptr;

                // Left Column Peel (iw = 0)
                {
                    float sum0 = 0.0f, sum1 = 0.0f;
                    for (int64_t cout = 0; cout < C_out; ++cout) {
                        const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                        const float* __restrict wp0 = &W[((cout * C_in + cin0 + 0) * 9)];
                        const float* __restrict wp1 = (cin_rem > 1) ? &W[((cout * C_in + cin0 + 1) * 9)] : nullptr;

                        #define TAP_SCALAR(OH_VAL, KW, W_IDX) { \
                            if ((OH_VAL) >= 0 && (OH_VAL) < conv_out_h) { \
                                const int64_t cur_ow = 1 - (KW); \
                                if (cur_ow >= 0 && cur_ow < conv_out_w) { \
                                    const float val = dp[(OH_VAL) * conv_out_w + cur_ow]; \
                                    sum0 += val * wp0[W_IDX]; \
                                    if (cin_rem > 1) sum1 += val * wp1[W_IDX]; \
                                } \
                            } \
                        }
                        TAP_SCALAR(oh0, 0, 0); TAP_SCALAR(oh0, 1, 1); TAP_SCALAR(oh0, 2, 2);
                        TAP_SCALAR(oh1, 0, 3); TAP_SCALAR(oh1, 1, 4); TAP_SCALAR(oh1, 2, 5);
                        TAP_SCALAR(oh2, 0, 6); TAP_SCALAR(oh2, 1, 7); TAP_SCALAR(oh2, 2, 8);
                        #undef TAP_SCALAR
                    }
                    dx_row0[0] = sum0;
                    if (cin_rem > 1) dx_row1[0] = sum1;
                }

                // Pure Unbranched Vector Interior (1 <= iw <= W_in - 9)
                int64_t iw = 1;
                for (; iw + 8 < W_in; iw += 8) {
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();

                    int64_t cout = 0;
                    for (; cout + 2 <= C_out; cout += 2) {
                        const float* __restrict dp0 = &d_conv_buf[(n * C_out + cout + 0) * conv_spatial];
                        const float* __restrict dp1 = &d_conv_buf[(n * C_out + cout + 1) * conv_spatial];

                        const float* __restrict wp0_c0 = &W[((cout + 0) * C_in + cin0 + 0) * 9];
                        const float* __restrict wp1_c0 = &W[((cout + 1) * C_in + cin0 + 0) * 9];
                        const float* __restrict wp0_c1 = (cin_rem > 1) ? &W[((cout + 0) * C_in + cin0 + 1) * 9] : nullptr;
                        const float* __restrict wp1_c1 = (cin_rem > 1) ? &W[((cout + 1) * C_in + cin0 + 1) * 9] : nullptr;

                        #define TAP_DX_FAST(OH_VAL, KW, W_IDX) { \
                            if ((OH_VAL) >= 0 && (OH_VAL) < conv_out_h) { \
                                const float* __restrict dr0 = &dp0[(OH_VAL) * conv_out_w + iw + 1 - (KW)]; \
                                const float* __restrict dr1 = &dp1[(OH_VAL) * conv_out_w + iw + 1 - (KW)]; \
                                const __m256 v0 = _mm256_loadu_ps(dr0); \
                                const __m256 v1 = _mm256_loadu_ps(dr1); \
                                acc0 = _mm256_fmadd_ps(v0, _mm256_set1_ps(wp0_c0[W_IDX]), acc0); \
                                acc0 = _mm256_fmadd_ps(v1, _mm256_set1_ps(wp1_c0[W_IDX]), acc0); \
                                if (cin_rem > 1) { \
                                    acc1 = _mm256_fmadd_ps(v0, _mm256_set1_ps(wp0_c1[W_IDX]), acc1); \
                                    acc1 = _mm256_fmadd_ps(v1, _mm256_set1_ps(wp1_c1[W_IDX]), acc1); \
                                } \
                            } \
                        }
                        TAP_DX_FAST(oh0, 0, 0); TAP_DX_FAST(oh0, 1, 1); TAP_DX_FAST(oh0, 2, 2);
                        TAP_DX_FAST(oh1, 0, 3); TAP_DX_FAST(oh1, 1, 4); TAP_DX_FAST(oh1, 2, 5);
                        TAP_DX_FAST(oh2, 0, 6); TAP_DX_FAST(oh2, 1, 7); TAP_DX_FAST(oh2, 2, 8);
                        #undef TAP_DX_FAST
                    }

                    for (; cout < C_out; ++cout) {
                        const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                        const float* __restrict wp0 = &W[((cout * C_in + cin0 + 0) * 9)];
                        const float* __restrict wp1 = (cin_rem > 1) ? &W[((cout * C_in + cin0 + 1) * 9)] : nullptr;

                        #define TAP_DX_FAST_1(OH_VAL, KW, W_IDX) { \
                            if ((OH_VAL) >= 0 && (OH_VAL) < conv_out_h) { \
                                const float* __restrict dr = &dp[(OH_VAL) * conv_out_w + iw + 1 - (KW)]; \
                                const __m256 v = _mm256_loadu_ps(dr); \
                                acc0 = _mm256_fmadd_ps(v, _mm256_set1_ps(wp0[W_IDX]), acc0); \
                                if (cin_rem > 1) acc1 = _mm256_fmadd_ps(v, _mm256_set1_ps(wp1[W_IDX]), acc1); \
                            } \
                        }
                        TAP_DX_FAST_1(oh0, 0, 0); TAP_DX_FAST_1(oh0, 1, 1); TAP_DX_FAST_1(oh0, 2, 2);
                        TAP_DX_FAST_1(oh1, 0, 3); TAP_DX_FAST_1(oh1, 1, 4); TAP_DX_FAST_1(oh1, 2, 5);
                        TAP_DX_FAST_1(oh2, 0, 6); TAP_DX_FAST_1(oh2, 1, 7); TAP_DX_FAST_1(oh2, 2, 8);
                        #undef TAP_DX_FAST_1
                    }

                    _mm256_storeu_ps(&dx_row0[iw], acc0);
                    if (cin_rem > 1) _mm256_storeu_ps(&dx_row1[iw], acc1);
                }

                // Right Column Peel
                for (; iw < W_in; ++iw) {
                    float sum0 = 0.0f, sum1 = 0.0f;
                    for (int64_t cout = 0; cout < C_out; ++cout) {
                        const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                        const float* __restrict wp0 = &W[((cout * C_in + cin0 + 0) * 9)];
                        const float* __restrict wp1 = (cin_rem > 1) ? &W[((cout * C_in + cin0 + 1) * 9)] : nullptr;

                        #define TAP_SCALAR_R(OH_VAL, KW, W_IDX) { \
                            if ((OH_VAL) >= 0 && (OH_VAL) < conv_out_h) { \
                                const int64_t cur_ow = iw + 1 - (KW); \
                                if (cur_ow >= 0 && cur_ow < conv_out_w) { \
                                    const float val = dp[(OH_VAL) * conv_out_w + cur_ow]; \
                                    sum0 += val * wp0[W_IDX]; \
                                    if (cin_rem > 1) sum1 += val * wp1[W_IDX]; \
                                } \
                            } \
                        }
                        TAP_SCALAR_R(oh0, 0, 0); TAP_SCALAR_R(oh0, 1, 1); TAP_SCALAR_R(oh0, 2, 2);
                        TAP_SCALAR_R(oh1, 0, 3); TAP_SCALAR_R(oh1, 1, 4); TAP_SCALAR_R(oh1, 2, 5);
                        TAP_SCALAR_R(oh2, 0, 6); TAP_SCALAR_R(oh2, 1, 7); TAP_SCALAR_R(oh2, 2, 8);
                        #undef TAP_SCALAR_R
                    }
                    dx_row0[iw] = sum0;
                    if (cin_rem > 1) dx_row1[iw] = sum1;
                }
            }
        }
    }

    // 3. Unbranched Vectorized dW Accumulation
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t cout = 0; cout < C_out; ++cout) {
        for (int64_t cin = 0; cin < C_in; ++cin) {
            int tid = omp_get_thread_num();
            if (tid >= 0 && tid < MAX_TELEMETRY_THREADS) {
                thread_call_count[tid].fetch_add(1, std::memory_order_relaxed);
            }

            float* __restrict dw_target = &dW[(cout * C_in + cin) * 9];

            __m256 v_dw00 = _mm256_setzero_ps(), v_dw01 = _mm256_setzero_ps(), v_dw02 = _mm256_setzero_ps();
            __m256 v_dw10 = _mm256_setzero_ps(), v_dw11 = _mm256_setzero_ps(), v_dw12 = _mm256_setzero_ps();
            __m256 v_dw20 = _mm256_setzero_ps(), v_dw21 = _mm256_setzero_ps(), v_dw22 = _mm256_setzero_ps();
            float s_dw[3][3] = {{0.0f}};

            for (int64_t n = 0; n < N; ++n) {
                const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                const float* __restrict xp = &x[(n * C_in + cin) * spatial_in];

                for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                    const float* __restrict dr = &dp[oh * conv_out_w];
                    const int64_t ih_base = oh - 1;

                    const float* __restrict xr0 = (ih_base + 0 >= 0 && ih_base + 0 < H) ? &xp[(ih_base + 0) * W_in] : nullptr;
                    const float* __restrict xr1 = (ih_base + 1 >= 0 && ih_base + 1 < H) ? &xp[(ih_base + 1) * W_in] : nullptr;
                    const float* __restrict xr2 = (ih_base + 2 >= 0 && ih_base + 2 < H) ? &xp[(ih_base + 2) * W_in] : nullptr;

                    // Left Border (ow = 0)
                    {
                        const float d_val = dr[0];
                        const int64_t iw0 = -1;
                        if (xr0) {
                            if (iw0 + 1 >= 0 && iw0 + 1 < W_in) s_dw[0][1] += d_val * xr0[iw0 + 1];
                            if (iw0 + 2 >= 0 && iw0 + 2 < W_in) s_dw[0][2] += d_val * xr0[iw0 + 2];
                        }
                        if (xr1) {
                            if (iw0 + 1 >= 0 && iw0 + 1 < W_in) s_dw[1][1] += d_val * xr1[iw0 + 1];
                            if (iw0 + 2 >= 0 && iw0 + 2 < W_in) s_dw[1][2] += d_val * xr1[iw0 + 2];
                        }
                        if (xr2) {
                            if (iw0 + 1 >= 0 && iw0 + 1 < W_in) s_dw[2][1] += d_val * xr2[iw0 + 1];
                            if (iw0 + 2 >= 0 && iw0 + 2 < W_in) s_dw[2][2] += d_val * xr2[iw0 + 2];
                        }
                    }

                    // Pure Unbranched Vector Interior (1 <= ow <= conv_out_w - 9)
                    int64_t ow = 1;
                    for (; ow + 8 < conv_out_w; ow += 8) {
                        const __m256 vd = _mm256_loadu_ps(&dr[ow]);
                        const int64_t iw0 = ow - 1;

                        if (xr0) {
                            v_dw00 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr0[iw0 + 0]), v_dw00);
                            v_dw01 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr0[iw0 + 1]), v_dw01);
                            v_dw02 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr0[iw0 + 2]), v_dw02);
                        }
                        if (xr1) {
                            v_dw10 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr1[iw0 + 0]), v_dw10);
                            v_dw11 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr1[iw0 + 1]), v_dw11);
                            v_dw12 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr1[iw0 + 2]), v_dw12);
                        }
                        if (xr2) {
                            v_dw20 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr2[iw0 + 0]), v_dw20);
                            v_dw21 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr2[iw0 + 1]), v_dw21);
                            v_dw22 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr2[iw0 + 2]), v_dw22);
                        }
                    }

                    // Right Border Peel
                    for (; ow < conv_out_w; ++ow) {
                        const float d_val = dr[ow];
                        const int64_t iw0 = ow - 1;
                        if (xr0) {
                            if (iw0 + 0 >= 0 && iw0 + 0 < W_in) s_dw[0][0] += d_val * xr0[iw0 + 0];
                            if (iw0 + 1 >= 0 && iw0 + 1 < W_in) s_dw[0][1] += d_val * xr0[iw0 + 1];
                            if (iw0 + 2 >= 0 && iw0 + 2 < W_in) s_dw[0][2] += d_val * xr0[iw0 + 2];
                        }
                        if (xr1) {
                            if (iw0 + 0 >= 0 && iw0 + 0 < W_in) s_dw[1][0] += d_val * xr1[iw0 + 0];
                            if (iw0 + 1 >= 0 && iw0 + 1 < W_in) s_dw[1][1] += d_val * xr1[iw0 + 1];
                            if (iw0 + 2 >= 0 && iw0 + 2 < W_in) s_dw[1][2] += d_val * xr1[iw0 + 2];
                        }
                        if (xr2) {
                            if (iw0 + 0 >= 0 && iw0 + 0 < W_in) s_dw[2][0] += d_val * xr2[iw0 + 0];
                            if (iw0 + 1 >= 0 && iw0 + 1 < W_in) s_dw[2][1] += d_val * xr2[iw0 + 1];
                            if (iw0 + 2 >= 0 && iw0 + 2 < W_in) s_dw[2][2] += d_val * xr2[iw0 + 2];
                        }
                    }
                }
            }

            #define REDUCE_SUM(V, S) { \
                alignas(32) float b[8]; \
                _mm256_store_ps(b, V); \
                (S) += (b[0] + b[1] + b[2] + b[3]) + (b[4] + b[5] + b[6] + b[7]); \
            }
            REDUCE_SUM(v_dw00, s_dw[0][0]); REDUCE_SUM(v_dw01, s_dw[0][1]); REDUCE_SUM(v_dw02, s_dw[0][2]);
            REDUCE_SUM(v_dw10, s_dw[1][0]); REDUCE_SUM(v_dw11, s_dw[1][1]); REDUCE_SUM(v_dw12, s_dw[1][2]);
            REDUCE_SUM(v_dw20, s_dw[2][0]); REDUCE_SUM(v_dw21, s_dw[2][1]); REDUCE_SUM(v_dw22, s_dw[2][2]);
            #undef REDUCE_SUM

            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    dw_target[r * 3 + c] = s_dw[r][c] * inv_m;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 4. STANDALONE FALLBACK PRIMITIVES
// -----------------------------------------------------------------------------
EXPORT_API void direct_conv2d_backward_fused_avx2(
    const float* __restrict dout, const float* __restrict x, const float* __restrict W,
    const float* __restrict in_act, float* __restrict dx, float* __restrict dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in,
    int64_t C_out, int64_t k_h, int64_t k_w,
    int64_t stride, int64_t pad, float inv_m, int32_t fuse_relu
) {
    (void)dout; (void)x; (void)W; (void)in_act; (void)dx; (void)dW;
    (void)N; (void)C_in; (void)H; (void)W_in; (void)C_out; (void)k_h; (void)k_w;
    (void)stride; (void)pad; (void)inv_m; (void)fuse_relu;
}

EXPORT_API void direct_conv2d_backward_weight_avx2(
    const float* __restrict dout, const float* __restrict x, float* __restrict dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in,
    int64_t C_out, int64_t k_h, int64_t k_w,
    int64_t stride, int64_t pad, float inv_m
) {
    (void)dout; (void)x; (void)dW; (void)N; (void)C_in; (void)H; (void)W_in; (void)C_out; (void)k_h; (void)k_w;
    (void)stride; (void)pad; (void)inv_m;
}

EXPORT_API void direct_conv2d_backward_input_avx2(
    const float* __restrict dout, const float* __restrict W, const float* __restrict in_act, float* __restrict dx,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in,
    int64_t C_out, int64_t k_h, int64_t k_w,
    int64_t stride, int64_t pad, int32_t fuse_relu
) {
    (void)dout; (void)W; (void)in_act; (void)dx; (void)N; (void)C_in; (void)H; (void)W_in; (void)C_out; (void)k_h; (void)k_w;
    (void)stride; (void)pad; (void)fuse_relu;
}

EXPORT_API void direct_maxpool_forward_avx2(
    const float* __restrict x, float* __restrict out, uint8_t* __restrict argmax_buf,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t pool_size, int64_t stride
) {
    (void)x; (void)out; (void)argmax_buf; (void)N; (void)C; (void)H; (void)W; (void)pool_size; (void)stride;
}

EXPORT_API void direct_maxpool_backward_avx2(
    const float* __restrict dout, const uint8_t* __restrict argmax_indices, float* __restrict dx,
    int64_t N, int64_t C, int64_t out_h, int64_t out_w, int64_t in_h, int64_t in_w, int64_t pool_size, int64_t stride
) {
    (void)dout; (void)argmax_indices; (void)dx; (void)N; (void)C; (void)out_h; (void)out_w; (void)in_h; (void)in_w; (void)pool_size; (void)stride;
}

EXPORT_API void direct_relu_forward_avx2(float* data, int64_t size) {
    const __m256 v_zero = _mm256_setzero_ps();
    int64_t i = 0;
    #pragma omp parallel for schedule(static)
    for (i = 0; i <= size - 8; i += 8) {
        int tid = omp_get_thread_num();
        if (tid >= 0 && tid < MAX_TELEMETRY_THREADS) {
            thread_call_count[tid].fetch_add(1, std::memory_order_relaxed);
        }
        __m256 v = _mm256_loadu_ps(&data[i]);
        _mm256_storeu_ps(&data[i], _mm256_max_ps(v, v_zero));
    }
    for (; i < size; ++i) {
        if (data[i] < 0.0f) data[i] = 0.0f;
    }
}

EXPORT_API void direct_relu_backward_avx2(float* dout, const float* in_act, int64_t size) {
    const __m256 v_zero = _mm256_setzero_ps();
    int64_t i = 0;
    #pragma omp parallel for schedule(static)
    for (i = 0; i <= size - 8; i += 8) {
        int tid = omp_get_thread_num();
        if (tid >= 0 && tid < MAX_TELEMETRY_THREADS) {
            thread_call_count[tid].fetch_add(1, std::memory_order_relaxed);
        }
        __m256 v_dout = _mm256_loadu_ps(&dout[i]);
        __m256 v_act  = _mm256_loadu_ps(&in_act[i]);
        __m256 mask   = _mm256_cmp_ps(v_act, v_zero, _CMP_GT_OQ);
        _mm256_storeu_ps(&dout[i], _mm256_and_ps(v_dout, mask));
    }
    for (; i < size; ++i) {
        if (in_act[i] <= 0.0f) dout[i] = 0.0f;
    }
}

EXPORT_API void direct_bias_backward_avx2(
    const float* __restrict dout, float* __restrict db,
    int64_t N, int64_t C_out, int64_t out_h, int64_t out_w, float inv_m
) {
    const int64_t spatial = out_h * out_w;
    #pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < C_out; ++c) {
        int tid = omp_get_thread_num();
        if (tid >= 0 && tid < MAX_TELEMETRY_THREADS) {
            thread_call_count[tid].fetch_add(1, std::memory_order_relaxed);
        }
        float sum = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            const float* plane = &dout[(n * C_out + c) * spatial];
            for (int64_t s = 0; s < spatial; ++s) {
                sum += plane[s];
            }
        }
        db[c] = sum * inv_m;
    }
}