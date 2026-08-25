#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <omp.h>
#include <atomic>
#include <iostream>
#include <algorithm>

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
// 1. DYNAMIC FORWARD ENGINE (UNROLLED 3x3 FAST-PATH + GENERAL ARBITRARY KERNEL)
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
    const int64_t out_h = (H + 2 * pad - k_h) / stride + 1;
    const int64_t out_w = (W_in + 2 * pad - k_w) / stride + 1;
    const int64_t spatial_out = out_h * out_w;
    const int64_t spatial_in  = H * W_in;
    const int64_t k_spatial   = k_h * k_w;
    const __m256 v_zero = _mm256_setzero_ps();

    // SPECIALIZED FAST-PATH: 3x3 Kernels
    if (k_h == 3 && k_w == 3) {
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

                            #define TAP_FWD_3X3(ROW_PTR, KW, W_IDX) { \
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

                            TAP_FWD_3X3(r0, 0, 0); TAP_FWD_3X3(r0, 1, 1); TAP_FWD_3X3(r0, 2, 2);
                            TAP_FWD_3X3(r1, 0, 3); TAP_FWD_3X3(r1, 1, 4); TAP_FWD_3X3(r1, 2, 5);
                            TAP_FWD_3X3(r2, 0, 6); TAP_FWD_3X3(r2, 1, 7); TAP_FWD_3X3(r2, 2, 8);
                            #undef TAP_FWD_3X3
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

                            #define TAP_SCALAR_3X3(ROW_PTR, KW, W_IDX) { \
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
                            TAP_SCALAR_3X3(r0, 0, 0); TAP_SCALAR_3X3(r0, 1, 1); TAP_SCALAR_3X3(r0, 2, 2);
                            TAP_SCALAR_3X3(r1, 0, 3); TAP_SCALAR_3X3(r1, 1, 4); TAP_SCALAR_3X3(r1, 2, 5);
                            TAP_SCALAR_3X3(r2, 0, 6); TAP_SCALAR_3X3(r2, 1, 7); TAP_SCALAR_3X3(r2, 2, 8);
                            #undef TAP_SCALAR_3X3
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
        return;
    }

    // GENERALIZED PATH: Arbitrary Kernel Dimensions (k_h, k_w)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t cout = 0; cout < C_out; ++cout) {
            int tid = omp_get_thread_num();
            if (tid >= 0 && tid < MAX_TELEMETRY_THREADS) {
                thread_call_count[tid].fetch_add(1, std::memory_order_relaxed);
            }

            float* __restrict out_plane = &out[(n * C_out + cout) * spatial_out];
            const float b_val = bias ? bias[cout] : 0.0f;
            const __m256 vb = _mm256_set1_ps(b_val);

            for (int64_t oh = 0; oh < out_h; ++oh) {
                const int64_t ih_base = oh * stride - pad;
                float* __restrict out_row = &out_plane[oh * out_w];

                int64_t ow = 0;
                for (; ow + 8 <= out_w; ow += 8) {
                    __m256 acc = vb;
                    for (int64_t cin = 0; cin < C_in; ++cin) {
                        const float* __restrict xp = &x[(n * C_in + cin) * spatial_in];
                        const float* __restrict wp = &W[(cout * C_in + cin) * k_spatial];

                        for (int64_t kh = 0; kh < k_h; ++kh) {
                            const int64_t cur_ih = ih_base + kh;
                            if (cur_ih < 0 || cur_ih >= H) continue;

                            const float* __restrict in_row = &xp[cur_ih * W_in];
                            for (int64_t kw = 0; kw < k_w; ++kw) {
                                const float weight_scalar = wp[kh * k_w + kw];
                                const __m256 vw = _mm256_set1_ps(weight_scalar);

                                alignas(32) float tmp[8] = {0};
                                for (int s = 0; s < 8; ++s) {
                                    int64_t cur_iw = (ow + s) * stride - pad + kw;
                                    if (cur_iw >= 0 && cur_iw < W_in) {
                                        tmp[s] = in_row[cur_iw];
                                    }
                                }
                                __m256 vx = _mm256_load_ps(tmp);
                                acc = _mm256_fmadd_ps(vx, vw, acc);
                            }
                        }
                    }

                    if (fuse_relu) {
                        acc = _mm256_max_ps(acc, v_zero);
                    }
                    _mm256_storeu_ps(&out_row[ow], acc);
                }

                for (; ow < out_w; ++ow) {
                    float s = b_val;
                    for (int64_t cin = 0; cin < C_in; ++cin) {
                        const float* __restrict xp = &x[(n * C_in + cin) * spatial_in];
                        const float* __restrict wp = &W[(cout * C_in + cin) * k_spatial];

                        for (int64_t kh = 0; kh < k_h; ++kh) {
                            const int64_t cur_ih = ih_base + kh;
                            if (cur_ih < 0 || cur_ih >= H) continue;

                            const float* __restrict in_row = &xp[cur_ih * W_in];
                            for (int64_t kw = 0; kw < k_w; ++kw) {
                                const int64_t cur_iw = ow * stride - pad + kw;
                                if (cur_iw >= 0 && cur_iw < W_in) {
                                    s += in_row[cur_iw] * wp[kh * k_w + kw];
                                }
                            }
                        }
                    }
                    if (fuse_relu) {
                        s = s > 0.0f ? s : 0.0f;
                    }
                    out_row[ow] = s;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 2. FUSED COMPOSITE CONV BLOCK (FORWARD: Conv + Fused ReLU + Dynamic MaxPool)
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

    const int64_t conv_out_h = (H + 2 * conv_pad - k_h) / conv_stride + 1;
    const int64_t conv_out_w = (W_in + 2 * conv_pad - k_w) / conv_stride + 1;
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

                for (int64_t pw = 0; pw < pool_out_w; ++pw) {
                    const int64_t iw_base = pw * pool_stride;

                    float max_val = -1e30f;
                    uint8_t best_idx = 0;

                    for (int64_t kh = 0; kh < pool_size; ++kh) {
                        const float* __restrict r = &cr_plane[(ih_base + kh) * conv_out_w];
                        for (int64_t kw = 0; kw < pool_size; ++kw) {
                            const float v = r[iw_base + kw];
                            if (v > max_val) {
                                max_val = v;
                                best_idx = static_cast<uint8_t>(kh * pool_size + kw);
                            }
                        }
                    }

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
    const int64_t conv_out_h = (H + 2 * conv_pad - k_h) / conv_stride + 1;
    const int64_t conv_out_w = (W_in + 2 * conv_pad - k_w) / conv_stride + 1;
    const int64_t conv_spatial = conv_out_h * conv_out_w;
    const int64_t pool_spatial = pool_out_h * pool_out_w;
    const int64_t spatial_in  = H * W_in;
    const int64_t k_spatial   = k_h * k_w;

    // 1. Thread-Parallel Sparse Unpooling + ReLU Gate + Bias Accumulation
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
                const int64_t ih_base = ph * pool_stride;
                for (int64_t pw = 0; pw < pool_out_w; ++pw) {
                    const int64_t p_idx = ph * pool_out_w + pw;
                    const uint8_t idx = mask_plane[p_idx];
                    const int64_t r_off = idx / pool_size;
                    const int64_t c_off = idx % pool_size;
                    const int64_t c_idx = (ih_base + r_off) * conv_out_w + (pw * pool_stride + c_off);

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

    // SPECIALIZED FAST-PATH: 3x3, Stride 1, Pad 1
    if (k_h == 3 && k_w == 3 && conv_stride == 1 && conv_pad == 1) {
        // 2a. Unbranched 2-Way Cin / 2-Way Cout Tiled dx Backprop
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

                    // Left Column Peel
                    {
                        float sum0 = 0.0f, sum1 = 0.0f;
                        for (int64_t cout = 0; cout < C_out; ++cout) {
                            const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                            const float* __restrict wp0 = &W[((cout * C_in + cin0 + 0) * 9)];
                            const float* __restrict wp1 = (cin_rem > 1) ? &W[((cout * C_in + cin0 + 1) * 9)] : nullptr;

                            #define TAP_SCALAR_3X3(OH_VAL, KW, W_IDX) { \
                                if ((OH_VAL) >= 0 && (OH_VAL) < conv_out_h) { \
                                    const int64_t cur_ow = 1 - (KW); \
                                    if (cur_ow >= 0 && cur_ow < conv_out_w) { \
                                        const float val = dp[(OH_VAL) * conv_out_w + cur_ow]; \
                                        sum0 += val * wp0[W_IDX]; \
                                        if (cin_rem > 1) sum1 += val * wp1[W_IDX]; \
                                    } \
                                } \
                            }
                            TAP_SCALAR_3X3(oh0, 0, 0); TAP_SCALAR_3X3(oh0, 1, 1); TAP_SCALAR_3X3(oh0, 2, 2);
                            TAP_SCALAR_3X3(oh1, 0, 3); TAP_SCALAR_3X3(oh1, 1, 4); TAP_SCALAR_3X3(oh1, 2, 5);
                            TAP_SCALAR_3X3(oh2, 0, 6); TAP_SCALAR_3X3(oh2, 1, 7); TAP_SCALAR_3X3(oh2, 2, 8);
                            #undef TAP_SCALAR_3X3
                        }
                        dx_row0[0] = sum0;
                        if (cin_rem > 1) dx_row1[0] = sum1;
                    }

                    // Vectorized Interior
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

                            #define TAP_DX_FAST_3X3(OH_VAL, KW, W_IDX) { \
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
                            TAP_DX_FAST_3X3(oh0, 0, 0); TAP_DX_FAST_3X3(oh0, 1, 1); TAP_DX_FAST_3X3(oh0, 2, 2);
                            TAP_DX_FAST_3X3(oh1, 0, 3); TAP_DX_FAST_3X3(oh1, 1, 4); TAP_DX_FAST_3X3(oh1, 2, 5);
                            TAP_DX_FAST_3X3(oh2, 0, 6); TAP_DX_FAST_3X3(oh2, 1, 7); TAP_DX_FAST_3X3(oh2, 2, 8);
                            #undef TAP_DX_FAST_3X3
                        }

                        for (; cout < C_out; ++cout) {
                            const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                            const float* __restrict wp0 = &W[((cout * C_in + cin0 + 0) * 9)];
                            const float* __restrict wp1 = (cin_rem > 1) ? &W[((cout * C_in + cin0 + 1) * 9)] : nullptr;

                            #define TAP_DX_FAST_1_3X3(OH_VAL, KW, W_IDX) { \
                                if ((OH_VAL) >= 0 && (OH_VAL) < conv_out_h) { \
                                    const float* __restrict dr = &dp[(OH_VAL) * conv_out_w + iw + 1 - (KW)]; \
                                    const __m256 v = _mm256_loadu_ps(dr); \
                                    acc0 = _mm256_fmadd_ps(v, _mm256_set1_ps(wp0[W_IDX]), acc0); \
                                    if (cin_rem > 1) acc1 = _mm256_fmadd_ps(v, _mm256_set1_ps(wp1[W_IDX]), acc1); \
                                } \
                            }
                            TAP_DX_FAST_1_3X3(oh0, 0, 0); TAP_DX_FAST_1_3X3(oh0, 1, 1); TAP_DX_FAST_1_3X3(oh0, 2, 2);
                            TAP_DX_FAST_1_3X3(oh1, 0, 3); TAP_DX_FAST_1_3X3(oh1, 1, 4); TAP_DX_FAST_1_3X3(oh1, 2, 5);
                            TAP_DX_FAST_1_3X3(oh2, 0, 6); TAP_DX_FAST_1_3X3(oh2, 1, 7); TAP_DX_FAST_1_3X3(oh2, 2, 8);
                            #undef TAP_DX_FAST_1_3X3
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

                            #define TAP_SCALAR_R_3X3(OH_VAL, KW, W_IDX) { \
                                if ((OH_VAL) >= 0 && (OH_VAL) < conv_out_h) { \
                                    const int64_t cur_ow = iw + 1 - (KW); \
                                    if (cur_ow >= 0 && cur_ow < conv_out_w) { \
                                        const float val = dp[(OH_VAL) * conv_out_w + cur_ow]; \
                                        sum0 += val * wp0[W_IDX]; \
                                        if (cin_rem > 1) sum1 += val * wp1[W_IDX]; \
                                    } \
                                } \
                            }
                            TAP_SCALAR_R_3X3(oh0, 0, 0); TAP_SCALAR_R_3X3(oh0, 1, 1); TAP_SCALAR_R_3X3(oh0, 2, 2);
                            TAP_SCALAR_R_3X3(oh1, 0, 3); TAP_SCALAR_R_3X3(oh1, 1, 4); TAP_SCALAR_R_3X3(oh1, 2, 5);
                            TAP_SCALAR_R_3X3(oh2, 0, 6); TAP_SCALAR_R_3X3(oh2, 1, 7); TAP_SCALAR_R_3X3(oh2, 2, 8);
                            #undef TAP_SCALAR_R_3X3
                        }
                        dx_row0[iw] = sum0;
                        if (cin_rem > 1) dx_row1[iw] = sum1;
                    }
                }
            }
        }

        // 3a. Vectorized dW Accumulation
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

                        // Left Border
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

                        // Vectorized Interior
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
        return;
    }

    // GENERALIZED PATH: Arbitrary Kernel (k_h, k_w), Stride, and Pad
    std::memset(dx, 0, N * C_in * spatial_in * sizeof(float));
    std::memset(dW, 0, C_out * C_in * k_h * k_w * sizeof(float));

    // 2b. Generalized Input Gradient (dx)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t cin = 0; cin < C_in; ++cin) {
            int tid = omp_get_thread_num();
            if (tid >= 0 && tid < MAX_TELEMETRY_THREADS) {
                thread_call_count[tid].fetch_add(1, std::memory_order_relaxed);
            }

            float* __restrict dx_plane = &dx[(n * C_in + cin) * spatial_in];

            for (int64_t cout = 0; cout < C_out; ++cout) {
                const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                const float* __restrict wp = &W[(cout * C_in + cin) * k_spatial];

                for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                    for (int64_t ow = 0; ow < conv_out_w; ++ow) {
                        const float d_val = dp[oh * conv_out_w + ow];
                        if (d_val == 0.0f) continue;

                        const int64_t ih_base = oh * conv_stride - conv_pad;
                        const int64_t iw_base = ow * conv_stride - conv_pad;

                        for (int64_t kh = 0; kh < k_h; ++kh) {
                            const int64_t cur_ih = ih_base + kh;
                            if (cur_ih < 0 || cur_ih >= H) continue;

                            float* __restrict dx_row = &dx_plane[cur_ih * W_in];
                            for (int64_t kw = 0; kw < k_w; ++kw) {
                                const int64_t cur_iw = iw_base + kw;
                                if (cur_iw >= 0 && cur_iw < W_in) {
                                    dx_row[cur_iw] += d_val * wp[kh * k_w + kw];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 3b. Generalized Weight Gradient (dW)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t cout = 0; cout < C_out; ++cout) {
        for (int64_t cin = 0; cin < C_in; ++cin) {
            int tid = omp_get_thread_num();
            if (tid >= 0 && tid < MAX_TELEMETRY_THREADS) {
                thread_call_count[tid].fetch_add(1, std::memory_order_relaxed);
            }

            float* __restrict dw_target = &dW[(cout * C_in + cin) * k_spatial];

            for (int64_t n = 0; n < N; ++n) {
                const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                const float* __restrict xp = &x[(n * C_in + cin) * spatial_in];

                for (int64_t kh = 0; kh < k_h; ++kh) {
                    for (int64_t kw = 0; kw < k_w; ++kw) {
                        float sum = 0.0f;
                        for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                            const int64_t cur_ih = oh * conv_stride - conv_pad + kh;
                            if (cur_ih < 0 || cur_ih >= H) continue;

                            const float* __restrict xr = &xp[cur_ih * W_in];
                            const float* __restrict dr = &dp[oh * conv_out_w];

                            for (int64_t ow = 0; ow < conv_out_w; ++ow) {
                                const int64_t cur_iw = ow * conv_stride - conv_pad + kw;
                                if (cur_iw >= 0 && cur_iw < W_in) {
                                    sum += dr[ow] * xr[cur_iw];
                                }
                            }
                        }
                        dw_target[kh * k_w + kw] += sum;
                    }
                }
            }

            for (int64_t k = 0; k < k_spatial; ++k) {
                dw_target[k] *= inv_m;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 4. STANDALONE PRIMITIVES & ACTIVATIONS
// -----------------------------------------------------------------------------
EXPORT_API void direct_conv2d_backward_fused_avx2(
    const float* __restrict dout, const float* __restrict x, const float* __restrict W,
    const float* __restrict in_act, float* __restrict dx, float* __restrict dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in,
    int64_t C_out, int64_t k_h, int64_t k_w,
    int64_t stride, int64_t pad, float inv_m, int32_t fuse_relu
) {
    (void)in_act; (void)fuse_relu;
    // Re-use core backward engine directly
    const int64_t conv_out_h = (H + 2 * pad - k_h) / stride + 1;
    const int64_t conv_out_w = (W_in + 2 * pad - k_w) / stride + 1;
    direct_conv_block_backward_avx2(
        dout, nullptr, x, W, nullptr, const_cast<float*>(dout),
        dx, dW, nullptr, N, C_in, H, W_in, C_out, k_h, k_w,
        stride, pad, 1, 1, conv_out_h, conv_out_w, inv_m
    );
}

EXPORT_API void direct_conv2d_backward_weight_avx2(
    const float* __restrict dout, const float* __restrict x, float* __restrict dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in,
    int64_t C_out, int64_t k_h, int64_t k_w,
    int64_t stride, int64_t pad, float inv_m
) {
    const int64_t conv_out_h = (H + 2 * pad - k_h) / stride + 1;
    const int64_t conv_out_w = (W_in + 2 * pad - k_w) / stride + 1;
    direct_conv_block_backward_avx2(
        dout, nullptr, x, nullptr, nullptr, const_cast<float*>(dout),
        nullptr, dW, nullptr, N, C_in, H, W_in, C_out, k_h, k_w,
        stride, pad, 1, 1, conv_out_h, conv_out_w, inv_m
    );
}

EXPORT_API void direct_conv2d_backward_input_avx2(
    const float* __restrict dout, const float* __restrict W, const float* __restrict in_act, float* __restrict dx,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in,
    int64_t C_out, int64_t k_h, int64_t k_w,
    int64_t stride, int64_t pad, int32_t fuse_relu
) {
    (void)in_act; (void)fuse_relu;
    const int64_t conv_out_h = (H + 2 * pad - k_h) / stride + 1;
    const int64_t conv_out_w = (W_in + 2 * pad - k_w) / stride + 1;
    direct_conv_block_backward_avx2(
        dout, nullptr, nullptr, W, nullptr, const_cast<float*>(dout),
        dx, nullptr, nullptr, N, C_in, H, W_in, C_out, k_h, k_w,
        stride, pad, 1, 1, conv_out_h, conv_out_w, 1.0f
    );
}

EXPORT_API void direct_maxpool_forward_avx2(
    const float* __restrict x, float* __restrict out, uint8_t* __restrict argmax_buf,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t pool_size, int64_t stride
) {
    const int64_t out_h = (H - pool_size) / stride + 1;
    const int64_t out_w = (W - pool_size) / stride + 1;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            int tid = omp_get_thread_num();
            if (tid >= 0 && tid < MAX_TELEMETRY_THREADS) {
                thread_call_count[tid].fetch_add(1, std::memory_order_relaxed);
            }

            const float* __restrict in_plane = &x[(n * C + c) * H * W];
            float* __restrict out_plane      = &out[(n * C + c) * out_h * out_w];
            uint8_t* __restrict m_plane      = &argmax_buf[(n * C + c) * out_h * out_w];

            for (int64_t oh = 0; oh < out_h; ++oh) {
                const int64_t ih_base = oh * stride;
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    const int64_t iw_base = ow * stride;
                    float max_val = -1e30f;
                    uint8_t best_idx = 0;

                    for (int64_t kh = 0; kh < pool_size; ++kh) {
                        const float* __restrict r = &in_plane[(ih_base + kh) * W];
                        for (int64_t kw = 0; kw < pool_size; ++kw) {
                            const float v = r[iw_base + kw];
                            if (v > max_val) {
                                max_val = v;
                                best_idx = static_cast<uint8_t>(kh * pool_size + kw);
                            }
                        }
                    }

                    const int64_t p_idx = oh * out_w + ow;
                    out_plane[p_idx] = max_val;
                    m_plane[p_idx] = best_idx;
                }
            }
        }
    }
}

EXPORT_API void direct_maxpool_backward_avx2(
    const float* __restrict dout, const uint8_t* __restrict argmax_indices, float* __restrict dx,
    int64_t N, int64_t C, int64_t out_h, int64_t out_w, int64_t in_h, int64_t in_w, int64_t pool_size, int64_t stride
) {
    std::memset(dx, 0, N * C * in_h * in_w * sizeof(float));

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            int tid = omp_get_thread_num();
            if (tid >= 0 && tid < MAX_TELEMETRY_THREADS) {
                thread_call_count[tid].fetch_add(1, std::memory_order_relaxed);
            }

            const float* __restrict dp_plane = &dout[(n * C + c) * out_h * out_w];
            const uint8_t* __restrict m_plane = &argmax_indices[(n * C + c) * out_h * out_w];
            float* __restrict dx_plane = &dx[(n * C + c) * in_h * in_w];

            for (int64_t oh = 0; oh < out_h; ++oh) {
                const int64_t ih_base = oh * stride;
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    const int64_t p_idx = oh * out_w + ow;
                    const uint8_t idx = m_plane[p_idx];
                    const int64_t r_off = idx / pool_size;
                    const int64_t c_off = idx % pool_size;
                    dx_plane[(ih_base + r_off) * in_w + (ow * stride + c_off)] += dp_plane[p_idx];
                }
            }
        }
    }
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