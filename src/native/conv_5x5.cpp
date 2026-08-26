#include "diagnostics.h"

void conv2d_forward_5x5_avx2(
    const float* __restrict x, const float* __restrict W, const float* __restrict bias, float* __restrict out,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t C_out, int64_t stride, int64_t pad, int32_t fuse_relu
) {
    DIAG_INC(fwd_5x5);
    TIME_SCOPE(time_fwd_5x5_ns);
    const int64_t out_h = (H + 2 * pad - 5) / stride + 1;
    const int64_t out_w = (W_in + 2 * pad - 5) / stride + 1;
    const int64_t spatial_out = out_h * out_w;
    const int64_t spatial_in  = H * W_in;
    const __m256 v_zero = _mm256_setzero_ps();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t cout_blk = 0; cout_blk < (C_out + 3) / 4; ++cout_blk) {
            const int64_t cout0 = cout_blk * 4;
            const int64_t c_rem = (C_out - cout0 >= 4) ? 4 : (C_out - cout0);

            for (int64_t oh = 0; oh < out_h; ++oh) {
                const int64_t ih0 = oh * stride - pad + 0;
                const int64_t ih1 = oh * stride - pad + 1;
                const int64_t ih2 = oh * stride - pad + 2;
                const int64_t ih3 = oh * stride - pad + 3;
                const int64_t ih4 = oh * stride - pad + 4;

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
                        const float* __restrict wp0 = &W[((cout0 + 0) * C_in + cin) * 25];
                        const float* __restrict wp1 = (c_rem > 1) ? &W[((cout0 + 1) * C_in + cin) * 25] : nullptr;
                        const float* __restrict wp2 = (c_rem > 2) ? &W[((cout0 + 2) * C_in + cin) * 25] : nullptr;
                        const float* __restrict wp3 = (c_rem > 3) ? &W[((cout0 + 3) * C_in + cin) * 25] : nullptr;

                        const float* __restrict r0 = (ih0 >= 0 && ih0 < H) ? &xp[ih0 * W_in] : nullptr;
                        const float* __restrict r1 = (ih1 >= 0 && ih1 < H) ? &xp[ih1 * W_in] : nullptr;
                        const float* __restrict r2 = (ih2 >= 0 && ih2 < H) ? &xp[ih2 * W_in] : nullptr;
                        const float* __restrict r3 = (ih3 >= 0 && ih3 < H) ? &xp[ih3 * W_in] : nullptr;
                        const float* __restrict r4 = (ih4 >= 0 && ih4 < H) ? &xp[ih4 * W_in] : nullptr;

                        #define TAP_FWD_5X5(ROW_PTR, KW, W_IDX) { \
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

                        TAP_FWD_5X5(r0, 0, 0);  TAP_FWD_5X5(r0, 1, 1);  TAP_FWD_5X5(r0, 2, 2);  TAP_FWD_5X5(r0, 3, 3);  TAP_FWD_5X5(r0, 4, 4);
                        TAP_FWD_5X5(r1, 0, 5);  TAP_FWD_5X5(r1, 1, 6);  TAP_FWD_5X5(r1, 2, 7);  TAP_FWD_5X5(r1, 3, 8);  TAP_FWD_5X5(r1, 4, 9);
                        TAP_FWD_5X5(r2, 0, 10); TAP_FWD_5X5(r2, 1, 11); TAP_FWD_5X5(r2, 2, 12); TAP_FWD_5X5(r2, 3, 13); TAP_FWD_5X5(r2, 4, 14);
                        TAP_FWD_5X5(r3, 0, 15); TAP_FWD_5X5(r3, 1, 16); TAP_FWD_5X5(r3, 2, 17); TAP_FWD_5X5(r3, 3, 18); TAP_FWD_5X5(r3, 4, 19);
                        TAP_FWD_5X5(r4, 0, 20); TAP_FWD_5X5(r4, 1, 21); TAP_FWD_5X5(r4, 2, 22); TAP_FWD_5X5(r4, 3, 23); TAP_FWD_5X5(r4, 4, 24);
                        #undef TAP_FWD_5X5
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
                        const float* __restrict wp0 = &W[((cout0 + 0) * C_in + cin) * 25];
                        const float* __restrict wp1 = (c_rem > 1) ? &W[((cout0 + 1) * C_in + cin) * 25] : nullptr;
                        const float* __restrict wp2 = (c_rem > 2) ? &W[((cout0 + 2) * C_in + cin) * 25] : nullptr;
                        const float* __restrict wp3 = (c_rem > 3) ? &W[((cout0 + 3) * C_in + cin) * 25] : nullptr;

                        const float* __restrict r0 = (ih0 >= 0 && ih0 < H) ? &xp[ih0 * W_in] : nullptr;
                        const float* __restrict r1 = (ih1 >= 0 && ih1 < H) ? &xp[ih1 * W_in] : nullptr;
                        const float* __restrict r2 = (ih2 >= 0 && ih2 < H) ? &xp[ih2 * W_in] : nullptr;
                        const float* __restrict r3 = (ih3 >= 0 && ih3 < H) ? &xp[ih3 * W_in] : nullptr;
                        const float* __restrict r4 = (ih4 >= 0 && ih4 < H) ? &xp[ih4 * W_in] : nullptr;

                        #define TAP_SCALAR_5X5(ROW_PTR, KW, W_IDX) { \
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
                        TAP_SCALAR_5X5(r0, 0, 0);  TAP_SCALAR_5X5(r0, 1, 1);  TAP_SCALAR_5X5(r0, 2, 2);  TAP_SCALAR_5X5(r0, 3, 3);  TAP_SCALAR_5X5(r0, 4, 4);
                        TAP_SCALAR_5X5(r1, 0, 5);  TAP_SCALAR_5X5(r1, 1, 6);  TAP_SCALAR_5X5(r1, 2, 7);  TAP_SCALAR_5X5(r1, 3, 8);  TAP_SCALAR_5X5(r1, 4, 9);
                        TAP_SCALAR_5X5(r2, 0, 10); TAP_SCALAR_5X5(r2, 1, 11); TAP_SCALAR_5X5(r2, 2, 12); TAP_SCALAR_5X5(r2, 3, 13); TAP_SCALAR_5X5(r2, 4, 14);
                        TAP_SCALAR_5X5(r3, 0, 15); TAP_SCALAR_5X5(r3, 1, 16); TAP_SCALAR_5X5(r3, 2, 17); TAP_SCALAR_5X5(r3, 3, 18); TAP_SCALAR_5X5(r3, 4, 19);
                        TAP_SCALAR_5X5(r4, 0, 20); TAP_SCALAR_5X5(r4, 1, 21); TAP_SCALAR_5X5(r4, 2, 22); TAP_SCALAR_5X5(r4, 3, 23); TAP_SCALAR_5X5(r4, 4, 24);
                        #undef TAP_SCALAR_5X5
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

void conv2d_backward_5x5_avx2(
    const float* __restrict d_conv_buf, const float* __restrict x, const float* __restrict W,
    float* __restrict dx, float* __restrict dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t C_out, int64_t stride, int64_t pad, float inv_m
) {
    (void)stride;
    (void)pad;
    DIAG_INC(bwd_5x5);
    TIME_SCOPE(time_bwd_5x5_dx_ns);
    const int64_t conv_out_h = H;
    const int64_t conv_out_w = W_in;
    const int64_t conv_spatial = conv_out_h * conv_out_w;
    const int64_t spatial_in   = H * W_in;

    #pragma omp parallel
    {
        // ==========================================
        // 1. dX Backpropagation Pass (Same Thread Team)
        // ==========================================
        if (dx && W) {
            #pragma omp for collapse(2) schedule(static) nowait
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t cin_blk = 0; cin_blk < (C_in + 1) / 2; ++cin_blk) {
                    const int64_t cin0 = cin_blk * 2;
                    const int64_t cin_rem = (C_in - cin0 >= 2) ? 2 : 1;

                    float* __restrict dx_p0 = &dx[(n * C_in + cin0 + 0) * spatial_in];
                    float* __restrict dx_p1 = (cin_rem > 1) ? &dx[(n * C_in + cin0 + 1) * spatial_in] : nullptr;

                    if (cin_rem == 2) {
                        for (int64_t ih = 0; ih < H; ++ih) {
                            const int64_t oh0 = ih + 2 - 0;
                            const int64_t oh1 = ih + 2 - 1;
                            const int64_t oh2 = ih + 2 - 2;
                            const int64_t oh3 = ih + 2 - 3;
                            const int64_t oh4 = ih + 2 - 4;

                            float* __restrict dx_row0 = &dx_p0[ih * W_in];
                            float* __restrict dx_row1 = &dx_p1[ih * W_in];

                            // Left Border Peel (iw = 0, 1)
                            for (int64_t iw = 0; iw < 2 && iw < W_in; ++iw) {
                                float sum0 = 0.0f, sum1 = 0.0f;
                                for (int64_t cout = 0; cout < C_out; ++cout) {
                                    const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                                    const float* __restrict wp0 = &W[((cout * C_in + cin0 + 0) * 25)];
                                    const float* __restrict wp1 = &W[((cout * C_in + cin0 + 1) * 25)];

                                    #define TAP_SCALAR_DX_5X5_F(OH_VAL, KW, W_IDX) { \
                                        if ((OH_VAL) >= 0 && (OH_VAL) < conv_out_h) { \
                                            const int64_t cur_ow = iw + 2 - (KW); \
                                            if (cur_ow >= 0 && cur_ow < conv_out_w) { \
                                                const float val = dp[(OH_VAL) * conv_out_w + cur_ow]; \
                                                sum0 += val * wp0[W_IDX]; \
                                                sum1 += val * wp1[W_IDX]; \
                                            } \
                                        } \
                                    }
                                    TAP_SCALAR_DX_5X5_F(oh0, 0, 0);  TAP_SCALAR_DX_5X5_F(oh0, 1, 1);  TAP_SCALAR_DX_5X5_F(oh0, 2, 2);  TAP_SCALAR_DX_5X5_F(oh0, 3, 3);  TAP_SCALAR_DX_5X5_F(oh0, 4, 4);
                                    TAP_SCALAR_DX_5X5_F(oh1, 0, 5);  TAP_SCALAR_DX_5X5_F(oh1, 1, 6);  TAP_SCALAR_DX_5X5_F(oh1, 2, 7);  TAP_SCALAR_DX_5X5_F(oh1, 3, 8);  TAP_SCALAR_DX_5X5_F(oh1, 4, 9);
                                    TAP_SCALAR_DX_5X5_F(oh2, 0, 10); TAP_SCALAR_DX_5X5_F(oh2, 1, 11); TAP_SCALAR_DX_5X5_F(oh2, 2, 12); TAP_SCALAR_DX_5X5_F(oh2, 3, 13); TAP_SCALAR_DX_5X5_F(oh2, 4, 14);
                                    TAP_SCALAR_DX_5X5_F(oh3, 0, 15); TAP_SCALAR_DX_5X5_F(oh3, 1, 16); TAP_SCALAR_DX_5X5_F(oh3, 2, 17); TAP_SCALAR_DX_5X5_F(oh3, 3, 18); TAP_SCALAR_DX_5X5_F(oh3, 4, 19);
                                    TAP_SCALAR_DX_5X5_F(oh4, 0, 20); TAP_SCALAR_DX_5X5_F(oh4, 1, 21); TAP_SCALAR_DX_5X5_F(oh4, 2, 22); TAP_SCALAR_DX_5X5_F(oh4, 3, 23); TAP_SCALAR_DX_5X5_F(oh4, 4, 24);
                                    #undef TAP_SCALAR_DX_5X5_F
                                }
                                dx_row0[iw] = sum0;
                                dx_row1[iw] = sum1;
                            }

                            // Vectorized Interior
                            int64_t iw = 2;
                            for (; iw + 8 + 2 <= W_in; iw += 8) {
                                __m256 acc0 = _mm256_setzero_ps();
                                __m256 acc1 = _mm256_setzero_ps();

                                int64_t cout = 0;
                                for (; cout + 2 <= C_out; cout += 2) {
                                    const float* __restrict dp0 = &d_conv_buf[(n * C_out + cout + 0) * conv_spatial];
                                    const float* __restrict dp1 = &d_conv_buf[(n * C_out + cout + 1) * conv_spatial];

                                    const float* __restrict wp0_c0 = &W[((cout + 0) * C_in + cin0 + 0) * 25];
                                    const float* __restrict wp1_c0 = &W[((cout + 1) * C_in + cin0 + 0) * 25];
                                    const float* __restrict wp0_c1 = &W[((cout + 0) * C_in + cin0 + 1) * 25];
                                    const float* __restrict wp1_c1 = &W[((cout + 1) * C_in + cin0 + 1) * 25];

                                    #define TAP_DX_FAST_5X5_F(OH_VAL, KW, W_IDX) { \
                                        if ((OH_VAL) >= 0 && (OH_VAL) < conv_out_h) { \
                                            const float* __restrict dr0 = &dp0[(OH_VAL) * conv_out_w + iw + 2 - (KW)]; \
                                            const float* __restrict dr1 = &dp1[(OH_VAL) * conv_out_w + iw + 2 - (KW)]; \
                                            const __m256 v0 = _mm256_loadu_ps(dr0); \
                                            const __m256 v1 = _mm256_loadu_ps(dr1); \
                                            acc0 = _mm256_fmadd_ps(v0, _mm256_set1_ps(wp0_c0[W_IDX]), acc0); \
                                            acc0 = _mm256_fmadd_ps(v1, _mm256_set1_ps(wp1_c0[W_IDX]), acc0); \
                                            acc1 = _mm256_fmadd_ps(v0, _mm256_set1_ps(wp0_c1[W_IDX]), acc1); \
                                            acc1 = _mm256_fmadd_ps(v1, _mm256_set1_ps(wp1_c1[W_IDX]), acc1); \
                                        } \
                                    }
                                    TAP_DX_FAST_5X5_F(oh0, 0, 0);  TAP_DX_FAST_5X5_F(oh0, 1, 1);  TAP_DX_FAST_5X5_F(oh0, 2, 2);  TAP_DX_FAST_5X5_F(oh0, 3, 3);  TAP_DX_FAST_5X5_F(oh0, 4, 4);
                                    TAP_DX_FAST_5X5_F(oh1, 0, 5);  TAP_DX_FAST_5X5_F(oh1, 1, 6);  TAP_DX_FAST_5X5_F(oh1, 2, 7);  TAP_DX_FAST_5X5_F(oh1, 3, 8);  TAP_DX_FAST_5X5_F(oh1, 4, 9);
                                    TAP_DX_FAST_5X5_F(oh2, 0, 10); TAP_DX_FAST_5X5_F(oh2, 1, 11); TAP_DX_FAST_5X5_F(oh2, 2, 12); TAP_DX_FAST_5X5_F(oh2, 3, 13); TAP_DX_FAST_5X5_F(oh2, 4, 14);
                                    TAP_DX_FAST_5X5_F(oh3, 0, 15); TAP_DX_FAST_5X5_F(oh3, 1, 16); TAP_DX_FAST_5X5_F(oh3, 2, 17); TAP_DX_FAST_5X5_F(oh3, 3, 18); TAP_DX_FAST_5X5_F(oh3, 4, 19);
                                    TAP_DX_FAST_5X5_F(oh4, 0, 20); TAP_DX_FAST_5X5_F(oh4, 1, 21); TAP_DX_FAST_5X5_F(oh4, 2, 22); TAP_DX_FAST_5X5_F(oh4, 3, 23); TAP_DX_FAST_5X5_F(oh4, 4, 24);
                                    #undef TAP_DX_FAST_5X5_F
                                }

                                for (; cout < C_out; ++cout) {
                                    const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                                    const float* __restrict wp0 = &W[((cout * C_in + cin0 + 0) * 25)];
                                    const float* __restrict wp1 = &W[((cout * C_in + cin0 + 1) * 25)];

                                    #define TAP_DX_FAST_1_5X5_F(OH_VAL, KW, W_IDX) { \
                                        if ((OH_VAL) >= 0 && (OH_VAL) < conv_out_h) { \
                                            const float* __restrict dr = &dp[(OH_VAL) * conv_out_w + iw + 2 - (KW)]; \
                                            const __m256 v = _mm256_loadu_ps(dr); \
                                            acc0 = _mm256_fmadd_ps(v, _mm256_set1_ps(wp0[W_IDX]), acc0); \
                                            acc1 = _mm256_fmadd_ps(v, _mm256_set1_ps(wp1[W_IDX]), acc1); \
                                        } \
                                    }
                                    TAP_DX_FAST_1_5X5_F(oh0, 0, 0);  TAP_DX_FAST_1_5X5_F(oh0, 1, 1);  TAP_DX_FAST_1_5X5_F(oh0, 2, 2);  TAP_DX_FAST_1_5X5_F(oh0, 3, 3);  TAP_DX_FAST_1_5X5_F(oh0, 4, 4);
                                    TAP_DX_FAST_1_5X5_F(oh1, 0, 5);  TAP_DX_FAST_1_5X5_F(oh1, 1, 6);  TAP_DX_FAST_1_5X5_F(oh1, 2, 7);  TAP_DX_FAST_1_5X5_F(oh1, 3, 8);  TAP_DX_FAST_1_5X5_F(oh1, 4, 9);
                                    TAP_DX_FAST_1_5X5_F(oh2, 0, 10); TAP_DX_FAST_1_5X5_F(oh2, 1, 11); TAP_DX_FAST_1_5X5_F(oh2, 2, 12); TAP_DX_FAST_1_5X5_F(oh2, 3, 13); TAP_DX_FAST_1_5X5_F(oh2, 4, 14);
                                    TAP_DX_FAST_1_5X5_F(oh3, 0, 15); TAP_DX_FAST_1_5X5_F(oh3, 1, 16); TAP_DX_FAST_1_5X5_F(oh3, 2, 17); TAP_DX_FAST_1_5X5_F(oh3, 3, 18); TAP_DX_FAST_1_5X5_F(oh3, 4, 19);
                                    TAP_DX_FAST_1_5X5_F(oh4, 0, 20); TAP_DX_FAST_1_5X5_F(oh4, 1, 21); TAP_DX_FAST_1_5X5_F(oh4, 2, 22); TAP_DX_FAST_1_5X5_F(oh4, 3, 23); TAP_DX_FAST_1_5X5_F(oh4, 4, 24);
                                    #undef TAP_DX_FAST_1_5X5_F
                                }

                                _mm256_storeu_ps(&dx_row0[iw], acc0);
                                _mm256_storeu_ps(&dx_row1[iw], acc1);
                            }

                            // Right Border Peel
                            for (; iw < W_in; ++iw) {
                                float sum0 = 0.0f, sum1 = 0.0f;
                                for (int64_t cout = 0; cout < C_out; ++cout) {
                                    const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                                    const float* __restrict wp0 = &W[((cout * C_in + cin0 + 0) * 25)];
                                    const float* __restrict wp1 = &W[((cout * C_in + cin0 + 1) * 25)];

                                    #define TAP_SCALAR_R_5X5_F(OH_VAL, KW, W_IDX) { \
                                        if ((OH_VAL) >= 0 && (OH_VAL) < conv_out_h) { \
                                            const int64_t cur_ow = iw + 2 - (KW); \
                                            if (cur_ow >= 0 && cur_ow < conv_out_w) { \
                                                const float val = dp[(OH_VAL) * conv_out_w + cur_ow]; \
                                                sum0 += val * wp0[W_IDX]; \
                                                sum1 += val * wp1[W_IDX]; \
                                            } \
                                        } \
                                    }
                                    TAP_SCALAR_R_5X5_F(oh0, 0, 0);  TAP_SCALAR_R_5X5_F(oh0, 1, 1);  TAP_SCALAR_R_5X5_F(oh0, 2, 2);  TAP_SCALAR_R_5X5_F(oh0, 3, 3);  TAP_SCALAR_R_5X5_F(oh0, 4, 4);
                                    TAP_SCALAR_R_5X5_F(oh1, 0, 5);  TAP_SCALAR_R_5X5_F(oh1, 1, 6);  TAP_SCALAR_R_5X5_F(oh1, 2, 7);  TAP_SCALAR_R_5X5_F(oh1, 3, 8);  TAP_SCALAR_R_5X5_F(oh1, 4, 9);
                                    TAP_SCALAR_R_5X5_F(oh2, 0, 10); TAP_SCALAR_R_5X5_F(oh2, 1, 11); TAP_SCALAR_R_5X5_F(oh2, 2, 12); TAP_SCALAR_R_5X5_F(oh2, 3, 13); TAP_SCALAR_R_5X5_F(oh2, 4, 14);
                                    TAP_SCALAR_R_5X5_F(oh3, 0, 15); TAP_SCALAR_R_5X5_F(oh3, 1, 16); TAP_SCALAR_R_5X5_F(oh3, 2, 17); TAP_SCALAR_R_5X5_F(oh3, 3, 18); TAP_SCALAR_R_5X5_F(oh3, 4, 19);
                                    TAP_SCALAR_R_5X5_F(oh4, 0, 20); TAP_SCALAR_R_5X5_F(oh4, 1, 21); TAP_SCALAR_R_5X5_F(oh4, 2, 22); TAP_SCALAR_R_5X5_F(oh4, 3, 23); TAP_SCALAR_R_5X5_F(oh4, 4, 24);
                                    #undef TAP_SCALAR_R_5X5_F
                                }
                                dx_row0[iw] = sum0;
                                dx_row1[iw] = sum1;
                            }
                        }
                    } else {
                        for (int64_t ih = 0; ih < H; ++ih) {
                            const int64_t oh0 = ih + 2 - 0;
                            const int64_t oh1 = ih + 2 - 1;
                            const int64_t oh2 = ih + 2 - 2;
                            const int64_t oh3 = ih + 2 - 3;
                            const int64_t oh4 = ih + 2 - 4;

                            float* __restrict dx_row0 = &dx_p0[ih * W_in];

                            for (int64_t iw = 0; iw < 2 && iw < W_in; ++iw) {
                                float sum0 = 0.0f;
                                for (int64_t cout = 0; cout < C_out; ++cout) {
                                    const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                                    const float* __restrict wp0 = &W[((cout * C_in + cin0 + 0) * 25)];

                                    #define TAP_SCALAR_DX_5X5_1(OH_VAL, KW, W_IDX) { \
                                        if ((OH_VAL) >= 0 && (OH_VAL) < conv_out_h) { \
                                            const int64_t cur_ow = iw + 2 - (KW); \
                                            if (cur_ow >= 0 && cur_ow < conv_out_w) { \
                                                const float val = dp[(OH_VAL) * conv_out_w + cur_ow]; \
                                                sum0 += val * wp0[W_IDX]; \
                                            } \
                                        } \
                                    }
                                    TAP_SCALAR_DX_5X5_1(oh0, 0, 0);  TAP_SCALAR_DX_5X5_1(oh0, 1, 1);  TAP_SCALAR_DX_5X5_1(oh0, 2, 2);  TAP_SCALAR_DX_5X5_1(oh0, 3, 3);  TAP_SCALAR_DX_5X5_1(oh0, 4, 4);
                                    TAP_SCALAR_DX_5X5_1(oh1, 0, 5);  TAP_SCALAR_DX_5X5_1(oh1, 1, 6);  TAP_SCALAR_DX_5X5_1(oh1, 2, 7);  TAP_SCALAR_DX_5X5_1(oh1, 3, 8);  TAP_SCALAR_DX_5X5_1(oh1, 4, 9);
                                    TAP_SCALAR_DX_5X5_1(oh2, 0, 10); TAP_SCALAR_DX_5X5_1(oh2, 1, 11); TAP_SCALAR_DX_5X5_1(oh2, 2, 12); TAP_SCALAR_DX_5X5_1(oh2, 3, 13); TAP_SCALAR_DX_5X5_1(oh2, 4, 14);
                                    TAP_SCALAR_DX_5X5_1(oh3, 0, 15); TAP_SCALAR_DX_5X5_1(oh3, 1, 16); TAP_SCALAR_DX_5X5_1(oh3, 2, 17); TAP_SCALAR_DX_5X5_1(oh3, 3, 18); TAP_SCALAR_DX_5X5_1(oh3, 4, 19);
                                    TAP_SCALAR_DX_5X5_1(oh4, 0, 20); TAP_SCALAR_DX_5X5_1(oh4, 1, 21); TAP_SCALAR_DX_5X5_1(oh4, 2, 22); TAP_SCALAR_DX_5X5_1(oh4, 3, 23); TAP_SCALAR_DX_5X5_1(oh4, 4, 24);
                                    #undef TAP_SCALAR_DX_5X5_1
                                }
                                dx_row0[iw] = sum0;
                            }

                            int64_t iw = 2;
                            for (; iw + 8 + 2 <= W_in; iw += 8) {
                                __m256 acc0 = _mm256_setzero_ps();

                                int64_t cout = 0;
                                for (; cout + 2 <= C_out; cout += 2) {
                                    const float* __restrict dp0 = &d_conv_buf[(n * C_out + cout + 0) * conv_spatial];
                                    const float* __restrict dp1 = &d_conv_buf[(n * C_out + cout + 1) * conv_spatial];

                                    const float* __restrict wp0_c0 = &W[((cout + 0) * C_in + cin0 + 0) * 25];
                                    const float* __restrict wp1_c0 = &W[((cout + 1) * C_in + cin0 + 0) * 25];

                                    #define TAP_DX_FAST_5X5_1(OH_VAL, KW, W_IDX) { \
                                        if ((OH_VAL) >= 0 && (OH_VAL) < conv_out_h) { \
                                            const float* __restrict dr0 = &dp0[(OH_VAL) * conv_out_w + iw + 2 - (KW)]; \
                                            const float* __restrict dr1 = &dp1[(OH_VAL) * conv_out_w + iw + 2 - (KW)]; \
                                            const __m256 v0 = _mm256_loadu_ps(dr0); \
                                            const __m256 v1 = _mm256_loadu_ps(dr1); \
                                            acc0 = _mm256_fmadd_ps(v0, _mm256_set1_ps(wp0_c0[W_IDX]), acc0); \
                                            acc0 = _mm256_fmadd_ps(v1, _mm256_set1_ps(wp1_c0[W_IDX]), acc0); \
                                        } \
                                    }
                                    TAP_DX_FAST_5X5_1(oh0, 0, 0);  TAP_DX_FAST_5X5_1(oh0, 1, 1);  TAP_DX_FAST_5X5_1(oh0, 2, 2);  TAP_DX_FAST_5X5_1(oh0, 3, 3);  TAP_DX_FAST_5X5_1(oh0, 4, 4);
                                    TAP_DX_FAST_5X5_1(oh1, 0, 5);  TAP_DX_FAST_5X5_1(oh1, 1, 6);  TAP_DX_FAST_5X5_1(oh1, 2, 7);  TAP_DX_FAST_5X5_1(oh1, 3, 8);  TAP_DX_FAST_5X5_1(oh1, 4, 9);
                                    TAP_DX_FAST_5X5_1(oh2, 0, 10); TAP_DX_FAST_5X5_1(oh2, 1, 11); TAP_DX_FAST_5X5_1(oh2, 2, 12); TAP_DX_FAST_5X5_1(oh2, 3, 13); TAP_DX_FAST_5X5_1(oh2, 4, 14);
                                    TAP_DX_FAST_5X5_1(oh3, 0, 15); TAP_DX_FAST_5X5_1(oh3, 1, 16); TAP_DX_FAST_5X5_1(oh3, 2, 17); TAP_DX_FAST_5X5_1(oh3, 3, 18); TAP_DX_FAST_5X5_1(oh3, 4, 19);
                                    TAP_DX_FAST_5X5_1(oh4, 0, 20); TAP_DX_FAST_5X5_1(oh4, 1, 21); TAP_DX_FAST_5X5_1(oh4, 2, 22); TAP_DX_FAST_5X5_1(oh4, 3, 23); TAP_DX_FAST_5X5_1(oh4, 4, 24);
                                    #undef TAP_DX_FAST_5X5_1
                                }

                                for (; cout < C_out; ++cout) {
                                    const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                                    const float* __restrict wp0 = &W[((cout * C_in + cin0 + 0) * 25)];

                                    #define TAP_DX_FAST_1_5X5_1B(OH_VAL, KW, W_IDX) { \
                                        if ((OH_VAL) >= 0 && (OH_VAL) < conv_out_h) { \
                                            const float* __restrict dr = &dp[(OH_VAL) * conv_out_w + iw + 2 - (KW)]; \
                                            const __m256 v = _mm256_loadu_ps(dr); \
                                            acc0 = _mm256_fmadd_ps(v, _mm256_set1_ps(wp0[W_IDX]), acc0); \
                                        } \
                                    }
                                    TAP_DX_FAST_1_5X5_1B(oh0, 0, 0);  TAP_DX_FAST_1_5X5_1B(oh0, 1, 1);  TAP_DX_FAST_1_5X5_1B(oh0, 2, 2);  TAP_DX_FAST_1_5X5_1B(oh0, 3, 3);  TAP_DX_FAST_1_5X5_1B(oh0, 4, 4);
                                    TAP_DX_FAST_1_5X5_1B(oh1, 0, 5);  TAP_DX_FAST_1_5X5_1B(oh1, 1, 6);  TAP_DX_FAST_1_5X5_1B(oh1, 2, 7);  TAP_DX_FAST_1_5X5_1B(oh1, 3, 8);  TAP_DX_FAST_1_5X5_1B(oh1, 4, 9);
                                    TAP_DX_FAST_1_5X5_1B(oh2, 0, 10); TAP_DX_FAST_1_5X5_1B(oh2, 1, 11); TAP_DX_FAST_1_5X5_1B(oh2, 2, 12); TAP_DX_FAST_1_5X5_1B(oh2, 3, 13); TAP_DX_FAST_1_5X5_1B(oh2, 4, 14);
                                    TAP_DX_FAST_1_5X5_1B(oh3, 0, 15); TAP_DX_FAST_1_5X5_1B(oh3, 1, 16); TAP_DX_FAST_1_5X5_1B(oh3, 2, 17); TAP_DX_FAST_1_5X5_1B(oh3, 3, 18); TAP_DX_FAST_1_5X5_1B(oh3, 4, 19);
                                    TAP_DX_FAST_1_5X5_1B(oh4, 0, 20); TAP_DX_FAST_1_5X5_1B(oh4, 1, 21); TAP_DX_FAST_1_5X5_1B(oh4, 2, 22); TAP_DX_FAST_1_5X5_1B(oh4, 3, 23); TAP_DX_FAST_1_5X5_1B(oh4, 4, 24);
                                    #undef TAP_DX_FAST_1_5X5_1B
                                }

                                _mm256_storeu_ps(&dx_row0[iw], acc0);
                            }

                            for (; iw < W_in; ++iw) {
                                float sum0 = 0.0f;
                                for (int64_t cout = 0; cout < C_out; ++cout) {
                                    const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                                    const float* __restrict wp0 = &W[((cout * C_in + cin0 + 0) * 25)];

                                    #define TAP_SCALAR_R_5X5_1(OH_VAL, KW, W_IDX) { \
                                        if ((OH_VAL) >= 0 && (OH_VAL) < conv_out_h) { \
                                            const int64_t cur_ow = iw + 2 - (KW); \
                                            if (cur_ow >= 0 && cur_ow < conv_out_w) { \
                                                const float val = dp[(OH_VAL) * conv_out_w + cur_ow]; \
                                                sum0 += val * wp0[W_IDX]; \
                                            } \
                                        } \
                                    }
                                    TAP_SCALAR_R_5X5_1(oh0, 0, 0);  TAP_SCALAR_R_5X5_1(oh0, 1, 1);  TAP_SCALAR_R_5X5_1(oh0, 2, 2);  TAP_SCALAR_R_5X5_1(oh0, 3, 3);  TAP_SCALAR_R_5X5_1(oh0, 4, 4);
                                    TAP_SCALAR_R_5X5_1(oh1, 0, 5);  TAP_SCALAR_R_5X5_1(oh1, 1, 6);  TAP_SCALAR_R_5X5_1(oh1, 2, 7);  TAP_SCALAR_R_5X5_1(oh1, 3, 8);  TAP_SCALAR_R_5X5_1(oh1, 4, 9);
                                    TAP_SCALAR_R_5X5_1(oh2, 0, 10); TAP_SCALAR_R_5X5_1(oh2, 1, 11); TAP_SCALAR_R_5X5_1(oh2, 2, 12); TAP_SCALAR_R_5X5_1(oh2, 3, 13); TAP_SCALAR_R_5X5_1(oh2, 4, 14);
                                    TAP_SCALAR_R_5X5_1(oh3, 0, 15); TAP_SCALAR_R_5X5_1(oh3, 1, 16); TAP_SCALAR_R_5X5_1(oh3, 2, 17); TAP_SCALAR_R_5X5_1(oh3, 3, 18); TAP_SCALAR_R_5X5_1(oh3, 4, 19);
                                    TAP_SCALAR_R_5X5_1(oh4, 0, 20); TAP_SCALAR_R_5X5_1(oh4, 1, 21); TAP_SCALAR_R_5X5_1(oh4, 2, 22); TAP_SCALAR_R_5X5_1(oh4, 3, 23); TAP_SCALAR_R_5X5_1(oh4, 4, 24);
                                    #undef TAP_SCALAR_R_5X5_1
                                }
                                dx_row0[iw] = sum0;
                            }
                        }
                    }
                }
            }
        }

        #pragma omp barrier

        // ==========================================
        // 2. dW Weight Gradient Accumulation (Persistent Team)
        // ==========================================
        if (dW && x) {
            #pragma omp for collapse(2) schedule(static)
            for (int64_t cout = 0; cout < C_out; ++cout) {
                for (int64_t cin = 0; cin < C_in; ++cin) {
                    float* __restrict dw_target = &dW[(cout * C_in + cin) * 25];

                    // PASS 1: Rows 0..2
                    {
                        __m256 v_dw00 = _mm256_setzero_ps(), v_dw01 = _mm256_setzero_ps(), v_dw02 = _mm256_setzero_ps(), v_dw03 = _mm256_setzero_ps(), v_dw04 = _mm256_setzero_ps();
                        __m256 v_dw10 = _mm256_setzero_ps(), v_dw11 = _mm256_setzero_ps(), v_dw12 = _mm256_setzero_ps(), v_dw13 = _mm256_setzero_ps(), v_dw14 = _mm256_setzero_ps();
                        __m256 v_dw20 = _mm256_setzero_ps(), v_dw21 = _mm256_setzero_ps(), v_dw22 = _mm256_setzero_ps(), v_dw23 = _mm256_setzero_ps(), v_dw24 = _mm256_setzero_ps();
                        float s_dw[3][5] = {{0.0f}};

                        for (int64_t n = 0; n < N; ++n) {
                            const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                            const float* __restrict xp = &x[(n * C_in + cin) * spatial_in];

                            for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                                const float* __restrict dr = &dp[oh * conv_out_w];
                                const int64_t ih_base = oh - 2;

                                const float* __restrict xr0 = (ih_base + 0 >= 0 && ih_base + 0 < H) ? &xp[(ih_base + 0) * W_in] : nullptr;
                                const float* __restrict xr1 = (ih_base + 1 >= 0 && ih_base + 1 < H) ? &xp[(ih_base + 1) * W_in] : nullptr;
                                const float* __restrict xr2 = (ih_base + 2 >= 0 && ih_base + 2 < H) ? &xp[(ih_base + 2) * W_in] : nullptr;

                                for (int64_t ow = 0; ow < 2 && ow < conv_out_w; ++ow) {
                                    const float d_val = dr[ow];
                                    const int64_t iw_base = ow - 2;
                                    #define ACC_PEEL_L_P1(XR, R_IDX) { \
                                        if (XR) { \
                                            for (int kw = 0; kw < 5; ++kw) { \
                                                const int64_t cur_iw = iw_base + kw; \
                                                if (cur_iw >= 0 && cur_iw < W_in) s_dw[R_IDX][kw] += d_val * (XR)[cur_iw]; \
                                            } \
                                        } \
                                    }
                                    ACC_PEEL_L_P1(xr0, 0); ACC_PEEL_L_P1(xr1, 1); ACC_PEEL_L_P1(xr2, 2);
                                    #undef ACC_PEEL_L_P1
                                }

                                int64_t ow = 2;
                                for (; ow + 8 + 2 <= conv_out_w; ow += 8) {
                                    const __m256 vd = _mm256_loadu_ps(&dr[ow]);
                                    const int64_t iw_base = ow - 2;

                                    if (xr0) {
                                        v_dw00 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr0[iw_base + 0]), v_dw00);
                                        v_dw01 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr0[iw_base + 1]), v_dw01);
                                        v_dw02 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr0[iw_base + 2]), v_dw02);
                                        v_dw03 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr0[iw_base + 3]), v_dw03);
                                        v_dw04 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr0[iw_base + 4]), v_dw04);
                                    }
                                    if (xr1) {
                                        v_dw10 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr1[iw_base + 0]), v_dw10);
                                        v_dw11 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr1[iw_base + 1]), v_dw11);
                                        v_dw12 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr1[iw_base + 2]), v_dw12);
                                        v_dw13 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr1[iw_base + 3]), v_dw13);
                                        v_dw14 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr1[iw_base + 4]), v_dw14);
                                    }
                                    if (xr2) {
                                        v_dw20 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr2[iw_base + 0]), v_dw20);
                                        v_dw21 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr2[iw_base + 1]), v_dw21);
                                        v_dw22 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr2[iw_base + 2]), v_dw22);
                                        v_dw23 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr2[iw_base + 3]), v_dw23);
                                        v_dw24 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr2[iw_base + 4]), v_dw24);
                                    }
                                }

                                for (; ow < conv_out_w; ++ow) {
                                    const float d_val = dr[ow];
                                    const int64_t iw_base = ow - 2;
                                    #define ACC_PEEL_R_P1(XR, R_IDX) { \
                                        if (XR) { \
                                            for (int kw = 0; kw < 5; ++kw) { \
                                                const int64_t cur_iw = iw_base + kw; \
                                                if (cur_iw >= 0 && cur_iw < W_in) s_dw[R_IDX][kw] += d_val * (XR)[cur_iw]; \
                                            } \
                                        } \
                                    }
                                    ACC_PEEL_R_P1(xr0, 0); ACC_PEEL_R_P1(xr1, 1); ACC_PEEL_R_P1(xr2, 2);
                                    #undef ACC_PEEL_R_P1
                                }
                            }
                        }

                        #define REDUCE_SUM_5X5(V, S) { \
                            alignas(32) float b[8]; \
                            _mm256_store_ps(b, V); \
                            (S) += (b[0] + b[1] + b[2] + b[3]) + (b[4] + b[5] + b[6] + b[7]); \
                        }
                        REDUCE_SUM_5X5(v_dw00, s_dw[0][0]); REDUCE_SUM_5X5(v_dw01, s_dw[0][1]); REDUCE_SUM_5X5(v_dw02, s_dw[0][2]); REDUCE_SUM_5X5(v_dw03, s_dw[0][3]); REDUCE_SUM_5X5(v_dw04, s_dw[0][4]);
                        REDUCE_SUM_5X5(v_dw10, s_dw[1][0]); REDUCE_SUM_5X5(v_dw11, s_dw[1][1]); REDUCE_SUM_5X5(v_dw12, s_dw[1][2]); REDUCE_SUM_5X5(v_dw13, s_dw[1][3]); REDUCE_SUM_5X5(v_dw14, s_dw[1][4]);
                        REDUCE_SUM_5X5(v_dw20, s_dw[2][0]); REDUCE_SUM_5X5(v_dw21, s_dw[2][1]); REDUCE_SUM_5X5(v_dw22, s_dw[2][2]); REDUCE_SUM_5X5(v_dw23, s_dw[2][3]); REDUCE_SUM_5X5(v_dw24, s_dw[2][4]);
                        #undef REDUCE_SUM_5X5

                        for (int r = 0; r < 3; ++r) {
                            for (int c = 0; c < 5; ++c) {
                                dw_target[r * 5 + c] = s_dw[r][c] * inv_m;
                            }
                        }
                    }

                    // PASS 2: Rows 3..4
                    {
                        __m256 v_dw30 = _mm256_setzero_ps(), v_dw31 = _mm256_setzero_ps(), v_dw32 = _mm256_setzero_ps(), v_dw33 = _mm256_setzero_ps(), v_dw34 = _mm256_setzero_ps();
                        __m256 v_dw40 = _mm256_setzero_ps(), v_dw41 = _mm256_setzero_ps(), v_dw42 = _mm256_setzero_ps(), v_dw43 = _mm256_setzero_ps(), v_dw44 = _mm256_setzero_ps();
                        float s_dw[2][5] = {{0.0f}};

                        for (int64_t n = 0; n < N; ++n) {
                            const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                            const float* __restrict xp = &x[(n * C_in + cin) * spatial_in];

                            for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                                const float* __restrict dr = &dp[oh * conv_out_w];
                                const int64_t ih_base = oh - 2;

                                const float* __restrict xr3 = (ih_base + 3 >= 0 && ih_base + 3 < H) ? &xp[(ih_base + 3) * W_in] : nullptr;
                                const float* __restrict xr4 = (ih_base + 4 >= 0 && ih_base + 4 < H) ? &xp[(ih_base + 4) * W_in] : nullptr;

                                for (int64_t ow = 0; ow < 2 && ow < conv_out_w; ++ow) {
                                    const float d_val = dr[ow];
                                    const int64_t iw_base = ow - 2;
                                    #define ACC_PEEL_L_P2(XR, R_IDX) { \
                                        if (XR) { \
                                            for (int kw = 0; kw < 5; ++kw) { \
                                                const int64_t cur_iw = iw_base + kw; \
                                                if (cur_iw >= 0 && cur_iw < W_in) s_dw[R_IDX][kw] += d_val * (XR)[cur_iw]; \
                                            } \
                                        } \
                                    }
                                    ACC_PEEL_L_P2(xr3, 0); ACC_PEEL_L_P2(xr4, 1);
                                    #undef ACC_PEEL_L_P2
                                }

                                int64_t ow = 2;
                                for (; ow + 8 + 2 <= conv_out_w; ow += 8) {
                                    const __m256 vd = _mm256_loadu_ps(&dr[ow]);
                                    const int64_t iw_base = ow - 2;

                                    if (xr3) {
                                        v_dw30 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr3[iw_base + 0]), v_dw30);
                                        v_dw31 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr3[iw_base + 1]), v_dw31);
                                        v_dw32 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr3[iw_base + 2]), v_dw32);
                                        v_dw33 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr3[iw_base + 3]), v_dw33);
                                        v_dw34 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr3[iw_base + 4]), v_dw34);
                                    }
                                    if (xr4) {
                                        v_dw40 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr4[iw_base + 0]), v_dw40);
                                        v_dw41 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr4[iw_base + 1]), v_dw41);
                                        v_dw42 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr4[iw_base + 2]), v_dw42);
                                        v_dw43 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr4[iw_base + 3]), v_dw43);
                                        v_dw44 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(&xr4[iw_base + 4]), v_dw44);
                                    }
                                }

                                for (; ow < conv_out_w; ++ow) {
                                    const float d_val = dr[ow];
                                    const int64_t iw_base = ow - 2;
                                    #define ACC_PEEL_R_P2(XR, R_IDX) { \
                                        if (XR) { \
                                            for (int kw = 0; kw < 5; ++kw) { \
                                                const int64_t cur_iw = iw_base + kw; \
                                                if (cur_iw >= 0 && cur_iw < W_in) s_dw[R_IDX][kw] += d_val * (XR)[cur_iw]; \
                                            } \
                                        } \
                                    }
                                    ACC_PEEL_R_P2(xr3, 0); ACC_PEEL_R_P2(xr4, 1);
                                    #undef ACC_PEEL_R_P2
                                }
                            }
                        }

                        #define REDUCE_SUM_5X5_P2(V, S) { \
                            alignas(32) float b[8]; \
                            _mm256_store_ps(b, V); \
                            (S) += (b[0] + b[1] + b[2] + b[3]) + (b[4] + b[5] + b[6] + b[7]); \
                        }
                        REDUCE_SUM_5X5_P2(v_dw30, s_dw[0][0]); REDUCE_SUM_5X5_P2(v_dw31, s_dw[0][1]); REDUCE_SUM_5X5_P2(v_dw32, s_dw[0][2]); REDUCE_SUM_5X5_P2(v_dw33, s_dw[0][3]); REDUCE_SUM_5X5_P2(v_dw34, s_dw[0][4]);
                        REDUCE_SUM_5X5_P2(v_dw40, s_dw[1][0]); REDUCE_SUM_5X5_P2(v_dw41, s_dw[1][1]); REDUCE_SUM_5X5_P2(v_dw42, s_dw[1][2]); REDUCE_SUM_5X5_P2(v_dw43, s_dw[1][3]); REDUCE_SUM_5X5_P2(v_dw44, s_dw[1][4]);
                        #undef REDUCE_SUM_5X5_P2

                        for (int r = 0; r < 2; ++r) {
                            for (int c = 0; c < 5; ++c) {
                                dw_target[(r + 3) * 5 + c] = s_dw[r][c] * inv_m;
                            }
                        }
                    }
                }
            }
        }
    }
}