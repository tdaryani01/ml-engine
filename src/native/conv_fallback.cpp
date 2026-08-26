#include "diagnostics.h"
#include <immintrin.h>
#include <cstring>
#include <algorithm>
#include <cstdint>

static inline float hsum256_ps(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 vsum  = _mm_add_ps(vlow, vhigh);
    vsum = _mm_hadd_ps(vsum, vsum);
    vsum = _mm_hadd_ps(vsum, vsum);
    return _mm_cvtss_f32(vsum);
}

void conv2d_forward_fallback_avx2(
    const float* x, const float* W, const float* bias, float* out,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t k_h, int64_t k_w, int64_t stride, int64_t pad,
    int64_t out_w_stride, int32_t fuse_relu
) {
    DIAG_INC(fwd_fallback);
    TIME_SCOPE(time_fwd_fallback_ns);

    const int64_t out_h = (H + 2 * pad - k_h) / stride + 1;
    const int64_t out_w = (W_in + 2 * pad - k_w) / stride + 1;
    const int64_t spatial_out = out_h * out_w_stride;
    const int64_t spatial_in  = H * W_in_stride;
    const int64_t k_spatial   = k_h * k_w;
    const __m256 v_zero = _mm256_setzero_ps();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t cout = 0; cout < C_out; ++cout) {
            float* __restrict out_ptr = &out[(n * C_out + cout) * spatial_out];
            const __m256 vb = bias ? _mm256_set1_ps(bias[cout]) : v_zero;

            // Initialize entire output slice (including padding stride)
            for (int64_t i = 0; i < spatial_out; i += 8) {
                _mm256_storeu_ps(&out_ptr[i], vb);
            }

            for (int64_t cin = 0; cin < C_in; ++cin) {
                const float* __restrict xp = &x[(n * C_in + cin) * spatial_in];
                const float* __restrict wp = &W[(cout * C_in + cin) * k_spatial];

                for (int64_t oh = 0; oh < out_h; ++oh) {
                    float* __restrict out_row = &out_ptr[oh * out_w_stride];
                    const int64_t ih_base = oh * stride - pad;

                    for (int64_t kh = 0; kh < k_h; ++kh) {
                        const int64_t ih = ih_base + kh;
                        if (ih < 0 || ih >= H) continue;

                        const float* __restrict in_row = &xp[ih * W_in_stride];

                        for (int64_t kw = 0; kw < k_w; ++kw) {
                            const float w_val = wp[kh * k_w + kw];
                            const __m256 vw = _mm256_set1_ps(w_val);
                            const int64_t iw_base = -pad + kw;

                            if (stride == 1) {
                                // Full unmasked vector loop across aligned output width
                                int64_t ow = 0;
                                if (iw_base >= 0) {
                                    const float* __restrict in_ptr = &in_row[iw_base];
                                    for (; ow < out_w_stride; ow += 8) {
                                        __m256 vo = _mm256_loadu_ps(&out_row[ow]);
                                        __m256 vi = _mm256_loadu_ps(&in_ptr[ow]);
                                        _mm256_storeu_ps(&out_row[ow], _mm256_fmadd_ps(vi, vw, vo));
                                    }
                                } else {
                                    // Boundary overlap when iw_base < 0
                                    for (; ow < out_w; ++ow) {
                                        const int64_t iw = ow + iw_base;
                                        if (iw >= 0 && iw < W_in) {
                                            out_row[ow] += in_row[iw] * w_val;
                                        }
                                    }
                                }
                            } else {
                                for (int64_t ow = 0; ow < out_w; ++ow) {
                                    const int64_t iw = ow * stride + iw_base;
                                    if (iw >= 0 && iw < W_in) {
                                        out_row[ow] += in_row[iw] * w_val;
                                    }
                                }
                            }
                        }
                    }
                }

                if (fuse_relu) {
                    for (int64_t i = 0; i < spatial_out; i += 8) {
                        __m256 v = _mm256_loadu_ps(&out_ptr[i]);
                        _mm256_storeu_ps(&out_ptr[i], _mm256_max_ps(v, v_zero));
                    }
                }
            }
        }
    }
}

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <algorithm>

void conv2d_backward_fallback_avx2(
    const float* d_conv_buf, const float* x, const float* W,
    float* dx, float* dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t k_h, int64_t k_w, int64_t stride, int64_t pad,
    int64_t conv_out_w_stride, float inv_m
) {
    const int64_t conv_out_h = (H + 2 * pad - k_h) / stride + 1;
    const int64_t conv_out_w = (W_in + 2 * pad - k_w) / stride + 1;
    const int64_t conv_spatial = conv_out_h * conv_out_w_stride;
    const int64_t spatial_in   = H * W_in_stride;
    const int64_t k_spatial    = k_h * k_w;

    // ------------------------------------------------------------------------
    // 1. dX Backward Pass (AVX2 Vectorized with Clamped Bounds)
    // ------------------------------------------------------------------------
    if (dx && W) {
        std::memset(dx, 0, N * C_in * spatial_in * sizeof(float));

        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t cin = 0; cin < C_in; ++cin) {
                float* __restrict dx_p = &dx[(n * C_in + cin) * spatial_in];

                for (int64_t cout = 0; cout < C_out; ++cout) {
                    const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                    const float* __restrict wp = &W[(cout * C_in + cin) * k_spatial];

                    for (int64_t kh = 0; kh < k_h; ++kh) {
                        for (int64_t kw = 0; kw < k_w; ++kw) {
                            const float w_val = wp[kh * k_w + kw];
                            const __m256 vw = _mm256_set1_ps(w_val);

                            for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                                const int64_t ih = oh * stride - pad + kh;
                                if (ih < 0 || ih >= H) continue;

                                float* __restrict dx_row = &dx_p[ih * W_in_stride];
                                const float* __restrict dr_row = &dp[oh * conv_out_w_stride];

                                // Stride 1 Fast AVX2 Vectorized Path
                                if (stride == 1) {
                                    const int64_t iw_start = std::max((int64_t)0, -pad + kw);
                                    const int64_t ow_start = iw_start + pad - kw;
                                    const int64_t count = std::min(conv_out_w - ow_start, W_in - iw_start);

                                    if (count <= 0) continue;

                                    int64_t i = 0;
                                    for (; i + 7 < count; i += 8) {
                                        __m256 v_dr = _mm256_loadu_ps(&dr_row[ow_start + i]);
                                        __m256 v_dx = _mm256_loadu_ps(&dx_row[iw_start + i]);
                                        _mm256_storeu_ps(&dx_row[iw_start + i], _mm256_fmadd_ps(v_dr, vw, v_dx));
                                    }
                                    for (; i < count; ++i) {
                                        dx_row[iw_start + i] += dr_row[ow_start + i] * w_val;
                                    }
                                } else {
                                    for (int64_t ow = 0; ow < conv_out_w; ++ow) {
                                        const int64_t iw = ow * stride - pad + kw;
                                        if (iw >= 0 && iw < W_in) {
                                            dx_row[iw] += dr_row[ow] * w_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // 2. dW Backward Pass (AVX2 Vectorized Dot-Product Accumulation)
    // ------------------------------------------------------------------------
    if (dW && x) {
        std::memset(dW, 0, C_out * C_in * k_spatial * sizeof(float));

        const __m256 v_inv_m = _mm256_set1_ps(inv_m);

        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t cout = 0; cout < C_out; ++cout) {
            for (int64_t cin = 0; cin < C_in; ++cin) {
                float* __restrict dw_target = &dW[(cout * C_in + cin) * k_spatial];

                for (int64_t kh = 0; kh < k_h; ++kh) {
                    for (int64_t kw = 0; kw < k_w; ++kw) {
                        const int64_t tap_idx = kh * k_w + kw;
                        __m256 v_sum = _mm256_setzero_ps();
                        float scalar_acc = 0.0f;

                        for (int64_t n = 0; n < N; ++n) {
                            const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                            const float* __restrict xp = &x[(n * C_in + cin) * spatial_in];

                            for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                                const int64_t ih = oh * stride - pad + kh;
                                if (ih < 0 || ih >= H) continue;

                                const float* __restrict dr_row = &dp[oh * conv_out_w_stride];
                                const float* __restrict xr_row = &xp[ih * W_in_stride];

                                if (stride == 1) {
                                    const int64_t iw_start = std::max((int64_t)0, -pad + kw);
                                    const int64_t ow_start = iw_start + pad - kw;
                                    const int64_t count = std::min(conv_out_w - ow_start, W_in - iw_start);

                                    if (count <= 0) continue;

                                    int64_t i = 0;
                                    for (; i + 7 < count; i += 8) {
                                        __m256 v_dr = _mm256_loadu_ps(&dr_row[ow_start + i]);
                                        __m256 v_xr = _mm256_loadu_ps(&xr_row[iw_start + i]);
                                        v_sum = _mm256_fmadd_ps(v_dr, v_xr, v_sum);
                                    }
                                    for (; i < count; ++i) {
                                        scalar_acc += dr_row[ow_start + i] * xr_row[iw_start + i];
                                    }
                                } else {
                                    for (int64_t ow = 0; ow < conv_out_w; ++ow) {
                                        const int64_t iw = ow * stride - pad + kw;
                                        if (iw >= 0 && iw < W_in) {
                                            scalar_acc += dr_row[ow] * xr_row[iw];
                                        }
                                    }
                                }
                            }
                        }

                        // Horizontal sum of v_sum
                        __m128 vlow = _mm256_castps256_ps128(v_sum);
                        __m128 vhigh = _mm256_extractf128_ps(v_sum, 1);
                        __m128 v128 = _mm_add_ps(vlow, vhigh);
                        __m128 shuf = _mm_movehdup_ps(v128);
                        __m128 sums = _mm_add_ps(v128, shuf);
                        shuf = _mm_movehl_ps(shuf, sums);
                        sums = _mm_add_ss(sums, shuf);
                        float total_sum = _mm_cvtss_f32(sums) + scalar_acc;

                        dw_target[tap_idx] = total_sum * inv_m;
                    }
                }
            }
        }
    }
}