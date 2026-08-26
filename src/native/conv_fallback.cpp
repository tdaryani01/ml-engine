#include "diagnostics.h"
#include <immintrin.h>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <vector>

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
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t C_out,
    int64_t k_h, int64_t k_w, int64_t stride, int64_t pad, int32_t fuse_relu
) {
    DIAG_INC(fwd_fallback);
    TIME_SCOPE(time_fwd_fallback_ns);
    const int64_t out_h = (H + 2 * pad - k_h) / stride + 1;
    const int64_t out_w = (W_in + 2 * pad - k_w) / stride + 1;
    const int64_t spatial_out = out_h * out_w;
    const int64_t spatial_in  = H * W_in;
    const int64_t k_spatial   = k_h * k_w;
    
    // Native 30x30 padded dimensions (treating padding as part of the image)
    const int64_t H_pad = H + 2 * pad;
    const int64_t W_pad = W_in + 2 * pad;
    const int64_t spatial_pad = H_pad * W_pad;
    const __m256 v_zero = _mm256_setzero_ps();

    #pragma omp parallel
    {
        // Thread-local pre-padded tensor: 30x30 layout initialized to 0.0f ONCE
        std::vector<float> x_padded_slice(spatial_pad, 0.0f);

        #pragma omp for collapse(2) schedule(static)
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t cout = 0; cout < C_out; ++cout) {
                float* __restrict out_ptr = &out[(n * C_out + cout) * spatial_out];
                const __m256 vb = bias ? _mm256_set1_ps(bias[cout]) : v_zero;
                const float s_bias = bias ? bias[cout] : 0.0f;

                // Initialize output grid
                int64_t i = 0;
                for (; i + 8 <= spatial_out; i += 8) {
                    _mm256_storeu_ps(&out_ptr[i], vb);
                }
                for (; i < spatial_out; ++i) {
                    out_ptr[i] = s_bias;
                }

                for (int64_t cin = 0; cin < C_in; ++cin) {
                    const float* __restrict xp = &x[(n * C_in + cin) * spatial_in];
                    const float* __restrict wp = &W[(cout * C_in + cin) * k_spatial];

                    // --- PRE-PAD THE TENSOR ONCE (Treating it as a 30x30 unpadded image) ---
                    std::fill(x_padded_slice.begin(), x_padded_slice.end(), 0.0f);
                    float* __restrict dest_row_ptr = &x_padded_slice[pad * W_pad + pad];
                    for (int64_t h = 0; h < H; ++h) {
                        std::memcpy(dest_row_ptr + h * W_pad, xp + h * W_in, W_in * sizeof(float));
                    }
                    // ---------------------------------------------------------------------

                    for (int64_t oh = 0; oh < out_h; ++oh) {
                        const int64_t ih_base = oh * stride;
                        float* __restrict out_row = &out_ptr[oh * out_w];

                        for (int64_t kh = 0; kh < k_h; ++kh) {
                            // Now we read directly from the 30x30 padded grid with ZERO conditional checks
                            const float* __restrict in_row = &x_padded_slice[(ih_base + kh) * W_pad];
                            for (int64_t kw = 0; kw < k_w; ++kw) {
                                const float w_val = wp[kh * k_w + kw];
                                const __m256 vw = _mm256_set1_ps(w_val);
                                const float* __restrict in_ptr = &in_row[kw];

                                int64_t ow = 0;
                                if (stride == 1) {
                                    for (; ow + 8 <= out_w; ow += 8) {
                                        __m256 v_out = _mm256_loadu_ps(&out_row[ow]);
                                        __m256 v_in  = _mm256_loadu_ps(&in_ptr[ow]);
                                        v_out = _mm256_fmadd_ps(v_in, vw, v_out);
                                        _mm256_storeu_ps(&out_row[ow], v_out);
                                    }
                                }
                                for (; ow < out_w; ++ow) {
                                    out_row[ow] += in_ptr[ow * stride] * w_val;
                                }
                            }
                        }
                    }
                }

                if (fuse_relu) {
                    int64_t i = 0;
                    for (; i + 8 <= spatial_out; i += 8) {
                        __m256 v = _mm256_loadu_ps(&out_ptr[i]);
                        _mm256_storeu_ps(&out_ptr[i], _mm256_max_ps(v, v_zero));
                    }
                    for (; i < spatial_out; ++i) {
                        out_ptr[i] = std::max(0.0f, out_ptr[i]);
                    }
                }
            }
        }
    }
}

void conv2d_backward_fallback_avx2(
    const float* d_conv_buf, const float* x, const float* W,
    float* dx, float* dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t C_out,
    int64_t k_h, int64_t k_w, int64_t stride, int64_t pad, float inv_m
) {
    DIAG_INC(bwd_fallback);
    TIME_SCOPE(time_bwd_fallback_ns);
    const int64_t conv_out_h = (H + 2 * pad - k_h) / stride + 1;
    const int64_t conv_out_w = (W_in + 2 * pad - k_w) / stride + 1;
    const int64_t conv_spatial = conv_out_h * conv_out_w;
    const int64_t spatial_in   = H * W_in;
    const int64_t k_spatial    = k_h * k_w;
    const int64_t pad_h_bwd    = (k_h - 1) - pad;
    const int64_t pad_w_bwd    = (k_w - 1) - pad;
    const int64_t H_pad_dr     = conv_out_h + 2 * std::max<int64_t>(0, pad_h_bwd);
    const int64_t W_pad_dr     = conv_out_w + 2 * std::max<int64_t>(0, pad_w_bwd);
    const int64_t spatial_pad_dr = H_pad_dr * W_pad_dr;

    // 1. dX Pass with pre-padded gradient tensor
    if (dx && W) {
        std::memset(dx, 0, N * C_in * spatial_in * sizeof(float));

        #pragma omp parallel
        {
            std::vector<float> dr_padded_slice(spatial_pad_dr, 0.0f);

            #pragma omp for collapse(2) schedule(static)
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t cin = 0; cin < C_in; ++cin) {
                    float* __restrict dx_p = &dx[(n * C_in + cin) * spatial_in];

                    for (int64_t cout = 0; cout < C_out; ++cout) {
                        const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                        const float* __restrict wp = &W[(cout * C_in + cin) * k_spatial];

                        // Pre-pad gradient tensor once
                        std::fill(dr_padded_slice.begin(), dr_padded_slice.end(), 0.0f);
                        float* __restrict dest_dr_ptr = &dr_padded_slice[pad_h_bwd * W_pad_dr + pad_w_bwd];
                        for (int64_t h = 0; h < conv_out_h; ++h) {
                            std::memcpy(dest_dr_ptr + h * W_pad_dr, dp + h * conv_out_w, conv_out_w * sizeof(float));
                        }

                        for (int64_t ih = 0; ih < H; ++ih) {
                            float* __restrict dx_row = &dx_p[ih * W_in];

                            for (int64_t kh = 0; kh < k_h; ++kh) {
                                const float* __restrict dr_row = &dr_padded_slice[(ih + (k_h - 1 - kh)) * W_pad_dr];
                                for (int64_t kw = 0; kw < k_w; ++kw) {
                                    const float w_val = wp[kh * k_w + kw];
                                    const __m256 vw = _mm256_set1_ps(w_val);
                                    const float* __restrict in_dr = &dr_row[k_w - 1 - kw];

                                    int64_t iw = 0;
                                    if (stride == 1) {
                                        for (; iw + 8 <= W_in; iw += 8) {
                                            __m256 v_dx = _mm256_loadu_ps(&dx_row[iw]);
                                            __m256 v_dr = _mm256_loadu_ps(&in_dr[iw]);
                                            v_dx = _mm256_fmadd_ps(v_dr, vw, v_dx);
                                            _mm256_storeu_ps(&dx_row[iw], v_dx);
                                        }
                                    }
                                    for (; iw < W_in; ++iw) {
                                        dx_row[iw] += in_dr[iw] * w_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. dW Pass with pre-padded input tensor
    if (dW && x) {
        std::memset(dW, 0, C_out * C_in * k_spatial * sizeof(float));
        const int64_t H_pad_x = H + 2 * pad;
        const int64_t W_pad_x = W_in + 2 * pad;
        const int64_t spatial_pad_x = H_pad_x * W_pad_x;

        #pragma omp parallel
        {
            std::vector<float> x_padded_slice(spatial_pad_x, 0.0f);
            std::vector<float> dw_local(k_spatial, 0.0f);

            #pragma omp for collapse(2) schedule(static)
            for (int64_t cout = 0; cout < C_out; ++cout) {
                for (int64_t cin = 0; cin < C_in; ++cin) {
                    std::fill(dw_local.begin(), dw_local.end(), 0.0f);

                    for (int64_t n = 0; n < N; ++n) {
                        const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                        const float* __restrict xp = &x[(n * C_in + cin) * spatial_in];

                        // Pre-pad input tensor once
                        std::fill(x_padded_slice.begin(), x_padded_slice.end(), 0.0f);
                        float* __restrict dest_xp_ptr = &x_padded_slice[pad * W_pad_x + pad];
                        for (int64_t h = 0; h < H; ++h) {
                            std::memcpy(dest_xp_ptr + h * W_pad_x, xp + h * W_in, W_in * sizeof(float));
                        }

                        for (int64_t kh = 0; kh < k_h; ++kh) {
                            for (int64_t kw = 0; kw < k_w; ++kw) {
                                const int64_t tap_idx = kh * k_w + kw;
                                __m256 v_acc = _mm256_setzero_ps();
                                float s_acc = 0.0f;

                                for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                                    const float* __restrict dr_row = &dp[oh * conv_out_w];
                                    const float* __restrict xr_row = &x_padded_slice[(oh * stride + kh) * W_pad_x + kw];

                                    int64_t ow = 0;
                                    if (stride == 1) {
                                        for (; ow + 8 <= conv_out_w; ow += 8) {
                                            v_acc = _mm256_fmadd_ps(_mm256_loadu_ps(&dr_row[ow]), _mm256_loadu_ps(&xr_row[ow]), v_acc);
                                        }
                                    }
                                    for (; ow < conv_out_w; ++ow) {
                                        s_acc += dr_row[ow] * xr_row[ow * stride];
                                    }
                                }
                                dw_local[tap_idx] += hsum256_ps(v_acc) + s_acc;
                            }
                        }
                    }

                    float* __restrict dw_target = &dW[(cout * C_in + cin) * k_spatial];
                    for (int64_t k = 0; k < k_spatial; ++k) {
                        dw_target[k] = dw_local[k] * inv_m;
                    }
                }
            }
        }
    }
}