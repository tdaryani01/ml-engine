#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <omp.h>

static inline float _mm256_reduce_add_ps(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 v128  = _mm_add_ps(vlow, vhigh);
    __m128 shuf  = _mm_movehdup_ps(v128);
    __m128 sums  = _mm_add_ps(v128, shuf);
    shuf         = _mm_movehl_ps(shuf, sums);
    sums         = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// ========================================================================
// Forward Pass: Write-Once Accumulator + Pointer Stepping + Masked Tails
// ========================================================================
void conv2d_forward_fallback_avx2(
    const float* x, const float* W, const float* bias, float* out,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t k_h, int64_t k_w, int64_t stride, int64_t pad,
    int64_t out_w_stride, int32_t fuse_relu
) {
    // DIAG_INC(fwd_fallback);
    // TIME_SCOPE(time_fwd_fallback_ns);

    const int64_t out_h = (H + 2 * pad - k_h) / stride + 1;
    const int64_t out_w = (W_in + 2 * pad - k_w) / stride + 1;
    const int64_t spatial_out = out_h * out_w_stride;
    const int64_t spatial_in  = H * W_in_stride;
    const int64_t k_spatial   = k_h * k_w;
    
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t cout = 0; cout < C_out; ++cout) {
            float* __restrict out_ptr = &out[(n * C_out + cout) * spatial_out];
            const __m256 vb = bias ? _mm256_set1_ps(bias[cout]) : v_zero;

            // Initialize entire output slice (including padding stride)
            int64_t sp = 0;
            for (; sp + 7 < spatial_out; sp += 8) {
                _mm256_storeu_ps(&out_ptr[sp], vb);
            }
            for (; sp < spatial_out; ++sp) {
                out_ptr[sp] = bias ? bias[cout] : 0.0f;
            }

            for (int64_t kh = 0; kh < k_h; ++kh) {
                for (int64_t kw = 0; kw < k_w; ++kw) {
                    const int64_t iw_base = -pad + kw;
                    
                    // Safely bounds-check the row so we never need a scalar fallback!
                    const int64_t ow_start = (stride == 1) ? std::max((int64_t)0, -iw_base) : 0;
                    const int64_t ow_end   = (stride == 1) ? std::min(out_w, W_in - iw_base) : out_w;
                    const int64_t count    = ow_end - ow_start;
                    
                    if (stride == 1 && count <= 0) continue;
                    const int64_t iw_start = ow_start + iw_base;

                    for (int64_t oh = 0; oh < out_h; ++oh) {
                        const int64_t ih = oh * stride - pad + kh;
                        if (ih < 0 || ih >= H) continue;

                        float* __restrict out_row = &out_ptr[oh * out_w_stride];
                        
                        if (stride == 1) {
                            const float* xp_base_ih = &x[n * C_in * spatial_in + ih * W_in_stride + iw_start];
                            const float* wp_base_k  = &W[cout * C_in * k_spatial + kh * k_w + kw];

                            int64_t i = 0;
                            for (; i + 15 < count; i += 16) {
                                __m256 vo0 = _mm256_loadu_ps(&out_row[ow_start + i]);
                                __m256 vo1 = _mm256_loadu_ps(&out_row[ow_start + i + 8]);

                                const float* in_ptr = xp_base_ih + i;
                                const float* wp_ptr = wp_base_k;

                                for (int64_t cin = 0; cin < C_in; ++cin) {
                                    const __m256 vw = _mm256_set1_ps(*wp_ptr);
                                    __m256 vi0 = _mm256_loadu_ps(in_ptr);
                                    __m256 vi1 = _mm256_loadu_ps(in_ptr + 8);

                                    vo0 = _mm256_fmadd_ps(vi0, vw, vo0);
                                    vo1 = _mm256_fmadd_ps(vi1, vw, vo1);

                                    in_ptr += spatial_in;
                                    wp_ptr += k_spatial;
                                }
                                _mm256_storeu_ps(&out_row[ow_start + i], vo0);
                                _mm256_storeu_ps(&out_row[ow_start + i + 8], vo1);
                            }
                            
                            for (; i + 7 < count; i += 8) {
                                __m256 vo0 = _mm256_loadu_ps(&out_row[ow_start + i]);
                                const float* in_ptr = xp_base_ih + i;
                                const float* wp_ptr = wp_base_k;

                                for (int64_t cin = 0; cin < C_in; ++cin) {
                                    const __m256 vw = _mm256_set1_ps(*wp_ptr);
                                    __m256 vi0 = _mm256_loadu_ps(in_ptr);
                                    vo0 = _mm256_fmadd_ps(vi0, vw, vo0);
                                    
                                    in_ptr += spatial_in;
                                    wp_ptr += k_spatial;
                                }
                                _mm256_storeu_ps(&out_row[ow_start + i], vo0);
                            }
                            
                            // Masked Tail handling - No scalar loop needed!
                            int64_t rem = count - i;
                            if (rem > 0) {
                                __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(rem), v_idx);
                                __m256 vo0 = _mm256_maskload_ps(&out_row[ow_start + i], mask);
                                
                                const float* in_ptr = xp_base_ih + i;
                                const float* wp_ptr = wp_base_k;

                                for (int64_t cin = 0; cin < C_in; ++cin) {
                                    const __m256 vw = _mm256_set1_ps(*wp_ptr);
                                    __m256 vi0 = _mm256_maskload_ps(in_ptr, mask);
                                    vo0 = _mm256_fmadd_ps(vi0, vw, vo0);
                                    
                                    in_ptr += spatial_in;
                                    wp_ptr += k_spatial;
                                }
                                _mm256_maskstore_ps(&out_row[ow_start + i], mask, vo0);
                            }
                        } else {
                            // Stride > 1 Fallback
                            const float* xp_base_ih = &x[n * C_in * spatial_in + ih * W_in_stride];
                            const float* wp_base_k  = &W[cout * C_in * k_spatial + kh * k_w + kw];

                            for (int64_t ow = 0; ow < out_w; ++ow) {
                                const int64_t iw = ow * stride + iw_base;
                                if (iw >= 0 && iw < W_in) {
                                    float out_val = out_row[ow];
                                    const float* in_ptr = xp_base_ih + iw;
                                    const float* wp_ptr = wp_base_k;

                                    for (int64_t cin = 0; cin < C_in; ++cin) {
                                        out_val += (*in_ptr) * (*wp_ptr);
                                        in_ptr += spatial_in;
                                        wp_ptr += k_spatial;
                                    }
                                    out_row[ow] = out_val;
                                }
                            }
                        }
                    }
                }
            }

            // Fuse ReLU inline
            if (fuse_relu) {
                int64_t i = 0;
                for (; i + 7 < spatial_out; i += 8) {
                    __m256 v = _mm256_loadu_ps(&out_ptr[i]);
                    _mm256_storeu_ps(&out_ptr[i], _mm256_max_ps(v, v_zero));
                }
                for (; i < spatial_out; ++i) {
                    out_ptr[i] = std::max(out_ptr[i], 0.0f);
                }
            }
        }
    }
}

// ========================================================================
// Backward Pass: Pointer Stepping + Masked Tails + L1 Accumulators
// ========================================================================
void conv2d_backward_fallback_avx2(
    const float* d_conv_buf, const float* x, const float* W,
    float* dx, float* dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t k_h, int64_t k_w, int64_t stride, int64_t pad,
    int64_t conv_out_w_stride, float inv_m
) {
    const int64_t conv_out_h   = (H + 2 * pad - k_h) / stride + 1;
    const int64_t conv_out_w   = (W_in + 2 * pad - k_w) / stride + 1;
    const int64_t conv_spatial = conv_out_h * conv_out_w_stride;
    const int64_t spatial_in   = H * W_in_stride;
    const int64_t k_spatial    = k_h * k_w;
    
    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    // ------------------------------------------------------------------------
    // dX Backward
    // ------------------------------------------------------------------------
    if (dx && W) {
        std::memset(dx, 0, N * C_in * spatial_in * sizeof(float));

        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t cin = 0; cin < C_in; ++cin) {
                float* __restrict dx_p = &dx[(n * C_in + cin) * spatial_in];

                for (int64_t kh = 0; kh < k_h; ++kh) {
                    for (int64_t kw = 0; kw < k_w; ++kw) {
                        const int64_t iw_base = -pad + kw;
                        const int64_t ow_start = (stride == 1) ? std::max((int64_t)0, -iw_base) : 0;
                        const int64_t ow_end   = (stride == 1) ? std::min(conv_out_w, W_in - iw_base) : conv_out_w;
                        const int64_t count    = ow_end - ow_start;
                        
                        if (stride == 1 && count <= 0) continue;
                        const int64_t iw_start = ow_start + iw_base;

                        for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                            const int64_t ih = oh * stride - pad + kh;
                            if (ih < 0 || ih >= H) continue;

                            float* __restrict dx_row = &dx_p[ih * W_in_stride];

                            if (stride == 1) {
                                const float* dp_base_oh = &d_conv_buf[n * C_out * conv_spatial + oh * conv_out_w_stride + ow_start];
                                const float* wp_base_k  = &W[cin * k_spatial + kh * k_w + kw];

                                int64_t i = 0;
                                for (; i + 15 < count; i += 16) {
                                    __m256 v_dx0 = _mm256_loadu_ps(&dx_row[iw_start + i]);
                                    __m256 v_dx1 = _mm256_loadu_ps(&dx_row[iw_start + i + 8]);

                                    const float* dp_ptr = dp_base_oh + i;
                                    const float* wp_ptr = wp_base_k;

                                    for (int64_t cout = 0; cout < C_out; ++cout) {
                                        const __m256 vw = _mm256_set1_ps(*wp_ptr);
                                        __m256 r0 = _mm256_loadu_ps(dp_ptr);
                                        __m256 r1 = _mm256_loadu_ps(dp_ptr + 8);

                                        v_dx0 = _mm256_fmadd_ps(r0, vw, v_dx0);
                                        v_dx1 = _mm256_fmadd_ps(r1, vw, v_dx1);

                                        dp_ptr += conv_spatial;
                                        wp_ptr += C_in * k_spatial;
                                    }

                                    _mm256_storeu_ps(&dx_row[iw_start + i], v_dx0);
                                    _mm256_storeu_ps(&dx_row[iw_start + i + 8], v_dx1);
                                }
                                
                                for (; i + 7 < count; i += 8) {
                                    __m256 v_dx0 = _mm256_loadu_ps(&dx_row[iw_start + i]);
                                    
                                    const float* dp_ptr = dp_base_oh + i;
                                    const float* wp_ptr = wp_base_k;

                                    for (int64_t cout = 0; cout < C_out; ++cout) {
                                        const __m256 vw = _mm256_set1_ps(*wp_ptr);
                                        __m256 r0 = _mm256_loadu_ps(dp_ptr);
                                        v_dx0 = _mm256_fmadd_ps(r0, vw, v_dx0);
                                        
                                        dp_ptr += conv_spatial;
                                        wp_ptr += C_in * k_spatial;
                                    }
                                    
                                    _mm256_storeu_ps(&dx_row[iw_start + i], v_dx0);
                                }
                                
                                int64_t rem = count - i;
                                if (rem > 0) {
                                    __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(rem), v_idx);
                                    __m256 v_dx0 = _mm256_maskload_ps(&dx_row[iw_start + i], mask);
                                    
                                    const float* dp_ptr = dp_base_oh + i;
                                    const float* wp_ptr = wp_base_k;

                                    for (int64_t cout = 0; cout < C_out; ++cout) {
                                        const __m256 vw = _mm256_set1_ps(*wp_ptr);
                                        __m256 r0 = _mm256_maskload_ps(dp_ptr, mask);
                                        v_dx0 = _mm256_fmadd_ps(r0, vw, v_dx0);
                                        
                                        dp_ptr += conv_spatial;
                                        wp_ptr += C_in * k_spatial;
                                    }
                                    
                                    _mm256_maskstore_ps(&dx_row[iw_start + i], mask, v_dx0);
                                }
                            } else {
                                const float* dp_base_oh = &d_conv_buf[n * C_out * conv_spatial + oh * conv_out_w_stride];
                                const float* wp_base_k  = &W[cin * k_spatial + kh * k_w + kw];

                                for (int64_t ow = 0; ow < conv_out_w; ++ow) {
                                    const int64_t iw = ow * stride + iw_base;
                                    if (iw >= 0 && iw < W_in) {
                                        float dx_val = dx_row[iw];
                                        const float* dp_ptr = dp_base_oh + ow;
                                        const float* wp_ptr = wp_base_k;

                                        for (int64_t cout = 0; cout < C_out; ++cout) {
                                            dx_val += (*dp_ptr) * (*wp_ptr);
                                            dp_ptr += conv_spatial;
                                            wp_ptr += C_in * k_spatial;
                                        }
                                        dx_row[iw] = dx_val;
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
    // dW Backward
    // ------------------------------------------------------------------------
    if (dW && x) {
        std::memset(dW, 0, C_out * C_in * k_spatial * sizeof(float));

        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t cout = 0; cout < C_out; ++cout) {
            for (int64_t cin = 0; cin < C_in; ++cin) {
                float* __restrict dw_target = &dW[(cout * C_in + cin) * k_spatial];
                
                __m256 tap_sums_main[64];
                __m256 tap_sums_tail[64];
                float tap_scalars[64]; 
                
                for(int i = 0; i < 64; ++i) {
                    tap_sums_main[i] = _mm256_setzero_ps();
                    tap_sums_tail[i] = _mm256_setzero_ps();
                    tap_scalars[i]   = 0.0f;
                }

                for (int64_t n = 0; n < N; ++n) {
                    const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                    const float* __restrict xp = &x[(n * C_in + cin) * spatial_in];

                    for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                        const float* __restrict dr_row = &dp[oh * conv_out_w_stride];

                        for (int64_t kh = 0; kh < k_h; ++kh) {
                            const int64_t ih = oh * stride - pad + kh;
                            if (ih < 0 || ih >= H) continue;

                            const float* __restrict xr_row = &xp[ih * W_in_stride];

                            for (int64_t kw = 0; kw < k_w; ++kw) {
                                const int64_t tap_idx = kh * k_w + kw;

                                if (stride == 1) {
                                    const int64_t iw_start = std::max((int64_t)0, -pad + kw);
                                    const int64_t ow_start = iw_start + pad - kw;
                                    const int64_t count = std::min(conv_out_w - ow_start, W_in - iw_start);

                                    if (count <= 0) continue;

                                    int64_t i = 0;
                                    __m256 v_acc = tap_sums_main[tap_idx];
                                    
                                    for (; i + 15 < count; i += 16) {
                                        __m256 r0 = _mm256_loadu_ps(&dr_row[ow_start + i]);
                                        __m256 x0 = _mm256_loadu_ps(&xr_row[iw_start + i]);
                                        v_acc = _mm256_fmadd_ps(r0, x0, v_acc);
                                        
                                        __m256 r1 = _mm256_loadu_ps(&dr_row[ow_start + i + 8]);
                                        __m256 x1 = _mm256_loadu_ps(&xr_row[iw_start + i + 8]);
                                        v_acc = _mm256_fmadd_ps(r1, x1, v_acc);
                                    }
                                    for (; i + 7 < count; i += 8) {
                                        __m256 r0 = _mm256_loadu_ps(&dr_row[ow_start + i]);
                                        __m256 x0 = _mm256_loadu_ps(&xr_row[iw_start + i]);
                                        v_acc = _mm256_fmadd_ps(r0, x0, v_acc);
                                    }
                                    
                                    tap_sums_main[tap_idx] = v_acc;

                                    int64_t rem = count - i;
                                    if (rem > 0) {
                                        __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(rem), v_idx);
                                        __m256 r0 = _mm256_maskload_ps(&dr_row[ow_start + i], mask);
                                        __m256 x0 = _mm256_maskload_ps(&xr_row[iw_start + i], mask);
                                        tap_sums_tail[tap_idx] = _mm256_fmadd_ps(r0, x0, tap_sums_tail[tap_idx]);
                                    }
                                } else {
                                    float row_acc = 0.0f;
                                    for (int64_t ow = 0; ow < conv_out_w; ++ow) {
                                        const int64_t iw = ow * stride - pad + kw;
                                        if (iw >= 0 && iw < W_in) {
                                            row_acc += dr_row[ow] * xr_row[iw];
                                        }
                                    }
                                    tap_scalars[tap_idx] += row_acc;
                                }
                            }
                        }
                    }
                }

                for (int64_t k = 0; k < k_spatial; ++k) {
                    __m256 v_total = _mm256_add_ps(tap_sums_main[k], tap_sums_tail[k]);
                    float tap_sum = _mm256_reduce_add_ps(v_total) + tap_scalars[k];
                    dw_target[k] = tap_sum * inv_m;
                }
            }
        }
    }
}