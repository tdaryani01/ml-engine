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

            int64_t sp = 0;
            for (; sp + 7 < spatial_out; sp += 8) {
                _mm256_storeu_ps(&out_ptr[sp], vb);
            }
            for (; sp < spatial_out; ++sp) {
                out_ptr[sp] = bias ? bias[cout] : 0.0f;
            }

            for (int64_t oh = 0; oh < out_h; ++oh) {
                float* __restrict out_row = &out_ptr[oh * out_w_stride];
                const int64_t ih_base = oh * stride - pad;

                if (stride == 1) {
                    const int64_t ow_safe_start = std::min(out_w, pad);
                    const int64_t ow_safe_end   = std::max(ow_safe_start, out_w - pad);

                    // 1. Left Edge (100% Vectorized with Masks)
                    for (int64_t ow = 0; ow < ow_safe_start; ow += 8) {
                        int64_t count = std::min((int64_t)8, ow_safe_start - ow);
                        __m256i out_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(count), v_idx);
                        __m256 vo0 = _mm256_maskload_ps(&out_row[ow], out_mask);
                        
                        const float* xp_ptr = &x[n * C_in * spatial_in];
                        const float* wp_ptr = &W[cout * C_in * k_spatial];
                        for (int64_t cin = 0; cin < C_in; ++cin) {
                            for (int64_t kh = 0; kh < k_h; ++kh) {
                                const int64_t ih = ih_base + kh;
                                if (ih < 0 || ih >= H) continue;
                                const float* in_row = xp_ptr + ih * W_in_stride;
                                const float* w_row  = wp_ptr + kh * k_w;
                                for (int64_t kw = 0; kw < k_w; ++kw) {
                                    const int64_t iw_base_k = ow - pad + kw;
                                    __m256i viw = _mm256_add_epi32(_mm256_set1_epi32(iw_base_k), v_idx);
                                    __m256i m1 = _mm256_cmpgt_epi32(viw, _mm256_set1_epi32(-1));
                                    __m256i m2 = _mm256_cmpgt_epi32(_mm256_set1_epi32(W_in), viw);
                                    __m256i in_mask = _mm256_and_si256(_mm256_and_si256(m1, m2), out_mask);
                                    
                                    __m256 vw = _mm256_set1_ps(w_row[kw]);
                                    __m256 vi0 = _mm256_maskload_ps(&in_row[iw_base_k], in_mask);
                                    vo0 = _mm256_fmadd_ps(vi0, vw, vo0);
                                }
                            }
                            xp_ptr += spatial_in;
                            wp_ptr += k_spatial;
                        }
                        _mm256_maskstore_ps(&out_row[ow], out_mask, vo0);
                    }

                    // 2. Safe Middle (Zero Bounds Checks, Fast Unaligned Loads)
                    int64_t ow = ow_safe_start;
                    for (; ow + 15 < ow_safe_end; ow += 16) {
                        __m256 vo0 = _mm256_loadu_ps(&out_row[ow]);
                        __m256 vo1 = _mm256_loadu_ps(&out_row[ow + 8]);

                        const float* xp_ptr = &x[n * C_in * spatial_in];
                        const float* wp_ptr = &W[cout * C_in * k_spatial];
                        for (int64_t cin = 0; cin < C_in; ++cin) {
                            for (int64_t kh = 0; kh < k_h; ++kh) {
                                const int64_t ih = ih_base + kh;
                                if (ih < 0 || ih >= H) continue;

                                const float* in_row = xp_ptr + ih * W_in_stride;
                                const float* w_row  = wp_ptr + kh * k_w;
                                for (int64_t kw = 0; kw < k_w; ++kw) {
                                    const __m256 vw = _mm256_set1_ps(w_row[kw]);
                                    const int64_t iw = ow - pad + kw;
                                    
                                    __m256 vi0 = _mm256_loadu_ps(&in_row[iw]);
                                    __m256 vi1 = _mm256_loadu_ps(&in_row[iw + 8]);
                                    vo0 = _mm256_fmadd_ps(vi0, vw, vo0);
                                    vo1 = _mm256_fmadd_ps(vi1, vw, vo1);
                                }
                            }
                            xp_ptr += spatial_in;
                            wp_ptr += k_spatial;
                        }
                        _mm256_storeu_ps(&out_row[ow], vo0);
                        _mm256_storeu_ps(&out_row[ow + 8], vo1);
                    }
                    
                    for (; ow + 7 < ow_safe_end; ow += 8) {
                        __m256 vo0 = _mm256_loadu_ps(&out_row[ow]);
                        const float* xp_ptr = &x[n * C_in * spatial_in];
                        const float* wp_ptr = &W[cout * C_in * k_spatial];

                        for (int64_t cin = 0; cin < C_in; ++cin) {
                            for (int64_t kh = 0; kh < k_h; ++kh) {
                                const int64_t ih = ih_base + kh;
                                if (ih < 0 || ih >= H) continue;

                                const float* in_row = xp_ptr + ih * W_in_stride;
                                const float* w_row  = wp_ptr + kh * k_w;
                                for (int64_t kw = 0; kw < k_w; ++kw) {
                                    const __m256 vw = _mm256_set1_ps(w_row[kw]);
                                    const int64_t iw = ow - pad + kw;
                                    
                                    __m256 vi0 = _mm256_loadu_ps(&in_row[iw]);
                                    vo0 = _mm256_fmadd_ps(vi0, vw, vo0);
                                }
                            }
                            xp_ptr += spatial_in;
                            wp_ptr += k_spatial;
                        }
                        _mm256_storeu_ps(&out_row[ow], vo0);
                    }

                    // 3. Right Edge + Tails (100% Vectorized with Masks)
                    for (; ow < out_w; ow += 8) {
                        int64_t count = std::min((int64_t)8, out_w - ow);
                        __m256i out_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(count), v_idx);
                        __m256 vo0 = _mm256_maskload_ps(&out_row[ow], out_mask);
                        
                        const float* xp_ptr = &x[n * C_in * spatial_in];
                        const float* wp_ptr = &W[cout * C_in * k_spatial];
                        for (int64_t cin = 0; cin < C_in; ++cin) {
                            for (int64_t kh = 0; kh < k_h; ++kh) {
                                const int64_t ih = ih_base + kh;
                                if (ih < 0 || ih >= H) continue;
                                const float* in_row = xp_ptr + ih * W_in_stride;
                                const float* w_row  = wp_ptr + kh * k_w;
                                for (int64_t kw = 0; kw < k_w; ++kw) {
                                    const int64_t iw_base_k = ow - pad + kw;
                                    __m256i viw = _mm256_add_epi32(_mm256_set1_epi32(iw_base_k), v_idx);
                                    __m256i m1 = _mm256_cmpgt_epi32(viw, _mm256_set1_epi32(-1));
                                    __m256i m2 = _mm256_cmpgt_epi32(_mm256_set1_epi32(W_in), viw);
                                    __m256i in_mask = _mm256_and_si256(_mm256_and_si256(m1, m2), out_mask);
                                    
                                    __m256 vw = _mm256_set1_ps(w_row[kw]);
                                    __m256 vi0 = _mm256_maskload_ps(&in_row[iw_base_k], in_mask);
                                    vo0 = _mm256_fmadd_ps(vi0, vw, vo0);
                                }
                            }
                            xp_ptr += spatial_in;
                            wp_ptr += k_spatial;
                        }
                        _mm256_maskstore_ps(&out_row[ow], out_mask, vo0);
                    }
                } else {
                    // Stride > 1 Fallback
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        float val = out_row[ow];
                        const float* xp_ptr = &x[n * C_in * spatial_in];
                        const float* wp_ptr = &W[cout * C_in * k_spatial];
                        
                        for (int64_t cin = 0; cin < C_in; ++cin) {
                            for (int64_t kh = 0; kh < k_h; ++kh) {
                                const int64_t ih = ih_base + kh;
                                if (ih >= 0 && ih < H) {
                                    const float* in_row = xp_ptr + ih * W_in_stride;
                                    const float* w_row  = wp_ptr + kh * k_w;
                                    for (int64_t kw = 0; kw < k_w; ++kw) {
                                        const int64_t iw = ow * stride - pad + kw;
                                        if (iw >= 0 && iw < W_in) {
                                            val += in_row[iw] * w_row[kw];
                                        }
                                    }
                                }
                            }
                            xp_ptr += spatial_in;
                            wp_ptr += k_spatial;
                        }
                        out_row[ow] = val;
                    }
                }
            }

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
    // 1. dX Backward
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
    // 2. dW Backward: Register Accumulators (Zero Stack Spilling)
    // ------------------------------------------------------------------------
    if (dW && x) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t cout = 0; cout < C_out; ++cout) {
            for (int64_t cin = 0; cin < C_in; ++cin) {
                float* __restrict dw_target = &dW[(cout * C_in + cin) * k_spatial];
                
                for (int64_t kh = 0; kh < k_h; ++kh) {
                    for (int64_t kw = 0; kw < k_w; ++kw) {
                        const int64_t tap_idx = kh * k_w + kw;
                        
                        if (stride == 1) {
                            const int64_t iw_base = -pad + kw;
                            const int64_t ow_start = std::max((int64_t)0, -iw_base);
                            const int64_t ow_end   = std::min(conv_out_w, W_in - iw_base);
                            const int64_t count    = ow_end - ow_start;
                            
                            if (count <= 0) {
                                dw_target[tap_idx] = 0.0f;
                                continue;
                            }
                            
                            const int64_t iw_start = ow_start + iw_base;

                            __m256 v_acc0 = _mm256_setzero_ps();
                            __m256 v_acc1 = _mm256_setzero_ps();
                            __m256 v_tail = _mm256_setzero_ps();

                            for (int64_t n = 0; n < N; ++n) {
                                const float* dp_n = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                                const float* xp_n = &x[(n * C_in + cin) * spatial_in];

                                for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                                    const int64_t ih = oh * stride - pad + kh;
                                    if (ih < 0 || ih >= H) continue;

                                    const float* dr_row = &dp_n[oh * conv_out_w_stride];
                                    const float* xr_row = &xp_n[ih * W_in_stride];

                                    int64_t i = 0;
                                    for (; i + 15 < count; i += 16) {
                                        __m256 r0 = _mm256_loadu_ps(&dr_row[ow_start + i]);
                                        __m256 x0 = _mm256_loadu_ps(&xr_row[iw_start + i]);
                                        v_acc0 = _mm256_fmadd_ps(r0, x0, v_acc0);
                                        
                                        __m256 r1 = _mm256_loadu_ps(&dr_row[ow_start + i + 8]);
                                        __m256 x1 = _mm256_loadu_ps(&xr_row[iw_start + i + 8]);
                                        v_acc1 = _mm256_fmadd_ps(r1, x1, v_acc1);
                                    }
                                    for (; i + 7 < count; i += 8) {
                                        __m256 r0 = _mm256_loadu_ps(&dr_row[ow_start + i]);
                                        __m256 x0 = _mm256_loadu_ps(&xr_row[iw_start + i]);
                                        v_acc0 = _mm256_fmadd_ps(r0, x0, v_acc0);
                                    }
                                    
                                    int64_t rem = count - i;
                                    if (rem > 0) {
                                        __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(rem), v_idx);
                                        __m256 r0 = _mm256_maskload_ps(&dr_row[ow_start + i], mask);
                                        __m256 x0 = _mm256_maskload_ps(&xr_row[iw_start + i], mask);
                                        v_tail = _mm256_fmadd_ps(r0, x0, v_tail);
                                    }
                                }
                            }
                            
                            __m256 v_total = _mm256_add_ps(v_acc0, _mm256_add_ps(v_acc1, v_tail));
                            dw_target[tap_idx] = _mm256_reduce_add_ps(v_total) * inv_m;

                        } else {
                            float tap_sum = 0.0f;
                            const int64_t iw_base = -pad + kw;

                            for (int64_t n = 0; n < N; ++n) {
                                const float* dp_n = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                                const float* xp_n = &x[(n * C_in + cin) * spatial_in];

                                for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                                    const int64_t ih = oh * stride - pad + kh;
                                    if (ih < 0 || ih >= H) continue;

                                    const float* dr_row = &dp_n[oh * conv_out_w_stride];
                                    const float* xr_row = &xp_n[ih * W_in_stride];

                                    for (int64_t ow = 0; ow < conv_out_w; ++ow) {
                                        const int64_t iw = ow * stride + iw_base;
                                        if (iw >= 0 && iw < W_in) {
                                            tap_sum += dr_row[ow] * xr_row[iw];
                                        }
                                    }
                                }
                            }
                            dw_target[tap_idx] = tap_sum * inv_m;
                        }
                    }
                }
            }
        }
    }
}