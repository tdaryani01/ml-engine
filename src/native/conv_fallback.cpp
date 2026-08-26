#include "diagnostics.h"
#include <immintrin.h>
#include <cstring>
#include <algorithm>
#include <cstdint>

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
    const int64_t cout_blocks = (C_out + 3) / 4;
    const __m256 v_zero = _mm256_setzero_ps();

    #pragma omp parallel for collapse(3) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t cout_blk = 0; cout_blk < cout_blocks; ++cout_blk) {
            for (int64_t oh = 0; oh < out_h; ++oh) {
                const int64_t cout0 = cout_blk * 4;
                const int64_t c_rem = (C_out - cout0 >= 4) ? 4 : (C_out - cout0);
                const int64_t ih_base = oh * stride - pad;

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
                        const float* __restrict wp0 = &W[((cout0 + 0) * C_in + cin) * k_spatial];
                        const float* __restrict wp1 = (c_rem > 1) ? &W[((cout0 + 1) * C_in + cin) * k_spatial] : nullptr;
                        const float* __restrict wp2 = (c_rem > 2) ? &W[((cout0 + 2) * C_in + cin) * k_spatial] : nullptr;
                        const float* __restrict wp3 = (c_rem > 3) ? &W[((cout0 + 3) * C_in + cin) * k_spatial] : nullptr;

                        for (int64_t kh = 0; kh < k_h; ++kh) {
                            const int64_t cur_ih = ih_base + kh;
                            if (cur_ih < 0 || cur_ih >= H) continue;
                            const float* __restrict in_row = &xp[cur_ih * W_in];

                            for (int64_t kw = 0; kw < k_w; ++kw) {
                                const int64_t cur_iw = iw0 + kw;
                                __m256 vx;
                                if (stride == 1 && cur_iw >= 0 && (cur_iw + 8) <= W_in) {
                                    vx = _mm256_loadu_ps(&in_row[cur_iw]);
                                } else {
                                    alignas(32) float tmp[8] = {0};
                                    for (int s = 0; s < 8; ++s) {
                                        int64_t s_iw = (ow + s) * stride - pad + kw;
                                        if (s_iw >= 0 && s_iw < W_in) tmp[s] = in_row[s_iw];
                                    }
                                    vx = _mm256_load_ps(tmp);
                                }

                                const int64_t w_idx = kh * k_w + kw;
                                acc0 = _mm256_fmadd_ps(vx, _mm256_set1_ps(wp0[w_idx]), acc0);
                                if (c_rem > 1) acc1 = _mm256_fmadd_ps(vx, _mm256_set1_ps(wp1[w_idx]), acc1);
                                if (c_rem > 2) acc2 = _mm256_fmadd_ps(vx, _mm256_set1_ps(wp2[w_idx]), acc2);
                                if (c_rem > 3) acc3 = _mm256_fmadd_ps(vx, _mm256_set1_ps(wp3[w_idx]), acc3);
                            }
                        }
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
                        const float* __restrict wp0 = &W[((cout0 + 0) * C_in + cin) * k_spatial];
                        const float* __restrict wp1 = (c_rem > 1) ? &W[((cout0 + 1) * C_in + cin) * k_spatial] : nullptr;
                        const float* __restrict wp2 = (c_rem > 2) ? &W[((cout0 + 2) * C_in + cin) * k_spatial] : nullptr;
                        const float* __restrict wp3 = (c_rem > 3) ? &W[((cout0 + 3) * C_in + cin) * k_spatial] : nullptr;

                        for (int64_t kh = 0; kh < k_h; ++kh) {
                            const int64_t cur_ih = ih_base + kh;
                            if (cur_ih < 0 || cur_ih >= H) continue;
                            const float* __restrict in_row = &xp[cur_ih * W_in];

                            for (int64_t kw = 0; kw < k_w; ++kw) {
                                const int64_t cur_iw = ow * stride - pad + kw;
                                if (cur_iw >= 0 && cur_iw < W_in) {
                                    const float val = in_row[cur_iw];
                                    const int64_t w_idx = kh * k_w + kw;
                                    s0 += val * wp0[w_idx];
                                    if (c_rem > 1) s1 += val * wp1[w_idx];
                                    if (c_rem > 2) s2 += val * wp2[w_idx];
                                    if (c_rem > 3) s3 += val * wp3[w_idx];
                                }
                            }
                        }
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
    const int64_t cin_blocks   = (C_in + 1) / 2;

    // 1. dX Input Gradient Pass
    if (dx && W) {
        std::memset(dx, 0, N * C_in * spatial_in * sizeof(float));

        #pragma omp parallel for collapse(3) schedule(static)
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t cin_blk = 0; cin_blk < cin_blocks; ++cin_blk) {
                for (int64_t ih = 0; ih < H; ++ih) {
                    const int64_t cin0 = cin_blk * 2;
                    const int64_t cin_rem = (C_in - cin0 >= 2) ? 2 : 1;

                    float* __restrict dx_p0 = &dx[(n * C_in + cin0 + 0) * spatial_in + ih * W_in];
                    float* __restrict dx_p1 = (cin_rem == 2) ? &dx[(n * C_in + cin0 + 1) * spatial_in + ih * W_in] : nullptr;

                    if (cin_rem == 2) {
                        int64_t iw = 0;
                        if (stride == 1) {
                            for (; iw + 8 <= W_in; iw += 8) {
                                __m256 acc0 = _mm256_setzero_ps();
                                __m256 acc1 = _mm256_setzero_ps();

                                for (int64_t cout = 0; cout < C_out; ++cout) {
                                    const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                                    const float* __restrict wp0 = &W[((cout * C_in + cin0 + 0) * k_spatial)];
                                    const float* __restrict wp1 = &W[((cout * C_in + cin0 + 1) * k_spatial)];

                                    for (int64_t kh = 0; kh < k_h; ++kh) {
                                        const int64_t oh = ih + pad - kh;
                                        if (oh < 0 || oh >= conv_out_h) continue;

                                        const float* __restrict dr = &dp[oh * conv_out_w];
                                        for (int64_t kw = 0; kw < k_w; ++kw) {
                                            const int64_t ow_offset = iw + pad - kw;
                                            const int64_t w_idx = kh * k_w + kw;
                                            const __m256 w0 = _mm256_set1_ps(wp0[w_idx]);
                                            const __m256 w1 = _mm256_set1_ps(wp1[w_idx]);

                                            __m256 v;
                                            if (ow_offset >= 0 && (ow_offset + 8) <= conv_out_w) {
                                                v = _mm256_loadu_ps(&dr[ow_offset]);
                                            } else {
                                                alignas(32) float tmp[8] = {0};
                                                for (int s = 0; s < 8; ++s) {
                                                    int64_t cur_ow = ow_offset + s;
                                                    if (cur_ow >= 0 && cur_ow < conv_out_w) tmp[s] = dr[cur_ow];
                                                }
                                                v = _mm256_load_ps(tmp);
                                            }

                                            acc0 = _mm256_fmadd_ps(v, w0, acc0);
                                            acc1 = _mm256_fmadd_ps(v, w1, acc1);
                                        }
                                    }
                                }
                                _mm256_storeu_ps(&dx_p0[iw], acc0);
                                _mm256_storeu_ps(&dx_p1[iw], acc1);
                            }
                        }

                        for (; iw < W_in; ++iw) {
                            float sum0 = 0.0f, sum1 = 0.0f;
                            for (int64_t cout = 0; cout < C_out; ++cout) {
                                const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                                const float* __restrict wp0 = &W[((cout * C_in + cin0 + 0) * k_spatial)];
                                const float* __restrict wp1 = &W[((cout * C_in + cin0 + 1) * k_spatial)];

                                for (int64_t kh = 0; kh < k_h; ++kh) {
                                    const int64_t oh_raw = ih + pad - kh;
                                    if (oh_raw < 0 || (stride == 1 ? false : (oh_raw % stride != 0))) continue;
                                    const int64_t oh = (stride == 1) ? oh_raw : (oh_raw / stride);
                                    if (oh >= conv_out_h) continue;

                                    const float* __restrict dp_row = &dp[oh * conv_out_w];
                                    for (int64_t kw = 0; kw < k_w; ++kw) {
                                        const int64_t ow_raw = iw + pad - kw;
                                        if (ow_raw >= 0 && (stride == 1 ? true : (ow_raw % stride == 0))) {
                                            const int64_t ow = (stride == 1) ? ow_raw : (ow_raw / stride);
                                            if (ow < conv_out_w) {
                                                const float val = dp_row[ow];
                                                const int64_t w_idx = kh * k_w + kw;
                                                sum0 += val * wp0[w_idx];
                                                sum1 += val * wp1[w_idx];
                                            }
                                        }
                                    }
                                }
                            }
                            dx_p0[iw] = sum0;
                            dx_p1[iw] = sum1;
                        }
                    } else {
                        int64_t iw = 0;
                        if (stride == 1) {
                            for (; iw + 8 <= W_in; iw += 8) {
                                __m256 acc0 = _mm256_setzero_ps();

                                for (int64_t cout = 0; cout < C_out; ++cout) {
                                    const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                                    const float* __restrict wp0 = &W[((cout * C_in + cin0 + 0) * k_spatial)];

                                    for (int64_t kh = 0; kh < k_h; ++kh) {
                                        const int64_t oh = ih + pad - kh;
                                        if (oh < 0 || oh >= conv_out_h) continue;

                                        const float* __restrict dr = &dp[oh * conv_out_w];
                                        for (int64_t kw = 0; kw < k_w; ++kw) {
                                            const int64_t ow_offset = iw + pad - kw;
                                            const int64_t w_idx = kh * k_w + kw;
                                            const __m256 w0 = _mm256_set1_ps(wp0[w_idx]);

                                            __m256 v;
                                            if (ow_offset >= 0 && (ow_offset + 8) <= conv_out_w) {
                                                v = _mm256_loadu_ps(&dr[ow_offset]);
                                            } else {
                                                alignas(32) float tmp[8] = {0};
                                                for (int s = 0; s < 8; ++s) {
                                                    int64_t cur_ow = ow_offset + s;
                                                    if (cur_ow >= 0 && cur_ow < conv_out_w) tmp[s] = dr[cur_ow];
                                                }
                                                v = _mm256_load_ps(tmp);
                                            }
                                            acc0 = _mm256_fmadd_ps(v, w0, acc0);
                                        }
                                    }
                                }
                                _mm256_storeu_ps(&dx_p0[iw], acc0);
                            }
                        }

                        for (; iw < W_in; ++iw) {
                            float sum0 = 0.0f;
                            for (int64_t cout = 0; cout < C_out; ++cout) {
                                const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                                const float* __restrict wp0 = &W[((cout * C_in + cin0 + 0) * k_spatial)];

                                for (int64_t kh = 0; kh < k_h; ++kh) {
                                    const int64_t oh_raw = ih + pad - kh;
                                    if (oh_raw < 0 || (stride == 1 ? false : (oh_raw % stride != 0))) continue;
                                    const int64_t oh = (stride == 1) ? oh_raw : (oh_raw / stride);
                                    if (oh >= conv_out_h) continue;

                                    const float* __restrict dp_row = &dp[oh * conv_out_w];
                                    for (int64_t kw = 0; kw < k_w; ++kw) {
                                        const int64_t ow_raw = iw + pad - kw;
                                        if (ow_raw >= 0 && (stride == 1 ? true : (ow_raw % stride == 0))) {
                                            const int64_t ow = (stride == 1) ? ow_raw : (ow_raw / stride);
                                            if (ow < conv_out_w) {
                                                sum0 += dp_row[ow] * wp0[kh * k_w + kw];
                                            }
                                        }
                                    }
                                }
                            }
                            dx_p0[iw] = sum0;
                        }
                    }
                }
            }
        }
    }

    // 2. dW Weight Gradient Pass (Persistent Accumulator per Tap)
    if (dW && x) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t cout = 0; cout < C_out; ++cout) {
            for (int64_t cin = 0; cin < C_in; ++cin) {
                float* __restrict dw_target = &dW[(cout * C_in + cin) * k_spatial];

                for (int64_t k = 0; k < k_spatial; ++k) {
                    const int64_t kh = k / k_w;
                    const int64_t kw = k % k_w;
                    __m256 v_acc = _mm256_setzero_ps();
                    float s_acc = 0.0f;

                    for (int64_t n = 0; n < N; ++n) {
                        const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                        const float* __restrict xp = &x[(n * C_in + cin) * spatial_in];

                        for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                            const int64_t ih = oh * stride - pad + kh;
                            if (ih < 0 || ih >= H) continue;

                            const float* __restrict dr = &dp[oh * conv_out_w];
                            const float* __restrict xr = &xp[ih * W_in];

                            int64_t ow = 0;
                            if (stride == 1) {
                                for (; ow + 8 <= conv_out_w; ow += 8) {
                                    const int64_t iw0 = ow - pad + kw;
                                    if (iw0 >= 0 && (iw0 + 8) <= W_in) {
                                        v_acc = _mm256_fmadd_ps(_mm256_loadu_ps(&dr[ow]), _mm256_loadu_ps(&xr[iw0]), v_acc);
                                    } else {
                                        for (int s = 0; s < 8; ++s) {
                                            const int64_t cur_iw = iw0 + s;
                                            if (cur_iw >= 0 && cur_iw < W_in) s_acc += dr[ow + s] * xr[cur_iw];
                                        }
                                    }
                                }
                            }

                            for (; ow < conv_out_w; ++ow) {
                                const int64_t iw = ow * stride - pad + kw;
                                if (iw >= 0 && iw < W_in) s_acc += dr[ow] * xr[iw];
                            }
                        }
                    }

                    alignas(32) float b[8];
                    _mm256_store_ps(b, v_acc);
                    dw_target[k] = ((b[0]+b[1]+b[2]+b[3]) + (b[4]+b[5]+b[6]+b[7]) + s_acc) * inv_m;
                }
            }
        }
    }
}