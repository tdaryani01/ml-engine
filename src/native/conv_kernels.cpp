#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <omp.h>

#if defined(_MSC_VER)
    #define EXPORT_API extern "C" __declspec(dllexport)
#else
    #define EXPORT_API extern "C" __attribute__((visibility("default")))
#endif

// -----------------------------------------------------------------------------
// DIAGNOSTICS
// -----------------------------------------------------------------------------
EXPORT_API int get_omp_threads() {
    int count = 0;
    #pragma omp parallel
    {
        #pragma omp single
        count = omp_get_num_threads();
    }
    return count;
}

// -----------------------------------------------------------------------------
// 1. FORWARD CONVOLUTION (Flattened Multi-Thread Schedule)
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
    const __m256 v_zero = _mm256_setzero_ps();
    const int64_t cout_blocks = C_out / 4;
    const int64_t total_tasks = N * cout_blocks;

    #pragma omp parallel for schedule(static)
    for (int64_t task = 0; task < total_tasks; ++task) {
        const int64_t n = task / cout_blocks;
        const int64_t cb = task % cout_blocks;
        const int64_t cout_base = cb * 4;

        float* out0 = &out[((n * C_out + cout_base + 0) * out_h) * out_w];
        float* out1 = &out[((n * C_out + cout_base + 1) * out_h) * out_w];
        float* out2 = &out[((n * C_out + cout_base + 2) * out_h) * out_w];
        float* out3 = &out[((n * C_out + cout_base + 3) * out_h) * out_w];

        const __m256 v_b0 = bias ? _mm256_set1_ps(bias[cout_base + 0]) : v_zero;
        const __m256 v_b1 = bias ? _mm256_set1_ps(bias[cout_base + 1]) : v_zero;
        const __m256 v_b2 = bias ? _mm256_set1_ps(bias[cout_base + 2]) : v_zero;
        const __m256 v_b3 = bias ? _mm256_set1_ps(bias[cout_base + 3]) : v_zero;

        for (int64_t oh = 0; oh < out_h; ++oh) {
            const int64_t ih_base = oh * stride - pad;
            float* r0 = &out0[oh * out_w];
            float* r1 = &out1[oh * out_w];
            float* r2 = &out2[oh * out_w];
            float* r3 = &out3[oh * out_w];

            int64_t ow = 0;
            for (; ow + 8 <= out_w; ow += 8) {
                __m256 a0 = v_b0, a1 = v_b1, a2 = v_b2, a3 = v_b3;

                for (int64_t cin = 0; cin < C_in; ++cin) {
                    const float* x_chan = &x[(n * C_in + cin) * H * W_in];
                    const float* w0 = &W[(((cout_base + 0) * C_in + cin) * k_h) * k_w];
                    const float* w1 = &W[(((cout_base + 1) * C_in + cin) * k_h) * k_w];
                    const float* w2 = &W[(((cout_base + 2) * C_in + cin) * k_h) * k_w];
                    const float* w3 = &W[(((cout_base + 3) * C_in + cin) * k_h) * k_w];

                    for (int64_t kh = 0; kh < k_h; ++kh) {
                        const int64_t ih = ih_base + kh;
                        if (ih < 0 || ih >= H) continue;

                        const float* x_row = &x_chan[ih * W_in];
                        const int64_t k_off = kh * k_w;

                        for (int64_t kw = 0; kw < k_w; ++kw) {
                            const __m256 vw0 = _mm256_set1_ps(w0[k_off + kw]);
                            const __m256 vw1 = _mm256_set1_ps(w1[k_off + kw]);
                            const __m256 vw2 = _mm256_set1_ps(w2[k_off + kw]);
                            const __m256 vw3 = _mm256_set1_ps(w3[k_off + kw]);

                            const int64_t iw = ow * stride - pad + kw;
                            if (iw >= 0 && (iw + 8) <= W_in && stride == 1) {
                                const __m256 in_vec = _mm256_loadu_ps(&x_row[iw]);
                                a0 = _mm256_fmadd_ps(in_vec, vw0, a0);
                                a1 = _mm256_fmadd_ps(in_vec, vw1, a1);
                                a2 = _mm256_fmadd_ps(in_vec, vw2, a2);
                                a3 = _mm256_fmadd_ps(in_vec, vw3, a3);
                            } else {
                                alignas(32) float t0[8], t1[8], t2[8], t3[8];
                                _mm256_storeu_ps(t0, a0); _mm256_storeu_ps(t1, a1);
                                _mm256_storeu_ps(t2, a2); _mm256_storeu_ps(t3, a3);
                                for (int i = 0; i < 8; ++i) {
                                    int64_t s_iw = (ow + i) * stride - pad + kw;
                                    if (s_iw >= 0 && s_iw < W_in) {
                                        float px = x_row[s_iw];
                                        t0[i] += px * w0[k_off + kw];
                                        t1[i] += px * w1[k_off + kw];
                                        t2[i] += px * w2[k_off + kw];
                                        t3[i] += px * w3[k_off + kw];
                                    }
                                }
                                a0 = _mm256_loadu_ps(t0); a1 = _mm256_loadu_ps(t1);
                                a2 = _mm256_loadu_ps(t2); a3 = _mm256_loadu_ps(t3);
                            }
                        }
                    }
                }

                if (fuse_relu) {
                    a0 = _mm256_max_ps(a0, v_zero);
                    a1 = _mm256_max_ps(a1, v_zero);
                    a2 = _mm256_max_ps(a2, v_zero);
                    a3 = _mm256_max_ps(a3, v_zero);
                }

                _mm256_storeu_ps(&r0[ow], a0);
                _mm256_storeu_ps(&r1[ow], a1);
                _mm256_storeu_ps(&r2[ow], a2);
                _mm256_storeu_ps(&r3[ow], a3);
            }

            for (; ow < out_w; ++ow) {
                for (int c_step = 0; c_step < 4; ++c_step) {
                    int64_t c = cout_base + c_step;
                    float acc = bias ? bias[c] : 0.0f;
                    float* out_target = (c_step == 0) ? r0 : (c_step == 1) ? r1 : (c_step == 2) ? r2 : r3;
                    for (int64_t cin = 0; cin < C_in; ++cin) {
                        const float* x_chan = &x[(n * C_in + cin) * H * W_in];
                        const float* w_chan = &W[((c * C_in + cin) * k_h) * k_w];
                        for (int64_t kh = 0; kh < k_h; ++kh) {
                            int64_t ih = ih_base + kh;
                            if (ih < 0 || ih >= H) continue;
                            for (int64_t kw = 0; kw < k_w; ++kw) {
                                int64_t iw = ow * stride - pad + kw;
                                if (iw >= 0 && iw < W_in) {
                                    acc += x_chan[ih * W_in + iw] * w_chan[kh * k_w + kw];
                                }
                            }
                        }
                    }
                    if (fuse_relu && acc < 0.0f) acc = 0.0f;
                    out_target[ow] = acc;
                }
            }
        }
    }

    // Remainder output channels
    for (int64_t cout = cout_blocks * 4; cout < C_out; ++cout) {
        float b_val = bias ? bias[cout] : 0.0f;
        #pragma omp parallel for schedule(static)
        for (int64_t n = 0; n < N; ++n) {
            float* out_plane = &out[(n * C_out + cout) * out_h * out_w];
            for (int64_t oh = 0; oh < out_h; ++oh) {
                int64_t ih_base = oh * stride - pad;
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    float acc = b_val;
                    for (int64_t cin = 0; cin < C_in; ++cin) {
                        const float* x_chan = &x[(n * C_in + cin) * H * W_in];
                        const float* w_chan = &W[((cout * C_in + cin) * k_h) * k_w];
                        for (int64_t kh = 0; kh < k_h; ++kh) {
                            int64_t ih = ih_base + kh;
                            if (ih < 0 || ih >= H) continue;
                            for (int64_t kw = 0; kw < k_w; ++kw) {
                                int64_t iw = ow * stride - pad + kw;
                                if (iw >= 0 && iw < W_in) {
                                    acc += x_chan[ih * W_in + iw] * w_chan[kh * k_w + kw];
                                }
                            }
                        }
                    }
                    if (fuse_relu && acc < 0.0f) acc = 0.0f;
                    out_plane[oh * out_w + ow] = acc;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 2. PARALLELIZED WEIGHT GRADIENTS (dW) - (Flattened Cout x Cin Task Loop)
// -----------------------------------------------------------------------------
EXPORT_API void direct_conv2d_backward_weight_avx2(
    const float* __restrict dout,
    const float* __restrict x,
    float* __restrict dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in,
    int64_t C_out, int64_t k_h, int64_t k_w,
    int64_t stride, int64_t pad,
    float inv_m
) {
    const int64_t out_h = (H + 2 * pad - k_h) / stride + 1;
    const int64_t out_w = (W_in + 2 * pad - k_w) / stride + 1;
    const int64_t cout_blocks = C_out / 4;
    const int64_t total_tasks = cout_blocks * C_in;

    #pragma omp parallel for schedule(static)
    for (int64_t task = 0; task < total_tasks; ++task) {
        const int64_t cb = task / C_in;
        const int64_t cin = task % C_in;
        const int64_t c0 = cb * 4;

        float* dw0 = &dW[(((c0 + 0) * C_in + cin) * k_h) * k_w];
        float* dw1 = &dW[(((c0 + 1) * C_in + cin) * k_h) * k_w];
        float* dw2 = &dW[(((c0 + 2) * C_in + cin) * k_h) * k_w];
        float* dw3 = &dW[(((c0 + 3) * C_in + cin) * k_h) * k_w];

        for (int64_t kh = 0; kh < k_h; ++kh) {
            for (int64_t kw = 0; kw < k_w; ++kw) {
                const int64_t k_idx = kh * k_w + kw;
                __m256 v0 = _mm256_setzero_ps();
                __m256 v1 = _mm256_setzero_ps();
                __m256 v2 = _mm256_setzero_ps();
                __m256 v3 = _mm256_setzero_ps();
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                for (int64_t n = 0; n < N; ++n) {
                    const float* d0 = &dout[((n * C_out + c0 + 0) * out_h) * out_w];
                    const float* d1 = &dout[((n * C_out + c0 + 1) * out_h) * out_w];
                    const float* d2 = &dout[((n * C_out + c0 + 2) * out_h) * out_w];
                    const float* d3 = &dout[((n * C_out + c0 + 3) * out_h) * out_w];
                    const float* x_chan = &x[(n * C_in + cin) * H * W_in];

                    for (int64_t oh = 0; oh < out_h; ++oh) {
                        const int64_t ih = oh * stride - pad + kh;
                        if (ih < 0 || ih >= H) continue;

                        const float* r_d0 = &d0[oh * out_w];
                        const float* r_d1 = &d1[oh * out_w];
                        const float* r_d2 = &d2[oh * out_w];
                        const float* r_d3 = &d3[oh * out_w];
                        const float* x_row = &x_chan[ih * W_in];

                        int64_t ow = 0;
                        if (stride == 1) {
                            const int64_t iw_base = kw - pad;
                            for (; ow + 8 <= out_w; ow += 8) {
                                int64_t iw = iw_base + ow;
                                if (iw >= 0 && (iw + 8) <= W_in) {
                                    const __m256 in_val = _mm256_loadu_ps(&x_row[iw]);
                                    v0 = _mm256_fmadd_ps(_mm256_loadu_ps(&r_d0[ow]), in_val, v0);
                                    v1 = _mm256_fmadd_ps(_mm256_loadu_ps(&r_d1[ow]), in_val, v1);
                                    v2 = _mm256_fmadd_ps(_mm256_loadu_ps(&r_d2[ow]), in_val, v2);
                                    v3 = _mm256_fmadd_ps(_mm256_loadu_ps(&r_d3[ow]), in_val, v3);
                                } else {
                                    for (int i = 0; i < 8; ++i) {
                                        int64_t s_iw = iw_base + ow + i;
                                        if (s_iw >= 0 && s_iw < W_in) {
                                            float px = x_row[s_iw];
                                            s0 += r_d0[ow + i] * px;
                                            s1 += r_d1[ow + i] * px;
                                            s2 += r_d2[ow + i] * px;
                                            s3 += r_d3[ow + i] * px;
                                        }
                                    }
                                }
                            }
                        }

                        for (; ow < out_w; ++ow) {
                            const int64_t iw = ow * stride - pad + kw;
                            if (iw >= 0 && iw < W_in) {
                                float px = x_row[iw];
                                s0 += r_d0[ow] * px;
                                s1 += r_d1[ow] * px;
                                s2 += r_d2[ow] * px;
                                s3 += r_d3[ow] * px;
                            }
                        }
                    }
                }

                alignas(32) float b0[8], b1[8], b2[8], b3[8];
                _mm256_storeu_ps(b0, v0); _mm256_storeu_ps(b1, v1);
                _mm256_storeu_ps(b2, v2); _mm256_storeu_ps(b3, v3);
                for (int i = 0; i < 8; ++i) {
                    s0 += b0[i]; s1 += b1[i]; s2 += b2[i]; s3 += b3[i];
                }

                dw0[k_idx] = s0 * inv_m;
                dw1[k_idx] = s1 * inv_m;
                dw2[k_idx] = s2 * inv_m;
                dw3[k_idx] = s3 * inv_m;
            }
        }
    }

    for (int64_t cout = cout_blocks * 4; cout < C_out; ++cout) {
        #pragma omp parallel for schedule(static)
        for (int64_t cin = 0; cin < C_in; ++cin) {
            float* dw = &dW[((cout * C_in + cin) * k_h) * k_w];
            for (int64_t kh = 0; kh < k_h; ++kh) {
                for (int64_t kw = 0; kw < k_w; ++kw) {
                    float sum = 0.0f;
                    for (int64_t n = 0; n < N; ++n) {
                        const float* d = &dout[((n * C_out + cout) * out_h) * out_w];
                        const float* x_chan = &x[(n * C_in + cin) * H * W_in];
                        for (int64_t oh = 0; oh < out_h; ++oh) {
                            const int64_t ih = oh * stride - pad + kh;
                            if (ih < 0 || ih >= H) continue;
                            for (int64_t ow = 0; ow < out_w; ++ow) {
                                const int64_t iw = ow * stride - pad + kw;
                                if (iw >= 0 && iw < W_in) {
                                    sum += d[oh * out_w + ow] * x_chan[ih * W_in + iw];
                                }
                            }
                        }
                    }
                    dw[kh * k_w + kw] = sum * inv_m;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 3. PARALLELIZED INPUT GRADIENTS (Flattened N x Cin Task Loop)
// -----------------------------------------------------------------------------
EXPORT_API void direct_conv2d_backward_input_avx2(
    const float* __restrict dout,
    const float* __restrict W,
    float* __restrict dx,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in,
    int64_t C_out, int64_t k_h, int64_t k_w,
    int64_t stride, int64_t pad
) {
    const int64_t out_h = (H + 2 * pad - k_h) / stride + 1;
    const int64_t out_w = (W_in + 2 * pad - k_w) / stride + 1;
    const int64_t total_tasks = N * C_in;

    #pragma omp parallel for schedule(static)
    for (int64_t task = 0; task < total_tasks; ++task) {
        const int64_t n = task / C_in;
        const int64_t cin = task % C_in;

        float* dx_plane = &dx[(n * C_in + cin) * H * W_in];
        std::memset(dx_plane, 0, H * W_in * sizeof(float));

        for (int64_t cout = 0; cout < C_out; ++cout) {
            const float* dout_plane = &dout[(n * C_out + cout) * out_h * out_w];
            const float* w_chan = &W[((cout * C_in + cin) * k_h) * k_w];

            for (int64_t kh = 0; kh < k_h; ++kh) {
                const float* w_row = &w_chan[kh * k_w];

                for (int64_t kw = 0; kw < k_w; ++kw) {
                    const float w_val = w_row[kw];
                    if (w_val == 0.0f) continue;
                    const __m256 v_w = _mm256_set1_ps(w_val);
                    const int64_t iw_base = kw - pad;

                    for (int64_t oh = 0; oh < out_h; ++oh) {
                        const int64_t ih = oh * stride - pad + kh;
                        if (ih < 0 || ih >= H) continue;

                        const float* dout_row = &dout_plane[oh * out_w];
                        float* dx_row = &dx_plane[ih * W_in];

                        int64_t ow = 0;
                        if (stride == 1) {
                            for (; ow + 8 <= out_w; ow += 8) {
                                const int64_t iw = iw_base + ow;
                                if (iw >= 0 && (iw + 8) <= W_in) {
                                    __m256 v_dx = _mm256_loadu_ps(&dx_row[iw]);
                                    __m256 v_dout = _mm256_loadu_ps(&dout_row[ow]);
                                    v_dx = _mm256_fmadd_ps(v_dout, v_w, v_dx);
                                    _mm256_storeu_ps(&dx_row[iw], v_dx);
                                } else {
                                    for (int i = 0; i < 8; ++i) {
                                        const int64_t s_iw = iw_base + ow + i;
                                        if (s_iw >= 0 && s_iw < W_in) {
                                            dx_row[s_iw] += dout_row[ow + i] * w_val;
                                        }
                                    }
                                }
                            }
                        }

                        for (; ow < out_w; ++ow) {
                            const int64_t iw = ow * stride - pad + kw;
                            if (iw >= 0 && iw < W_in) {
                                dx_row[iw] += dout_row[ow] * w_val;
                            }
                        }
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 4. VECTORIZED IN-PLACE RELU
// -----------------------------------------------------------------------------
EXPORT_API void direct_relu_forward_avx2(float* data, int64_t size) {
    const __m256 v_zero = _mm256_setzero_ps();
    int64_t i = 0;
    #pragma omp parallel for schedule(static)
    for (i = 0; i <= size - 8; i += 8) {
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
        __m256 v_dout = _mm256_loadu_ps(&dout[i]);
        __m256 v_act = _mm256_loadu_ps(&in_act[i]);
        __m256 mask = _mm256_cmp_ps(v_act, v_zero, _CMP_GT_OQ);
        _mm256_storeu_ps(&dout[i], _mm256_and_ps(v_dout, mask));
    }
    for (; i < size; ++i) {
        if (in_act[i] <= 0.0f) dout[i] = 0.0f;
    }
}

// -----------------------------------------------------------------------------
// 5. VECTORIZED MAXPOOL (Flattened Planes Task Loop)
// -----------------------------------------------------------------------------
EXPORT_API void direct_maxpool_forward_avx2(
    const float* __restrict x,
    float* __restrict out,
    int64_t* __restrict argmax_buf,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t pool_size, int64_t stride
) {
    const int64_t out_h = (H - pool_size) / stride + 1;
    const int64_t out_w = (W - pool_size) / stride + 1;
    const int64_t total_planes = N * C;

    #pragma omp parallel for schedule(static)
    for (int64_t p = 0; p < total_planes; ++p) {
        const float* x_plane = &x[p * H * W];
        float* out_plane = &out[p * out_h * out_w];
        int64_t* mask_plane = &argmax_buf[p * out_h * out_w * 2];

        for (int64_t oh = 0; oh < out_h; ++oh) {
            const int64_t ih_base = oh * stride;
            for (int64_t ow = 0; ow < out_w; ++ow) {
                const int64_t iw_base = ow * stride;
                float max_val = -1e30f;
                int64_t best_h = 0, best_w = 0;

                for (int64_t ph = 0; ph < pool_size; ++ph) {
                    const int64_t ih = ih_base + ph;
                    const float* x_row = &x_plane[ih * W];
                    for (int64_t pw = 0; pw < pool_size; ++pw) {
                        const int64_t iw = iw_base + pw;
                        const float val = x_row[iw];
                        if (val > max_val) {
                            max_val = val;
                            best_h = ih;
                            best_w = iw;
                        }
                    }
                }
                const int64_t out_idx = oh * out_w + ow;
                out_plane[out_idx] = max_val;
                mask_plane[2 * out_idx + 0] = best_h;
                mask_plane[2 * out_idx + 1] = best_w;
            }
        }
    }
}

EXPORT_API void direct_maxpool_backward_avx2(
    const float* __restrict dout,
    const int64_t* __restrict argmax_indices,
    float* __restrict dx,
    int64_t N, int64_t C, int64_t out_h, int64_t out_w,
    int64_t in_h, int64_t in_w
) {
    const int64_t total_planes = N * C;

    #pragma omp parallel for schedule(static)
    for (int64_t p = 0; p < total_planes; ++p) {
        const float* dout_plane = &dout[p * out_h * out_w];
        const int64_t* mask_plane = &argmax_indices[p * out_h * out_w * 2];
        float* dx_plane = &dx[p * in_h * in_w];

        std::memset(dx_plane, 0, in_h * in_w * sizeof(float));

        for (int64_t i = 0; i < out_h * out_w; ++i) {
            const int64_t h_idx = mask_plane[2 * i];
            const int64_t w_idx = mask_plane[2 * i + 1];
            dx_plane[h_idx * in_w + w_idx] += dout_plane[i];
        }
    }
}

// -----------------------------------------------------------------------------
// 6. DIRECT BIAS GRADIENT ACCUMULATION
// -----------------------------------------------------------------------------
EXPORT_API void direct_bias_backward_avx2(
    const float* __restrict dout,
    float* __restrict db,
    int64_t N, int64_t C_out, int64_t out_h, int64_t out_w,
    float inv_m
) {
    const int64_t spatial = out_h * out_w;

    #pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < C_out; ++c) {
        float sum = 0.0f;
        __m256 v_sum = _mm256_setzero_ps();

        for (int64_t n = 0; n < N; ++n) {
            const float* plane = &dout[(n * C_out + c) * spatial];
            int64_t s = 0;
            for (; s + 8 <= spatial; s += 8) {
                __m256 v = _mm256_loadu_ps(&plane[s]);
                v_sum = _mm256_add_ps(v_sum, v);
            }
            for (; s < spatial; ++s) {
                sum += plane[s];
            }
        }

        alignas(32) float buf[8];
        _mm256_storeu_ps(buf, v_sum);
        for (int i = 0; i < 8; ++i) sum += buf[i];

        db[c] = sum * inv_m;
    }
}