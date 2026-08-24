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

EXPORT_API int get_omp_threads() {
    return 4;
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
// 1. FAST CONTIGUOUS IM2COL & COL2IM
// -----------------------------------------------------------------------------
static inline void im2col_kernel(
    const float* __restrict x,
    float* __restrict col,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t k_h, int64_t k_w,
    int64_t stride, int64_t pad
) {
    const int64_t out_h = (H + 2 * pad - k_h) / stride + 1;
    const int64_t out_w = (W + 2 * pad - k_w) / stride + 1;
    const int64_t spatial_out = out_h * out_w;
    const int64_t spatial_k = k_h * k_w;
    const int64_t K = C * spatial_k;

    #pragma omp parallel for num_threads(4) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        float* __restrict col_n = &col[n * spatial_out * K];
        const float* __restrict x_n = &x[n * C * H * W];

        for (int64_t c = 0; c < C; ++c) {
            const float* __restrict x_c = &x_n[c * H * W];
            const int64_t col_c_base = c * spatial_k;

            for (int64_t kh = 0; kh < k_h; ++kh) {
                for (int64_t kw = 0; kw < k_w; ++kw) {
                    const int64_t col_idx = col_c_base + kh * k_w + kw;

                    for (int64_t oh = 0; oh < out_h; ++oh) {
                        const int64_t ih = oh * stride - pad + kh;
                        const int64_t row_base = oh * out_w;

                        if (ih >= 0 && ih < H) {
                            const float* __restrict x_row = &x_c[ih * W];
                            for (int64_t ow = 0; ow < out_w; ++ow) {
                                const int64_t iw = ow * stride - pad + kw;
                                col_n[(row_base + ow) * K + col_idx] = (iw >= 0 && iw < W) ? x_row[iw] : 0.0f;
                            }
                        } else {
                            for (int64_t ow = 0; ow < out_w; ++ow) {
                                col_n[(row_base + ow) * K + col_idx] = 0.0f;
                            }
                        }
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 2. CONVOLUTION FORWARD
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
    const int64_t K = C_in * k_h * k_w;
    const int64_t total_rows = N * spatial_out;

    float* col = (float*)_aligned_malloc(total_rows * K * sizeof(float), 32);
    im2col_kernel(x, col, N, C_in, H, W_in, k_h, k_w, stride, pad);

    #pragma omp parallel for num_threads(4) schedule(static)
    for (int64_t m = 0; m < total_rows; ++m) {
        const int64_t n = m / spatial_out;
        const int64_t sp = m % spatial_out;
        const float* __restrict a_row = &col[m * K];

        for (int64_t cout = 0; cout < C_out; ++cout) {
            float sum = bias ? bias[cout] : 0.0f;
            const float* __restrict w_row = &W[cout * K];

            __m256 acc = _mm256_setzero_ps();
            int64_t k = 0;
            for (; k + 8 <= K; k += 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(&a_row[k]), _mm256_loadu_ps(&w_row[k]), acc);
            }
            alignas(32) float b[8];
            _mm256_store_ps(b, acc);
            sum += (b[0] + b[1] + b[2] + b[3]) + (b[4] + b[5] + b[6] + b[7]);

            for (; k < K; ++k) {
                sum += a_row[k] * w_row[k];
            }

            if (fuse_relu && sum < 0.0f) sum = 0.0f;
            out[(n * C_out + cout) * spatial_out + sp] = sum;
        }
    }

    _aligned_free(col);
}

// -----------------------------------------------------------------------------
// 3. FUSED DIRECT BACKWARD (dx + dW)
// -----------------------------------------------------------------------------
EXPORT_API void direct_conv2d_backward_fused_avx2(
    const float* __restrict dout,
    const float* __restrict x,
    const float* __restrict W,
    const float* __restrict in_act,
    float* __restrict dx,
    float* __restrict dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in,
    int64_t C_out, int64_t k_h, int64_t k_w,
    int64_t stride, int64_t pad,
    float inv_m,
    int32_t fuse_relu
) {
    const int64_t out_h = (H + 2 * pad - k_h) / stride + 1;
    const int64_t out_w = (W_in + 2 * pad - k_w) / stride + 1;
    const __m256 v_zero = _mm256_setzero_ps();

    const int64_t total_in_tasks = N * C_in;
    #pragma omp parallel for num_threads(4) schedule(static)
    for (int64_t task = 0; task < total_in_tasks; ++task) {
        const int64_t n   = task / C_in;
        const int64_t cin = task % C_in;
        float* __restrict dx_plane = &dx[(n * C_in + cin) * H * W_in];
        const float* __restrict act_plane = in_act ? &in_act[(n * C_in + cin) * H * W_in] : nullptr;

        for (int64_t ih = 0; ih < H; ++ih) {
            float* __restrict dx_row = &dx_plane[ih * W_in];
            const float* __restrict act_row = act_plane ? &act_plane[ih * W_in] : nullptr;

            int64_t iw = 0;
            for (; iw + 8 <= W_in; iw += 8) {
                __m256 acc = _mm256_setzero_ps();
                int64_t cout = 0;

                for (; cout + 2 <= C_out; cout += 2) {
                    const float* __restrict d_p0 = &dout[(n * C_out + cout) * out_h * out_w];
                    const float* __restrict d_p1 = &dout[(n * C_out + cout + 1) * out_h * out_w];
                    const float* __restrict w_c0 = &W[((cout * C_in + cin) * k_h) * k_w];
                    const float* __restrict w_c1 = &W[(((cout + 1) * C_in + cin) * k_h) * k_w];

                    for (int64_t kh = 0; kh < k_h; ++kh) {
                        const int64_t oh = ih + pad - kh;
                        if (oh < 0 || oh >= out_h) continue;

                        const float* __restrict d_r0 = &d_p0[oh * out_w];
                        const float* __restrict d_r1 = &d_p1[oh * out_w];
                        const float* __restrict w_r0 = &w_c0[kh * k_w];
                        const float* __restrict w_r1 = &w_c1[kh * k_w];

                        for (int64_t kw = 0; kw < k_w; ++kw) {
                            const float w0 = w_r0[kw];
                            const float w1 = w_r1[kw];
                            const int64_t ow = iw + pad - kw;

                            if (stride == 1 && ow >= 0 && (ow + 8) <= out_w) {
                                if (w0 != 0.0f) acc = _mm256_fmadd_ps(_mm256_loadu_ps(&d_r0[ow]), _mm256_set1_ps(w0), acc);
                                if (w1 != 0.0f) acc = _mm256_fmadd_ps(_mm256_loadu_ps(&d_r1[ow]), _mm256_set1_ps(w1), acc);
                            } else {
                                alignas(32) float t0[8] = {0}, t1[8] = {0};
                                for (int s = 0; s < 8; ++s) {
                                    int64_t s_ow = ow + s;
                                    if (s_ow >= 0 && s_ow < out_w) {
                                        t0[s] = d_r0[s_ow];
                                        t1[s] = d_r1[s_ow];
                                    }
                                }
                                if (w0 != 0.0f) acc = _mm256_fmadd_ps(_mm256_load_ps(t0), _mm256_set1_ps(w0), acc);
                                if (w1 != 0.0f) acc = _mm256_fmadd_ps(_mm256_load_ps(t1), _mm256_set1_ps(w1), acc);
                            }
                        }
                    }
                }

                for (; cout < C_out; ++cout) {
                    const float* __restrict d_plane = &dout[(n * C_out + cout) * out_h * out_w];
                    const float* __restrict w_chan  = &W[((cout * C_in + cin) * k_h) * k_w];

                    for (int64_t kh = 0; kh < k_h; ++kh) {
                        const int64_t oh = ih + pad - kh;
                        if (oh < 0 || oh >= out_h) continue;

                        const float* __restrict d_row = &d_plane[oh * out_w];
                        const float* __restrict w_row = &w_chan[kh * k_w];

                        for (int64_t kw = 0; kw < k_w; ++kw) {
                            const float w_val = w_row[kw];
                            if (w_val == 0.0f) continue;
                            const int64_t ow = iw + pad - kw;

                            if (stride == 1 && ow >= 0 && (ow + 8) <= out_w) {
                                acc = _mm256_fmadd_ps(_mm256_loadu_ps(&d_row[ow]), _mm256_set1_ps(w_val), acc);
                            } else {
                                alignas(32) float t_buf[8] = {0};
                                for (int s = 0; s < 8; ++s) {
                                    int64_t s_ow = ow + s;
                                    if (s_ow >= 0 && s_ow < out_w) t_buf[s] = d_row[s_ow];
                                }
                                acc = _mm256_fmadd_ps(_mm256_load_ps(t_buf), _mm256_set1_ps(w_val), acc);
                            }
                        }
                    }
                }

                if (fuse_relu && act_row) {
                    __m256 mask = _mm256_cmp_ps(_mm256_loadu_ps(&act_row[iw]), v_zero, _CMP_GT_OQ);
                    acc = _mm256_and_ps(acc, mask);
                }

                _mm256_storeu_ps(&dx_row[iw], acc);
            }

            for (; iw < W_in; ++iw) {
                float sum = 0.0f;
                for (int64_t cout = 0; cout < C_out; ++cout) {
                    const float* __restrict d_plane = &dout[(n * C_out + cout) * out_h * out_w];
                    const float* __restrict w_chan  = &W[((cout * C_in + cin) * k_h) * k_w];

                    for (int64_t kh = 0; kh < k_h; ++kh) {
                        const int64_t oh = ih + pad - kh;
                        if (oh < 0 || oh >= out_h) continue;

                        for (int64_t kw = 0; kw < k_w; ++kw) {
                            const int64_t ow = iw + pad - kw;
                            if (ow >= 0 && ow < out_w) {
                                sum += d_plane[oh * out_w + ow] * w_chan[kh * k_w + kw];
                            }
                        }
                    }
                }
                if (fuse_relu && act_row && act_row[iw] <= 0.0f) sum = 0.0f;
                dx_row[iw] = sum;
            }
        }
    }

    const int64_t total_w_tasks = C_out * C_in;
    #pragma omp parallel for num_threads(4) schedule(static)
    for (int64_t task = 0; task < total_w_tasks; ++task) {
        const int64_t cout = task / C_in;
        const int64_t cin  = task % C_in;
        float* __restrict dw_plane = &dW[((cout * C_in + cin) * k_h) * k_w];

        __m256 k_acc[3][3];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                k_acc[i][j] = _mm256_setzero_ps();
            }
        }

        for (int64_t n = 0; n < N; ++n) {
            const float* __restrict d_plane = &dout[(n * C_out + cout) * out_h * out_w];
            const float* __restrict x_plane = &x[(n * C_in + cin) * H * W_in];

            for (int64_t oh = 0; oh < out_h; ++oh) {
                const float* __restrict d_row = &d_plane[oh * out_w];
                const int64_t ih_base = oh * stride - pad;

                int64_t ow = 0;
                for (; ow + 8 <= out_w; ow += 8) {
                    __m256 v_d = _mm256_loadu_ps(&d_row[ow]);
                    const int64_t iw_base = ow * stride - pad;

                    for (int64_t kh = 0; kh < k_h; ++kh) {
                        const int64_t ih = ih_base + kh;
                        if (ih < 0 || ih >= H) continue;

                        const float* __restrict x_row = &x_plane[ih * W_in];
                        for (int64_t kw = 0; kw < k_w; ++kw) {
                            const int64_t iw = iw_base + kw;
                            if (stride == 1 && iw >= 0 && (iw + 8) <= W_in) {
                                k_acc[kh][kw] = _mm256_fmadd_ps(v_d, _mm256_loadu_ps(&x_row[iw]), k_acc[kh][kw]);
                            } else {
                                alignas(32) float t_buf[8] = {0};
                                for (int s = 0; s < 8; ++s) {
                                    int64_t s_iw = iw + s * stride;
                                    if (s_iw >= 0 && s_iw < W_in) t_buf[s] = x_row[s_iw];
                                }
                                k_acc[kh][kw] = _mm256_fmadd_ps(v_d, _mm256_load_ps(t_buf), k_acc[kh][kw]);
                            }
                        }
                    }
                }
            }
        }

        for (int64_t kh = 0; kh < k_h; ++kh) {
            for (int64_t kw = 0; kw < k_w; ++kw) {
                alignas(32) float b[8];
                _mm256_store_ps(b, k_acc[kh][kw]);
                dw_plane[kh * k_w + kw] = ((b[0] + b[1] + b[2] + b[3]) + (b[4] + b[5] + b[6] + b[7])) * inv_m;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 4. MAXPOOLING & ACTIVATIONS (COMPACT uint8_t ARGMAX)
// -----------------------------------------------------------------------------
EXPORT_API void direct_maxpool_forward_avx2(
    const float* __restrict x,
    float* __restrict out,
    uint8_t* __restrict argmax_buf,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t pool_size, int64_t stride
) {
    const int64_t out_h = (H - pool_size) / stride + 1;
    const int64_t out_w = (W - pool_size) / stride + 1;
    const int64_t total_planes = N * C;

    #pragma omp parallel for num_threads(4) schedule(static)
    for (int64_t p = 0; p < total_planes; ++p) {
        const float* x_plane = &x[p * H * W];
        float* out_plane     = &out[p * out_h * out_w];
        uint8_t* mask_plane  = &argmax_buf[p * out_h * out_w];

        for (int64_t oh = 0; oh < out_h; ++oh) {
            const int64_t ih_base = oh * stride;
            for (int64_t ow = 0; ow < out_w; ++ow) {
                const int64_t iw_base = ow * stride;
                float max_val = -1e30f;
                uint8_t best_idx = 0;

                for (int64_t ph = 0; ph < pool_size; ++ph) {
                    const int64_t ih = ih_base + ph;
                    const float* x_row = &x_plane[ih * W];
                    for (int64_t pw = 0; pw < pool_size; ++pw) {
                        const int64_t iw = iw_base + pw;
                        const float val = x_row[iw];
                        if (val > max_val) {
                            max_val = val;
                            best_idx = (uint8_t)(ph * pool_size + pw);
                        }
                    }
                }
                const int64_t out_idx = oh * out_w + ow;
                out_plane[out_idx] = max_val;
                mask_plane[out_idx] = best_idx;
            }
        }
    }
}

EXPORT_API void direct_maxpool_backward_avx2(
    const float* __restrict dout,
    const uint8_t* __restrict argmax_indices,
    float* __restrict dx,
    int64_t N, int64_t C, int64_t out_h, int64_t out_w,
    int64_t in_h, int64_t in_w,
    int64_t pool_size, int64_t stride
) {
    const int64_t total_planes = N * C;

    #pragma omp parallel for num_threads(4) schedule(static)
    for (int64_t p = 0; p < total_planes; ++p) {
        const float* dout_plane = &dout[p * out_h * out_w];
        const uint8_t* mask_plane = &argmax_indices[p * out_h * out_w];
        float* dx_plane = &dx[p * in_h * in_w];

        std::memset(dx_plane, 0, in_h * in_w * sizeof(float));

        for (int64_t oh = 0; oh < out_h; ++oh) {
            const int64_t ih_base = oh * stride;
            for (int64_t ow = 0; ow < out_w; ++ow) {
                const int64_t out_idx = oh * out_w + ow;
                const uint8_t idx = mask_plane[out_idx];
                const int64_t ph = idx / pool_size;
                const int64_t pw = idx % pool_size;
                dx_plane[(ih_base + ph) * in_w + (ow * stride + pw)] += dout_plane[out_idx];
            }
        }
    }
}

EXPORT_API void direct_relu_forward_avx2(float* data, int64_t size) {
    const __m256 v_zero = _mm256_setzero_ps();
    int64_t i = 0;
    #pragma omp parallel for num_threads(4) schedule(static)
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
    #pragma omp parallel for num_threads(4) schedule(static)
    for (i = 0; i <= size - 8; i += 8) {
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
    const float* __restrict dout,
    float* __restrict db,
    int64_t N, int64_t C_out, int64_t out_h, int64_t out_w,
    float inv_m
) {
    const int64_t spatial = out_h * out_w;

    #pragma omp parallel for num_threads(4) schedule(static)
    for (int64_t c = 0; c < C_out; ++c) {
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

// -----------------------------------------------------------------------------
// 5. COMPOSITE CONV BLOCK (FORWARD & FUSED SINGLE-PASS BACKWARD)
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

    direct_maxpool_forward_avx2(
        out_conv_relu, out_pool, argmax_buf,
        N, C_out, conv_out_h, conv_out_w,
        pool_size, pool_stride
    );
}

// Single-pass fused: Unpool + ReLU Gating + Bias Gradient Accumulation
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

    if (db) {
        std::memset(db, 0, C_out * sizeof(float));
    }

    // 1. FUSED UNPOOL + RELU GATE + BIAS SUM
    #pragma omp parallel for num_threads(4) schedule(static)
    for (int64_t cout = 0; cout < C_out; ++cout) {
        float channel_bias_acc = 0.0f;

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
                    const int64_t ch = ih_base + r_off;
                    const int64_t cw = pw * pool_stride + c_off;
                    const int64_t c_idx = ch * conv_out_w + cw;

                    float grad = dp_plane[p_idx];
                    // In-place ReLU gate
                    if (act_plane && act_plane[c_idx] <= 0.0f) {
                        grad = 0.0f;
                    }

                    d_plane[c_idx] += grad;
                    channel_bias_acc += grad;
                }
            }
        }

        if (db) {
            db[cout] = channel_bias_acc * inv_m;
        }
    }

    // 2. BACKWARD CONV2D (dx + dW)
    direct_conv2d_backward_fused_avx2(
        d_conv_buf, x, W, nullptr, dx, dW,
        N, C_in, H, W_in, C_out, k_h, k_w,
        conv_stride, conv_pad, inv_m, 0
    );
}

EXPORT_API void direct_conv2d_backward_weight_avx2(
    const float* __restrict dout,
    const float* __restrict x,
    float* __restrict dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in,
    int64_t C_out, int64_t k_h, int64_t k_w,
    int64_t stride, int64_t pad,
    float inv_m
) {
    (void)dout; (void)x; (void)dW;
    (void)N; (void)C_in; (void)H; (void)W_in;
    (void)C_out; (void)k_h; (void)k_w;
    (void)stride; (void)pad; (void)inv_m;
}

EXPORT_API void direct_conv2d_backward_input_avx2(
    const float* __restrict dout,
    const float* __restrict W,
    const float* __restrict in_act,
    float* __restrict dx,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in,
    int64_t C_out, int64_t k_h, int64_t k_w,
    int64_t stride, int64_t pad,
    int32_t fuse_relu
) {
    (void)dout; (void)W; (void)in_act; (void)dx;
    (void)N; (void)C_in; (void)H; (void)W_in;
    (void)C_out; (void)k_h; (void)k_w;
    (void)stride; (void)pad; (void)fuse_relu;
}