#include "diagnostics.h"
#include <omp.h>
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <atomic>

#ifdef ENABLE_ENGINE_DIAGNOSTICS
KernelTelemetry g_diag = {};
#endif

// -----------------------------------------------------------------------------
// Specialized Kernel Prototypes
// -----------------------------------------------------------------------------
void conv2d_forward_1x1_avx2(
    const float* __restrict x, const float* __restrict W, const float* __restrict bias, float* __restrict out,
    int64_t N, int64_t C_in, int64_t HW, int64_t C_out,
    int64_t stride, int32_t fuse_relu
);
void conv2d_backward_1x1_avx2(
    const float* __restrict dout, const float* __restrict x, const float* __restrict W,
    float* __restrict dx, float* __restrict dW,
    int64_t N, int64_t C_in, int64_t HW, int64_t C_out,
    int64_t stride, float inv_m
);

void conv2d_forward_3x3_avx2(
    const float* __restrict x, const float* __restrict W, const float* __restrict bias, float* __restrict out,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t C_out,
    int64_t stride, int64_t pad, int32_t fuse_relu
);
void conv2d_backward_3x3_avx2(
    const float* __restrict dout, const float* __restrict x, const float* __restrict W,
    float* __restrict dx, float* __restrict dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t C_out,
    int64_t stride, int64_t pad, float inv_m
);

void conv2d_forward_5x5_avx2(
    const float* __restrict x, const float* __restrict W, const float* __restrict bias, float* __restrict out,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t C_out,
    int64_t stride, int64_t pad, int32_t fuse_relu
);
void conv2d_backward_5x5_avx2(
    const float* __restrict dout, const float* __restrict x, const float* __restrict W,
    float* __restrict dx, float* __restrict dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t C_out,
    int64_t stride, int64_t pad, float inv_m
);

void conv2d_forward_fallback_avx2(
    const float* x, const float* W, const float* bias, float* out,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t C_out,
    int64_t k_h, int64_t k_w, int64_t stride, int64_t pad, int32_t fuse_relu
);
void conv2d_backward_fallback_avx2(
    const float* d_conv_buf, const float* x, const float* W,
    float* dx, float* dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t C_out,
    int64_t k_h, int64_t k_w, int64_t stride, int64_t pad, float inv_m
);

// -----------------------------------------------------------------------------
// Telemetry & Route Logging (Prints route decision once per configuration)
// -----------------------------------------------------------------------------
static void log_routing_decision(const char* pass_type, const char* kernel_name, int64_t k_h, int64_t k_w, int64_t stride, int64_t pad) {
    static std::atomic<uint32_t> logged_mask{0};
    uint32_t bit_id = 0;
    if (std::strcmp(kernel_name, "1x1_SPECIALIZED") == 0) bit_id = 1;
    else if (std::strcmp(kernel_name, "3x3_SPECIALIZED") == 0) bit_id = 2;
    else if (std::strcmp(kernel_name, "5x5_SPECIALIZED") == 0) bit_id = 4;
    else bit_id = 8;

    if (std::strcmp(pass_type, "BWD") == 0) bit_id <<= 4;

    if (!(logged_mask.fetch_or(bit_id) & bit_id)) {
        std::printf("[ENGINE_DISPATCH] %s -> %s | Geometry: [%lldx%lld], Stride: %lld, Pad: %lld\n",
                    pass_type, kernel_name, (long long)k_h, (long long)k_w, (long long)stride, (long long)pad);
        std::fflush(stdout);
    }
}

// -----------------------------------------------------------------------------
// Explicit Dispatch Decision Matrix
// -----------------------------------------------------------------------------
static inline void dispatch_forward(
    const float* x, const float* W, const float* bias, float* out,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t C_out,
    int64_t k_h, int64_t k_w, int64_t stride, int64_t pad, int32_t fuse_relu
) {
    if (k_h == 1 && k_w == 1 && pad == 0) {
        log_routing_decision("FWD", "1x1_SPECIALIZED", k_h, k_w, stride, pad);
        conv2d_forward_1x1_avx2(x, W, bias, out, N, C_in, H * W_in, C_out, stride, fuse_relu);
    } else if (k_h == 3 && k_w == 3 && pad == 1) {
        log_routing_decision("FWD", "3x3_SPECIALIZED", k_h, k_w, stride, pad);
        conv2d_forward_3x3_avx2(x, W, bias, out, N, C_in, H, W_in, C_out, stride, pad, fuse_relu);
    } else if (k_h == 5 && k_w == 5 && pad == 2) {
        log_routing_decision("FWD", "5x5_SPECIALIZED", k_h, k_w, stride, pad);
        conv2d_forward_5x5_avx2(x, W, bias, out, N, C_in, H, W_in, C_out, stride, pad, fuse_relu);
    } else {
        log_routing_decision("FWD", "GENERIC_FALLBACK", k_h, k_w, stride, pad);
        conv2d_forward_fallback_avx2(x, W, bias, out, N, C_in, H, W_in, C_out, k_h, k_w, stride, pad, fuse_relu);
    }
}

static inline void dispatch_backward(
    const float* dout, const float* x, const float* W,
    float* dx, float* dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t C_out,
    int64_t k_h, int64_t k_w, int64_t stride, int64_t pad, float inv_m
) {
    if (k_h == 1 && k_w == 1 && pad == 0) {
        log_routing_decision("BWD", "1x1_SPECIALIZED", k_h, k_w, stride, pad);
        conv2d_backward_1x1_avx2(dout, x, W, dx, dW, N, C_in, H * W_in, C_out, stride, inv_m);
    } else if (k_h == 3 && k_w == 3 && pad == 1) {
        log_routing_decision("BWD", "3x3_SPECIALIZED", k_h, k_w, stride, pad);
        conv2d_backward_3x3_avx2(dout, x, W, dx, dW, N, C_in, H, W_in, C_out, stride, pad, inv_m);
    } else if (k_h == 5 && k_w == 5 && pad == 2) {
        log_routing_decision("BWD", "5x5_SPECIALIZED", k_h, k_w, stride, pad);
        conv2d_backward_5x5_avx2(dout, x, W, dx, dW, N, C_in, H, W_in, C_out, stride, pad, inv_m);
    } else {
        log_routing_decision("BWD", "GENERIC_FALLBACK", k_h, k_w, stride, pad);
        conv2d_backward_fallback_avx2(dout, x, W, dx, dW, N, C_in, H, W_in, C_out, k_h, k_w, stride, pad, inv_m);
    }
}

void maxpool2d_backward_avx2(
    const float* dout_pool, const uint8_t* argmax_buf, float* d_conv_buf,
    int64_t N, int64_t C, int64_t pool_h, int64_t pool_w,
    int64_t conv_h, int64_t conv_w, int64_t pool_size, int64_t pool_stride
) {
    const int64_t conv_spatial = conv_h * conv_w;
    const int64_t pool_spatial = pool_h * pool_w;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            float* __restrict dp = &d_conv_buf[(n * C + c) * conv_spatial];
            const float* __restrict dout_p = &dout_pool[(n * C + c) * pool_spatial];
            const uint8_t* __restrict arg_p = &argmax_buf[(n * C + c) * pool_spatial];

            std::memset(dp, 0, conv_spatial * sizeof(float));

            for (int64_t ph = 0; ph < pool_h; ++ph) {
                for (int64_t pw = 0; pw < pool_w; ++pw) {
                    const int64_t p_idx = ph * pool_w + pw;
                    const uint8_t max_idx = arg_p[p_idx];
                    const float grad = dout_p[p_idx];

                    const int64_t kh = max_idx / pool_size;
                    const int64_t kw = max_idx % pool_size;
                    const int64_t ih = ph * pool_stride + kh;
                    const int64_t iw = pw * pool_stride + kw;

                    if (ih >= 0 && ih < conv_h && iw >= 0 && iw < conv_w) {
                        dp[ih * conv_w + iw] += grad;
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Exported C Interfaces
// -----------------------------------------------------------------------------
extern "C" {

__declspec(dllexport) int32_t get_omp_threads(void) {
    return omp_get_max_threads();
}

__declspec(dllexport) void log_engine_runtime_diagnostics(
    void* p1, void* p2, void* p3,
    int64_t i1, int64_t i2, int64_t i3, int64_t i4,
    int64_t i5, int64_t i6, int64_t i7
) {
    (void)p1; (void)p2; (void)p3;
    (void)i1; (void)i2; (void)i3; (void)i4;
    (void)i5; (void)i6; (void)i7;
}

__declspec(dllexport) void direct_conv_block_forward_avx2(
    const float* x,
    const float* W,
    const float* bias,
    float* out_conv,
    float* out_pool,
    uint8_t* argmax_buf,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in,
    int64_t C_out, int64_t k_h, int64_t k_w,
    int64_t conv_stride, int64_t conv_pad,
    int64_t pool_size, int64_t pool_stride
) {
    dispatch_forward(
        x, W, bias, out_conv,
        N, C_in, H, W_in, C_out,
        k_h, k_w, conv_stride, conv_pad, 1
    );

    if (out_pool && argmax_buf) {
        const int64_t conv_out_h = (H + 2 * conv_pad - k_h) / conv_stride + 1;
        const int64_t conv_out_w = (W_in + 2 * conv_pad - k_w) / conv_stride + 1;
        const int64_t pool_out_h = (conv_out_h - pool_size) / pool_stride + 1;
        const int64_t pool_out_w = (conv_out_w - pool_size) / pool_stride + 1;
        const int64_t conv_spatial = conv_out_h * conv_out_w;
        const int64_t pool_spatial = pool_out_h * pool_out_w;

        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C_out; ++c) {
                const float* __restrict cp = &out_conv[(n * C_out + c) * conv_spatial];
                float* __restrict pp = &out_pool[(n * C_out + c) * pool_spatial];
                uint8_t* __restrict ap = &argmax_buf[(n * C_out + c) * pool_spatial];

                for (int64_t ph = 0; ph < pool_out_h; ++ph) {
                    for (int64_t pw = 0; pw < pool_out_w; ++pw) {
                        float max_val = -1e30f;
                        uint8_t max_idx = 0;

                        for (int64_t kh = 0; kh < pool_size; ++kh) {
                            for (int64_t kw = 0; kw < pool_size; ++kw) {
                                const int64_t ih = ph * pool_stride + kh;
                                const int64_t iw = pw * pool_stride + kw;
                                const float val = cp[ih * conv_out_w + iw];
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = (uint8_t)(kh * pool_size + kw);
                                }
                            }
                        }

                        const int64_t p_idx = ph * pool_out_w + pw;
                        pp[p_idx] = max_val;
                        ap[p_idx] = max_idx;
                    }
                }
            }
        }
    }
}

__declspec(dllexport) void direct_conv_block_backward_avx2(
    const float* dout_pool,
    const uint8_t* argmax_buf,
    const float* x,
    const float* W,
    const float* conv_act,
    float* d_conv_buf,
    float* dx_buf,
    float* dW_buf,
    float* db_buf,
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

    if (dout_pool && argmax_buf && d_conv_buf) {
        maxpool2d_backward_avx2(
            dout_pool, argmax_buf, d_conv_buf,
            N, C_out, pool_out_h, pool_out_w,
            conv_out_h, conv_out_w, pool_size, pool_stride
        );
    }

    if (conv_act && d_conv_buf) {
        const int64_t total_conv_elements = N * C_out * conv_spatial;
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < total_conv_elements; ++i) {
            if (conv_act[i] <= 0.0f) {
                d_conv_buf[i] = 0.0f;
            }
        }
    }

    if (db_buf && d_conv_buf) {
        #pragma omp parallel for schedule(static)
        for (int64_t cout = 0; cout < C_out; ++cout) {
            double sum = 0.0;
            for (int64_t n = 0; n < N; ++n) {
                const float* dp = &d_conv_buf[(n * C_out + cout) * conv_spatial];
                for (int64_t s = 0; s < conv_spatial; ++s) {
                    sum += (double)dp[s];
                }
            }
            db_buf[cout] = (float)(sum * (double)inv_m);
        }
    }

    dispatch_backward(
        d_conv_buf, x, W, dx_buf, dW_buf,
        N, C_in, H, W_in, C_out, k_h, k_w,
        conv_stride, conv_pad, inv_m
    );
}

__declspec(dllexport) void direct_conv2d_forward_avx2(
    const float* x, const float* W, const float* bias, float* out,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t C_out,
    int64_t k_h, int64_t k_w, int64_t stride, int64_t pad, int32_t fuse_relu
) {
    dispatch_forward(x, W, bias, out, N, C_in, H, W_in, C_out, k_h, k_w, stride, pad, fuse_relu);
}

__declspec(dllexport) void direct_conv2d_backward_fused_avx2(
    const float* dout, const float* x, const float* W, const float* in_act,
    float* dx_buf, float* dW_buf,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t C_out,
    int64_t k_h, int64_t k_w, int64_t stride, int64_t pad,
    float inv_m, int32_t fuse_relu
) {
    (void)in_act; (void)fuse_relu;
    dispatch_backward(dout, x, W, dx_buf, dW_buf, N, C_in, H, W_in, C_out, k_h, k_w, stride, pad, inv_m);
}

__declspec(dllexport) void direct_conv2d_backward_weight_avx2(
    const float* d_conv_buf, const float* x, float* dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t C_out,
    int64_t k_h, int64_t k_w, int64_t stride, int64_t pad, float inv_m
) {
    dispatch_backward(d_conv_buf, x, nullptr, nullptr, dW, N, C_in, H, W_in, C_out, k_h, k_w, stride, pad, inv_m);
}

__declspec(dllexport) void direct_conv2d_backward_input_avx2(
    const float* d_conv_buf, const float* W, const float* in_act, float* dx,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t C_out,
    int64_t k_h, int64_t k_w, int64_t stride, int64_t pad, int32_t fuse_relu
) {
    (void)in_act; (void)fuse_relu;
    dispatch_backward(d_conv_buf, nullptr, W, dx, nullptr, N, C_in, H, W_in, C_out, k_h, k_w, stride, pad, 1.0f);
}

__declspec(dllexport) void direct_relu_forward_avx2(float* x, int64_t size) {
    const __m256 v_zero = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&x[i]);
        _mm256_storeu_ps(&x[i], _mm256_max_ps(v, v_zero));
    }
    for (; i < size; ++i) {
        x[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

__declspec(dllexport) void direct_relu_backward_avx2(float* dout, const float* in_act, int64_t size) {
    int64_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 vd = _mm256_loadu_ps(&dout[i]);
        __m256 va = _mm256_loadu_ps(&in_act[i]);
        __m256 mask = _mm256_cmp_ps(va, _mm256_setzero_ps(), _CMP_GT_OQ);
        _mm256_storeu_ps(&dout[i], _mm256_and_ps(vd, mask));
    }
    for (; i < size; ++i) {
        if (in_act[i] <= 0.0f) dout[i] = 0.0f;
    }
}

__declspec(dllexport) void direct_maxpool_forward_avx2(
    const float* x, float* out_pool, uint8_t* argmax_buf,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t pool_size, int64_t pool_stride
) {
    const int64_t out_h = (H - pool_size) / pool_stride + 1;
    const int64_t out_w = (W - pool_size) / pool_stride + 1;
    const int64_t in_spatial = H * W;
    const int64_t out_spatial = out_h * out_w;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const float* __restrict xp = &x[(n * C + c) * in_spatial];
            float* __restrict pp = &out_pool[(n * C + c) * out_spatial];
            uint8_t* __restrict ap = &argmax_buf[(n * C + c) * out_spatial];

            for (int64_t ph = 0; ph < out_h; ++ph) {
                for (int64_t pw = 0; pw < out_w; ++pw) {
                    float max_val = -1e30f;
                    uint8_t max_idx = 0;
                    for (int64_t kh = 0; kh < pool_size; ++kh) {
                        for (int64_t kw = 0; kw < pool_size; ++kw) {
                            const float val = xp[(ph * pool_stride + kh) * W + pw * pool_stride + kw];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = (uint8_t)(kh * pool_size + kw);
                            }
                        }
                    }
                    pp[ph * out_w + pw] = max_val;
                    ap[ph * out_w + pw] = max_idx;
                }
            }
        }
    }
}

__declspec(dllexport) void direct_maxpool_backward_avx2(
    const float* dout_pool, const uint8_t* argmax_buf, float* dx_buf,
    int64_t N, int64_t C, int64_t out_h, int64_t out_w,
    int64_t in_h, int64_t in_w, int64_t pool_size, int64_t pool_stride
) {
    maxpool2d_backward_avx2(
        dout_pool, argmax_buf, dx_buf,
        N, C, out_h, out_w, in_h, in_w, pool_size, pool_stride
    );
}

__declspec(dllexport) void direct_bias_backward_avx2(
    const float* dout, float* db,
    int64_t N, int64_t C_out, int64_t out_h, int64_t out_w,
    float inv_m
) {
    const int64_t spatial = out_h * out_w;
    #pragma omp parallel for schedule(static)
    for (int64_t cout = 0; cout < C_out; ++cout) {
        double sum = 0.0;
        for (int64_t n = 0; n < N; ++n) {
            const float* dp = &dout[(n * C_out + cout) * spatial];
            for (int64_t s = 0; s < spatial; ++s) {
                sum += (double)dp[s];
            }
        }
        db[cout] = (float)(sum * (double)inv_m);
    }
}

}