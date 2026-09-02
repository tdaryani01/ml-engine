#include "diagnostics.h"
#include <omp.h>
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <stdexcept>

#ifdef ENABLE_ENGINE_DIAGNOSTICS
KernelTelemetry g_diag = {};
#endif

// -----------------------------------------------------------------------------
// Fallback kernel prototypes (sole conv path)
// -----------------------------------------------------------------------------
void conv2d_forward_fallback_avx2(
    const float* x, const float* W, const float* bias, float* out,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t k_h, int64_t k_w, int64_t stride, int64_t pad,
    int64_t out_w_stride, int32_t fuse_relu
);
void conv2d_backward_fallback_avx2(
    const float* d_conv_buf, const float* x, const float* W,
    float* dx, float* dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t k_h, int64_t k_w, int64_t stride, int64_t pad,
    int64_t conv_out_w_stride, float inv_m
);

// -----------------------------------------------------------------------------
// Telemetry & Route Logging
// -----------------------------------------------------------------------------
static void log_routing_decision(const char* pass_type, const char* kernel_name, int64_t k_h, int64_t k_w, int64_t stride, int64_t pad) {
    static std::atomic<uint32_t> logged_mask{0};
    const uint32_t bit_id = (std::strcmp(pass_type, "BWD") == 0) ? 0x10u : 0x01u;

    if (!(logged_mask.fetch_or(bit_id) & bit_id)) {
        std::printf("[ENGINE_DISPATCH] %s -> %s | Geometry: [%lldx%lld], Stride: %lld, Pad: %lld\n",
                    pass_type, kernel_name, (long long)k_h, (long long)k_w, (long long)stride, (long long)pad);
        std::fflush(stdout);
    }
}

// -----------------------------------------------------------------------------
// Dispatch: generic fallback only (Stride1Specialist plugins inside conv_fallback)
// -----------------------------------------------------------------------------
static inline void dispatch_forward(
    const float* x, const float* W, const float* bias, float* out,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride, int64_t C_out,
    int64_t k_h, int64_t k_w, int64_t stride, int64_t pad, int64_t out_w_stride, int32_t fuse_relu
) {
    log_routing_decision("FWD", "GENERIC_FALLBACK", k_h, k_w, stride, pad);
    conv2d_forward_fallback_avx2(
        x, W, bias, out,
        N, C_in, H, W_in, W_in_stride,
        C_out, k_h, k_w, stride, pad,
        out_w_stride, fuse_relu
    );
}

static inline void dispatch_backward(
    const float* dout, const float* x, const float* W,
    float* dx, float* dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride, int64_t C_out,
    int64_t k_h, int64_t k_w, int64_t stride, int64_t pad, int64_t conv_out_w_stride, float inv_m
) {
    log_routing_decision("BWD", "GENERIC_FALLBACK", k_h, k_w, stride, pad);
    conv2d_backward_fallback_avx2(
        dout, x, W, dx, dW,
        N, C_in, H, W_in, W_in_stride,
        C_out, k_h, k_w, stride, pad,
        conv_out_w_stride, inv_m
    );
}

void maxpool2d_backward_avx2(
    const float* dout_pool, const uint8_t* argmax_buf, float* d_conv_buf,
    int64_t N, int64_t C, int64_t pool_h, int64_t pool_w,
    int64_t conv_h, int64_t conv_w, int64_t conv_w_stride,
    int64_t pool_size, int64_t pool_stride
) {
    const int64_t conv_spatial = conv_h * conv_w_stride;
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
                        dp[ih * conv_w_stride + iw] += grad;
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Exported C Interfaces (With Clean Status Code Boundary & Try/Catch)
// -----------------------------------------------------------------------------
extern "C" {

__declspec(dllexport) void log_engine_runtime_diagnostics(
    void* p1, void* p2, void* p3,
    int64_t i1, int64_t i2, int64_t i3, int64_t i4,
    int64_t i5, int64_t i6, int64_t i7
) {
    (void)p1; (void)p2; (void)p3;
    (void)i1; (void)i2; (void)i3; (void)i4;
    (void)i5; (void)i6; (void)i7;
}

__declspec(dllexport) int32_t direct_conv_block_forward_avx2(
    const float* x,
    const float* W,
    const float* bias,
    float* out_conv,
    float* out_pool,
    uint8_t* argmax_buf,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t k_h, int64_t k_w,
    int64_t conv_stride, int64_t conv_pad, int64_t conv_out_w_stride,
    int64_t pool_size, int64_t pool_stride
) {
    try {
        if (!x || !W || !out_conv) {
            std::fprintf(stderr, "[ENGINE_ERROR] Null pointer passed to direct_conv_block_forward_avx2 (x=%p, W=%p, out=%p)\n", x, W, out_conv);
            return -1;
        }
        if (N <= 0 || C_in <= 0 || H <= 0 || W_in <= 0 || C_out <= 0) {
            std::fprintf(stderr, "[ENGINE_ERROR] Invalid tensor shape in direct_conv_block_forward_avx2 [N=%lld, Cin=%lld, H=%lld, Win=%lld, Cout=%lld]\n",
                         (long long)N, (long long)C_in, (long long)H, (long long)W_in, (long long)C_out);
            return -2;
        }

        dispatch_forward(
            x, W, bias, out_conv,
            N, C_in, H, W_in, W_in_stride, C_out,
            k_h, k_w, conv_stride, conv_pad, conv_out_w_stride, 1
        );

        if (out_pool && argmax_buf) {
            const int64_t conv_out_h = (H + 2 * conv_pad - k_h) / conv_stride + 1;
            const int64_t conv_out_w = (W_in + 2 * conv_pad - k_w) / conv_stride + 1;
            const int64_t pool_out_h = (conv_out_h - pool_size) / pool_stride + 1;
            const int64_t pool_out_w = (conv_out_w - pool_size) / pool_stride + 1;
    const int64_t conv_spatial = conv_out_h * conv_out_w_stride;
    const int64_t pool_spatial = pool_out_h * pool_out_w;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C_out; ++c) {
            const float* __restrict cp = &out_conv[(n * C_out + c) * conv_spatial];
            float* __restrict pp = &out_pool[(n * C_out + c) * pool_spatial];
            uint8_t* __restrict ap = &argmax_buf[(n * C_out + c) * pool_spatial];

            if (pool_size == 2 && pool_stride == 2) {
                // 2x2 Fast Path - Perfectly Unrolled, Zero Index Math, Register Bound
                for (int64_t ph = 0; ph < pool_out_h; ++ph) {
                    const float* __restrict r0 = cp + (ph * 2) * conv_out_w_stride;
                    const float* __restrict r1 = r0 + conv_out_w_stride;
                    
                    float* __restrict p_row = pp + ph * pool_out_w;
                    uint8_t* __restrict a_row = ap + ph * pool_out_w;

                    for (int64_t pw = 0; pw < pool_out_w; ++pw) {
                        const int64_t iw = pw * 2;
                        
                        // Force compiler to keep these in XMM registers
                        const float v0 = r0[iw];
                        const float v1 = r0[iw + 1];
                        const float v2 = r1[iw];
                        const float v3 = r1[iw + 1];

                        float max_val = v0;
                        uint8_t max_idx = 0;

                        if (v1 > max_val) { max_val = v1; max_idx = 1; }
                        if (v2 > max_val) { max_val = v2; max_idx = 2; }
                        if (v3 > max_val) { max_val = v3; max_idx = 3; }

                        p_row[pw] = max_val;
                        a_row[pw] = max_idx;
                    }
                }
            } else {
                // Generic Pointer-Stepped Fallback (Zero nested imul)
                for (int64_t ph = 0; ph < pool_out_h; ++ph) {
                    const float* __restrict in_row_base = cp + (ph * pool_stride) * conv_out_w_stride;
                    float* __restrict p_row = pp + ph * pool_out_w;
                    uint8_t* __restrict a_row = ap + ph * pool_out_w;

                    for (int64_t pw = 0; pw < pool_out_w; ++pw) {
                        const float* __restrict in_ptr = in_row_base + (pw * pool_stride);
                        
                        float max_val = -1e30f;
                        uint8_t max_idx = 0;

                        for (int64_t kh = 0; kh < pool_size; ++kh) {
                            const float* __restrict k_row = in_ptr + kh * conv_out_w_stride;
                            for (int64_t kw = 0; kw < pool_size; ++kw) {
                                const float val = k_row[kw];
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = (uint8_t)(kh * pool_size + kw);
                                }
                            }
                        }
                        p_row[pw] = max_val;
                        a_row[pw] = max_idx;
                    }
                }
            }
        }
    }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ENGINE_EXCEPTION] direct_conv_block_forward_avx2 caught: %s\n", e.what());
        return -99;
    } catch (...) {
        std::fprintf(stderr, "[ENGINE_EXCEPTION] direct_conv_block_forward_avx2 caught unknown native exception.\n");
        return -100;
    }
}

__declspec(dllexport) int32_t direct_conv_block_backward_avx2(
    const float* dout_pool,
    const uint8_t* argmax_buf,
    const float* x,
    const float* W,
    const float* conv_act,
    float* d_conv_buf,
    float* dx_buf,
    float* dW_buf,
    float* db_buf,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t k_h, int64_t k_w,
    int64_t conv_stride, int64_t conv_pad, int64_t conv_out_w_stride,
    int64_t pool_size, int64_t pool_stride,
    int64_t pool_out_h, int64_t pool_out_w,
    float inv_m
) {
    try {
        const int64_t conv_out_h = (H + 2 * conv_pad - k_h) / conv_stride + 1;
        const int64_t conv_out_w = (W_in + 2 * conv_pad - k_w) / conv_stride + 1;
        const int64_t conv_spatial = conv_out_h * conv_out_w_stride;

        if (dout_pool && argmax_buf && d_conv_buf) {
            maxpool2d_backward_avx2(
                dout_pool, argmax_buf, d_conv_buf,
                N, C_out, pool_out_h, pool_out_w,
                conv_out_h, conv_out_w, conv_out_w_stride,
                pool_size, pool_stride
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
                    for (int64_t h = 0; h < conv_out_h; ++h) {
                        const float* row = &dp[h * conv_out_w_stride];
                        for (int64_t w = 0; w < conv_out_w; ++w) {
                            sum += (double)row[w];
                        }
                    }
                }
                db_buf[cout] = (float)(sum * (double)inv_m);
            }
        }

        dispatch_backward(
            d_conv_buf, x, W, dx_buf, dW_buf,
            N, C_in, H, W_in, W_in_stride, C_out, k_h, k_w,
            conv_stride, conv_pad, conv_out_w_stride, inv_m
        );
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ENGINE_EXCEPTION] direct_conv_block_backward_avx2 caught: %s\n", e.what());
        return -99;
    } catch (...) {
        std::fprintf(stderr, "[ENGINE_EXCEPTION] direct_conv_block_backward_avx2 caught unknown native exception.\n");
        return -100;
    }
}

__declspec(dllexport) int32_t direct_conv2d_forward_avx2(
    const float* x, const float* W, const float* bias, float* out,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride, int64_t C_out,
    int64_t k_h, int64_t k_w, int64_t stride, int64_t pad, int64_t out_w_stride, int32_t fuse_relu
) {
    try {
        dispatch_forward(
            x, W, bias, out,
            N, C_in, H, W_in, W_in_stride, C_out,
            k_h, k_w, stride, pad, out_w_stride, fuse_relu
        );
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ENGINE_EXCEPTION] direct_conv2d_forward_avx2 caught: %s\n", e.what());
        return -99;
    } catch (...) {
        std::fprintf(stderr, "[ENGINE_EXCEPTION] direct_conv2d_forward_avx2 caught unknown native exception.\n");
        return -100;
    }
}

__declspec(dllexport) int32_t direct_conv2d_backward_fused_avx2(
    const float* dout, const float* x, const float* W, const float* in_act,
    float* dx_buf, float* dW_buf,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride, int64_t C_out,
    int64_t k_h, int64_t k_w, int64_t stride, int64_t pad, int64_t conv_out_w_stride,
    float inv_m, int32_t fuse_relu
) {
    (void)in_act; (void)fuse_relu;
    try {
        dispatch_backward(
            dout, x, W, dx_buf, dW_buf,
            N, C_in, H, W_in, W_in_stride, C_out,
            k_h, k_w, stride, pad, conv_out_w_stride, inv_m
        );
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ENGINE_EXCEPTION] direct_conv2d_backward_fused_avx2 caught: %s\n", e.what());
        return -99;
    } catch (...) {
        std::fprintf(stderr, "[ENGINE_EXCEPTION] direct_conv2d_backward_fused_avx2 caught unknown native exception.\n");
        return -100;
    }
}

__declspec(dllexport) int32_t direct_conv2d_backward_weight_avx2(
    const float* d_conv_buf, const float* x, float* dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride, int64_t C_out,
    int64_t k_h, int64_t k_w, int64_t stride, int64_t pad, int64_t conv_out_w_stride, float inv_m
) {
    try {
        dispatch_backward(
            d_conv_buf, x, nullptr, nullptr, dW,
            N, C_in, H, W_in, W_in_stride, C_out,
            k_h, k_w, stride, pad, conv_out_w_stride, inv_m
        );
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ENGINE_EXCEPTION] direct_conv2d_backward_weight_avx2 caught: %s\n", e.what());
        return -99;
    } catch (...) {
        std::fprintf(stderr, "[ENGINE_EXCEPTION] direct_conv2d_backward_weight_avx2 caught unknown native exception.\n");
        return -100;
    }
}

__declspec(dllexport) int32_t direct_conv2d_backward_input_avx2(
    const float* d_conv_buf, const float* W, const float* in_act, float* dx,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride, int64_t C_out,
    int64_t k_h, int64_t k_w, int64_t stride, int64_t pad, int64_t conv_out_w_stride, int32_t fuse_relu
) {
    (void)in_act; (void)fuse_relu;
    try {
        dispatch_backward(
            d_conv_buf, nullptr, W, dx, nullptr,
            N, C_in, H, W_in, W_in_stride, C_out,
            k_h, k_w, stride, pad, conv_out_w_stride, 1.0f
        );
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ENGINE_EXCEPTION] direct_conv2d_backward_input_avx2 caught: %s\n", e.what());
        return -99;
    } catch (...) {
        std::fprintf(stderr, "[ENGINE_EXCEPTION] direct_conv2d_backward_input_avx2 caught unknown native exception.\n");
        return -100;
    }
}

__declspec(dllexport) int32_t direct_relu_forward_avx2(float* x, int64_t size) {
    try {
        if (!x) return -1;
        const __m256 v_zero = _mm256_setzero_ps();
        int64_t i = 0;
        for (; i + 8 <= size; i += 8) {
            __m256 v = _mm256_loadu_ps(&x[i]);
            _mm256_storeu_ps(&x[i], _mm256_max_ps(v, v_zero));
        }
        for (; i < size; ++i) {
            x[i] = x[i] > 0.0f ? x[i] : 0.0f;
        }
        return 0;
    } catch (...) {
        return -100;
    }
}

__declspec(dllexport) int32_t direct_relu_backward_avx2(float* dout, const float* in_act, int64_t size) {
    try {
        if (!dout || !in_act) return -1;
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
        return 0;
    } catch (...) {
        return -100;
    }
}

__declspec(dllexport) int32_t direct_maxpool_forward_avx2(
    const float* x, float* out_pool, uint8_t* argmax_buf,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t pool_size, int64_t pool_stride
) {
    try {
        if (!x || !out_pool || !argmax_buf) return -1;
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
        return 0;
    } catch (...) {
        return -100;
    }
}

__declspec(dllexport) int32_t direct_maxpool_backward_avx2(
    const float* dout_pool, const uint8_t* argmax_buf, float* dx_buf,
    int64_t N, int64_t C, int64_t out_h, int64_t out_w,
    int64_t in_h, int64_t in_w, int64_t pool_size, int64_t pool_stride
) {
    try {
        if (!dout_pool || !argmax_buf || !dx_buf) return -1;
        maxpool2d_backward_avx2(
            dout_pool, argmax_buf, dx_buf,
            N, C, out_h, out_w, in_h, in_w, in_w, pool_size, pool_stride
        );
        return 0;
    } catch (...) {
        return -100;
    }
}

__declspec(dllexport) int32_t direct_bias_backward_avx2(
    const float* dout, float* db,
    int64_t N, int64_t C_out, int64_t out_h, int64_t out_w,
    float inv_m
) {
    try {
        if (!dout || !db) return -1;
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
        return 0;
    } catch (...) {
        return -100;
    }
}

}