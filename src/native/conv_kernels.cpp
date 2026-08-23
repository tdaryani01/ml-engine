#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <windows.h>

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
    std::cout << "\n================ [C++ DLL RUNTIME AUDIT] ================" << std::endl;
    std::cout << "[Thread Diagnostics]" << std::endl;
    std::cout << "  - Configured OMP Max Threads: " << omp_get_max_threads() << std::endl;
    
    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        std::cout << "  - Active Parallel Threads   : " << omp_get_num_threads() << std::endl;
    }

    uintptr_t x_addr   = reinterpret_cast<uintptr_t>(x);
    uintptr_t W_addr   = reinterpret_cast<uintptr_t>(W);
    uintptr_t out_addr = reinterpret_cast<uintptr_t>(out);

    std::cout << "\n[Memory Layout & Pointer Alignment]" << std::endl;
    std::cout << "  - Input Pointer  (x)  : 0x" << std::hex << x_addr << std::dec 
              << " | 32-Byte AVX Aligned: " << ((x_addr % 32 == 0) ? "YES" : "NO (Misaligned)") << std::endl;
    std::cout << "  - Weight Pointer (W)  : 0x" << std::hex << W_addr << std::dec 
              << " | 32-Byte AVX Aligned: " << ((W_addr % 32 == 0) ? "YES" : "NO (Misaligned)") << std::endl;
    std::cout << "  - Output Pointer (out): 0x" << std::hex << out_addr << std::dec 
              << " | 32-Byte AVX Aligned: " << ((out_addr % 32 == 0) ? "YES" : "NO (Misaligned)") << std::endl;

    std::cout << "=========================================================\n" << std::endl;
}

// -----------------------------------------------------------------------------
// 1. FORWARD CONVOLUTION (Pure Register Accumulation)
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
    const int64_t total_tasks = N * C_out;

    #pragma omp parallel for num_threads(4) schedule(static)
    for (int64_t task = 0; task < total_tasks; ++task) {
        const int64_t n = task / C_out;
        const int64_t cout = task % C_out;

        float* __restrict out_plane = &out[(n * C_out + cout) * out_h * out_w];
        const float b_val = bias ? bias[cout] : 0.0f;
        const __m256 v_b = _mm256_set1_ps(b_val);

        for (int64_t oh = 0; oh < out_h; ++oh) {
            const int64_t ih_base = oh * stride - pad;
            float* __restrict r_out = &out_plane[oh * out_w];

            int64_t ow = 0;
            // Vectorized 8-pixel chunk with accumulator in register
            for (; ow + 8 <= out_w; ow += 8) {
                __m256 acc = v_b;
                const int64_t iw_base = ow * stride - pad;

                for (int64_t cin = 0; cin < C_in; ++cin) {
                    const float* __restrict x_chan = &x[(n * C_in + cin) * H * W_in];
                    const float* __restrict w_chan = &W[((cout * C_in + cin) * k_h) * k_w];

                    for (int64_t kh = 0; kh < k_h; ++kh) {
                        const int64_t ih = ih_base + kh;
                        if (ih < 0 || ih >= H) continue;

                        const float* __restrict x_row = &x_chan[ih * W_in];
                        const float* __restrict w_row = &w_chan[kh * k_w];

                        #pragma unroll
                        for (int64_t kw = 0; kw < k_w; ++kw) {
                            const float w_val = w_row[kw];
                            if (w_val == 0.0f) continue;
                            const __m256 vw = _mm256_set1_ps(w_val);
                            const int64_t iw = iw_base + kw;

                            if (iw >= 0 && (iw + 8) <= W_in && stride == 1) {
                                acc = _mm256_fmadd_ps(_mm256_loadu_ps(&x_row[iw]), vw, acc);
                            } else {
                                alignas(32) float t_buf[8] = {0};
                                for (int s = 0; s < 8; ++s) {
                                    int64_t s_iw = iw + s;
                                    if (s_iw >= 0 && s_iw < W_in) t_buf[s] = x_row[s_iw];
                                }
                                acc = _mm256_fmadd_ps(_mm256_load_ps(t_buf), vw, acc);
                            }
                        }
                    }
                }

                if (fuse_relu) acc = _mm256_max_ps(acc, v_zero);
                _mm256_storeu_ps(&r_out[ow], acc);
            }

            // Remainder scalar chunk (accumulate in scalar float, store once)
            for (; ow < out_w; ++ow) {
                float acc = b_val;
                const int64_t iw_base = ow * stride - pad;

                for (int64_t cin = 0; cin < C_in; ++cin) {
                    const float* __restrict x_chan = &x[(n * C_in + cin) * H * W_in];
                    const float* __restrict w_chan = &W[((cout * C_in + cin) * k_h) * k_w];

                    for (int64_t kh = 0; kh < k_h; ++kh) {
                        const int64_t ih = ih_base + kh;
                        if (ih < 0 || ih >= H) continue;

                        const float* __restrict x_row = &x_chan[ih * W_in];
                        const float* __restrict w_row = &w_chan[kh * k_w];

                        #pragma unroll
                        for (int64_t kw = 0; kw < k_w; ++kw) {
                            const int64_t iw = iw_base + kw;
                            if (iw >= 0 && iw < W_in) {
                                acc += x_row[iw] * w_row[kw];
                            }
                        }
                    }
                }
                if (fuse_relu && acc < 0.0f) acc = 0.0f;
                r_out[ow] = acc;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 2. BACKWARD WEIGHT GRADIENTS (dW)
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
    const int64_t total_tasks = C_out * C_in;

    #pragma omp parallel for num_threads(4) schedule(static)
    for (int64_t task = 0; task < total_tasks; ++task) {
        const int64_t cout = task / C_in;
        const int64_t cin  = task % C_in;
        float* __restrict dw_plane = &dW[((cout * C_in + cin) * k_h) * k_w];

        for (int64_t kh = 0; kh < k_h; ++kh) {
            for (int64_t kw = 0; kw < k_w; ++kw) {
                __m256 vacc = _mm256_setzero_ps();
                float scalar_acc = 0.0f;

                for (int64_t n = 0; n < N; ++n) {
                    const float* __restrict d_plane = &dout[(n * C_out + cout) * out_h * out_w];
                    const float* __restrict x_plane = &x[(n * C_in + cin) * H * W_in];

                    for (int64_t oh = 0; oh < out_h; ++oh) {
                        const int64_t ih = oh * stride - pad + kh;
                        if (ih < 0 || ih >= H) continue;

                        const float* __restrict d_row = &d_plane[oh * out_w];
                        const float* __restrict x_row = &x_plane[ih * W_in];

                        int64_t ow = 0;
                        if (stride == 1) {
                            for (; ow + 8 <= out_w; ow += 8) {
                                const int64_t iw = ow - pad + kw;
                                if (iw >= 0 && (iw + 8) <= W_in) {
                                    __m256 v_d = _mm256_loadu_ps(&d_row[ow]);
                                    __m256 v_x = _mm256_loadu_ps(&x_row[iw]);
                                    vacc = _mm256_fmadd_ps(v_d, v_x, vacc);
                                } else {
                                    for (int s = 0; s < 8; ++s) {
                                        const int64_t s_iw = iw + s;
                                        if (s_iw >= 0 && s_iw < W_in) {
                                            scalar_acc += d_row[ow + s] * x_row[s_iw];
                                        }
                                    }
                                }
                            }
                        }

                        for (; ow < out_w; ++ow) {
                            const int64_t iw = ow * stride - pad + kw;
                            if (iw >= 0 && iw < W_in) {
                                scalar_acc += d_row[ow] * x_row[iw];
                            }
                        }
                    }
                }

                alignas(32) float b[8];
                _mm256_store_ps(b, vacc);
                float s = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + b[7] + scalar_acc;
                dw_plane[kh * k_w + kw] = s * inv_m;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 3. BACKWARD INPUT GRADIENTS (dx)
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

    #pragma omp parallel for num_threads(4) schedule(static)
    for (int64_t task = 0; task < total_tasks; ++task) {
        const int64_t n   = task / C_in;
        const int64_t cin = task % C_in;
        float* __restrict dx_plane = &dx[(n * C_in + cin) * H * W_in];

        std::memset(dx_plane, 0, H * W_in * sizeof(float));

        for (int64_t cout = 0; cout < C_out; ++cout) {
            const float* __restrict d_plane = &dout[(n * C_out + cout) * out_h * out_w];
            const float* __restrict w_chan = &W[((cout * C_in + cin) * k_h) * k_w];

            for (int64_t kh = 0; kh < k_h; ++kh) {
                const float* __restrict w_row = &w_chan[kh * k_w];

                for (int64_t kw = 0; kw < k_w; ++kw) {
                    const float w_val = w_row[kw];
                    if (w_val == 0.0f) continue;
                    const __m256 vw = _mm256_set1_ps(w_val);

                    for (int64_t oh = 0; oh < out_h; ++oh) {
                        const int64_t ih = oh * stride - pad + kh;
                        if (ih < 0 || ih >= H) continue;

                        const float* __restrict d_row  = &d_plane[oh * out_w];
                        float* __restrict dx_row = &dx_plane[ih * W_in];

                        int64_t ow = 0;
                        if (stride == 1) {
                            for (; ow + 8 <= out_w; ow += 8) {
                                const int64_t iw = ow - pad + kw;
                                if (iw >= 0 && (iw + 8) <= W_in) {
                                    __m256 v_dx = _mm256_loadu_ps(&dx_row[iw]);
                                    __m256 v_d  = _mm256_loadu_ps(&d_row[ow]);
                                    _mm256_storeu_ps(&dx_row[iw], _mm256_fmadd_ps(v_d, vw, v_dx));
                                } else {
                                    for (int s = 0; s < 8; ++s) {
                                        const int64_t s_iw = iw + s;
                                        if (s_iw >= 0 && s_iw < W_in) {
                                            dx_row[s_iw] += d_row[ow + s] * w_val;
                                        }
                                    }
                                }
                            }
                        }

                        for (; ow < out_w; ++ow) {
                            const int64_t iw = ow * stride - pad + kw;
                            if (iw >= 0 && iw < W_in) {
                                dx_row[iw] += d_row[ow] * w_val;
                            }
                        }
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 4. IN-PLACE RELU
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// 5. MAXPOOL
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

    #pragma omp parallel for num_threads(4) schedule(static)
    for (int64_t p = 0; p < total_planes; ++p) {
        const float* x_plane = &x[p * H * W];
        float* out_plane     = &out[p * out_h * out_w];
        int64_t* mask_plane  = &argmax_buf[p * out_h * out_w * 2];

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

    #pragma omp parallel for num_threads(4) schedule(static)
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
// 6. BIAS ACCUMULATION
// -----------------------------------------------------------------------------
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