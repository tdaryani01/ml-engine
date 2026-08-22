#include <immintrin.h>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <omp.h>

#if defined(_MSC_VER)
    #define EXPORT_API extern "C" __declspec(dllexport)
#else
    #define EXPORT_API extern "C" __attribute__((visibility("default")))
#endif

EXPORT_API void direct_conv2d_forward_avx2(
    const float* __restrict x,
    const float* __restrict W,
    const float* __restrict bias,
    float* __restrict out,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in,
    int64_t C_out, int64_t k_h, int64_t k_w,
    int64_t stride, int64_t pad
) {
    const int64_t out_h = (H + 2 * pad - k_h) / stride + 1;
    const int64_t out_w = (W_in + 2 * pad - k_w) / stride + 1;
    const int64_t total_tasks = N * C_out;

    #pragma omp parallel for schedule(static)
    for (int64_t task = 0; task < total_tasks; ++task) {
        const int64_t n = task / C_out;
        const int64_t cout = task % C_out;
        const float b_val = bias ? bias[cout] : 0.0f;

        float* out_plane = &out[(n * C_out + cout) * out_h * out_w];

        for (int64_t oh = 0; oh < out_h; ++oh) {
            const int64_t ih_base = oh * stride - pad;
            float* out_row = &out_plane[oh * out_w];

            // Initialize row with bias
            for (int64_t ow = 0; ow < out_w; ++ow) {
                out_row[ow] = b_val;
            }

            for (int64_t cin = 0; cin < C_in; ++cin) {
                const float* x_chan = &x[(n * C_in + cin) * H * W_in];
                const float* w_chan = &W[((cout * C_in + cin) * k_h) * k_w];

                for (int64_t kh = 0; kh < k_h; ++kh) {
                    const int64_t ih = ih_base + kh;
                    if (ih < 0 || ih >= H) continue;

                    const float* x_row = &x_chan[ih * W_in];
                    const float* w_row = &w_chan[kh * k_w];

                    for (int64_t kw = 0; kw < k_w; ++kw) {
                        const float w_val = w_row[kw];
                        if (w_val == 0.0f) continue;
                        const __m256 v_w = _mm256_set1_ps(w_val);

                        const int64_t iw_base = kw - pad;

                        // AVX2 vectorization across output width when stride == 1
                        int64_t ow = 0;
                        if (stride == 1) {
                            for (; ow + 8 <= out_w; ow += 8) {
                                int64_t iw = iw_base + ow;
                                if (iw >= 0 && (iw + 8) <= W_in) {
                                    __m256 v_out = _mm256_loadu_ps(&out_row[ow]);
                                    __m256 v_in = _mm256_loadu_ps(&x_row[iw]);
                                    v_out = _mm256_fmadd_ps(v_in, v_w, v_out);
                                    _mm256_storeu_ps(&out_row[ow], v_out);
                                } else {
                                    // Boundary fallback
                                    for (int i = 0; i < 8; ++i) {
                                        int64_t single_iw = iw_base + ow + i;
                                        if (single_iw >= 0 && single_iw < W_in) {
                                            out_row[ow + i] += x_row[single_iw] * w_val;
                                        }
                                    }
                                }
                            }
                        }

                        // Scalar remainder / strided path
                        for (; ow < out_w; ++ow) {
                            const int64_t iw = ow * stride + iw_base;
                            if (iw >= 0 && iw < W_in) {
                                out_row[ow] += x_row[iw] * w_val;
                            }
                        }
                    }
                }
            }
        }
    }
}