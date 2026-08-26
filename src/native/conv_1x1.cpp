#include "diagnostics.h"

void conv2d_forward_1x1_avx2(
    const float* __restrict x, const float* __restrict W, const float* __restrict bias, float* __restrict out,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t C_out, int32_t fuse_relu
) {
    DIAG_INC(fwd_1x1);
    TIME_SCOPE(time_fwd_1x1_ns);
    const int64_t spatial = H * W_in;
    const __m256 v_zero = _mm256_setzero_ps();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t cout = 0; cout < C_out; ++cout) {
            float* __restrict out_p = &out[(n * C_out + cout) * spatial];
            const __m256 vb = bias ? _mm256_set1_ps(bias[cout]) : _mm256_setzero_ps();
            const float b_scalar = bias ? bias[cout] : 0.0f;

            int64_t s = 0;
            for (; s + 8 <= spatial; s += 8) {
                __m256 acc = vb;
                for (int64_t cin = 0; cin < C_in; ++cin) {
                    const float* __restrict xp = &x[(n * C_in + cin) * spatial];
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(&xp[s]), _mm256_set1_ps(W[cout * C_in + cin]), acc);
                }
                if (fuse_relu) acc = _mm256_max_ps(acc, v_zero);
                _mm256_storeu_ps(&out_p[s], acc);
            }
            for (; s < spatial; ++s) {
                float sum = b_scalar;
                for (int64_t cin = 0; cin < C_in; ++cin) sum += x[(n * C_in + cin) * spatial + s] * W[cout * C_in + cin];
                if (fuse_relu && sum < 0.0f) sum = 0.0f;
                out_p[s] = sum;
            }
        }
    }
}

void conv2d_backward_1x1_avx2(
    const float* __restrict d_conv_buf, const float* __restrict x, const float* __restrict W,
    float* __restrict dx, float* __restrict dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t C_out, float inv_m
) {
    DIAG_INC(bwd_1x1);
    TIME_SCOPE(time_bwd_1x1_ns);
    const int64_t spatial = H * W_in;

    if (dx && W) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t cin = 0; cin < C_in; ++cin) {
                float* __restrict dx_p = &dx[(n * C_in + cin) * spatial];
                int64_t s = 0;
                for (; s + 8 <= spatial; s += 8) {
                    __m256 acc = _mm256_setzero_ps();
                    for (int64_t cout = 0; cout < C_out; ++cout) {
                        const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * spatial];
                        acc = _mm256_fmadd_ps(_mm256_loadu_ps(&dp[s]), _mm256_set1_ps(W[cout * C_in + cin]), acc);
                    }
                    _mm256_storeu_ps(&dx_p[s], acc);
                }
                for (; s < spatial; ++s) {
                    float sum = 0.0f;
                    for (int64_t cout = 0; cout < C_out; ++cout) sum += d_conv_buf[(n * C_out + cout) * spatial + s] * W[cout * C_in + cin];
                    dx_p[s] = sum;
                }
            }
        }
    }

    if (dW && x) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t cout = 0; cout < C_out; ++cout) {
            for (int64_t cin = 0; cin < C_in; ++cin) {
                __m256 v_dw = _mm256_setzero_ps();
                float s_dw = 0.0f;
                for (int64_t n = 0; n < N; ++n) {
                    const float* __restrict dp = &d_conv_buf[(n * C_out + cout) * spatial];
                    const float* __restrict xp = &x[(n * C_in + cin) * spatial];
                    int64_t s = 0;
                    for (; s + 8 <= spatial; s += 8) {
                        v_dw = _mm256_fmadd_ps(_mm256_loadu_ps(&dp[s]), _mm256_loadu_ps(&xp[s]), v_dw);
                    }
                    for (; s < spatial; ++s) s_dw += dp[s] * xp[s];
                }
                dW[cout * C_in + cin] = (s_dw + reduce_add_avx2(v_dw)) * inv_m;
            }
        }
    }
}