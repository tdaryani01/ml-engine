// src/native/conv_im2col_gemm.cpp
// Fused im2col + OpenBLAS SGEMM convolution (IM2COL_GEMM backend).
//
// Threading contract (bin/libopenblas USE_OPENMP=1 + OMP_MAX_ACTIVE_LEVELS=1):
//   One shared LLVM OpenMP pool: im2col/col2im/fuse parallel, serial GEMM fans out same pool.
//   Phases: parallel im2col → serial GEMM → parallel col2im/fuse (never nested).
#include "blas_dynamic.h"
#include "im2col_telemetry.h"
#include "omp_config.h"

extern "C" {
int32_t im2col_avx2(
    const float* x, float* out,
    int64_t N, int64_t C, int64_t H, int64_t W_logical, int64_t W_stride,
    int64_t k_h, int64_t k_w, int64_t stride, int64_t pad);
int32_t col2im_avx2(
    const float* col, float* dx,
    int64_t N, int64_t C, int64_t H, int64_t W_logical, int64_t W_stride,
    int64_t k_h, int64_t k_w, int64_t stride, int64_t pad);
}

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <omp.h>

namespace {

inline int64_t conv_out_dim(int64_t in, int64_t k, int64_t pad, int64_t stride) {
    return (in + 2 * pad - k) / stride + 1;
}

inline int32_t omp_threads_for_phase() {
    ml_omp_before_parallel();
    return get_omp_threads();
}

void fuse_forward_transpose_bias(
    const float* gemm_out,
    const float* bias,
    float* out,
    int64_t N,
    int64_t C_out,
    int64_t out_h,
    int64_t out_w,
    int64_t out_w_stride
) {
    const int64_t spatial = out_h * out_w;
    const int32_t omp_n = omp_threads_for_phase();
    #pragma omp parallel for num_threads(omp_n) collapse(3) schedule(static) if(omp_n > 1)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t h = 0; h < out_h; ++h) {
            for (int64_t w = 0; w < out_w; ++w) {
                const int64_t row_idx = n * spatial + h * out_w + w;
                for (int64_t c = 0; c < C_out; ++c) {
                    out[((n * C_out + c) * out_h + h) * out_w_stride + w] =
                        gemm_out[row_idx * C_out + c] + bias[c];
                }
            }
        }
    }
}

void fuse_dout_transpose(
    const float* dout,
    float* dout_trans,
    int64_t N,
    int64_t C_out,
    int64_t out_h,
    int64_t out_w,
    int64_t dout_w_stride
) {
    im2col_telemetry::on_fuse_dout_transpose();
    const int64_t spatial = out_h * out_w;
    const int64_t chan_stride = out_h * dout_w_stride;
    const __m256i gather_step = _mm256_set_epi32(
        static_cast<int>(7 * chan_stride),
        static_cast<int>(6 * chan_stride),
        static_cast<int>(5 * chan_stride),
        static_cast<int>(4 * chan_stride),
        static_cast<int>(3 * chan_stride),
        static_cast<int>(2 * chan_stride),
        static_cast<int>(chan_stride),
        0);

    const int32_t omp_n = omp_threads_for_phase();
    #pragma omp parallel for num_threads(omp_n) collapse(3) schedule(static) if(omp_n > 1)
    for (int64_t n = 0; n < N; ++n) {
        const int64_t row_n_offset = n * spatial;
        const int64_t n_plane = n * C_out * out_h * dout_w_stride;
        for (int64_t h = 0; h < out_h; ++h) {
            const int64_t row_h_offset = row_n_offset + h * out_w;
            const int64_t nh_base = n_plane + h * dout_w_stride;
            for (int64_t w = 0; w < out_w; ++w) {
                const int64_t row_idx = row_h_offset + w;
                const float* src_base = dout + nh_base + w;
                float* dst = dout_trans + row_idx * C_out;

                int64_t c = 0;
                if (C_out >= 8) {
                    for (; c + 8 <= C_out; c += 8) {
                        const __m256 vals = _mm256_i32gather_ps(
                            src_base + c * chan_stride, gather_step, 4);
                        _mm256_storeu_ps(dst + c, vals);
                    }
                }
                for (; c < C_out; ++c) {
                    dst[c] = src_base[c * chan_stride];
                }
            }
        }
    }
}

void fuse_dout_transpose_bias(
    const float* dout,
    float* dout_trans,
    float* db,
    int64_t N,
    int64_t C_out,
    int64_t out_h,
    int64_t out_w,
    int64_t dout_w_stride,
    float inv_m
) {
    fuse_dout_transpose(dout, dout_trans, N, C_out, out_h, out_w, dout_w_stride);
    const int64_t rows = N * out_h * out_w;
    const int32_t omp_n = omp_threads_for_phase();
    #pragma omp parallel for num_threads(omp_n) schedule(static) if(omp_n > 1)
    for (int64_t c = 0; c < C_out; ++c) {
        double sum = 0.0;
        const int64_t col_off = c;
        for (int64_t r = 0; r < rows; ++r) {
            sum += static_cast<double>(dout_trans[r * C_out + col_off]);
        }
        db[c] = static_cast<float>(sum * static_cast<double>(inv_m));
    }
}

void relu_fwd_inplace(float* out, int64_t elems) {
    const int32_t omp_n = omp_threads_for_phase();
    #pragma omp parallel for num_threads(omp_n) schedule(static) if(omp_n > 1)
    for (int64_t i = 0; i < elems; ++i) {
        if (out[i] < 0.0f) {
            out[i] = 0.0f;
        }
    }
}

void relu_bwd_inplace(float* dx, const float* in_act, int64_t elems) {
    const int32_t omp_n = omp_threads_for_phase();
    #pragma omp parallel for num_threads(omp_n) schedule(static) if(omp_n > 1)
    for (int64_t i = 0; i < elems; ++i) {
        if (in_act[i] <= 0.0f) {
            dx[i] = 0.0f;
        }
    }
}

int32_t require_blas() {
    if (!blas_runtime_ready()) {
        return -100;
    }
    return 0;
}

}  // namespace

extern "C" {

__declspec(dllexport) int32_t fuse_dout_transpose_bias_avx2(
    const float* dout,
    float* dout_trans,
    float* db,
    int64_t N,
    int64_t C_out,
    int64_t out_h,
    int64_t out_w,
    int64_t dout_w_stride,
    float inv_m
) {
    if (!dout || !dout_trans || !db || N <= 0 || C_out <= 0 || out_h <= 0 || out_w <= 0) {
        return -1;
    }
    fuse_dout_transpose_bias(
        dout, dout_trans, db, N, C_out, out_h, out_w, dout_w_stride, inv_m);
    return 0;
}

__declspec(dllexport) int32_t conv2d_forward_im2col_gemm_avx2(
    const float* x,
    const float* W_fwd,
    const float* bias,
    float* out,
    float* col_buf,
    float* gemm_buf,
    int64_t N,
    int64_t C_in,
    int64_t H,
    int64_t W_in,
    int64_t W_in_stride,
    int64_t C_out,
    int64_t k_h,
    int64_t k_w,
    int64_t stride,
    int64_t pad,
    int64_t out_w_stride,
    int32_t fuse_relu
) {
    if (!x || !W_fwd || !bias || !out || !col_buf || !gemm_buf) {
        return -1;
    }
    if (N <= 0 || C_in <= 0 || H <= 0 || W_in <= 0 || C_out <= 0 || k_h <= 0 || k_w <= 0 || stride <= 0 || pad < 0) {
        return -2;
    }
    const int32_t blas_rc = require_blas();
    if (blas_rc != 0) {
        return blas_rc;
    }

    const int64_t out_h = conv_out_dim(H, k_h, pad, stride);
    const int64_t out_w = conv_out_dim(W_in, k_w, pad, stride);
    if (out_h <= 0 || out_w <= 0) {
        return -3;
    }

    const int64_t k_dim = C_in * k_h * k_w;
    const int64_t m_dim = N * out_h * out_w;

    const int32_t im2col_rc = im2col_avx2(
        x, col_buf, N, C_in, H, W_in, W_in_stride, k_h, k_w, stride, pad);
    if (im2col_rc != 0) {
        return im2col_rc;
    }

    // Serial phase only — never call inside #pragma omp parallel (OpenBLAS → 1 thread).
    blas_gemm_forward(col_buf, W_fwd, gemm_buf, m_dim, C_out, k_dim);

    fuse_forward_transpose_bias(gemm_buf, bias, out, N, C_out, out_h, out_w, out_w_stride);
    if (out_w_stride > out_w) {
        const int32_t omp_n = omp_threads_for_phase();
        #pragma omp parallel for num_threads(omp_n) collapse(4) schedule(static) if(omp_n > 1)
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C_out; ++c) {
                for (int64_t h = 0; h < out_h; ++h) {
                    for (int64_t w = out_w; w < out_w_stride; ++w) {
                        out[((n * C_out + c) * out_h + h) * out_w_stride + w] = 0.0f;
                    }
                }
            }
        }
    }

    if (fuse_relu) {
        relu_fwd_inplace(out, N * C_out * out_h * out_w_stride);
    }
    return 0;
}

__declspec(dllexport) int32_t conv2d_backward_weight_im2col_gemm_avx2(
    const float* dout,
    const float* x,
    float* dW,
    float* col_buf,
    float* dout_trans_buf,
    int64_t N,
    int64_t C_in,
    int64_t H,
    int64_t W_in,
    int64_t W_in_stride,
    int64_t C_out,
    int64_t k_h,
    int64_t k_w,
    int64_t stride,
    int64_t pad,
    int64_t dout_w_stride,
    float inv_m,
    int32_t col_valid,
    int32_t dout_trans_valid
) {
    if (!dout || !x || !dW || !col_buf || !dout_trans_buf) {
        return -1;
    }
    const int32_t blas_rc = require_blas();
    if (blas_rc != 0) {
        return blas_rc;
    }

    const int64_t out_h = conv_out_dim(H, k_h, pad, stride);
    const int64_t out_w = conv_out_dim(W_in, k_w, pad, stride);
    const int64_t m_dim = N * out_h * out_w;
    const int64_t k_dim = C_in * k_h * k_w;

    if (!dout_trans_valid) {
        fuse_dout_transpose(dout, dout_trans_buf, N, C_out, out_h, out_w, dout_w_stride);
    }

    if (!col_valid) {
        const int32_t im2col_rc = im2col_avx2(
            x, col_buf, N, C_in, H, W_in, W_in_stride, k_h, k_w, stride, pad);
        if (im2col_rc != 0) {
            return im2col_rc;
        }
    }

    // Serial phase only (im2col parallel region has ended).
    blas_gemm_param_grad(dout_trans_buf, col_buf, dW, m_dim, C_out, k_dim, inv_m);
    return 0;
}

__declspec(dllexport) int32_t conv2d_backward_input_im2col_gemm_avx2(
    const float* dout,
    const float* W,
    const float* in_act,
    float* dx,
    float* dout_trans_buf,
    float* dcol_buf,
    int64_t N,
    int64_t C_in,
    int64_t H,
    int64_t W_in,
    int64_t W_in_stride,
    int64_t C_out,
    int64_t k_h,
    int64_t k_w,
    int64_t stride,
    int64_t pad,
    int64_t dout_w_stride,
    int32_t fuse_relu,
    int32_t dout_trans_valid
) {
    if (!dout || !W || !dx || !dout_trans_buf || !dcol_buf) {
        return -1;
    }
    if (fuse_relu && !in_act) {
        return -2;
    }
    const int32_t blas_rc = require_blas();
    if (blas_rc != 0) {
        return blas_rc;
    }

    const int64_t out_h = conv_out_dim(H, k_h, pad, stride);
    const int64_t out_w = conv_out_dim(W_in, k_w, pad, stride);
    const int64_t m_dim = N * out_h * out_w;
    const int64_t k_dim = C_in * k_h * k_w;

    if (!dout_trans_valid) {
        fuse_dout_transpose(dout, dout_trans_buf, N, C_out, out_h, out_w, dout_w_stride);
    }
    // Serial phase only (fuse_dout_transpose parallel region has ended).
    blas_gemm_backward_input(dout_trans_buf, W, dcol_buf, m_dim, C_out, k_dim);

    const int32_t col2im_rc = col2im_avx2(
        dcol_buf, dx, N, C_in, H, W_in, W_in_stride, k_h, k_w, stride, pad);
    if (col2im_rc != 0) {
        return col2im_rc;
    }

    if (fuse_relu) {
        relu_bwd_inplace(dx, in_act, N * C_in * H * W_in_stride);
    }
    return 0;
}

}  // extern "C"
