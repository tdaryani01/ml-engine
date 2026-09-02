// src/native/im2col.cpp
// Image-to-column and column-to-image primitives for conv2d GEMM paths.
// Layout matches utils/im2col_fast.py (numba reference): row-major col[N*out_h*out_w, C*k_h*k_w].
//
// im2col: output-stationary nest (out_y/out_x outer) — sequential col-row writes, no memset.
// col2im: per-(n,c) parallel; sequential col-channel reads per output pixel (not input-stationary).
// Interior tiles (full k×k window in-bounds) skip per-pixel bounds checks for any k/stride/pad.
#include "omp_config.h"
#include "im2col_telemetry.h"
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <omp.h>

namespace {

inline int64_t im2col_out_dim(int64_t in, int64_t k, int64_t pad, int64_t stride) {
    return (in + 2 * pad - k) / stride + 1;
}

// True when the entire k_h×k_w window at (out_y, out_x) lies inside the input tensor.
inline bool full_tile_interior(
    int64_t out_y,
    int64_t out_x,
    int64_t k_h,
    int64_t k_w,
    int64_t stride,
    int64_t pad,
    int64_t H,
    int64_t W
) {
    const int64_t in_y_lo = out_y * stride - pad;
    const int64_t in_x_lo = out_x * stride - pad;
    return in_y_lo >= 0 && in_x_lo >= 0 &&
           in_y_lo + k_h <= H && in_x_lo + k_w <= W;
}

inline bool y_tile_interior(int64_t out_y, int64_t k_h, int64_t stride, int64_t pad, int64_t H) {
    const int64_t in_y_lo = out_y * stride - pad;
    return in_y_lo >= 0 && in_y_lo + k_h <= H;
}

inline bool x_tile_interior(int64_t out_x, int64_t k_w, int64_t stride, int64_t pad, int64_t W) {
    const int64_t in_x_lo = out_x * stride - pad;
    return in_x_lo >= 0 && in_x_lo + k_w <= W;
}

void im2col_row_fast(
    const float* x,
    float* row,
    int64_t n,
    int64_t C,
    int64_t H,
    int64_t W_logical,
    int64_t W_stride,
    int64_t k_h,
    int64_t k_w,
    int64_t stride,
    int64_t pad,
    int64_t out_y,
    int64_t out_x
) {
    const int64_t in_y0 = out_y * stride - pad;
    const int64_t in_x0 = out_x * stride - pad;
    float* dst = row;
    (void)W_logical;
    for (int64_t c = 0; c < C; ++c) {
        const int64_t x_plane = (n * C + c) * H;
        for (int64_t ky = 0; ky < k_h; ++ky) {
            const float* x_row = x + (x_plane + in_y0 + ky) * W_stride + in_x0;
            for (int64_t kx = 0; kx < k_w; ++kx) {
                *dst++ = x_row[kx];
            }
        }
    }
}

void im2col_row_padded(
    const float* x,
    float* row,
    int64_t n,
    int64_t C,
    int64_t H,
    int64_t W_logical,
    int64_t W_stride,
    int64_t k_h,
    int64_t k_w,
    int64_t stride,
    int64_t pad,
    int64_t out_y,
    int64_t out_x
) {
    float* dst = row;
    for (int64_t c = 0; c < C; ++c) {
        const int64_t x_plane = (n * C + c) * H;
        for (int64_t ky = 0; ky < k_h; ++ky) {
            const int64_t in_y = out_y * stride - pad + ky;
            for (int64_t kx = 0; kx < k_w; ++kx) {
                const int64_t in_x = out_x * stride - pad + kx;
                float val = 0.0f;
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W_logical) {
                    val = x[(x_plane + in_y) * W_stride + in_x];
                }
                *dst++ = val;
            }
        }
    }
}

void col2im_channel_fast(
    const float* row_c,
    float* dx,
    int64_t n,
    int64_t c,
    int64_t C,
    int64_t H,
    int64_t W_logical,
    int64_t W_stride,
    int64_t k_h,
    int64_t k_w,
    int64_t stride,
    int64_t pad,
    int64_t out_y,
    int64_t out_x
) {
    const float* src = row_c;
    const int64_t x_plane = (n * C + c) * H;
    const int64_t in_y0 = out_y * stride - pad;
    const int64_t in_x0 = out_x * stride - pad;
    (void)W_logical;
    for (int64_t ky = 0; ky < k_h; ++ky) {
        float* dx_row = dx + (x_plane + in_y0 + ky) * W_stride + in_x0;
        for (int64_t kx = 0; kx < k_w; ++kx) {
            dx_row[kx] += *src++;
        }
    }
}

// Y in-bounds; clip kx once (edge strips). row_c layout: [ky][kx] row-major k_h×k_w.
void col2im_channel_xclip(
    const float* row_c,
    float* dx,
    int64_t n,
    int64_t c,
    int64_t C,
    int64_t H,
    int64_t W_logical,
    int64_t W_stride,
    int64_t k_h,
    int64_t k_w,
    int64_t stride,
    int64_t pad,
    int64_t out_y,
    int64_t out_x
) {
    const int64_t x_plane = (n * C + c) * H;
    const int64_t in_y0 = out_y * stride - pad;
    const int64_t in_x0 = out_x * stride - pad;
    int64_t kx_lo = 0;
    int64_t kx_hi = k_w;
    if (in_x0 < 0) {
        kx_lo = -in_x0;
    }
    if (in_x0 + k_w > W_logical) {
        kx_hi = W_logical - in_x0;
    }
    const int64_t kx_count = kx_hi - kx_lo;
    for (int64_t ky = 0; ky < k_h; ++ky) {
        const float* src = row_c + ky * k_w + kx_lo;
        float* dx_row = dx + (x_plane + in_y0 + ky) * W_stride + in_x0 + kx_lo;
        for (int64_t i = 0; i < kx_count; ++i) {
            dx_row[i] += src[i];
        }
    }
    (void)C;
}

// X in-bounds; clip ky once (edge strips).
void col2im_channel_yclip(
    const float* row_c,
    float* dx,
    int64_t n,
    int64_t c,
    int64_t C,
    int64_t H,
    int64_t W_logical,
    int64_t W_stride,
    int64_t k_h,
    int64_t k_w,
    int64_t stride,
    int64_t pad,
    int64_t out_y,
    int64_t out_x
) {
    const int64_t x_plane = (n * C + c) * H;
    const int64_t in_y0 = out_y * stride - pad;
    const int64_t in_x0 = out_x * stride - pad;
    int64_t ky_lo = 0;
    int64_t ky_hi = k_h;
    if (in_y0 < 0) {
        ky_lo = -in_y0;
    }
    if (in_y0 + k_h > H) {
        ky_hi = H - in_y0;
    }
    for (int64_t ky = ky_lo; ky < ky_hi; ++ky) {
        const float* src = row_c + ky * k_w;
        float* dx_row = dx + (x_plane + in_y0 + ky) * W_stride + in_x0;
        for (int64_t kx = 0; kx < k_w; ++kx) {
            dx_row[kx] += src[kx];
        }
    }
    (void)C;
    (void)W_logical;
}

void col2im_channel_padded(
    const float* row_c,
    float* dx,
    int64_t n,
    int64_t c,
    int64_t C,
    int64_t H,
    int64_t W_logical,
    int64_t W_stride,
    int64_t k_h,
    int64_t k_w,
    int64_t stride,
    int64_t pad,
    int64_t out_y,
    int64_t out_x
) {
    const float* src = row_c;
    const int64_t x_plane = (n * C + c) * H;
    for (int64_t ky = 0; ky < k_h; ++ky) {
        const int64_t in_y = out_y * stride - pad + ky;
        for (int64_t kx = 0; kx < k_w; ++kx) {
            const int64_t in_x = out_x * stride - pad + kx;
            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W_logical) {
                dx[(x_plane + in_y) * W_stride + in_x] += *src;
            }
            ++src;
        }
    }
}

void col2im_scatter_nc(
    const float* col,
    float* dx,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W_logical,
    int64_t W_stride,
    int64_t k_h,
    int64_t k_w,
    int64_t stride,
    int64_t pad,
    int64_t n,
    int64_t c,
    int64_t spatial_out,
    int64_t out_h,
    int64_t out_w,
    int64_t col_cols,
    int64_t spatial_k,
    uint64_t& tiles_fast,
    uint64_t& tiles_xclip,
    uint64_t& tiles_yclip,
    uint64_t& tiles_corner
) {
    const int64_t n_row_base = n * spatial_out;
    const int64_t col_c_base = c * spatial_k;
    for (int64_t out_y = 0; out_y < out_h; ++out_y) {
        const int64_t row_y_base = n_row_base + out_y * out_w;
        const bool y_ok = y_tile_interior(out_y, k_h, stride, pad, H);
        for (int64_t out_x = 0; out_x < out_w; ++out_x) {
            const float* row_c = col + (row_y_base + out_x) * col_cols + col_c_base;
            if (y_ok && x_tile_interior(out_x, k_w, stride, pad, W_logical)) {
                ++tiles_fast;
                col2im_channel_fast(
                    row_c, dx, n, c, C, H, W_logical, W_stride,
                    k_h, k_w, stride, pad, out_y, out_x);
            } else if (y_ok) {
                ++tiles_xclip;
                col2im_channel_xclip(
                    row_c, dx, n, c, C, H, W_logical, W_stride,
                    k_h, k_w, stride, pad, out_y, out_x);
            } else if (x_tile_interior(out_x, k_w, stride, pad, W_logical)) {
                ++tiles_yclip;
                col2im_channel_yclip(
                    row_c, dx, n, c, C, H, W_logical, W_stride,
                    k_h, k_w, stride, pad, out_y, out_x);
            } else {
                ++tiles_corner;
                col2im_channel_padded(
                    row_c, dx, n, c, C, H, W_logical, W_stride,
                    k_h, k_w, stride, pad, out_y, out_x);
            }
        }
    }
}

void col2im_scatter(
    const float* col,
    float* dx,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W_logical,
    int64_t W_stride,
    int64_t k_h,
    int64_t k_w,
    int64_t stride,
    int64_t pad
) {
    const int64_t out_h = im2col_out_dim(H, k_h, pad, stride);
    const int64_t out_w = im2col_out_dim(W_logical, k_w, pad, stride);
    const int64_t spatial_k = k_h * k_w;
    const int64_t spatial_out = out_h * out_w;
    const int64_t col_cols = C * spatial_k;

    uint64_t tiles_fast = 0;
    uint64_t tiles_xclip = 0;
    uint64_t tiles_yclip = 0;
    uint64_t tiles_corner = 0;

    ml_omp_before_parallel();
    const int32_t omp_n = get_omp_threads();
    if (omp_n > 1) {
        #pragma omp parallel for num_threads(omp_n) collapse(2) schedule(static) reduction(+:tiles_fast,tiles_xclip,tiles_yclip,tiles_corner)
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                col2im_scatter_nc(
                    col, dx, N, C, H, W_logical, W_stride, k_h, k_w, stride, pad,
                    n, c, spatial_out, out_h, out_w, col_cols, spatial_k,
                    tiles_fast, tiles_xclip, tiles_yclip, tiles_corner);
            }
        }
    } else {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                col2im_scatter_nc(
                    col, dx, N, C, H, W_logical, W_stride, k_h, k_w, stride, pad,
                    n, c, spatial_out, out_h, out_w, col_cols, spatial_k,
                    tiles_fast, tiles_xclip, tiles_yclip, tiles_corner);
            }
        }
    }
    im2col_telemetry::add_col2im_tiles(tiles_fast, tiles_xclip, tiles_yclip, tiles_corner);
}

void im2col_tile(
    const float* x,
    float* out,
    int64_t n,
    int64_t C,
    int64_t H,
    int64_t W_logical,
    int64_t W_stride,
    int64_t k_h,
    int64_t k_w,
    int64_t stride,
    int64_t pad,
    int64_t out_h,
    int64_t out_w,
    int64_t spatial_out,
    int64_t col_cols,
    int64_t out_y,
    uint64_t& tiles_fast,
    uint64_t& tiles_padded
) {
    const int64_t n_row_base = n * spatial_out;
    for (int64_t out_x = 0; out_x < out_w; ++out_x) {
        const int64_t row_idx = n_row_base + out_y * out_w + out_x;
        float* row = out + row_idx * col_cols;
        if (full_tile_interior(out_y, out_x, k_h, k_w, stride, pad, H, W_logical)) {
            ++tiles_fast;
            im2col_row_fast(
                x, row, n, C, H, W_logical, W_stride,
                k_h, k_w, stride, pad, out_y, out_x);
        } else {
            ++tiles_padded;
            im2col_row_padded(
                x, row, n, C, H, W_logical, W_stride,
                k_h, k_w, stride, pad, out_y, out_x);
        }
    }
}

}  // namespace

extern "C" {

__declspec(dllexport) int32_t im2col_avx2(
    const float* x,
    float* out,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W_logical,
    int64_t W_stride,
    int64_t k_h,
    int64_t k_w,
    int64_t stride,
    int64_t pad
) {
    if (!x || !out || N <= 0 || C <= 0 || H <= 0 || W_logical <= 0 || k_h <= 0 || k_w <= 0 || stride <= 0 || pad < 0) {
        return -1;
    }
    if (W_stride < W_logical) {
        return -3;
    }

    const int64_t out_h = im2col_out_dim(H, k_h, pad, stride);
    const int64_t out_w = im2col_out_dim(W_logical, k_w, pad, stride);
    if (out_h <= 0 || out_w <= 0) {
        return -2;
    }

    const int64_t spatial_k = k_h * k_w;
    const int64_t col_cols = C * spatial_k;
    const int64_t spatial_out = out_h * out_w;

    im2col_telemetry::on_im2col_call();
    im2col_telemetry::record_ptr_alignment(x, out, nullptr);
    if (im2col_telemetry::ptr_misaligned(out)) {
        im2col_telemetry::record_out_misalign();
    }

    uint64_t tiles_fast = 0;
    uint64_t tiles_padded = 0;

    ml_omp_before_parallel();
    const int32_t omp_n = get_omp_threads();
    if (omp_n > 1) {
        #pragma omp parallel for num_threads(omp_n) collapse(2) schedule(static) reduction(+:tiles_fast,tiles_padded)
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t out_y = 0; out_y < out_h; ++out_y) {
                im2col_tile(
                    x, out, n, C, H, W_logical, W_stride, k_h, k_w, stride, pad,
                    out_h, out_w, spatial_out, col_cols, out_y, tiles_fast, tiles_padded);
            }
        }
    } else {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t out_y = 0; out_y < out_h; ++out_y) {
                im2col_tile(
                    x, out, n, C, H, W_logical, W_stride, k_h, k_w, stride, pad,
                    out_h, out_w, spatial_out, col_cols, out_y, tiles_fast, tiles_padded);
            }
        }
    }

    im2col_telemetry::add_im2col_tiles(tiles_fast, tiles_padded);
    return 0;
}

__declspec(dllexport) int32_t col2im_avx2(
    const float* col,
    float* dx,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W_logical,
    int64_t W_stride,
    int64_t k_h,
    int64_t k_w,
    int64_t stride,
    int64_t pad
) {
    if (!col || !dx || N <= 0 || C <= 0 || H <= 0 || W_logical <= 0 || k_h <= 0 || k_w <= 0 || stride <= 0 || pad < 0) {
        return -1;
    }
    if (W_stride < W_logical) {
        return -3;
    }

    const int64_t out_h = im2col_out_dim(H, k_h, pad, stride);
    const int64_t out_w = im2col_out_dim(W_logical, k_w, pad, stride);
    if (out_h <= 0 || out_w <= 0) {
        return -2;
    }

    uint64_t memset_bytes = 0;
    const char* memset_env = std::getenv("ML_ENGINE_COL2IM_MEMSET");
    const bool do_memset = memset_env &&
        (std::strcmp(memset_env, "1") == 0 ||
         std::strcmp(memset_env, "true") == 0 ||
         std::strcmp(memset_env, "TRUE") == 0 ||
         std::strcmp(memset_env, "yes") == 0);
    if (do_memset) {
        const int64_t dx_elems = N * C * H * W_stride;
        memset_bytes = static_cast<uint64_t>(dx_elems) * sizeof(float);
        std::memset(dx, 0, static_cast<size_t>(dx_elems) * sizeof(float));
    }
    im2col_telemetry::on_col2im_call(memset_bytes);
    im2col_telemetry::record_ptr_alignment(nullptr, col, dx);

    col2im_scatter(col, dx, N, C, H, W_logical, W_stride, k_h, k_w, stride, pad);
    return 0;
}

}  // extern "C"
