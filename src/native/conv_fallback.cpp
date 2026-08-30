#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <omp.h>

static inline float _mm256_reduce_add_ps(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 v128  = _mm_add_ps(vlow, vhigh);
    __m128 shuf  = _mm_movehdup_ps(v128);
    __m128 sums  = _mm_add_ps(v128, shuf);
    shuf         = _mm_movehl_ps(shuf, sums);
    sums         = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// Fixed-size forward work document: 4 output channels x 8 output columns x 1 row.
static constexpr int64_t FWD_TILE_OW   = 8;
static constexpr int64_t FWD_TILE_COUT = 4;

struct ConvFwdTileDoc {
    int64_t n;
    int64_t cout0;
    int64_t oh;
    int64_t ow;
    int8_t  cout_count;
    int8_t  ow_count;
    int8_t  middle_zone;
};

static inline ConvFwdTileDoc decode_fwd_tile_doc(
    int64_t tid, int64_t N, int64_t C_out, int64_t out_h, int64_t out_w,
    int64_t ow_safe_start, int64_t ow_safe_end
) {
    const int64_t cout_blks = (C_out + FWD_TILE_COUT - 1) / FWD_TILE_COUT;
    const int64_t ow_tiles  = (out_w + FWD_TILE_OW - 1) / FWD_TILE_OW;

    int64_t t = tid;
    const int64_t ow_tile = t % ow_tiles; t /= ow_tiles;
    const int64_t oh       = t % out_h;   t /= out_h;
    const int64_t cout_blk = t % cout_blks; t /= cout_blks;
    const int64_t n        = t;

    const int64_t cout0 = cout_blk * FWD_TILE_COUT;
    const int64_t ow    = ow_tile * FWD_TILE_OW;

    ConvFwdTileDoc doc{};
    doc.n          = n;
    doc.cout0      = cout0;
    doc.oh         = oh;
    doc.ow         = ow;
    doc.cout_count = (int8_t)std::min((int64_t)FWD_TILE_COUT, C_out - cout0);
    doc.ow_count   = (int8_t)std::min((int64_t)FWD_TILE_OW, out_w - ow);
    doc.middle_zone = (int8_t)(
        doc.ow_count == FWD_TILE_OW &&
        ow >= ow_safe_start &&
        (ow + FWD_TILE_OW) <= ow_safe_end
    );
    return doc;
}

// Algorithm kernel: one fixed-layout tile, no OpenMP.
static void process_fwd_tile_stride1(
    const ConvFwdTileDoc& doc,
    const float* __restrict x,
    const float* __restrict W,
    float* __restrict out,
    int64_t C_in, int64_t C_out, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t k_h, int64_t k_w, int64_t pad,
    int64_t spatial_in, int64_t spatial_out, int64_t k_spatial, int64_t out_w_stride
) {
    const int64_t ih_base = doc.oh - pad;
    const int64_t c_rem   = doc.cout_count;

    float* __restrict out_r0 = &out[(doc.n * C_out + doc.cout0 + 0) * spatial_out + doc.oh * out_w_stride + doc.ow];
    float* __restrict out_r1 = (c_rem > 1) ? &out[(doc.n * C_out + doc.cout0 + 1) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;
    float* __restrict out_r2 = (c_rem > 2) ? &out[(doc.n * C_out + doc.cout0 + 2) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;
    float* __restrict out_r3 = (c_rem > 3) ? &out[(doc.n * C_out + doc.cout0 + 3) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;

    __m256 vo0 = _mm256_loadu_ps(out_r0);
    __m256 vo1 = (c_rem > 1) ? _mm256_loadu_ps(out_r1) : _mm256_setzero_ps();
    __m256 vo2 = (c_rem > 2) ? _mm256_loadu_ps(out_r2) : _mm256_setzero_ps();
    __m256 vo3 = (c_rem > 3) ? _mm256_loadu_ps(out_r3) : _mm256_setzero_ps();

    const float* xp_base = &x[doc.n * C_in * spatial_in];

    if (doc.middle_zone) {
        for (int64_t cin = 0; cin < C_in; ++cin) {
            const float* __restrict xp  = xp_base + cin * spatial_in;
            const float* __restrict wp0 = &W[((doc.cout0 + 0) * C_in + cin) * k_spatial];
            const float* __restrict wp1 = (c_rem > 1) ? &W[((doc.cout0 + 1) * C_in + cin) * k_spatial] : nullptr;
            const float* __restrict wp2 = (c_rem > 2) ? &W[((doc.cout0 + 2) * C_in + cin) * k_spatial] : nullptr;
            const float* __restrict wp3 = (c_rem > 3) ? &W[((doc.cout0 + 3) * C_in + cin) * k_spatial] : nullptr;

            for (int64_t kh = 0; kh < k_h; ++kh) {
                const int64_t ih = ih_base + kh;
                if (ih < 0 || ih >= H) continue;

                const float* __restrict in_row = xp + ih * W_in_stride;
                for (int64_t kw = 0; kw < k_w; ++kw) {
                    const int64_t iw = doc.ow - pad + kw;
                    const __m256 vx  = _mm256_loadu_ps(&in_row[iw]);
                    const __m256 vw0 = _mm256_set1_ps(wp0[kh * k_w + kw]);
                    vo0 = _mm256_fmadd_ps(vx, vw0, vo0);
                    if (c_rem > 1) vo1 = _mm256_fmadd_ps(vx, _mm256_set1_ps(wp1[kh * k_w + kw]), vo1);
                    if (c_rem > 2) vo2 = _mm256_fmadd_ps(vx, _mm256_set1_ps(wp2[kh * k_w + kw]), vo2);
                    if (c_rem > 3) vo3 = _mm256_fmadd_ps(vx, _mm256_set1_ps(wp3[kh * k_w + kw]), vo3);
                }
            }
        }
    } else {
        const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        __m256i out_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);

        for (int64_t cin = 0; cin < C_in; ++cin) {
            const float* __restrict xp  = xp_base + cin * spatial_in;
            const float* __restrict wp0 = &W[((doc.cout0 + 0) * C_in + cin) * k_spatial];
            const float* __restrict wp1 = (c_rem > 1) ? &W[((doc.cout0 + 1) * C_in + cin) * k_spatial] : nullptr;
            const float* __restrict wp2 = (c_rem > 2) ? &W[((doc.cout0 + 2) * C_in + cin) * k_spatial] : nullptr;
            const float* __restrict wp3 = (c_rem > 3) ? &W[((doc.cout0 + 3) * C_in + cin) * k_spatial] : nullptr;

            for (int64_t kh = 0; kh < k_h; ++kh) {
                const int64_t ih = ih_base + kh;
                if (ih < 0 || ih >= H) continue;

                const float* __restrict in_row = xp + ih * W_in_stride;
                for (int64_t kw = 0; kw < k_w; ++kw) {
                    const int64_t iw_base_k = doc.ow - pad + kw;
                    __m256i viw = _mm256_add_epi32(_mm256_set1_epi32((int)iw_base_k), v_idx);
                    __m256i m1 = _mm256_cmpgt_epi32(viw, _mm256_set1_epi32(-1));
                    __m256i m2 = _mm256_cmpgt_epi32(_mm256_set1_epi32(W_in), viw);
                    __m256i in_mask = _mm256_and_si256(_mm256_and_si256(m1, m2), out_mask);

                    const __m256 vx = _mm256_maskload_ps(&in_row[iw_base_k], in_mask);
                    vo0 = _mm256_fmadd_ps(vx, _mm256_set1_ps(wp0[kh * k_w + kw]), vo0);
                    if (c_rem > 1) vo1 = _mm256_fmadd_ps(vx, _mm256_set1_ps(wp1[kh * k_w + kw]), vo1);
                    if (c_rem > 2) vo2 = _mm256_fmadd_ps(vx, _mm256_set1_ps(wp2[kh * k_w + kw]), vo2);
                    if (c_rem > 3) vo3 = _mm256_fmadd_ps(vx, _mm256_set1_ps(wp3[kh * k_w + kw]), vo3);
                }
            }
        }

        _mm256_maskstore_ps(out_r0, out_mask, vo0);
        if (c_rem > 1) _mm256_maskstore_ps(out_r1, out_mask, vo1);
        if (c_rem > 2) _mm256_maskstore_ps(out_r2, out_mask, vo2);
        if (c_rem > 3) _mm256_maskstore_ps(out_r3, out_mask, vo3);
        return;
    }

    _mm256_storeu_ps(out_r0, vo0);
    if (c_rem > 1) _mm256_storeu_ps(out_r1, vo1);
    if (c_rem > 2) _mm256_storeu_ps(out_r2, vo2);
    if (c_rem > 3)     _mm256_storeu_ps(out_r3, vo3);
}

// Fixed-size backward work documents (8-wide output columns x 1 row).
struct ConvBwdDxTileDoc {
    int64_t n;
    int64_t cin;
    int64_t oh;
    int64_t ow;
    int8_t  ow_count;
    int8_t  middle_zone;
};

static inline ConvBwdDxTileDoc decode_bwd_dx_tile_doc(
    int64_t tid, int64_t N, int64_t C_in, int64_t H, int64_t W_in
) {
    const int64_t iw_tiles = (W_in + FWD_TILE_OW - 1) / FWD_TILE_OW;

    int64_t t = tid;
    const int64_t iw_tile = t % iw_tiles; t /= iw_tiles;
    const int64_t ih       = t % H;        t /= H;
    const int64_t cin      = t % C_in;     t /= C_in;
    const int64_t n        = t;

    const int64_t iw = iw_tile * FWD_TILE_OW;

    ConvBwdDxTileDoc doc{};
    doc.n     = n;
    doc.cin   = cin;
    doc.oh    = ih;
    doc.ow    = iw;
    doc.ow_count = (int8_t)std::min((int64_t)FWD_TILE_OW, W_in - iw);
    doc.middle_zone = (int8_t)(doc.ow_count == FWD_TILE_OW);
    return doc;
}

static inline __m256 fmadd_dx_cout_sum_stride1(
    __m256 v_dx, __m256 r0,
    const float* __restrict W,
    int64_t C_out, int64_t C_in, int64_t cin, int64_t k_spatial,
    int64_t tap
) {
    int64_t cout = 0;
    for (; cout + 3 < C_out; cout += 4) {
        const float* __restrict w_base = &W[(cout * C_in + cin) * k_spatial + tap];
        const float w_sum = w_base[0 * C_in * k_spatial]
                          + w_base[1 * C_in * k_spatial]
                          + w_base[2 * C_in * k_spatial]
                          + w_base[3 * C_in * k_spatial];
        v_dx = _mm256_fmadd_ps(r0, _mm256_set1_ps(w_sum), v_dx);
    }
    for (; cout < C_out; ++cout) {
        v_dx = _mm256_fmadd_ps(
            r0, _mm256_set1_ps(W[(cout * C_in + cin) * k_spatial + tap]), v_dx);
    }
    return v_dx;
}

static void process_bwd_dx_tile_stride1(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict d_conv_buf,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t C_out, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t k_h, int64_t k_w, int64_t pad,
    int64_t spatial_in, int64_t conv_spatial, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w, int64_t conv_out_w_stride
) {
    float* __restrict dx_row = &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];

    if (doc.middle_zone) {
        __m256 v_dx = _mm256_loadu_ps(dx_row);

        for (int64_t kh = 0; kh < k_h; ++kh) {
            const int64_t oh = doc.oh + pad - kh;
            if (oh < 0 || oh >= conv_out_h) continue;

            const float* __restrict dp_row = &d_conv_buf[doc.n * C_out * conv_spatial + oh * conv_out_w_stride];
            for (int64_t kw = 0; kw < k_w; ++kw) {
                const int64_t ow = doc.ow + pad - kw;
                if (ow < 0 || (ow + FWD_TILE_OW) > conv_out_w) continue;

                const __m256 r0 = _mm256_loadu_ps(&dp_row[ow]);
                const int64_t tap = kh * k_w + kw;
                v_dx = fmadd_dx_cout_sum_stride1(
                    v_dx, r0, W, C_out, C_in, doc.cin, k_spatial, tap);
            }
        }

        _mm256_storeu_ps(dx_row, v_dx);
        return;
    }

    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    __m256 v_dx = _mm256_maskload_ps(dx_row, dx_mask);

    for (int64_t kh = 0; kh < k_h; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        if (oh < 0 || oh >= conv_out_h) continue;

        const float* __restrict dp_row = &d_conv_buf[doc.n * C_out * conv_spatial + oh * conv_out_w_stride];
        for (int64_t kw = 0; kw < k_w; ++kw) {
            const int64_t ow_base = doc.ow + pad - kw;
            __m256i vow = _mm256_add_epi32(_mm256_set1_epi32((int)ow_base), v_idx);
            __m256i m1 = _mm256_cmpgt_epi32(vow, _mm256_set1_epi32(-1));
            __m256i m2 = _mm256_cmpgt_epi32(_mm256_set1_epi32(conv_out_w), vow);
            __m256i valid = _mm256_and_si256(_mm256_and_si256(m1, m2), dx_mask);

            alignas(32) float dr_tmp[8] = {0};
            for (int lane = 0; lane < doc.ow_count; ++lane) {
                const int64_t ow = ow_base + lane;
                if (ow >= 0 && ow < conv_out_w) {
                    dr_tmp[lane] = dp_row[ow];
                }
            }
            const __m256 r0 = _mm256_maskload_ps(dr_tmp, valid);
            const int64_t tap = kh * k_w + kw;
            v_dx = fmadd_dx_cout_sum_stride1(
                v_dx, r0, W, C_out, C_in, doc.cin, k_spatial, tap);
        }
    }

    _mm256_maskstore_ps(dx_row, dx_mask, v_dx);
}

static inline void decode_dw_nci_task(
    int64_t task_id, int64_t N, int64_t C_out, int64_t C_in,
    int64_t& n, int64_t& cout, int64_t& cin
) {
    int64_t t = task_id;
    cin  = t % C_in;  t /= C_in;
    cout = t % C_out; t /= C_out;
    n    = t;
}

static void process_dw_nci_task(
    int64_t n, int64_t cout, int64_t cin,
    float* __restrict dw_slice,
    const float* __restrict d_conv_buf,
    const float* __restrict x,
    int64_t C_in, int64_t C_out, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t k_h, int64_t k_w, int64_t stride, int64_t pad,
    int64_t spatial_in, int64_t conv_spatial, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w, int64_t conv_out_w_stride,
    const __m256i& v_idx
) {
    const float* __restrict dp_n = &d_conv_buf[(n * C_out + cout) * conv_spatial];
    const float* __restrict xp_n = &x[(n * C_in + cin) * spatial_in];

    for (int64_t kh = 0; kh < k_h; ++kh) {
        int64_t kw = 0;

        if (stride == 1) {
            for (; kw + 3 < k_w; kw += 4) {
                const int64_t iw_base_0 = -pad + kw;
                const int64_t ow_start = std::max((int64_t)0, -iw_base_0);
                const int64_t ow_end   = std::min(conv_out_w, W_in - (-pad + kw + 3));
                const int64_t count    = ow_end - ow_start;
                if (count <= 0) break;

                const int64_t iw_start_0 = ow_start + iw_base_0;

                __m256 v_acc0 = _mm256_setzero_ps();
                __m256 v_acc1 = _mm256_setzero_ps();
                __m256 v_acc2 = _mm256_setzero_ps();
                __m256 v_acc3 = _mm256_setzero_ps();
                __m256 v_t0 = _mm256_setzero_ps();
                __m256 v_t1 = _mm256_setzero_ps();
                __m256 v_t2 = _mm256_setzero_ps();
                __m256 v_t3 = _mm256_setzero_ps();

                for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                    const int64_t ih = oh - pad + kh;
                    if (ih < 0 || ih >= H) continue;

                    const float* __restrict dr_row = &dp_n[oh * conv_out_w_stride];
                    const float* __restrict xr_row = &xp_n[ih * W_in_stride];

                    int64_t i = 0;
                    for (; i + 7 < count; i += 8) {
                        __m256 r0 = _mm256_loadu_ps(&dr_row[ow_start + i]);
                        __m256 x0 = _mm256_loadu_ps(&xr_row[iw_start_0 + i]);
                        __m256 x1 = _mm256_loadu_ps(&xr_row[iw_start_0 + i + 1]);
                        __m256 x2 = _mm256_loadu_ps(&xr_row[iw_start_0 + i + 2]);
                        __m256 x3 = _mm256_loadu_ps(&xr_row[iw_start_0 + i + 3]);
                        v_acc0 = _mm256_fmadd_ps(r0, x0, v_acc0);
                        v_acc1 = _mm256_fmadd_ps(r0, x1, v_acc1);
                        v_acc2 = _mm256_fmadd_ps(r0, x2, v_acc2);
                        v_acc3 = _mm256_fmadd_ps(r0, x3, v_acc3);
                    }

                    int64_t rem = count - i;
                    if (rem > 0) {
                        __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(rem), v_idx);
                        __m256 r0 = _mm256_maskload_ps(&dr_row[ow_start + i], mask);
                        __m256 x0 = _mm256_maskload_ps(&xr_row[iw_start_0 + i], mask);
                        __m256 x1 = _mm256_maskload_ps(&xr_row[iw_start_0 + i + 1], mask);
                        __m256 x2 = _mm256_maskload_ps(&xr_row[iw_start_0 + i + 2], mask);
                        __m256 x3 = _mm256_maskload_ps(&xr_row[iw_start_0 + i + 3], mask);
                        v_t0 = _mm256_fmadd_ps(r0, x0, v_t0);
                        v_t1 = _mm256_fmadd_ps(r0, x1, v_t1);
                        v_t2 = _mm256_fmadd_ps(r0, x2, v_t2);
                        v_t3 = _mm256_fmadd_ps(r0, x3, v_t3);
                    }
                }

                const int64_t base_idx = kh * k_w + kw;
                dw_slice[base_idx + 0] += _mm256_reduce_add_ps(_mm256_add_ps(v_acc0, v_t0));
                dw_slice[base_idx + 1] += _mm256_reduce_add_ps(_mm256_add_ps(v_acc1, v_t1));
                dw_slice[base_idx + 2] += _mm256_reduce_add_ps(_mm256_add_ps(v_acc2, v_t2));
                dw_slice[base_idx + 3] += _mm256_reduce_add_ps(_mm256_add_ps(v_acc3, v_t3));
            }
        }

        for (; kw < k_w; ++kw) {
            const int64_t tap_idx = kh * k_w + kw;

            if (stride == 1) {
                const int64_t iw_base = -pad + kw;
                const int64_t ow_start = std::max((int64_t)0, -iw_base);
                const int64_t ow_end   = std::min(conv_out_w, W_in - iw_base);
                const int64_t count    = ow_end - ow_start;
                if (count <= 0) continue;

                const int64_t iw_start = ow_start + iw_base;
                __m256 v_acc0 = _mm256_setzero_ps();
                __m256 v_tail = _mm256_setzero_ps();

                for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                    const int64_t ih = oh - pad + kh;
                    if (ih < 0 || ih >= H) continue;

                    const float* __restrict dr_row = &dp_n[oh * conv_out_w_stride];
                    const float* __restrict xr_row = &xp_n[ih * W_in_stride];

                    int64_t i = 0;
                    for (; i + 7 < count; i += 8) {
                        __m256 r0 = _mm256_loadu_ps(&dr_row[ow_start + i]);
                        __m256 x0 = _mm256_loadu_ps(&xr_row[iw_start + i]);
                        v_acc0 = _mm256_fmadd_ps(r0, x0, v_acc0);
                    }

                    int64_t rem = count - i;
                    if (rem > 0) {
                        __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(rem), v_idx);
                        __m256 r0 = _mm256_maskload_ps(&dr_row[ow_start + i], mask);
                        __m256 x0 = _mm256_maskload_ps(&xr_row[iw_start + i], mask);
                        v_tail = _mm256_fmadd_ps(r0, x0, v_tail);
                    }
                }
                dw_slice[tap_idx] += _mm256_reduce_add_ps(_mm256_add_ps(v_acc0, v_tail));

            } else {
                float tap_sum = 0.0f;
                const int64_t iw_base = -pad + kw;

                for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                    const int64_t ih = oh * stride - pad + kh;
                    if (ih < 0 || ih >= H) continue;

                    const float* __restrict dr_row = &dp_n[oh * conv_out_w_stride];
                    const float* __restrict xr_row = &xp_n[ih * W_in_stride];

                    for (int64_t ow = 0; ow < conv_out_w; ++ow) {
                        const int64_t iw = ow * stride + iw_base;
                        if (iw >= 0 && iw < W_in) {
                            tap_sum += dr_row[ow] * xr_row[iw];
                        }
                    }
                }
                dw_slice[tap_idx] += tap_sum;
            }
        }
    }
}

// ========================================================================
// Forward Pass: Bias Init + Tiled Batch Dispatch + Optional ReLU
// ========================================================================
void conv2d_forward_fallback_avx2(
    const float* x, const float* W, const float* bias, float* out,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t k_h, int64_t k_w, int64_t stride, int64_t pad,
    int64_t out_w_stride, int32_t fuse_relu
) {
    const int64_t out_h = (H + 2 * pad - k_h) / stride + 1;
    const int64_t out_w = (W_in + 2 * pad - k_w) / stride + 1;
    const int64_t spatial_out = out_h * out_w_stride;
    const int64_t spatial_in  = H * W_in_stride;
    const int64_t k_spatial   = k_h * k_w;

    const __m256 v_zero = _mm256_setzero_ps();

    // Phase 1: bias broadcast (unchanged layout, cheap)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t cout = 0; cout < C_out; ++cout) {
            float* __restrict out_ptr = &out[(n * C_out + cout) * spatial_out];
            const __m256 vb = bias ? _mm256_set1_ps(bias[cout]) : v_zero;

            int64_t sp = 0;
            for (; sp + 7 < spatial_out; sp += 8) {
                _mm256_storeu_ps(&out_ptr[sp], vb);
            }
            for (; sp < spatial_out; ++sp) {
                out_ptr[sp] = bias ? bias[cout] : 0.0f;
            }
        }
    }

    // Phase 2: uniform tile batch (algorithm separated from dispatch)
    if (stride == 1) {
        const int64_t cout_blks = (C_out + FWD_TILE_COUT - 1) / FWD_TILE_COUT;
        const int64_t ow_tiles  = (out_w + FWD_TILE_OW - 1) / FWD_TILE_OW;
        const int64_t tile_count = N * cout_blks * out_h * ow_tiles;
        const int64_t ow_safe_start = std::min(out_w, pad);
        const int64_t ow_safe_end   = std::max(ow_safe_start, out_w - pad);

        #pragma omp parallel for schedule(dynamic, 1)
        for (int64_t tid = 0; tid < tile_count; ++tid) {
            const ConvFwdTileDoc doc = decode_fwd_tile_doc(
                tid, N, C_out, out_h, out_w, ow_safe_start, ow_safe_end
            );
            process_fwd_tile_stride1(
                doc, x, W, out,
                C_in, C_out, H, W_in, W_in_stride,
                k_h, k_w, pad,
                spatial_in, spatial_out, k_spatial, out_w_stride
            );
        }
    } else {
        #pragma omp parallel for collapse(2) schedule(dynamic, 1)
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t cout = 0; cout < C_out; ++cout) {
                float* __restrict out_ptr = &out[(n * C_out + cout) * spatial_out];

                for (int64_t oh = 0; oh < out_h; ++oh) {
                    float* __restrict out_row = &out_ptr[oh * out_w_stride];
                    const int64_t ih_base = oh * stride - pad;

                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        float val = out_row[ow];
                        const float* xp_ptr = &x[n * C_in * spatial_in];
                        const float* wp_ptr = &W[cout * C_in * k_spatial];

                        for (int64_t cin = 0; cin < C_in; ++cin) {
                            for (int64_t kh = 0; kh < k_h; ++kh) {
                                const int64_t ih = ih_base + kh;
                                if (ih >= 0 && ih < H) {
                                    const float* in_row = xp_ptr + ih * W_in_stride;
                                    const float* w_row  = wp_ptr + kh * k_w;
                                    for (int64_t kw = 0; kw < k_w; ++kw) {
                                        const int64_t iw = ow * stride - pad + kw;
                                        if (iw >= 0 && iw < W_in) {
                                            val += in_row[iw] * w_row[kw];
                                        }
                                    }
                                }
                            }
                            xp_ptr += spatial_in;
                            wp_ptr += k_spatial;
                        }
                        out_row[ow] = val;
                    }
                }
            }
        }
    }

    // Phase 3: optional ReLU
    if (fuse_relu) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t cout = 0; cout < C_out; ++cout) {
                float* __restrict out_ptr = &out[(n * C_out + cout) * spatial_out];
                int64_t i = 0;
                for (; i + 7 < spatial_out; i += 8) {
                    __m256 v = _mm256_loadu_ps(&out_ptr[i]);
                    _mm256_storeu_ps(&out_ptr[i], _mm256_max_ps(v, v_zero));
                }
                for (; i < spatial_out; ++i) {
                    out_ptr[i] = std::max(out_ptr[i], 0.0f);
                }
            }
        }
    }
}

// ========================================================================
// Backward Pass: streamed dX + dW work queue (single OpenMP team)
// ========================================================================
void conv2d_backward_fallback_avx2(
    const float* d_conv_buf, const float* x, const float* W,
    float* dx, float* dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t k_h, int64_t k_w, int64_t stride, int64_t pad,
    int64_t conv_out_w_stride, float inv_m
) {
    const int64_t conv_out_h   = (H + 2 * pad - k_h) / stride + 1;
    const int64_t conv_out_w   = (W_in + 2 * pad - k_w) / stride + 1;
    const int64_t conv_spatial = conv_out_h * conv_out_w_stride;
    const int64_t spatial_in   = H * W_in_stride;
    const int64_t k_spatial    = k_h * k_w;

    const bool do_dx = (dx && W);
    const bool do_dw = (dW && x);

    if (do_dx) {
        std::memset(dx, 0, (size_t)(N * C_in * spatial_in) * sizeof(float));
    }
    if (do_dw) {
        std::memset(dW, 0, (size_t)(C_out * C_in * k_spatial) * sizeof(float));
    }

    const int64_t dw_count = C_out * C_in * k_spatial;
    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    const int64_t iw_tiles = (W_in + FWD_TILE_OW - 1) / FWD_TILE_OW;
    const int64_t dx_tile_count = N * C_in * H * iw_tiles;
    const int64_t dw_task_count = N * C_out * C_in;
    const bool stream_dx_dw = (do_dx && do_dw && stride == 1);

    #pragma omp parallel
    {
        thread_local static int64_t tls_dw_cap = 0;
        thread_local static float*  tls_priv_dW = nullptr;

        if (do_dw) {
            if (dw_count > tls_dw_cap) {
                std::free(tls_priv_dW);
                tls_priv_dW = (float*)std::malloc((size_t)dw_count * sizeof(float));
                tls_dw_cap = tls_priv_dW ? dw_count : 0;
            }
            if (tls_priv_dW) {
                std::memset(tls_priv_dW, 0, (size_t)dw_count * sizeof(float));
            }
        }

        if (stream_dx_dw && tls_priv_dW) {
            const int64_t work_total = dx_tile_count + dw_task_count;

            #pragma omp for schedule(dynamic, 1)
            for (int64_t wid = 0; wid < work_total; ++wid) {
                if (wid < dx_tile_count) {
                    const ConvBwdDxTileDoc doc = decode_bwd_dx_tile_doc(wid, N, C_in, H, W_in);
                    process_bwd_dx_tile_stride1(
                        doc, d_conv_buf, W, dx,
                        C_in, C_out, H, W_in, W_in_stride,
                        k_h, k_w, pad,
                        spatial_in, conv_spatial, k_spatial,
                        conv_out_h, conv_out_w, conv_out_w_stride
                    );
                } else {
                    const int64_t task_id = wid - dx_tile_count;
                    int64_t n, cout, cin;
                    decode_dw_nci_task(task_id, N, C_out, C_in, n, cout, cin);
                    float* __restrict dw_slice = &tls_priv_dW[(cout * C_in + cin) * k_spatial];
                    process_dw_nci_task(
                        n, cout, cin, dw_slice,
                        d_conv_buf, x,
                        C_in, C_out, H, W_in, W_in_stride,
                        k_h, k_w, stride, pad,
                        spatial_in, conv_spatial, k_spatial,
                        conv_out_h, conv_out_w, conv_out_w_stride,
                        v_idx
                    );
                }
            }

            #pragma omp critical(dw_batch_merge)
            {
                for (int64_t i = 0; i < dw_count; ++i) {
                    dW[i] += tls_priv_dW[i];
                }
            }
        } else {
            if (do_dx) {
                if (stride == 1) {
                    #pragma omp for schedule(dynamic, 1)
                    for (int64_t tid = 0; tid < dx_tile_count; ++tid) {
                        const ConvBwdDxTileDoc doc = decode_bwd_dx_tile_doc(tid, N, C_in, H, W_in);
                        process_bwd_dx_tile_stride1(
                            doc, d_conv_buf, W, dx,
                            C_in, C_out, H, W_in, W_in_stride,
                            k_h, k_w, pad,
                            spatial_in, conv_spatial, k_spatial,
                            conv_out_h, conv_out_w, conv_out_w_stride
                        );
                    }
                } else {
                    #pragma omp for collapse(2) schedule(dynamic, 1)
                    for (int64_t n = 0; n < N; ++n) {
                        for (int64_t cin = 0; cin < C_in; ++cin) {
                            float* __restrict dx_p = &dx[(n * C_in + cin) * spatial_in];

                            for (int64_t kh = 0; kh < k_h; ++kh) {
                                for (int64_t kw = 0; kw < k_w; ++kw) {
                                    const int64_t iw_base = -pad + kw;

                                    for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                                        const int64_t ih = oh * stride - pad + kh;
                                        if (ih < 0 || ih >= H) continue;

                                        float* __restrict dx_row = &dx_p[ih * W_in_stride];
                                        const float* dp_base_oh = &d_conv_buf[n * C_out * conv_spatial + oh * conv_out_w_stride];
                                        const float* wp_base_k  = &W[(cin * k_spatial) + kh * k_w + kw];

                                        for (int64_t ow = 0; ow < conv_out_w; ++ow) {
                                            const int64_t iw = ow * stride + iw_base;
                                            if (iw >= 0 && iw < W_in) {
                                                float dx_val = dx_row[iw];
                                                const float* dp_ptr = dp_base_oh + ow;
                                                const float* wp_ptr = wp_base_k;

                                                for (int64_t cout = 0; cout < C_out; ++cout) {
                                                    dx_val += (*dp_ptr) * (*wp_ptr);
                                                    dp_ptr += conv_spatial;
                                                    wp_ptr += C_in * k_spatial;
                                                }
                                                dx_row[iw] = dx_val;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (do_dw && tls_priv_dW) {
                #pragma omp for schedule(dynamic, 1)
                for (int64_t task_id = 0; task_id < dw_task_count; ++task_id) {
                    int64_t n, cout, cin;
                    decode_dw_nci_task(task_id, N, C_out, C_in, n, cout, cin);
                    float* __restrict dw_slice = &tls_priv_dW[(cout * C_in + cin) * k_spatial];
                    process_dw_nci_task(
                        n, cout, cin, dw_slice,
                        d_conv_buf, x,
                        C_in, C_out, H, W_in, W_in_stride,
                        k_h, k_w, stride, pad,
                        spatial_in, conv_spatial, k_spatial,
                        conv_out_h, conv_out_w, conv_out_w_stride,
                        v_idx
                    );
                }

                #pragma omp critical(dw_batch_merge)
                {
                    for (int64_t i = 0; i < dw_count; ++i) {
                        dW[i] += tls_priv_dW[i];
                    }
                }
            }
        }
    }

    if (do_dw) {
        for (int64_t i = 0; i < dw_count; ++i) {
            dW[i] *= inv_m;
        }
    }
}