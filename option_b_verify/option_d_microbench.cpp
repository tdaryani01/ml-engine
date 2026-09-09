// Standalone microbenchmark: does C_out-blocking (Option D) beat the current
// NCHW sliding-window crawl (Approach A, same as production stride1_bwd_dx_
// tile_c8_k7 / fmadd_dx_cout4) for layer 1's backward-dX?
//
// Layer 1 dims (config.yaml conv[0]): C_in=3, C_out=8, K=7, pad=1, stride=1,
// H_in=W_in=28, H_out=W_out=24, N=32. C_in=3 fails the C_in%8==0 gate used by
// Option B, so this layer stays on the crawl path in production today.
//
// Approach A: mimics conv_fallback.cpp's stride1_bwd_dx_accum_cout4_k7 exactly
//             (vector lanes = 8 spatial output pixels, dY loaded via loadu at
//             kw offsets, weight is a scalar broadcast, C_out unrolled 4-at-a-time).
// Approach D: C_out-blocked. Vector lanes = 8 output channels (C_out), all at
//             ONE fixed spatial position. dY pre-transposed to [N][H_out][W_out]
//             [C_out] (channel-last, so 8 C_out values at one pixel are
//             contiguous/aligned). W pre-transposed to [C_in][K][K][C_out]
//             (also channel-last). Requires one horizontal reduction per
//             output element (dX is now the *reduced* result, not the vector
//             lanes) -- this is the main extra cost vs Approach A.
//
// Both are checked against a naive scalar reference before any timing is
// trusted.

#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>

static constexpr int64_t N = 32;
static constexpr int64_t C_in = 3;
static constexpr int64_t C_out = 8;
static constexpr int64_t K = 7;
static constexpr int64_t PAD = 1;
static constexpr int64_t H_in = 28, W_in = 28;
static constexpr int64_t H_out = 24, W_out = 24; // (28+2*1-7)/1+1 = 24

// ---------------- Reference (naive scalar) ----------------
void reference_dx(const float* dY, const float* W, float* dX) {
    std::memset(dX, 0, sizeof(float) * N * C_in * H_in * W_in);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t cin = 0; cin < C_in; ++cin) {
            for (int64_t ih = 0; ih < H_in; ++ih) {
                for (int64_t iw = 0; iw < W_in; ++iw) {
                    float acc = 0.0f;
                    for (int64_t cout = 0; cout < C_out; ++cout) {
                        for (int64_t kh = 0; kh < K; ++kh) {
                            int64_t oh = ih - kh + PAD;
                            if (oh < 0 || oh >= H_out) continue;
                            for (int64_t kw = 0; kw < K; ++kw) {
                                int64_t ow = iw - kw + PAD;
                                if (ow < 0 || ow >= W_out) continue;
                                float dy = dY[((n * C_out + cout) * H_out + oh) * W_out + ow];
                                float w = W[((cout * C_in + cin) * K + kh) * K + kw];
                                acc += dy * w;
                            }
                        }
                    }
                    dX[((n * C_in + cin) * H_in + ih) * W_in + iw] = acc;
                }
            }
        }
    }
}

// ---------------- Approach A: NCHW crawl (current engine code, K=7, C_out=8) ----------------
static inline __m256 fmadd_dx_cout4(
    __m256 v_dx, __m256 r0, __m256 r1, __m256 r2, __m256 r3,
    float w0, float w1, float w2, float w3
) {
    return _mm256_fmadd_ps(
        r0, _mm256_set1_ps(w0),
        _mm256_fmadd_ps(r1, _mm256_set1_ps(w1),
            _mm256_fmadd_ps(r2, _mm256_set1_ps(w2),
                _mm256_fmadd_ps(r3, _mm256_set1_ps(w3), v_dx))));
}

static constexpr int64_t PAD_L = K - 1 - PAD; // 5
static constexpr int64_t OW_TILE = 8;
static constexpr int64_t DY_ROW_STRIDE = PAD_L + W_out + OW_TILE;

void build_dy_pad(const float* dY, float* dY_pad) {
    std::memset(dY_pad, 0, sizeof(float) * N * C_out * H_out * DY_ROW_STRIDE);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C_out; ++c)
            for (int64_t oh = 0; oh < H_out; ++oh)
                std::memcpy(
                    &dY_pad[((n * C_out + c) * H_out + oh) * DY_ROW_STRIDE + PAD_L],
                    &dY[((n * C_out + c) * H_out + oh) * W_out],
                    sizeof(float) * W_out
                );
}

void approach_a_dx(const float* dY_pad, const float* W, float* dX) {
    std::memset(dX, 0, sizeof(float) * N * C_in * H_in * W_in);
    alignas(32) float dx_tile[OW_TILE];
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t cin = 0; cin < C_in; ++cin) {
            for (int64_t ih = 0; ih < H_in; ++ih) {
                for (int64_t iw_tile = 0; iw_tile < W_in; iw_tile += OW_TILE) {
                    int64_t tile_w = std::min<int64_t>(OW_TILE, W_in - iw_tile);
                    __m256 v_dx = _mm256_setzero_ps();
                    int64_t ow0 = iw_tile + PAD + PAD_L;
                    for (int64_t cout = 0; cout + 3 < C_out; cout += 4) {
                        for (int64_t kh = 0; kh < K; ++kh) {
                            int64_t oh = ih - kh + PAD;
                            if (oh < 0 || oh >= H_out) continue;
                            const float* dy0 = &dY_pad[((n * C_out + cout + 0) * H_out + oh) * DY_ROW_STRIDE];
                            const float* dy1 = &dY_pad[((n * C_out + cout + 1) * H_out + oh) * DY_ROW_STRIDE];
                            const float* dy2 = &dY_pad[((n * C_out + cout + 2) * H_out + oh) * DY_ROW_STRIDE];
                            const float* dy3 = &dY_pad[((n * C_out + cout + 3) * H_out + oh) * DY_ROW_STRIDE];
                            const float* w0p = &W[((cout + 0) * C_in + cin) * K * K + kh * K];
                            const float* w1p = &W[((cout + 1) * C_in + cin) * K * K + kh * K];
                            const float* w2p = &W[((cout + 2) * C_in + cin) * K * K + kh * K];
                            const float* w3p = &W[((cout + 3) * C_in + cin) * K * K + kh * K];
                            for (int64_t kw = 0; kw < K; ++kw) {
                                v_dx = fmadd_dx_cout4(
                                    v_dx,
                                    _mm256_loadu_ps(dy0 + ow0 - kw), _mm256_loadu_ps(dy1 + ow0 - kw),
                                    _mm256_loadu_ps(dy2 + ow0 - kw), _mm256_loadu_ps(dy3 + ow0 - kw),
                                    w0p[kw], w1p[kw], w2p[kw], w3p[kw]
                                );
                            }
                        }
                    }
                    _mm256_storeu_ps(dx_tile, v_dx);
                    for (int64_t t = 0; t < tile_w; ++t) {
                        dX[((n * C_in + cin) * H_in + ih) * W_in + iw_tile + t] = dx_tile[t];
                    }
                }
            }
        }
    }
}

// ---------------- Approach D: C_out-blocked ----------------
// dY_b layout: [N][H_out][W_out][C_out] (channel-last, C_out=8 contiguous)
void transpose_dy_cout_last(const float* dY, float* dY_b) {
    for (int64_t n = 0; n < N; ++n)
        for (int64_t oh = 0; oh < H_out; ++oh)
            for (int64_t ow = 0; ow < W_out; ++ow)
                for (int64_t cout = 0; cout < C_out; ++cout)
                    dY_b[((n * H_out + oh) * W_out + ow) * C_out + cout] =
                        dY[((n * C_out + cout) * H_out + oh) * W_out + ow];
}

// W_b layout: [C_in][K][K][C_out] (channel-last, C_out=8 contiguous)
void transpose_w_cout_last(const float* W, float* W_b) {
    for (int64_t cin = 0; cin < C_in; ++cin)
        for (int64_t kh = 0; kh < K; ++kh)
            for (int64_t kw = 0; kw < K; ++kw)
                for (int64_t cout = 0; cout < C_out; ++cout)
                    W_b[((cin * K + kh) * K + kw) * C_out + cout] =
                        W[((cout * C_in + cin) * K + kh) * K + kw];
}

static inline float reduce_add_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

void approach_d_dx(const float* dY_b, const float* W_b, float* dX) {
    std::memset(dX, 0, sizeof(float) * N * C_in * H_in * W_in);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t cin = 0; cin < C_in; ++cin) {
            const float* __restrict w_cin = &W_b[cin * K * K * C_out];
            for (int64_t ih = 0; ih < H_in; ++ih) {
                for (int64_t iw = 0; iw < W_in; ++iw) {
                    __m256 acc = _mm256_setzero_ps();
                    for (int64_t kh = 0; kh < K; ++kh) {
                        int64_t oh = ih - kh + PAD;
                        if (oh < 0 || oh >= H_out) continue;
                        const float* __restrict dy_row = &dY_b[(n * H_out + oh) * W_out * C_out];
                        const float* __restrict w_kh = w_cin + kh * K * C_out;
                        for (int64_t kw = 0; kw < K; ++kw) {
                            int64_t ow = iw - kw + PAD;
                            if (ow < 0 || ow >= W_out) continue;
                            __m256 dyv = _mm256_loadu_ps(dy_row + ow * C_out);
                            __m256 wv = _mm256_loadu_ps(w_kh + kw * C_out);
                            acc = _mm256_fmadd_ps(dyv, wv, acc);
                        }
                    }
                    dX[((n * C_in + cin) * H_in + ih) * W_in + iw] = reduce_add_ps(acc);
                }
            }
        }
    }
}

float max_abs_diff(const float* a, const float* b, size_t count) {
    float m = 0.0f;
    for (size_t i = 0; i < count; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> dY(N * C_out * H_out * W_out);
    std::vector<float> W(C_out * C_in * K * K);
    for (auto& v : dY) v = dist(rng);
    for (auto& v : W) v = dist(rng);

    std::vector<float> dX_ref(N * C_in * H_in * W_in);
    reference_dx(dY.data(), W.data(), dX_ref.data());

    // --- Approach A ---
    std::vector<float> dY_pad(N * C_out * H_out * DY_ROW_STRIDE);
    build_dy_pad(dY.data(), dY_pad.data());
    std::vector<float> dX_a(N * C_in * H_in * W_in);
    approach_a_dx(dY_pad.data(), W.data(), dX_a.data());
    float err_a = max_abs_diff(dX_ref.data(), dX_a.data(), dX_ref.size());
    printf("Approach A max abs err vs reference: %e\n", err_a);

    // --- Approach D ---
    std::vector<float> dY_b(N * H_out * W_out * C_out);
    transpose_dy_cout_last(dY.data(), dY_b.data());
    std::vector<float> W_b(C_in * K * K * C_out);
    transpose_w_cout_last(W.data(), W_b.data());
    std::vector<float> dX_d(N * C_in * H_in * W_in);
    approach_d_dx(dY_b.data(), W_b.data(), dX_d.data());
    float err_d = max_abs_diff(dX_ref.data(), dX_d.data(), dX_ref.size());
    printf("Approach D max abs err vs reference: %e\n", err_d);

    if (err_a > 1e-3f || err_d > 1e-3f) {
        printf("CORRECTNESS FAILURE -- refusing to trust timing.\n");
        return 1;
    }

    // --- Timing ---
    const int REPS = 30;
    const int WARMUP = 3;

    for (int i = 0; i < WARMUP; ++i) approach_a_dx(dY_pad.data(), W.data(), dX_a.data());
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < REPS; ++i) approach_a_dx(dY_pad.data(), W.data(), dX_a.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double a_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / REPS;

    // Include transpose cost in Approach D's timed total -- it has to happen
    // every call in production (weights change every step; dY is fresh every
    // call), so amortizing it away would be misleading.
    for (int i = 0; i < WARMUP; ++i) {
        transpose_dy_cout_last(dY.data(), dY_b.data());
        transpose_w_cout_last(W.data(), W_b.data());
        approach_d_dx(dY_b.data(), W_b.data(), dX_d.data());
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < REPS; ++i) {
        transpose_dy_cout_last(dY.data(), dY_b.data());
        transpose_w_cout_last(W.data(), W_b.data());
        approach_d_dx(dY_b.data(), W_b.data(), dX_d.data());
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double d_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / REPS;

    // Also report Approach D compute-only (no transpose), to separate "is the
    // gather+reduce kernel itself faster" from "is the transpose overhead too
    // costly", since dY transpose could plausibly be fused into an existing
    // pass in production (unlike this microbench).
    for (int i = 0; i < WARMUP; ++i) approach_d_dx(dY_b.data(), W_b.data(), dX_d.data());
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < REPS; ++i) approach_d_dx(dY_b.data(), W_b.data(), dX_d.data());
    auto t5 = std::chrono::high_resolution_clock::now();
    double d_compute_only_ms = std::chrono::duration<double, std::milli>(t5 - t4).count() / REPS;

    printf("\nApproach A (current crawl):         %.4f ms/call\n", a_ms);
    printf("Approach D (cout-blocked, +transpose): %.4f ms/call  (%+.1f%% vs A)\n",
           d_ms, (d_ms / a_ms - 1.0) * 100.0);
    printf("Approach D compute-only (no transpose): %.4f ms/call  (%+.1f%% vs A)\n",
           d_compute_only_ms, (d_compute_only_ms / a_ms - 1.0) * 100.0);

    return 0;
}
