#include <immintrin.h>
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#ifndef __forceinline
#define __forceinline inline __attribute__((always_inline))
#endif
#endif
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <omp.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// Diagnostic: BWD_DX_MOCK_EDGES=1 skips left/right dX tiles (dx stays 0 there).
static bool bwd_dx_mock_edges_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("BWD_DX_MOCK_EDGES");
        cached = (env && env[0] == '1' && env[1] == '\0') ? 1 : 0;
    }
    return cached != 0;
}

static void log_bwd_dx_mock_edges_once() {
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
        std::printf("[BWD_DX_MOCK] LR edge tiles skipped (dx left zero)\n");
        std::fflush(stdout);
    }
}

static bool bwd_queue_stats_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("BWD_QUEUE_STATS");
        cached = (env && env[0] == '1' && env[1] == '\0') ? 1 : 0;
    }
    return cached != 0;
}

static bool fwd_queue_stats_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("FWD_QUEUE_STATS");
        cached = (env && env[0] == '1' && env[1] == '\0') ? 1 : 0;
    }
    return cached != 0;
}

static constexpr int QUEUE_STATS_MAX_THREADS = 64;

struct BwdQueueThreadStats {
    int64_t dx = 0;
    int64_t dw = 0;
    uint64_t dx_cycles = 0;
    uint64_t dw_cycles = 0;
    uint64_t dx_min_cycles = UINT64_MAX;
    uint64_t dx_max_cycles = 0;
    uint64_t dw_min_cycles = UINT64_MAX;
    uint64_t dw_max_cycles = 0;
};

static inline uint64_t bwd_rdtsc() {
    return static_cast<uint64_t>(__rdtsc());
}

static double bwd_tsc_ghz() {
    static double ghz = 0.0;
    if (ghz > 0.0) {
        return ghz;
    }
    if (const char* env = std::getenv("BWD_TSC_GHZ")) {
        ghz = std::atof(env);
        if (ghz > 0.0) {
            return ghz;
        }
    }
#ifdef _WIN32
    LARGE_INTEGER freq{};
    LARGE_INTEGER q0{};
    LARGE_INTEGER q1{};
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&q0);
    const uint64_t c0 = bwd_rdtsc();
    const LONGLONG target = q0.QuadPart + (freq.QuadPart / 10); // ~100 ms busy
    do {
        QueryPerformanceCounter(&q1);
    } while (q1.QuadPart < target);
    const uint64_t c1 = bwd_rdtsc();
    const double sec = static_cast<double>(q1.QuadPart - q0.QuadPart) / static_cast<double>(freq.QuadPart);
#else
    const uint64_t c0 = bwd_rdtsc();
    const auto t0 = std::chrono::steady_clock::now();
    const auto until = t0 + std::chrono::milliseconds(100);
    while (std::chrono::steady_clock::now() < until) {
        /* spin */
    }
    const uint64_t c1 = bwd_rdtsc();
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
#endif
    ghz = (sec > 0.0) ? (static_cast<double>(c1 - c0) / sec / 1e9) : 3.0;
    return ghz;
}

static inline double bwd_cycles_to_ns(uint64_t cycles) {
    const double ghz = bwd_tsc_ghz();
    return ghz > 0.0 ? (static_cast<double>(cycles) / ghz) : 0.0;
}

struct BwdOverlapProof {
    std::atomic<uint64_t> dx_first{UINT64_MAX};
    std::atomic<uint64_t> dw_first{UINT64_MAX};
    std::atomic<uint64_t> dx_last{0};
    std::atomic<uint64_t> dw_last{0};
    std::atomic<int> dx_active{0};
    std::atomic<int> dw_active{0};
    std::atomic<int64_t> concurrent_enters{0};
};

static inline void bwd_overlap_note_dx_start(BwdOverlapProof* p) {
    if (!p) {
        return;
    }
    const uint64_t t = bwd_rdtsc();
    uint64_t prev = p->dx_first.load(std::memory_order_relaxed);
    while (t < prev && !p->dx_first.compare_exchange_weak(prev, t, std::memory_order_relaxed)) {}
    if (p->dw_active.load(std::memory_order_relaxed) > 0) {
        p->concurrent_enters.fetch_add(1, std::memory_order_relaxed);
    }
    p->dx_active.fetch_add(1, std::memory_order_relaxed);
}

static inline void bwd_overlap_note_dx_end(BwdOverlapProof* p) {
    if (!p) {
        return;
    }
    const uint64_t t = bwd_rdtsc();
    uint64_t prev = p->dx_last.load(std::memory_order_relaxed);
    while (t > prev && !p->dx_last.compare_exchange_weak(prev, t, std::memory_order_relaxed)) {}
    p->dx_active.fetch_sub(1, std::memory_order_relaxed);
}

static inline void bwd_overlap_note_dw_start(BwdOverlapProof* p) {
    if (!p) {
        return;
    }
    const uint64_t t = bwd_rdtsc();
    uint64_t prev = p->dw_first.load(std::memory_order_relaxed);
    while (t < prev && !p->dw_first.compare_exchange_weak(prev, t, std::memory_order_relaxed)) {}
    if (p->dx_active.load(std::memory_order_relaxed) > 0) {
        p->concurrent_enters.fetch_add(1, std::memory_order_relaxed);
    }
    p->dw_active.fetch_add(1, std::memory_order_relaxed);
}

static inline void bwd_overlap_note_dw_end(BwdOverlapProof* p) {
    if (!p) {
        return;
    }
    const uint64_t t = bwd_rdtsc();
    uint64_t prev = p->dw_last.load(std::memory_order_relaxed);
    while (t > prev && !p->dw_last.compare_exchange_weak(prev, t, std::memory_order_relaxed)) {}
    p->dw_active.fetch_sub(1, std::memory_order_relaxed);
}

static void log_bwd_overlap_proof(const BwdOverlapProof* p) {
    const uint64_t dx0 = p->dx_first.load();
    const uint64_t dx1 = p->dx_last.load();
    const uint64_t dw0 = p->dw_first.load();
    const uint64_t dw1 = p->dw_last.load();
    const bool overlap = (dx0 != UINT64_MAX && dw0 != UINT64_MAX && dx0 < dw1 && dw0 < dx1);
    uint64_t overlap_cycles = 0;
    if (overlap) {
        const uint64_t start = std::max(dx0, dw0);
        const uint64_t end = std::min(dx1, dw1);
        if (end > start) {
            overlap_cycles = end - start;
        }
    }
    const double ghz = bwd_tsc_ghz();
    std::printf(
        "[BWD_OVERLAP] dx=[%llu,%llu] dw=[%llu,%llu] overlap=%s "
        "overlap_cycles=%llu (%.1f us) concurrent_tile_starts=%lld\n",
        (unsigned long long)dx0, (unsigned long long)dx1,
        (unsigned long long)dw0, (unsigned long long)dw1,
        overlap ? "YES" : "NO",
        (unsigned long long)overlap_cycles,
        overlap_cycles / (ghz * 1000.0),
        (long long)p->concurrent_enters.load()
    );
    std::fflush(stdout);
}

static inline void decode_stream_work_item(
    int64_t wid, int64_t dx_count, int64_t dw_count,
    bool& is_dx, int64_t& local_id
);

static void log_bwd_queue_plan(
    int64_t N, int64_t C_in, int64_t C_out, int64_t H, int64_t W_in,
    int64_t dx_count, int64_t dw_count, int64_t work_total,
    int64_t chunk, int nthreads
) {
    std::printf(
        "[BWD_QUEUE_PLAN] N=%lld Cin=%lld Cout=%lld H=%lld W=%lld | "
        "queue_len=%lld dx=%lld dw=%lld chunk=%lld threads=%d\n",
        (long long)N, (long long)C_in, (long long)C_out,
        (long long)H, (long long)W_in,
        (long long)work_total, (long long)dx_count, (long long)dw_count,
        (long long)chunk, nthreads
    );

    const int64_t pattern_len = std::min((int64_t)48, work_total);
    std::printf("[BWD_QUEUE_PLAN] wid[0..%lld): ", (long long)pattern_len);
    for (int64_t wid = 0; wid < pattern_len; ++wid) {
        bool is_dx = false;
        int64_t local_id = 0;
        decode_stream_work_item(wid, dx_count, dw_count, is_dx, local_id);
        std::printf("%c", is_dx ? 'd' : 'w');
    }
    std::printf("\n");

    int64_t dw_hist[9] = {};
    int64_t max_dw = 0;
    int64_t chunks_all_dw = 0;
    const int64_t num_chunks = (work_total + chunk - 1) / chunk;
    for (int64_t c = 0; c < num_chunks; ++c) {
        const int64_t c0 = c * chunk;
        const int64_t c1 = std::min(c0 + chunk, work_total);
        int64_t dx_in = 0;
        int64_t dw_in = 0;
        for (int64_t wid = c0; wid < c1; ++wid) {
            bool is_dx = false;
            int64_t local_id = 0;
            decode_stream_work_item(wid, dx_count, dw_count, is_dx, local_id);
            if (is_dx) ++dx_in;
            else ++dw_in;
        }
        if (dw_in <= 8) ++dw_hist[dw_in];
        if (dw_in > max_dw) max_dw = dw_in;
        if (dw_in == (c1 - c0)) ++chunks_all_dw;
    }

    std::printf(
        "[BWD_QUEUE_PLAN] omp_chunk_dw_hist (dx+dw=%lld slots): "
        "0dw=%lld 1dw=%lld 2dw=%lld 3dw=%lld 4dw=%lld "
        "5dw=%lld 6dw=%lld 7dw=%lld 8dw=%lld | max_dw=%lld all_dw_chunks=%lld/%lld\n",
        (long long)chunk,
        (long long)dw_hist[0], (long long)dw_hist[1], (long long)dw_hist[2],
        (long long)dw_hist[3], (long long)dw_hist[4], (long long)dw_hist[5],
        (long long)dw_hist[6], (long long)dw_hist[7], (long long)dw_hist[8],
        (long long)max_dw, (long long)chunks_all_dw, (long long)num_chunks
    );
    std::fflush(stdout);
}

static void log_bwd_queue_runtime(
    const BwdQueueThreadStats* stats, int nthreads,
    int64_t expect_dx, int64_t expect_dw
) {
    int64_t sum_dx = 0;
    int64_t sum_dw = 0;
    std::printf("[BWD_QUEUE_RUNTIME] per_thread (actual work executed):\n");
    for (int t = 0; t < nthreads; ++t) {
        const int64_t dx = stats[t].dx;
        const int64_t dw = stats[t].dw;
        sum_dx += dx;
        sum_dw += dw;
        std::printf(
            "  t%d: dx=%lld dw=%lld total=%lld\n",
            t, (long long)dx, (long long)dw, (long long)(dx + dw)
        );
    }
    std::printf(
        "[BWD_QUEUE_RUNTIME] sum dx=%lld dw=%lld (expect dx=%lld dw=%lld)\n",
        (long long)sum_dx, (long long)sum_dw,
        (long long)expect_dx, (long long)expect_dw
    );

    uint64_t dx_cycles = 0;
    uint64_t dw_cycles = 0;
    uint64_t dx_min = UINT64_MAX;
    uint64_t dx_max = 0;
    uint64_t dw_min = UINT64_MAX;
    uint64_t dw_max = 0;
    for (int t = 0; t < nthreads; ++t) {
        dx_cycles += stats[t].dx_cycles;
        dw_cycles += stats[t].dw_cycles;
        if (stats[t].dx_min_cycles < dx_min) {
            dx_min = stats[t].dx_min_cycles;
        }
        if (stats[t].dx_max_cycles > dx_max) {
            dx_max = stats[t].dx_max_cycles;
        }
        if (stats[t].dw_min_cycles < dw_min) {
            dw_min = stats[t].dw_min_cycles;
        }
        if (stats[t].dw_max_cycles > dw_max) {
            dw_max = stats[t].dw_max_cycles;
        }
    }
    if (dx_min == UINT64_MAX) {
        dx_min = 0;
    }
    if (dw_min == UINT64_MAX) {
        dw_min = 0;
    }

    const double ghz = bwd_tsc_ghz();
    const uint64_t dx_avg_cycles = sum_dx > 0 ? (dx_cycles / static_cast<uint64_t>(sum_dx)) : 0;
    const uint64_t dw_avg_cycles = sum_dw > 0 ? (dw_cycles / static_cast<uint64_t>(sum_dw)) : 0;
    const double dx_avg_ns = bwd_cycles_to_ns(dx_avg_cycles);
    const double dw_avg_ns = bwd_cycles_to_ns(dw_avg_cycles);
    const double ratio = (dx_avg_cycles > 0)
        ? (static_cast<double>(dw_avg_cycles) / static_cast<double>(dx_avg_cycles))
        : 0.0;

    std::printf("[BWD_QUEUE_TIMING] tsc_ghz=%.3f (override with BWD_TSC_GHZ)\n", ghz);
    std::printf(
        "[BWD_QUEUE_TIMING] dx: n=%lld total_cycles=%llu avg_cycles=%llu avg_ns=%.1f min_cycles=%llu max_cycles=%llu\n",
        (long long)sum_dx, (unsigned long long)dx_cycles,
        (unsigned long long)dx_avg_cycles, dx_avg_ns,
        (unsigned long long)dx_min, (unsigned long long)dx_max
    );
    std::printf(
        "[BWD_QUEUE_TIMING] dw: n=%lld total_cycles=%llu avg_cycles=%llu avg_ns=%.1f min_cycles=%llu max_cycles=%llu\n",
        (long long)sum_dw, (unsigned long long)dw_cycles,
        (unsigned long long)dw_avg_cycles, dw_avg_ns,
        (unsigned long long)dw_min, (unsigned long long)dw_max
    );
    std::printf(
        "[BWD_QUEUE_TIMING] dw/dx avg_cycles_per_slot=%.2fx\n",
        ratio
    );
    std::fflush(stdout);
}

static void log_fwd_queue_runtime(
    const int64_t* tiles_per_thread, int nthreads,
    int64_t tile_count, int64_t chunk
) {
    int64_t sum = 0;
    int64_t min_t = tile_count;
    int64_t max_t = 0;
    std::printf(
        "[FWD_QUEUE_RUNTIME] tile_count=%lld chunk=%lld threads=%d\n",
        (long long)tile_count, (long long)chunk, nthreads
    );
    for (int t = 0; t < nthreads; ++t) {
        const int64_t n = tiles_per_thread[t];
        sum += n;
        if (n < min_t) min_t = n;
        if (n > max_t) max_t = n;
        std::printf("  t%d: tiles=%lld\n", t, (long long)n);
    }
    std::printf(
        "[FWD_QUEUE_RUNTIME] sum=%lld min=%lld max=%lld spread=%lld\n",
        (long long)sum, (long long)min_t, (long long)max_t,
        (long long)(max_t - min_t)
    );
    std::fflush(stdout);
}

static inline float _mm256_reduce_add_ps(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 v128  = _mm_add_ps(vlow, vhigh);
    v128         = _mm_hadd_ps(v128, v128);
    v128         = _mm_hadd_ps(v128, v128);
    return _mm_cvtss_f32(v128);
}

static constexpr int64_t FWD_TILE_OW   = 8;
static constexpr int64_t FWD_TILE_COUT = 4;

static inline __m256i bwd_dw_lane_mask(int64_t ow_count) {
    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    return _mm256_cmpgt_epi32(_mm256_set1_epi32((int)ow_count), v_idx);
}

static inline bool bwd_dw_ow_strip_is_interior(
    int64_t ow, int64_t ow_count, int64_t k_w, int64_t pad,
    int64_t conv_out_w, int64_t W_in
) {
    if (ow_count != FWD_TILE_OW) return false;
    if (ow < 0 || (ow + FWD_TILE_OW) > conv_out_w) return false;
    const int64_t iw_lo = ow - pad;
    const int64_t iw_hi = ow + (k_w - 1 - pad) + FWD_TILE_OW;
    return iw_lo >= 0 && iw_hi <= W_in;
}

struct BwdDwOwStripInfo {
    int64_t ow;
    int64_t ow_count;
    bool full_ow;
    bool strip_interior;
};

static inline void bwd_dw_build_strip_info(
    BwdDwOwStripInfo* strips, int& n_strips,
    int64_t conv_out_w, int64_t W_in, int64_t k_w, int64_t pad
) {
    const int64_t ow_tiles = (conv_out_w + FWD_TILE_OW - 1) / FWD_TILE_OW;
    n_strips = 0;
    for (int64_t t = 0; t < ow_tiles; ++t) {
        const int64_t ow = t * FWD_TILE_OW;
        const int64_t ow_count = std::min((int64_t)FWD_TILE_OW, conv_out_w - ow);
        strips[n_strips++] = {
            ow,
            ow_count,
            ow_count == FWD_TILE_OW,
            bwd_dw_ow_strip_is_interior(ow, ow_count, k_w, pad, conv_out_w, W_in)
        };
    }
}

static inline void bwd_dw_fmadd_kw4(
    __m256 dy8, const float* __restrict x_row, int64_t ow, int64_t pad, int64_t kw,
    __m256& v0, __m256& v1, __m256& v2, __m256& v3
) {
    const int64_t iw0 = ow - pad + kw;
    v0 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 0), v0);
    v1 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 1), v1);
    v2 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 2), v2);
    v3 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 3), v3);
}

// Fixed-size forward work document: 4 output channels x 8 output columns x 1 row.
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

// --- Stride-1 padded-buffer kernel specialists (template<int K>) ---
// Only explicit specializations are implemented. Runtime dispatch via
// stride1_specialist_k() / stride1_try_*_specialist().

struct ConvBwdDxTileDoc;

template<int K>
struct Stride1Specialist;

static inline int64_t stride1_specialist_k(int64_t k_h, int64_t k_w) {
    if (k_h != k_w) {
        return 0;
    }
    switch (k_h) {
        case 1: return 1;
        case 2: return 2;
        case 3: return 3;
        case 4: return 4;
        case 5: return 5;
        case 6: return 6;
        case 7: return 7;
        default: return 0;
    }
}

static inline bool stride1_fwd_builds_x_pad(int64_t k_h, int64_t k_w) {
    return stride1_specialist_k(k_h, k_w) != 0;
}

template<int K>
static inline void bwd_dx_k_kh_bounds(
    int64_t ih, int64_t pad, int64_t conv_out_h, int64_t& kh_lo, int64_t& kh_hi
) {
    kh_lo = ih + pad - conv_out_h + 1;
    if (kh_lo < 0) {
        kh_lo = 0;
    }
    kh_hi = ih + pad + 1;
    if (kh_hi > K) {
        kh_hi = K;
    }
}

static inline void bwd_dx_store_tile(
    float* __restrict dx_row, __m256 v_dx, bool full_dx, __m256i dx_mask
) {
    if (full_dx) {
        _mm256_storeu_ps(dx_row, v_dx);
    } else {
        _mm256_maskstore_ps(dx_row, dx_mask, v_dx);
    }
}

template<>
struct Stride1Specialist<1> {
    static constexpr int K = 1;

    static void fwd_tile(
        const ConvFwdTileDoc& doc,
        const float* __restrict x_pad_buf,
        int64_t x_pad_l, int64_t x_row_stride,
        const float* __restrict W,
        float* __restrict out,
        int64_t C_in, int64_t C_out, int64_t H,
        int64_t pad, int64_t k_spatial, int64_t spatial_out, int64_t out_w_stride
    ) {
        const int64_t ih_base = doc.oh - pad;
        const int64_t c_rem   = doc.cout_count;
        const int64_t x_plane = H * x_row_stride;

        float* __restrict out_r0 = &out[(doc.n * C_out + doc.cout0 + 0) * spatial_out + doc.oh * out_w_stride + doc.ow];
        float* __restrict out_r1 = (c_rem > 1) ? &out[(doc.n * C_out + doc.cout0 + 1) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;
        float* __restrict out_r2 = (c_rem > 2) ? &out[(doc.n * C_out + doc.cout0 + 2) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;
        float* __restrict out_r3 = (c_rem > 3) ? &out[(doc.n * C_out + doc.cout0 + 3) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;

        __m256 vo0 = _mm256_loadu_ps(out_r0);
        __m256 vo1 = (c_rem > 1) ? _mm256_loadu_ps(out_r1) : _mm256_setzero_ps();
        __m256 vo2 = (c_rem > 2) ? _mm256_loadu_ps(out_r2) : _mm256_setzero_ps();
        __m256 vo3 = (c_rem > 3) ? _mm256_loadu_ps(out_r3) : _mm256_setzero_ps();

        const bool full_ow = (doc.ow_count == FWD_TILE_OW);
        const __m256i out_mask = bwd_dw_lane_mask(doc.ow_count);

        const float* __restrict xp_base = &x_pad_buf[doc.n * C_in * x_plane];
        const int64_t iw0 = x_pad_l + doc.ow - pad;

        for (int64_t cin = 0; cin < C_in; ++cin) {
            const float* __restrict xp  = xp_base + cin * x_plane;
            const float* __restrict wp0 = &W[((doc.cout0 + 0) * C_in + cin) * k_spatial];
            const float* __restrict wp1 = (c_rem > 1) ? &W[((doc.cout0 + 1) * C_in + cin) * k_spatial] : nullptr;
            const float* __restrict wp2 = (c_rem > 2) ? &W[((doc.cout0 + 2) * C_in + cin) * k_spatial] : nullptr;
            const float* __restrict wp3 = (c_rem > 3) ? &W[((doc.cout0 + 3) * C_in + cin) * k_spatial] : nullptr;

            for (int64_t kh = 0; kh < K; ++kh) {
                const int64_t ih = ih_base + kh;
                if (ih < 0 || ih >= H) {
                    continue;
                }

                const float* __restrict in_row = xp + ih * x_row_stride;
                const __m256 vx0 = _mm256_loadu_ps(in_row + iw0 + 0);

                const float* __restrict w0 = wp0 + kh * K;
                vo0 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w0[0]), vo0);

                if (c_rem > 1) {
                    const float* __restrict w1 = wp1 + kh * K;
                    vo1 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w1[0]), vo1);
                }
                if (c_rem > 2) {
                    const float* __restrict w2 = wp2 + kh * K;
                    vo2 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w2[0]), vo2);
                }
                if (c_rem > 3) {
                    const float* __restrict w3 = wp3 + kh * K;
                    vo3 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w3[0]), vo3);
                }
            }
        }

        if (full_ow) {
            _mm256_storeu_ps(out_r0, vo0);
            if (c_rem > 1) _mm256_storeu_ps(out_r1, vo1);
            if (c_rem > 2) _mm256_storeu_ps(out_r2, vo2);
            if (c_rem > 3) _mm256_storeu_ps(out_r3, vo3);
        } else {
            _mm256_maskstore_ps(out_r0, out_mask, vo0);
            if (c_rem > 1) _mm256_maskstore_ps(out_r1, out_mask, vo1);
            if (c_rem > 2) _mm256_maskstore_ps(out_r2, out_mask, vo2);
            if (c_rem > 3) _mm256_maskstore_ps(out_r3, out_mask, vo3);
        }
    }

    static void bwd_dx_tile(
        const ConvBwdDxTileDoc& doc,
        const float* __restrict dy_pad_buf,
        int64_t dy_pad_l, int64_t dy_row_stride,
        const float* __restrict W,
        float* __restrict dx,
        int64_t C_in, int64_t C_out, int64_t W_in_stride,
        int64_t pad, int64_t spatial_in, int64_t k_spatial,
        int64_t conv_out_h, int64_t conv_out_w
    );

    static void dw_nci(
        int64_t n, int64_t cout, int64_t cin,
        float* __restrict dw_slice,
        const float* __restrict dy_pad_buf,
        const float* __restrict x_pad_buf,
        int64_t C_in, int64_t C_out,
        int64_t dy_pad_l, int64_t dy_row_stride,
        int64_t x_pad_l, int64_t x_row_stride,
        int64_t H, int64_t pad,
        int64_t conv_out_h, int64_t conv_out_w
    );
};

template<>
struct Stride1Specialist<2> {
    static constexpr int K = 2;

    static void fwd_tile(
        const ConvFwdTileDoc& doc,
        const float* __restrict x_pad_buf,
        int64_t x_pad_l, int64_t x_row_stride,
        const float* __restrict W,
        float* __restrict out,
        int64_t C_in, int64_t C_out, int64_t H,
        int64_t pad, int64_t k_spatial, int64_t spatial_out, int64_t out_w_stride
    ) {
        const int64_t ih_base = doc.oh - pad;
        const int64_t c_rem   = doc.cout_count;
        const int64_t x_plane = H * x_row_stride;

        float* __restrict out_r0 = &out[(doc.n * C_out + doc.cout0 + 0) * spatial_out + doc.oh * out_w_stride + doc.ow];
        float* __restrict out_r1 = (c_rem > 1) ? &out[(doc.n * C_out + doc.cout0 + 1) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;
        float* __restrict out_r2 = (c_rem > 2) ? &out[(doc.n * C_out + doc.cout0 + 2) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;
        float* __restrict out_r3 = (c_rem > 3) ? &out[(doc.n * C_out + doc.cout0 + 3) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;

        __m256 vo0 = _mm256_loadu_ps(out_r0);
        __m256 vo1 = (c_rem > 1) ? _mm256_loadu_ps(out_r1) : _mm256_setzero_ps();
        __m256 vo2 = (c_rem > 2) ? _mm256_loadu_ps(out_r2) : _mm256_setzero_ps();
        __m256 vo3 = (c_rem > 3) ? _mm256_loadu_ps(out_r3) : _mm256_setzero_ps();

        const bool full_ow = (doc.ow_count == FWD_TILE_OW);
        const __m256i out_mask = bwd_dw_lane_mask(doc.ow_count);

        const float* __restrict xp_base = &x_pad_buf[doc.n * C_in * x_plane];
        const int64_t iw0 = x_pad_l + doc.ow - pad;

        for (int64_t cin = 0; cin < C_in; ++cin) {
            const float* __restrict xp  = xp_base + cin * x_plane;
            const float* __restrict wp0 = &W[((doc.cout0 + 0) * C_in + cin) * k_spatial];
            const float* __restrict wp1 = (c_rem > 1) ? &W[((doc.cout0 + 1) * C_in + cin) * k_spatial] : nullptr;
            const float* __restrict wp2 = (c_rem > 2) ? &W[((doc.cout0 + 2) * C_in + cin) * k_spatial] : nullptr;
            const float* __restrict wp3 = (c_rem > 3) ? &W[((doc.cout0 + 3) * C_in + cin) * k_spatial] : nullptr;

            for (int64_t kh = 0; kh < K; ++kh) {
                const int64_t ih = ih_base + kh;
                if (ih < 0 || ih >= H) {
                    continue;
                }

                const float* __restrict in_row = xp + ih * x_row_stride;
                const __m256 vx0 = _mm256_loadu_ps(in_row + iw0 + 0);
                const __m256 vx1 = _mm256_loadu_ps(in_row + iw0 + 1);

                const float* __restrict w0 = wp0 + kh * K;
                vo0 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w0[0]), vo0);
                vo0 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w0[1]), vo0);

                if (c_rem > 1) {
                    const float* __restrict w1 = wp1 + kh * K;
                    vo1 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w1[0]), vo1);
                    vo1 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w1[1]), vo1);
                }
                if (c_rem > 2) {
                    const float* __restrict w2 = wp2 + kh * K;
                    vo2 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w2[0]), vo2);
                    vo2 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w2[1]), vo2);
                }
                if (c_rem > 3) {
                    const float* __restrict w3 = wp3 + kh * K;
                    vo3 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w3[0]), vo3);
                    vo3 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w3[1]), vo3);
                }
            }
        }

        if (full_ow) {
            _mm256_storeu_ps(out_r0, vo0);
            if (c_rem > 1) _mm256_storeu_ps(out_r1, vo1);
            if (c_rem > 2) _mm256_storeu_ps(out_r2, vo2);
            if (c_rem > 3) _mm256_storeu_ps(out_r3, vo3);
        } else {
            _mm256_maskstore_ps(out_r0, out_mask, vo0);
            if (c_rem > 1) _mm256_maskstore_ps(out_r1, out_mask, vo1);
            if (c_rem > 2) _mm256_maskstore_ps(out_r2, out_mask, vo2);
            if (c_rem > 3) _mm256_maskstore_ps(out_r3, out_mask, vo3);
        }
    }

    static void bwd_dx_tile(
        const ConvBwdDxTileDoc& doc,
        const float* __restrict dy_pad_buf,
        int64_t dy_pad_l, int64_t dy_row_stride,
        const float* __restrict W,
        float* __restrict dx,
        int64_t C_in, int64_t C_out, int64_t W_in_stride,
        int64_t pad, int64_t spatial_in, int64_t k_spatial,
        int64_t conv_out_h, int64_t conv_out_w
    );

    static void dw_nci(
        int64_t n, int64_t cout, int64_t cin,
        float* __restrict dw_slice,
        const float* __restrict dy_pad_buf,
        const float* __restrict x_pad_buf,
        int64_t C_in, int64_t C_out,
        int64_t dy_pad_l, int64_t dy_row_stride,
        int64_t x_pad_l, int64_t x_row_stride,
        int64_t H, int64_t pad,
        int64_t conv_out_h, int64_t conv_out_w
    );
};

template<>
struct Stride1Specialist<3> {
    static constexpr int K = 3;

    static void fwd_tile(
        const ConvFwdTileDoc& doc,
        const float* __restrict x_pad_buf,
        int64_t x_pad_l, int64_t x_row_stride,
        const float* __restrict W,
        float* __restrict out,
        int64_t C_in, int64_t C_out, int64_t H,
        int64_t pad, int64_t k_spatial, int64_t spatial_out, int64_t out_w_stride
    ) {
        const int64_t ih_base = doc.oh - pad;
        const int64_t c_rem   = doc.cout_count;
        const int64_t x_plane = H * x_row_stride;

        float* __restrict out_r0 = &out[(doc.n * C_out + doc.cout0 + 0) * spatial_out + doc.oh * out_w_stride + doc.ow];
        float* __restrict out_r1 = (c_rem > 1) ? &out[(doc.n * C_out + doc.cout0 + 1) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;
        float* __restrict out_r2 = (c_rem > 2) ? &out[(doc.n * C_out + doc.cout0 + 2) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;
        float* __restrict out_r3 = (c_rem > 3) ? &out[(doc.n * C_out + doc.cout0 + 3) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;

        __m256 vo0 = _mm256_loadu_ps(out_r0);
        __m256 vo1 = (c_rem > 1) ? _mm256_loadu_ps(out_r1) : _mm256_setzero_ps();
        __m256 vo2 = (c_rem > 2) ? _mm256_loadu_ps(out_r2) : _mm256_setzero_ps();
        __m256 vo3 = (c_rem > 3) ? _mm256_loadu_ps(out_r3) : _mm256_setzero_ps();

        const bool full_ow = (doc.ow_count == FWD_TILE_OW);
        const __m256i out_mask = bwd_dw_lane_mask(doc.ow_count);

        const float* __restrict xp_base = &x_pad_buf[doc.n * C_in * x_plane];
        const int64_t iw0 = x_pad_l + doc.ow - pad;

        for (int64_t cin = 0; cin < C_in; ++cin) {
            const float* __restrict xp  = xp_base + cin * x_plane;
            const float* __restrict wp0 = &W[((doc.cout0 + 0) * C_in + cin) * k_spatial];
            const float* __restrict wp1 = (c_rem > 1) ? &W[((doc.cout0 + 1) * C_in + cin) * k_spatial] : nullptr;
            const float* __restrict wp2 = (c_rem > 2) ? &W[((doc.cout0 + 2) * C_in + cin) * k_spatial] : nullptr;
            const float* __restrict wp3 = (c_rem > 3) ? &W[((doc.cout0 + 3) * C_in + cin) * k_spatial] : nullptr;

            for (int64_t kh = 0; kh < K; ++kh) {
                const int64_t ih = ih_base + kh;
                if (ih < 0 || ih >= H) {
                    continue;
                }

                const float* __restrict in_row = xp + ih * x_row_stride;
                const __m256 vx0 = _mm256_loadu_ps(in_row + iw0 + 0);
                const __m256 vx1 = _mm256_loadu_ps(in_row + iw0 + 1);
                const __m256 vx2 = _mm256_loadu_ps(in_row + iw0 + 2);

                const float* __restrict w0 = wp0 + kh * K;
                vo0 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w0[0]), vo0);
                vo0 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w0[1]), vo0);
                vo0 = _mm256_fmadd_ps(vx2, _mm256_set1_ps(w0[2]), vo0);

                if (c_rem > 1) {
                    const float* __restrict w1 = wp1 + kh * K;
                    vo1 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w1[0]), vo1);
                    vo1 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w1[1]), vo1);
                    vo1 = _mm256_fmadd_ps(vx2, _mm256_set1_ps(w1[2]), vo1);
                }
                if (c_rem > 2) {
                    const float* __restrict w2 = wp2 + kh * K;
                    vo2 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w2[0]), vo2);
                    vo2 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w2[1]), vo2);
                    vo2 = _mm256_fmadd_ps(vx2, _mm256_set1_ps(w2[2]), vo2);
                }
                if (c_rem > 3) {
                    const float* __restrict w3 = wp3 + kh * K;
                    vo3 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w3[0]), vo3);
                    vo3 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w3[1]), vo3);
                    vo3 = _mm256_fmadd_ps(vx2, _mm256_set1_ps(w3[2]), vo3);
                }
            }
        }

        if (full_ow) {
            _mm256_storeu_ps(out_r0, vo0);
            if (c_rem > 1) _mm256_storeu_ps(out_r1, vo1);
            if (c_rem > 2) _mm256_storeu_ps(out_r2, vo2);
            if (c_rem > 3) _mm256_storeu_ps(out_r3, vo3);
        } else {
            _mm256_maskstore_ps(out_r0, out_mask, vo0);
            if (c_rem > 1) _mm256_maskstore_ps(out_r1, out_mask, vo1);
            if (c_rem > 2) _mm256_maskstore_ps(out_r2, out_mask, vo2);
            if (c_rem > 3) _mm256_maskstore_ps(out_r3, out_mask, vo3);
        }
    }

    static void bwd_dx_tile(
        const ConvBwdDxTileDoc& doc,
        const float* __restrict dy_pad_buf,
        int64_t dy_pad_l, int64_t dy_row_stride,
        const float* __restrict W,
        float* __restrict dx,
        int64_t C_in, int64_t C_out, int64_t W_in_stride,
        int64_t pad, int64_t spatial_in, int64_t k_spatial,
        int64_t conv_out_h, int64_t conv_out_w
    );

    static void dw_nci(
        int64_t n, int64_t cout, int64_t cin,
        float* __restrict dw_slice,
        const float* __restrict dy_pad_buf,
        const float* __restrict x_pad_buf,
        int64_t C_in, int64_t C_out,
        int64_t dy_pad_l, int64_t dy_row_stride,
        int64_t x_pad_l, int64_t x_row_stride,
        int64_t H, int64_t pad,
        int64_t conv_out_h, int64_t conv_out_w
    );
};

template<>
struct Stride1Specialist<4> {
    static constexpr int K = 4;

    static void fwd_tile(
        const ConvFwdTileDoc& doc,
        const float* __restrict x_pad_buf,
        int64_t x_pad_l, int64_t x_row_stride,
        const float* __restrict W,
        float* __restrict out,
        int64_t C_in, int64_t C_out, int64_t H,
        int64_t pad, int64_t k_spatial, int64_t spatial_out, int64_t out_w_stride
    ) {
        const int64_t ih_base = doc.oh - pad;
        const int64_t c_rem   = doc.cout_count;
        const int64_t x_plane = H * x_row_stride;

        float* __restrict out_r0 = &out[(doc.n * C_out + doc.cout0 + 0) * spatial_out + doc.oh * out_w_stride + doc.ow];
        float* __restrict out_r1 = (c_rem > 1) ? &out[(doc.n * C_out + doc.cout0 + 1) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;
        float* __restrict out_r2 = (c_rem > 2) ? &out[(doc.n * C_out + doc.cout0 + 2) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;
        float* __restrict out_r3 = (c_rem > 3) ? &out[(doc.n * C_out + doc.cout0 + 3) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;

        __m256 vo0 = _mm256_loadu_ps(out_r0);
        __m256 vo1 = (c_rem > 1) ? _mm256_loadu_ps(out_r1) : _mm256_setzero_ps();
        __m256 vo2 = (c_rem > 2) ? _mm256_loadu_ps(out_r2) : _mm256_setzero_ps();
        __m256 vo3 = (c_rem > 3) ? _mm256_loadu_ps(out_r3) : _mm256_setzero_ps();

        const bool full_ow = (doc.ow_count == FWD_TILE_OW);
        const __m256i out_mask = bwd_dw_lane_mask(doc.ow_count);

        const float* __restrict xp_base = &x_pad_buf[doc.n * C_in * x_plane];
        const int64_t iw0 = x_pad_l + doc.ow - pad;

        for (int64_t cin = 0; cin < C_in; ++cin) {
            const float* __restrict xp  = xp_base + cin * x_plane;
            const float* __restrict wp0 = &W[((doc.cout0 + 0) * C_in + cin) * k_spatial];
            const float* __restrict wp1 = (c_rem > 1) ? &W[((doc.cout0 + 1) * C_in + cin) * k_spatial] : nullptr;
            const float* __restrict wp2 = (c_rem > 2) ? &W[((doc.cout0 + 2) * C_in + cin) * k_spatial] : nullptr;
            const float* __restrict wp3 = (c_rem > 3) ? &W[((doc.cout0 + 3) * C_in + cin) * k_spatial] : nullptr;

            for (int64_t kh = 0; kh < K; ++kh) {
                const int64_t ih = ih_base + kh;
                if (ih < 0 || ih >= H) {
                    continue;
                }

                const float* __restrict in_row = xp + ih * x_row_stride;
                const __m256 vx0 = _mm256_loadu_ps(in_row + iw0 + 0);
                const __m256 vx1 = _mm256_loadu_ps(in_row + iw0 + 1);
                const __m256 vx2 = _mm256_loadu_ps(in_row + iw0 + 2);
                const __m256 vx3 = _mm256_loadu_ps(in_row + iw0 + 3);

                const float* __restrict w0 = wp0 + kh * K;
                vo0 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w0[0]), vo0);
                vo0 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w0[1]), vo0);
                vo0 = _mm256_fmadd_ps(vx2, _mm256_set1_ps(w0[2]), vo0);
                vo0 = _mm256_fmadd_ps(vx3, _mm256_set1_ps(w0[3]), vo0);

                if (c_rem > 1) {
                    const float* __restrict w1 = wp1 + kh * K;
                    vo1 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w1[0]), vo1);
                    vo1 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w1[1]), vo1);
                    vo1 = _mm256_fmadd_ps(vx2, _mm256_set1_ps(w1[2]), vo1);
                    vo1 = _mm256_fmadd_ps(vx3, _mm256_set1_ps(w1[3]), vo1);
                }
                if (c_rem > 2) {
                    const float* __restrict w2 = wp2 + kh * K;
                    vo2 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w2[0]), vo2);
                    vo2 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w2[1]), vo2);
                    vo2 = _mm256_fmadd_ps(vx2, _mm256_set1_ps(w2[2]), vo2);
                    vo2 = _mm256_fmadd_ps(vx3, _mm256_set1_ps(w2[3]), vo2);
                }
                if (c_rem > 3) {
                    const float* __restrict w3 = wp3 + kh * K;
                    vo3 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w3[0]), vo3);
                    vo3 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w3[1]), vo3);
                    vo3 = _mm256_fmadd_ps(vx2, _mm256_set1_ps(w3[2]), vo3);
                    vo3 = _mm256_fmadd_ps(vx3, _mm256_set1_ps(w3[3]), vo3);
                }
            }
        }

        if (full_ow) {
            _mm256_storeu_ps(out_r0, vo0);
            if (c_rem > 1) _mm256_storeu_ps(out_r1, vo1);
            if (c_rem > 2) _mm256_storeu_ps(out_r2, vo2);
            if (c_rem > 3) _mm256_storeu_ps(out_r3, vo3);
        } else {
            _mm256_maskstore_ps(out_r0, out_mask, vo0);
            if (c_rem > 1) _mm256_maskstore_ps(out_r1, out_mask, vo1);
            if (c_rem > 2) _mm256_maskstore_ps(out_r2, out_mask, vo2);
            if (c_rem > 3) _mm256_maskstore_ps(out_r3, out_mask, vo3);
        }
    }

    static void bwd_dx_tile(
        const ConvBwdDxTileDoc& doc,
        const float* __restrict dy_pad_buf,
        int64_t dy_pad_l, int64_t dy_row_stride,
        const float* __restrict W,
        float* __restrict dx,
        int64_t C_in, int64_t C_out, int64_t W_in_stride,
        int64_t pad, int64_t spatial_in, int64_t k_spatial,
        int64_t conv_out_h, int64_t conv_out_w
    );

    static void dw_nci(
        int64_t n, int64_t cout, int64_t cin,
        float* __restrict dw_slice,
        const float* __restrict dy_pad_buf,
        const float* __restrict x_pad_buf,
        int64_t C_in, int64_t C_out,
        int64_t dy_pad_l, int64_t dy_row_stride,
        int64_t x_pad_l, int64_t x_row_stride,
        int64_t H, int64_t pad,
        int64_t conv_out_h, int64_t conv_out_w
    );
};

template<>
struct Stride1Specialist<6> {
    static constexpr int K = 6;

    // Forward: x_pad => loadu every kw; fully unrolled kh×kw.
    static void fwd_tile(
    const ConvFwdTileDoc& doc,
    const float* __restrict x_pad_buf,
    int64_t x_pad_l, int64_t x_row_stride,
    const float* __restrict W,
    float* __restrict out,
    int64_t C_in, int64_t C_out, int64_t H,
    int64_t pad, int64_t k_spatial, int64_t spatial_out, int64_t out_w_stride
) {
    const int64_t ih_base = doc.oh - pad;
    const int64_t c_rem   = doc.cout_count;
    const int64_t x_plane = H * x_row_stride;

    float* __restrict out_r0 = &out[(doc.n * C_out + doc.cout0 + 0) * spatial_out + doc.oh * out_w_stride + doc.ow];
    float* __restrict out_r1 = (c_rem > 1) ? &out[(doc.n * C_out + doc.cout0 + 1) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;
    float* __restrict out_r2 = (c_rem > 2) ? &out[(doc.n * C_out + doc.cout0 + 2) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;
    float* __restrict out_r3 = (c_rem > 3) ? &out[(doc.n * C_out + doc.cout0 + 3) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;

    __m256 vo0 = _mm256_loadu_ps(out_r0);
    __m256 vo1 = (c_rem > 1) ? _mm256_loadu_ps(out_r1) : _mm256_setzero_ps();
    __m256 vo2 = (c_rem > 2) ? _mm256_loadu_ps(out_r2) : _mm256_setzero_ps();
    __m256 vo3 = (c_rem > 3) ? _mm256_loadu_ps(out_r3) : _mm256_setzero_ps();

    const bool full_ow = (doc.ow_count == FWD_TILE_OW);
    const __m256i out_mask = bwd_dw_lane_mask(doc.ow_count);

    const float* __restrict xp_base = &x_pad_buf[doc.n * C_in * x_plane];
    const int64_t iw0 = x_pad_l + doc.ow - pad;

    for (int64_t cin = 0; cin < C_in; ++cin) {
        const float* __restrict xp  = xp_base + cin * x_plane;
        const float* __restrict wp0 = &W[((doc.cout0 + 0) * C_in + cin) * k_spatial];
        const float* __restrict wp1 = (c_rem > 1) ? &W[((doc.cout0 + 1) * C_in + cin) * k_spatial] : nullptr;
        const float* __restrict wp2 = (c_rem > 2) ? &W[((doc.cout0 + 2) * C_in + cin) * k_spatial] : nullptr;
        const float* __restrict wp3 = (c_rem > 3) ? &W[((doc.cout0 + 3) * C_in + cin) * k_spatial] : nullptr;

        for (int64_t kh = 0; kh < K; ++kh) {
            const int64_t ih = ih_base + kh;
            if (ih < 0 || ih >= H) {
                continue;
            }

            const float* __restrict in_row = xp + ih * x_row_stride;
            const __m256 vx0 = _mm256_loadu_ps(in_row + iw0 + 0);
            const __m256 vx1 = _mm256_loadu_ps(in_row + iw0 + 1);
            const __m256 vx2 = _mm256_loadu_ps(in_row + iw0 + 2);
            const __m256 vx3 = _mm256_loadu_ps(in_row + iw0 + 3);
            const __m256 vx4 = _mm256_loadu_ps(in_row + iw0 + 4);
            const __m256 vx5 = _mm256_loadu_ps(in_row + iw0 + 5);

            const float* __restrict w0 = wp0 + kh * K;
            vo0 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w0[0]), vo0);
            vo0 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w0[1]), vo0);
            vo0 = _mm256_fmadd_ps(vx2, _mm256_set1_ps(w0[2]), vo0);
            vo0 = _mm256_fmadd_ps(vx3, _mm256_set1_ps(w0[3]), vo0);
            vo0 = _mm256_fmadd_ps(vx4, _mm256_set1_ps(w0[4]), vo0);
            vo0 = _mm256_fmadd_ps(vx5, _mm256_set1_ps(w0[5]), vo0);

            if (c_rem > 1) {
                const float* __restrict w1 = wp1 + kh * K;
                vo1 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w1[0]), vo1);
                vo1 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w1[1]), vo1);
                vo1 = _mm256_fmadd_ps(vx2, _mm256_set1_ps(w1[2]), vo1);
                vo1 = _mm256_fmadd_ps(vx3, _mm256_set1_ps(w1[3]), vo1);
                vo1 = _mm256_fmadd_ps(vx4, _mm256_set1_ps(w1[4]), vo1);
                vo1 = _mm256_fmadd_ps(vx5, _mm256_set1_ps(w1[5]), vo1);
            }
            if (c_rem > 2) {
                const float* __restrict w2 = wp2 + kh * K;
                vo2 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w2[0]), vo2);
                vo2 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w2[1]), vo2);
                vo2 = _mm256_fmadd_ps(vx2, _mm256_set1_ps(w2[2]), vo2);
                vo2 = _mm256_fmadd_ps(vx3, _mm256_set1_ps(w2[3]), vo2);
                vo2 = _mm256_fmadd_ps(vx4, _mm256_set1_ps(w2[4]), vo2);
                vo2 = _mm256_fmadd_ps(vx5, _mm256_set1_ps(w2[5]), vo2);
            }
            if (c_rem > 3) {
                const float* __restrict w3 = wp3 + kh * K;
                vo3 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w3[0]), vo3);
                vo3 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w3[1]), vo3);
                vo3 = _mm256_fmadd_ps(vx2, _mm256_set1_ps(w3[2]), vo3);
                vo3 = _mm256_fmadd_ps(vx3, _mm256_set1_ps(w3[3]), vo3);
                vo3 = _mm256_fmadd_ps(vx4, _mm256_set1_ps(w3[4]), vo3);
                vo3 = _mm256_fmadd_ps(vx5, _mm256_set1_ps(w3[5]), vo3);
            }
        }
    }

    if (full_ow) {
        _mm256_storeu_ps(out_r0, vo0);
        if (c_rem > 1) _mm256_storeu_ps(out_r1, vo1);
        if (c_rem > 2) _mm256_storeu_ps(out_r2, vo2);
        if (c_rem > 3) _mm256_storeu_ps(out_r3, vo3);
    } else {
        _mm256_maskstore_ps(out_r0, out_mask, vo0);
        if (c_rem > 1) _mm256_maskstore_ps(out_r1, out_mask, vo1);
        if (c_rem > 2) _mm256_maskstore_ps(out_r2, out_mask, vo2);
        if (c_rem > 3) _mm256_maskstore_ps(out_r3, out_mask, vo3);
    }
    }

    static void bwd_dx_tile(
        const ConvBwdDxTileDoc& doc,
        const float* __restrict dy_pad_buf,
        int64_t dy_pad_l, int64_t dy_row_stride,
        const float* __restrict W,
        float* __restrict dx,
        int64_t C_in, int64_t C_out, int64_t W_in_stride,
        int64_t pad, int64_t spatial_in, int64_t k_spatial,
        int64_t conv_out_h, int64_t conv_out_w
    );

    static void dw_nci(
        int64_t n, int64_t cout, int64_t cin,
        float* __restrict dw_slice,
        const float* __restrict dy_pad_buf,
        const float* __restrict x_pad_buf,
        int64_t C_in, int64_t C_out,
        int64_t dy_pad_l, int64_t dy_row_stride,
        int64_t x_pad_l, int64_t x_row_stride,
        int64_t H, int64_t pad,
        int64_t conv_out_h, int64_t conv_out_w
    );
};

template<>
struct Stride1Specialist<5> {
    static constexpr int K = 5;

    static void fwd_tile(
        const ConvFwdTileDoc& doc,
        const float* __restrict x_pad_buf,
        int64_t x_pad_l, int64_t x_row_stride,
        const float* __restrict W,
        float* __restrict out,
        int64_t C_in, int64_t C_out, int64_t H,
        int64_t pad, int64_t k_spatial, int64_t spatial_out, int64_t out_w_stride
    ) {
        const int64_t ih_base = doc.oh - pad;
        const int64_t c_rem   = doc.cout_count;
        const int64_t x_plane = H * x_row_stride;

        float* __restrict out_r0 = &out[(doc.n * C_out + doc.cout0 + 0) * spatial_out + doc.oh * out_w_stride + doc.ow];
        float* __restrict out_r1 = (c_rem > 1) ? &out[(doc.n * C_out + doc.cout0 + 1) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;
        float* __restrict out_r2 = (c_rem > 2) ? &out[(doc.n * C_out + doc.cout0 + 2) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;
        float* __restrict out_r3 = (c_rem > 3) ? &out[(doc.n * C_out + doc.cout0 + 3) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;

        __m256 vo0 = _mm256_loadu_ps(out_r0);
        __m256 vo1 = (c_rem > 1) ? _mm256_loadu_ps(out_r1) : _mm256_setzero_ps();
        __m256 vo2 = (c_rem > 2) ? _mm256_loadu_ps(out_r2) : _mm256_setzero_ps();
        __m256 vo3 = (c_rem > 3) ? _mm256_loadu_ps(out_r3) : _mm256_setzero_ps();

        const bool full_ow = (doc.ow_count == FWD_TILE_OW);
        const __m256i out_mask = bwd_dw_lane_mask(doc.ow_count);

        const float* __restrict xp_base = &x_pad_buf[doc.n * C_in * x_plane];
        const int64_t iw0 = x_pad_l + doc.ow - pad;

        for (int64_t cin = 0; cin < C_in; ++cin) {
            const float* __restrict xp  = xp_base + cin * x_plane;
            const float* __restrict wp0 = &W[((doc.cout0 + 0) * C_in + cin) * k_spatial];
            const float* __restrict wp1 = (c_rem > 1) ? &W[((doc.cout0 + 1) * C_in + cin) * k_spatial] : nullptr;
            const float* __restrict wp2 = (c_rem > 2) ? &W[((doc.cout0 + 2) * C_in + cin) * k_spatial] : nullptr;
            const float* __restrict wp3 = (c_rem > 3) ? &W[((doc.cout0 + 3) * C_in + cin) * k_spatial] : nullptr;

            for (int64_t kh = 0; kh < K; ++kh) {
                const int64_t ih = ih_base + kh;
                if (ih < 0 || ih >= H) {
                    continue;
                }

                const float* __restrict in_row = xp + ih * x_row_stride;
                const __m256 vx0 = _mm256_loadu_ps(in_row + iw0 + 0);
                const __m256 vx1 = _mm256_loadu_ps(in_row + iw0 + 1);
                const __m256 vx2 = _mm256_loadu_ps(in_row + iw0 + 2);
                const __m256 vx3 = _mm256_loadu_ps(in_row + iw0 + 3);
                const __m256 vx4 = _mm256_loadu_ps(in_row + iw0 + 4);

                const float* __restrict w0 = wp0 + kh * K;
                vo0 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w0[0]), vo0);
                vo0 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w0[1]), vo0);
                vo0 = _mm256_fmadd_ps(vx2, _mm256_set1_ps(w0[2]), vo0);
                vo0 = _mm256_fmadd_ps(vx3, _mm256_set1_ps(w0[3]), vo0);
                vo0 = _mm256_fmadd_ps(vx4, _mm256_set1_ps(w0[4]), vo0);

                if (c_rem > 1) {
                    const float* __restrict w1 = wp1 + kh * K;
                    vo1 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w1[0]), vo1);
                    vo1 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w1[1]), vo1);
                    vo1 = _mm256_fmadd_ps(vx2, _mm256_set1_ps(w1[2]), vo1);
                    vo1 = _mm256_fmadd_ps(vx3, _mm256_set1_ps(w1[3]), vo1);
                    vo1 = _mm256_fmadd_ps(vx4, _mm256_set1_ps(w1[4]), vo1);
                }
                if (c_rem > 2) {
                    const float* __restrict w2 = wp2 + kh * K;
                    vo2 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w2[0]), vo2);
                    vo2 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w2[1]), vo2);
                    vo2 = _mm256_fmadd_ps(vx2, _mm256_set1_ps(w2[2]), vo2);
                    vo2 = _mm256_fmadd_ps(vx3, _mm256_set1_ps(w2[3]), vo2);
                    vo2 = _mm256_fmadd_ps(vx4, _mm256_set1_ps(w2[4]), vo2);
                }
                if (c_rem > 3) {
                    const float* __restrict w3 = wp3 + kh * K;
                    vo3 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w3[0]), vo3);
                    vo3 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w3[1]), vo3);
                    vo3 = _mm256_fmadd_ps(vx2, _mm256_set1_ps(w3[2]), vo3);
                    vo3 = _mm256_fmadd_ps(vx3, _mm256_set1_ps(w3[3]), vo3);
                    vo3 = _mm256_fmadd_ps(vx4, _mm256_set1_ps(w3[4]), vo3);
                }
            }
        }

        if (full_ow) {
            _mm256_storeu_ps(out_r0, vo0);
            if (c_rem > 1) _mm256_storeu_ps(out_r1, vo1);
            if (c_rem > 2) _mm256_storeu_ps(out_r2, vo2);
            if (c_rem > 3) _mm256_storeu_ps(out_r3, vo3);
        } else {
            _mm256_maskstore_ps(out_r0, out_mask, vo0);
            if (c_rem > 1) _mm256_maskstore_ps(out_r1, out_mask, vo1);
            if (c_rem > 2) _mm256_maskstore_ps(out_r2, out_mask, vo2);
            if (c_rem > 3) _mm256_maskstore_ps(out_r3, out_mask, vo3);
        }
    }

    static void bwd_dx_tile(
        const ConvBwdDxTileDoc& doc,
        const float* __restrict dy_pad_buf,
        int64_t dy_pad_l, int64_t dy_row_stride,
        const float* __restrict W,
        float* __restrict dx,
        int64_t C_in, int64_t C_out, int64_t W_in_stride,
        int64_t pad, int64_t spatial_in, int64_t k_spatial,
        int64_t conv_out_h, int64_t conv_out_w
    );

    static void dw_nci(
        int64_t n, int64_t cout, int64_t cin,
        float* __restrict dw_slice,
        const float* __restrict dy_pad_buf,
        const float* __restrict x_pad_buf,
        int64_t C_in, int64_t C_out,
        int64_t dy_pad_l, int64_t dy_row_stride,
        int64_t x_pad_l, int64_t x_row_stride,
        int64_t H, int64_t pad,
        int64_t conv_out_h, int64_t conv_out_w
    );
};

template<>
struct Stride1Specialist<7> {
    static constexpr int K = 7;

    static void fwd_tile(
        const ConvFwdTileDoc& doc,
        const float* __restrict x_pad_buf,
        int64_t x_pad_l, int64_t x_row_stride,
        const float* __restrict W,
        float* __restrict out,
        int64_t C_in, int64_t C_out, int64_t H,
        int64_t pad, int64_t k_spatial, int64_t spatial_out, int64_t out_w_stride
    ) {
        const int64_t ih_base = doc.oh - pad;
        const int64_t c_rem   = doc.cout_count;
        const int64_t x_plane = H * x_row_stride;

        float* __restrict out_r0 = &out[(doc.n * C_out + doc.cout0 + 0) * spatial_out + doc.oh * out_w_stride + doc.ow];
        float* __restrict out_r1 = (c_rem > 1) ? &out[(doc.n * C_out + doc.cout0 + 1) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;
        float* __restrict out_r2 = (c_rem > 2) ? &out[(doc.n * C_out + doc.cout0 + 2) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;
        float* __restrict out_r3 = (c_rem > 3) ? &out[(doc.n * C_out + doc.cout0 + 3) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;

        __m256 vo0 = _mm256_loadu_ps(out_r0);
        __m256 vo1 = (c_rem > 1) ? _mm256_loadu_ps(out_r1) : _mm256_setzero_ps();
        __m256 vo2 = (c_rem > 2) ? _mm256_loadu_ps(out_r2) : _mm256_setzero_ps();
        __m256 vo3 = (c_rem > 3) ? _mm256_loadu_ps(out_r3) : _mm256_setzero_ps();

        const bool full_ow = (doc.ow_count == FWD_TILE_OW);
        const __m256i out_mask = bwd_dw_lane_mask(doc.ow_count);

        const float* __restrict xp_base = &x_pad_buf[doc.n * C_in * x_plane];
        const int64_t iw0 = x_pad_l + doc.ow - pad;

        for (int64_t cin = 0; cin < C_in; ++cin) {
            const float* __restrict xp  = xp_base + cin * x_plane;
            const float* __restrict wp0 = &W[((doc.cout0 + 0) * C_in + cin) * k_spatial];
            const float* __restrict wp1 = (c_rem > 1) ? &W[((doc.cout0 + 1) * C_in + cin) * k_spatial] : nullptr;
            const float* __restrict wp2 = (c_rem > 2) ? &W[((doc.cout0 + 2) * C_in + cin) * k_spatial] : nullptr;
            const float* __restrict wp3 = (c_rem > 3) ? &W[((doc.cout0 + 3) * C_in + cin) * k_spatial] : nullptr;

            for (int64_t kh = 0; kh < K; ++kh) {
                const int64_t ih = ih_base + kh;
                if (ih < 0 || ih >= H) {
                    continue;
                }

                const float* __restrict in_row = xp + ih * x_row_stride;
                const __m256 vx0 = _mm256_loadu_ps(in_row + iw0 + 0);
                const __m256 vx1 = _mm256_loadu_ps(in_row + iw0 + 1);
                const __m256 vx2 = _mm256_loadu_ps(in_row + iw0 + 2);
                const __m256 vx3 = _mm256_loadu_ps(in_row + iw0 + 3);
                const __m256 vx4 = _mm256_loadu_ps(in_row + iw0 + 4);
                const __m256 vx5 = _mm256_loadu_ps(in_row + iw0 + 5);
                const __m256 vx6 = _mm256_loadu_ps(in_row + iw0 + 6);

                const float* __restrict w0 = wp0 + kh * K;
                vo0 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w0[0]), vo0);
                vo0 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w0[1]), vo0);
                vo0 = _mm256_fmadd_ps(vx2, _mm256_set1_ps(w0[2]), vo0);
                vo0 = _mm256_fmadd_ps(vx3, _mm256_set1_ps(w0[3]), vo0);
                vo0 = _mm256_fmadd_ps(vx4, _mm256_set1_ps(w0[4]), vo0);
                vo0 = _mm256_fmadd_ps(vx5, _mm256_set1_ps(w0[5]), vo0);
                vo0 = _mm256_fmadd_ps(vx6, _mm256_set1_ps(w0[6]), vo0);

                if (c_rem > 1) {
                    const float* __restrict w1 = wp1 + kh * K;
                    vo1 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w1[0]), vo1);
                    vo1 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w1[1]), vo1);
                    vo1 = _mm256_fmadd_ps(vx2, _mm256_set1_ps(w1[2]), vo1);
                    vo1 = _mm256_fmadd_ps(vx3, _mm256_set1_ps(w1[3]), vo1);
                    vo1 = _mm256_fmadd_ps(vx4, _mm256_set1_ps(w1[4]), vo1);
                    vo1 = _mm256_fmadd_ps(vx5, _mm256_set1_ps(w1[5]), vo1);
                    vo1 = _mm256_fmadd_ps(vx6, _mm256_set1_ps(w1[6]), vo1);
                }
                if (c_rem > 2) {
                    const float* __restrict w2 = wp2 + kh * K;
                    vo2 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w2[0]), vo2);
                    vo2 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w2[1]), vo2);
                    vo2 = _mm256_fmadd_ps(vx2, _mm256_set1_ps(w2[2]), vo2);
                    vo2 = _mm256_fmadd_ps(vx3, _mm256_set1_ps(w2[3]), vo2);
                    vo2 = _mm256_fmadd_ps(vx4, _mm256_set1_ps(w2[4]), vo2);
                    vo2 = _mm256_fmadd_ps(vx5, _mm256_set1_ps(w2[5]), vo2);
                    vo2 = _mm256_fmadd_ps(vx6, _mm256_set1_ps(w2[6]), vo2);
                }
                if (c_rem > 3) {
                    const float* __restrict w3 = wp3 + kh * K;
                    vo3 = _mm256_fmadd_ps(vx0, _mm256_set1_ps(w3[0]), vo3);
                    vo3 = _mm256_fmadd_ps(vx1, _mm256_set1_ps(w3[1]), vo3);
                    vo3 = _mm256_fmadd_ps(vx2, _mm256_set1_ps(w3[2]), vo3);
                    vo3 = _mm256_fmadd_ps(vx3, _mm256_set1_ps(w3[3]), vo3);
                    vo3 = _mm256_fmadd_ps(vx4, _mm256_set1_ps(w3[4]), vo3);
                    vo3 = _mm256_fmadd_ps(vx5, _mm256_set1_ps(w3[5]), vo3);
                    vo3 = _mm256_fmadd_ps(vx6, _mm256_set1_ps(w3[6]), vo3);
                }
            }
        }

        if (full_ow) {
            _mm256_storeu_ps(out_r0, vo0);
            if (c_rem > 1) _mm256_storeu_ps(out_r1, vo1);
            if (c_rem > 2) _mm256_storeu_ps(out_r2, vo2);
            if (c_rem > 3) _mm256_storeu_ps(out_r3, vo3);
        } else {
            _mm256_maskstore_ps(out_r0, out_mask, vo0);
            if (c_rem > 1) _mm256_maskstore_ps(out_r1, out_mask, vo1);
            if (c_rem > 2) _mm256_maskstore_ps(out_r2, out_mask, vo2);
            if (c_rem > 3) _mm256_maskstore_ps(out_r3, out_mask, vo3);
        }
    }

    static void bwd_dx_tile(
        const ConvBwdDxTileDoc& doc,
        const float* __restrict dy_pad_buf,
        int64_t dy_pad_l, int64_t dy_row_stride,
        const float* __restrict W,
        float* __restrict dx,
        int64_t C_in, int64_t C_out, int64_t W_in_stride,
        int64_t pad, int64_t spatial_in, int64_t k_spatial,
        int64_t conv_out_h, int64_t conv_out_w
    );

    static void dw_nci(
        int64_t n, int64_t cout, int64_t cin,
        float* __restrict dw_slice,
        const float* __restrict dy_pad_buf,
        const float* __restrict x_pad_buf,
        int64_t C_in, int64_t C_out,
        int64_t dy_pad_l, int64_t dy_row_stride,
        int64_t x_pad_l, int64_t x_row_stride,
        int64_t H, int64_t pad,
        int64_t conv_out_h, int64_t conv_out_w
    );
};

static inline bool stride1_try_fwd_specialist(
    int64_t k_h, int64_t k_w,
    const ConvFwdTileDoc& doc,
    const float* __restrict x_pad_buf,
    int64_t x_pad_l, int64_t x_row_stride,
    const float* __restrict W,
    float* __restrict out,
    int64_t C_in, int64_t C_out, int64_t H,
    int64_t pad, int64_t k_spatial, int64_t spatial_out, int64_t out_w_stride
) {
    if (!x_pad_buf) {
        return false;
    }
    switch (stride1_specialist_k(k_h, k_w)) {
        case 1:
            Stride1Specialist<1>::fwd_tile(
                doc, x_pad_buf, x_pad_l, x_row_stride,
                W, out, C_in, C_out, H, pad, k_spatial, spatial_out, out_w_stride
            );
            return true;
        case 2:
            Stride1Specialist<2>::fwd_tile(
                doc, x_pad_buf, x_pad_l, x_row_stride,
                W, out, C_in, C_out, H, pad, k_spatial, spatial_out, out_w_stride
            );
            return true;
        case 3:
            Stride1Specialist<3>::fwd_tile(
                doc, x_pad_buf, x_pad_l, x_row_stride,
                W, out, C_in, C_out, H, pad, k_spatial, spatial_out, out_w_stride
            );
            return true;
        case 4:
            Stride1Specialist<4>::fwd_tile(
                doc, x_pad_buf, x_pad_l, x_row_stride,
                W, out, C_in, C_out, H, pad, k_spatial, spatial_out, out_w_stride
            );
            return true;
        case 5:
            Stride1Specialist<5>::fwd_tile(
                doc, x_pad_buf, x_pad_l, x_row_stride,
                W, out, C_in, C_out, H, pad, k_spatial, spatial_out, out_w_stride
            );
            return true;
        case 6:
            Stride1Specialist<6>::fwd_tile(
                doc, x_pad_buf, x_pad_l, x_row_stride,
                W, out, C_in, C_out, H, pad, k_spatial, spatial_out, out_w_stride
            );
            return true;
        case 7:
            Stride1Specialist<7>::fwd_tile(
                doc, x_pad_buf, x_pad_l, x_row_stride,
                W, out, C_in, C_out, H, pad, k_spatial, spatial_out, out_w_stride
            );
            return true;
        default:
            return false;
    }
}

static inline bool stride1_try_bwd_dx_specialist(
    int64_t k_h, int64_t k_w,
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t C_out, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    if (!dy_pad_buf) {
        return false;
    }
    switch (stride1_specialist_k(k_h, k_w)) {
        case 1:
            Stride1Specialist<1>::bwd_dx_tile(
                doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
                C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
                conv_out_h, conv_out_w
            );
            return true;
        case 2:
            Stride1Specialist<2>::bwd_dx_tile(
                doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
                C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
                conv_out_h, conv_out_w
            );
            return true;
        case 3:
            Stride1Specialist<3>::bwd_dx_tile(
                doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
                C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
                conv_out_h, conv_out_w
            );
            return true;
        case 4:
            Stride1Specialist<4>::bwd_dx_tile(
                doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
                C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
                conv_out_h, conv_out_w
            );
            return true;
        case 5:
            Stride1Specialist<5>::bwd_dx_tile(
                doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
                C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
                conv_out_h, conv_out_w
            );
            return true;
        case 6:
            Stride1Specialist<6>::bwd_dx_tile(
                doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
                C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
                conv_out_h, conv_out_w
            );
            return true;
        case 7:
            Stride1Specialist<7>::bwd_dx_tile(
                doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
                C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
                conv_out_h, conv_out_w
            );
            return true;
        default:
            return false;
    }
}

static inline bool stride1_try_dw_specialist(
    int64_t k_h, int64_t k_w,
    int64_t n, int64_t cout, int64_t cin,
    float* __restrict dw_slice,
    const float* __restrict dy_pad_buf,
    const float* __restrict x_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    int64_t x_pad_l, int64_t x_row_stride,
    int64_t C_in, int64_t C_out, int64_t H,
    int64_t pad, int64_t conv_out_h, int64_t conv_out_w
) {
    if (!dy_pad_buf || !x_pad_buf) {
        return false;
    }
    switch (stride1_specialist_k(k_h, k_w)) {
        case 1:
            Stride1Specialist<1>::dw_nci(
                n, cout, cin, dw_slice,
                dy_pad_buf, x_pad_buf,
                C_in, C_out,
                dy_pad_l, dy_row_stride,
                x_pad_l, x_row_stride,
                H, pad,
                conv_out_h, conv_out_w
            );
            return true;
        case 2:
            Stride1Specialist<2>::dw_nci(
                n, cout, cin, dw_slice,
                dy_pad_buf, x_pad_buf,
                C_in, C_out,
                dy_pad_l, dy_row_stride,
                x_pad_l, x_row_stride,
                H, pad,
                conv_out_h, conv_out_w
            );
            return true;
        case 3:
            Stride1Specialist<3>::dw_nci(
                n, cout, cin, dw_slice,
                dy_pad_buf, x_pad_buf,
                C_in, C_out,
                dy_pad_l, dy_row_stride,
                x_pad_l, x_row_stride,
                H, pad,
                conv_out_h, conv_out_w
            );
            return true;
        case 4:
            Stride1Specialist<4>::dw_nci(
                n, cout, cin, dw_slice,
                dy_pad_buf, x_pad_buf,
                C_in, C_out,
                dy_pad_l, dy_row_stride,
                x_pad_l, x_row_stride,
                H, pad,
                conv_out_h, conv_out_w
            );
            return true;
        case 5:
            Stride1Specialist<5>::dw_nci(
                n, cout, cin, dw_slice,
                dy_pad_buf, x_pad_buf,
                C_in, C_out,
                dy_pad_l, dy_row_stride,
                x_pad_l, x_row_stride,
                H, pad,
                conv_out_h, conv_out_w
            );
            return true;
        case 6:
            Stride1Specialist<6>::dw_nci(
                n, cout, cin, dw_slice,
                dy_pad_buf, x_pad_buf,
                C_in, C_out,
                dy_pad_l, dy_row_stride,
                x_pad_l, x_row_stride,
                H, pad,
                conv_out_h, conv_out_w
            );
            return true;
        case 7:
            Stride1Specialist<7>::dw_nci(
                n, cout, cin, dw_slice,
                dy_pad_buf, x_pad_buf,
                C_in, C_out,
                dy_pad_l, dy_row_stride,
                x_pad_l, x_row_stride,
                H, pad,
                conv_out_h, conv_out_w
            );
            return true;
        default:
            return false;
    }
}

// Algorithm kernel: one fixed-layout tile, no OpenMP.
static void process_fwd_tile_stride1(
    const ConvFwdTileDoc& doc,
    const float* __restrict x,
    const float* __restrict W,
    float* __restrict out,
    int64_t C_in, int64_t C_out, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t k_h, int64_t k_w, int64_t pad,
    int64_t spatial_in, int64_t spatial_out, int64_t k_spatial, int64_t out_w_stride,
    const float* __restrict x_pad_buf,
    int64_t x_pad_l, int64_t x_row_stride
) {
    if (stride1_try_fwd_specialist(
            k_h, k_w, doc, x_pad_buf, x_pad_l, x_row_stride,
            W, out, C_in, C_out, H, pad, k_spatial, spatial_out, out_w_stride)) {
        return;
    }

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

static inline bool bwd_dx_tile_is_interior(
    const ConvBwdDxTileDoc& doc, int64_t k_w, int64_t pad, int64_t conv_out_w
) {
    if (doc.ow_count != FWD_TILE_OW) return false;
    const int64_t ow_min = doc.ow + pad - (k_w - 1);
    return ow_min >= 0 && (doc.ow + pad + FWD_TILE_OW) <= conv_out_w;
}

static inline int64_t bwd_dy_pad_l(int64_t k_w, int64_t pad) {
    const int64_t v = k_w - 1 - pad;
    return v > 0 ? v : 0;
}

static inline int64_t bwd_x_pad_l(int64_t pad) {
    return pad;
}

static inline int64_t bwd_dy_row_stride(
    int64_t k_w, int64_t pad, int64_t W_in, int64_t conv_out_w
) {
    const int64_t pad_l = bwd_dy_pad_l(k_w, pad);
    const int64_t pad_r = (W_in - 1) + pad + FWD_TILE_OW - conv_out_w;
    return pad_l + conv_out_w + (pad_r > 0 ? pad_r : 0);
}

static inline int64_t bwd_x_row_stride(
    int64_t k_w, int64_t pad, int64_t W_in, int64_t conv_out_w
) {
    const int64_t pad_l = bwd_x_pad_l(pad);
    const int64_t max_iw = (conv_out_w - 1) - pad + (k_w - 1) + (FWD_TILE_OW - 1);
    const int64_t need = pad_l + max_iw + 1;
    const int64_t min_stride = pad_l + W_in;
    return need > min_stride ? need : min_stride;
}

// Zero-padded rows: logical index i -> padded[i + pad_l].
static void build_bwd_row_pad_buf(
    const float* __restrict src, float* __restrict dst,
    int64_t nplanes, int64_t nrows, int64_t row_w,
    int64_t src_row_stride, int64_t pad_l, int64_t row_stride
) {
    const int64_t src_plane = nrows * src_row_stride;
    const int64_t dst_plane = nrows * row_stride;
    const __m256 z = _mm256_setzero_ps();

    #pragma omp parallel for collapse(2) schedule(dynamic, 8)
    for (int64_t nc = 0; nc < nplanes; ++nc) {
        for (int64_t row = 0; row < nrows; ++row) {
            float* __restrict dst_row = &dst[nc * dst_plane + row * row_stride];
            const float* __restrict src_row = &src[nc * src_plane + row * src_row_stride];

            int64_t i = 0;
            for (; i + 7 < pad_l; i += 8) {
                _mm256_storeu_ps(dst_row + i, z);
            }
            for (; i < pad_l; ++i) {
                dst_row[i] = 0.0f;
            }

            i = 0;
            for (; i + 7 < row_w; i += 8) {
                _mm256_storeu_ps(dst_row + pad_l + i, _mm256_loadu_ps(src_row + i));
            }
            for (; i < row_w; ++i) {
                dst_row[pad_l + i] = src_row[i];
            }

            const int64_t tail = pad_l + row_w;
            for (i = tail; i + 7 < row_stride; i += 8) {
                _mm256_storeu_ps(dst_row + i, z);
            }
            for (; i < row_stride; ++i) {
                dst_row[i] = 0.0f;
            }
        }
    }
}

static inline void build_dy_pad_buf(
    const float* __restrict src, float* __restrict dst,
    int64_t N, int64_t C_out, int64_t conv_out_h, int64_t conv_out_w,
    int64_t src_row_stride, int64_t dy_pad_l, int64_t dy_row_stride
) {
    build_bwd_row_pad_buf(
        src, dst, N * C_out, conv_out_h, conv_out_w,
        src_row_stride, dy_pad_l, dy_row_stride
    );
}

static inline void build_x_pad_buf(
    const float* __restrict src, float* __restrict dst,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in,
    int64_t src_row_stride, int64_t x_pad_l, int64_t x_row_stride
) {
    build_bwd_row_pad_buf(
        src, dst, N * C_in, H, W_in,
        src_row_stride, x_pad_l, x_row_stride
    );
}

// 8-wide dY strip has no valid ow → padded column is all zero; skip loads/FMA.
static inline bool bwd_dw_dy_strip_has_overlap(
    int64_t i, int64_t ow_start, int64_t conv_out_w
) {
    const int64_t ow_lo = ow_start + i;
    const int64_t ow_hi = ow_lo + 8;
    return ow_lo < conv_out_w && ow_hi > 0;
}

// Strip has at least one lane where both ow and iw are in valid range.
static inline bool bwd_dw_strip_has_overlap(
    int64_t i, int64_t ow_start, int64_t iw_start,
    int64_t conv_out_w, int64_t W_in
) {
    if (!bwd_dw_dy_strip_has_overlap(i, ow_start, conv_out_w)) return false;
    const int64_t iw_lo = iw_start + i;
    const int64_t iw_hi = iw_lo + 8;
    if (iw_lo >= W_in || iw_hi <= 0) return false;
    return true;
}

// Structural pad column: entire 8-wide strip lies outside valid ow/iw (halo geometry).
// NOT the same as a data cell that happens to be 0.0f — those still run through FMA.
static inline bool bwd_dw_strip_is_pad_column(
    int64_t i, int64_t ow_start, int64_t iw_start,
    int64_t conv_out_w, int64_t W_in
) {
    return !bwd_dw_strip_has_overlap(i, ow_start, iw_start, conv_out_w, W_in);
}

// Skip leading/trailing full pad columns once per tap (outside the oh loop).
static inline void bwd_dw_strip_i_bounds(
    int64_t count, int64_t ow_start, int64_t iw_start,
    int64_t conv_out_w, int64_t W_in,
    int64_t& i_begin, int64_t& i_end
) {
    i_begin = 0;
    i_end = count;
    while (i_begin < i_end &&
           bwd_dw_strip_is_pad_column(i_begin, ow_start, iw_start, conv_out_w, W_in)) {
        i_begin += 8;
    }
    while (i_end > i_begin) {
        const int64_t last = i_end - 8;
        if (last < i_begin ||
            !bwd_dw_strip_is_pad_column(last, ow_start, iw_start, conv_out_w, W_in)) {
            break;
        }
        i_end = last;
    }
}

// 4-kw unroll: strip is skippable only when all four taps are structural pad columns.
static inline bool bwd_dw_strip_is_pad_column4(
    int64_t i, int64_t ow_start, int64_t iw_start_0,
    int64_t conv_out_w, int64_t W_in
) {
    return bwd_dw_strip_is_pad_column(i, ow_start, iw_start_0 + 0, conv_out_w, W_in) &&
           bwd_dw_strip_is_pad_column(i, ow_start, iw_start_0 + 1, conv_out_w, W_in) &&
           bwd_dw_strip_is_pad_column(i, ow_start, iw_start_0 + 2, conv_out_w, W_in) &&
           bwd_dw_strip_is_pad_column(i, ow_start, iw_start_0 + 3, conv_out_w, W_in);
}

static inline void bwd_dw_strip_i_bounds4(
    int64_t count, int64_t ow_start, int64_t iw_start_0,
    int64_t conv_out_w, int64_t W_in,
    int64_t& i_begin, int64_t& i_end
) {
    i_begin = 0;
    i_end = count;
    while (i_begin < i_end &&
           bwd_dw_strip_is_pad_column4(i_begin, ow_start, iw_start_0, conv_out_w, W_in)) {
        i_begin += 8;
    }
    while (i_end > i_begin) {
        const int64_t last = i_end - 8;
        if (last < i_begin ||
            !bwd_dw_strip_is_pad_column4(last, ow_start, iw_start_0, conv_out_w, W_in)) {
            break;
        }
        i_end = last;
    }
}

// 8-wide dY window for one kw: zeros via blend, no stack halo (unpadded fallback).
static __forceinline __m256 load_dy_window_8(
    const float* __restrict dy_row,
    int64_t ow_base, int64_t conv_out_w, bool full_dx,
    __m256i dx_mask, __m256i v_idx, __m256i v_cow
) {
    const __m256 z = _mm256_setzero_ps();
    if (full_dx && ow_base >= 0 && (ow_base + FWD_TILE_OW) <= conv_out_w) {
        return _mm256_loadu_ps(dy_row + ow_base);
    }
    __m256i ow = _mm256_add_epi32(_mm256_set1_epi32((int)ow_base), v_idx);
    __m256i ok = _mm256_and_si256(
        _mm256_and_si256(_mm256_cmpgt_epi32(ow, _mm256_set1_epi32(-1)), _mm256_cmpgt_epi32(v_cow, ow)),
        dx_mask
    );
    if (ow_base >= 0) {
        return _mm256_maskload_ps(dy_row + ow_base, ok);
    }
    const __m256i v_last = _mm256_sub_epi32(v_cow, _mm256_set1_epi32(1));
    __m256i safe = _mm256_min_epi32(
        _mm256_max_epi32(ow, _mm256_setzero_si256()),
        v_last
    );
    __m256 g = _mm256_i32gather_ps(dy_row, safe, 4);
    __m256i m = _mm256_slli_epi32(ok, 31);
    return _mm256_blendv_ps(z, g, _mm256_castsi256_ps(m));
}

static __forceinline __m256 load_x_window_8(
    const float* __restrict x_row,
    int64_t iw_base, int64_t W_in, bool full_ow,
    __m256i ow_mask, __m256i v_idx
) {
    const __m256 z = _mm256_setzero_ps();
    if (full_ow && iw_base >= 0 && (iw_base + FWD_TILE_OW) <= W_in) {
        return _mm256_loadu_ps(x_row + iw_base);
    }
    __m256i iw = _mm256_add_epi32(_mm256_set1_epi32((int)iw_base), v_idx);
    const __m256i v_win = _mm256_set1_epi32((int)W_in);
    __m256i ok = _mm256_and_si256(
        _mm256_and_si256(_mm256_cmpgt_epi32(iw, _mm256_set1_epi32(-1)), _mm256_cmpgt_epi32(v_win, iw)),
        ow_mask
    );
    if (iw_base >= 0) {
        return _mm256_maskload_ps(x_row + iw_base, ok);
    }
    const __m256i v_last = _mm256_sub_epi32(v_win, _mm256_set1_epi32(1));
    __m256i safe = _mm256_min_epi32(
        _mm256_max_epi32(iw, _mm256_setzero_si256()),
        v_last
    );
    __m256 g = _mm256_i32gather_ps(x_row, safe, 4);
    __m256i m = _mm256_slli_epi32(ok, 31);
    return _mm256_blendv_ps(z, g, _mm256_castsi256_ps(m));
}

static __forceinline __m256 fmadd_dx_cout4(
    __m256 v_dx, __m256 r0, __m256 r1, __m256 r2, __m256 r3,
    float w0, float w1, float w2, float w3
) {
    return _mm256_fmadd_ps(
        r0, _mm256_set1_ps(w0),
        _mm256_fmadd_ps(
            r1, _mm256_set1_ps(w1),
            _mm256_fmadd_ps(
                r2, _mm256_set1_ps(w2),
                _mm256_fmadd_ps(r3, _mm256_set1_ps(w3), v_dx)
            )
        )
    );
}

static __forceinline __m256 fmadd_dx_cout1(__m256 v_dx, __m256 r0, float w0) {
    return _mm256_fmadd_ps(r0, _mm256_set1_ps(w0), v_dx);
}

// One cout×4 group: fully unrolled kw against padded dY row offsets (K=6).
static __forceinline void stride1_bwd_dx_accum_cout4_k6(
    __m256& v_dx,
    int64_t ow0,
    const float* __restrict dy0,
    const float* __restrict dy1,
    const float* __restrict dy2,
    const float* __restrict dy3,
    const float* __restrict wp0,
    const float* __restrict wp1,
    const float* __restrict wp2,
    const float* __restrict wp3
) {
    v_dx = fmadd_dx_cout4(
        v_dx,
        _mm256_loadu_ps(dy0 + ow0 - 0), _mm256_loadu_ps(dy1 + ow0 - 0),
        _mm256_loadu_ps(dy2 + ow0 - 0), _mm256_loadu_ps(dy3 + ow0 - 0),
        wp0[0], wp1[0], wp2[0], wp3[0]
    );
    v_dx = fmadd_dx_cout4(
        v_dx,
        _mm256_loadu_ps(dy0 + ow0 - 1), _mm256_loadu_ps(dy1 + ow0 - 1),
        _mm256_loadu_ps(dy2 + ow0 - 1), _mm256_loadu_ps(dy3 + ow0 - 1),
        wp0[1], wp1[1], wp2[1], wp3[1]
    );
    v_dx = fmadd_dx_cout4(
        v_dx,
        _mm256_loadu_ps(dy0 + ow0 - 2), _mm256_loadu_ps(dy1 + ow0 - 2),
        _mm256_loadu_ps(dy2 + ow0 - 2), _mm256_loadu_ps(dy3 + ow0 - 2),
        wp0[2], wp1[2], wp2[2], wp3[2]
    );
    v_dx = fmadd_dx_cout4(
        v_dx,
        _mm256_loadu_ps(dy0 + ow0 - 3), _mm256_loadu_ps(dy1 + ow0 - 3),
        _mm256_loadu_ps(dy2 + ow0 - 3), _mm256_loadu_ps(dy3 + ow0 - 3),
        wp0[3], wp1[3], wp2[3], wp3[3]
    );
    v_dx = fmadd_dx_cout4(
        v_dx,
        _mm256_loadu_ps(dy0 + ow0 - 4), _mm256_loadu_ps(dy1 + ow0 - 4),
        _mm256_loadu_ps(dy2 + ow0 - 4), _mm256_loadu_ps(dy3 + ow0 - 4),
        wp0[4], wp1[4], wp2[4], wp3[4]
    );
    v_dx = fmadd_dx_cout4(
        v_dx,
        _mm256_loadu_ps(dy0 + ow0 - 5), _mm256_loadu_ps(dy1 + ow0 - 5),
        _mm256_loadu_ps(dy2 + ow0 - 5), _mm256_loadu_ps(dy3 + ow0 - 5),
        wp0[5], wp1[5], wp2[5], wp3[5]
    );
}

// C_out=8: two fixed cout×4 groups, no cout loop.
static void stride1_bwd_dx_tile_c8_k6(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    static constexpr int K = Stride1Specialist<6>::K;
    (void)C_in;

    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, K, pad, conv_out_w)) {
        return;
    }

    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];

    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);

    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);

    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * 8 * dy_plane];
    const int64_t ow0 = doc.ow + pad + dy_pad_l;

    int64_t kh_lo = 0;
    int64_t kh_hi = K;
    bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);

    for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
        const float* __restrict w_kh = w_cin + kh * K;

        stride1_bwd_dx_accum_cout4_k6(
            v_dx, ow0,
            dy_oh + 0 * dy_plane, dy_oh + 1 * dy_plane,
            dy_oh + 2 * dy_plane, dy_oh + 3 * dy_plane,
            w_kh + 0 * w_cout_stride, w_kh + 1 * w_cout_stride,
            w_kh + 2 * w_cout_stride, w_kh + 3 * w_cout_stride
        );
        stride1_bwd_dx_accum_cout4_k6(
            v_dx, ow0,
            dy_oh + 4 * dy_plane, dy_oh + 5 * dy_plane,
            dy_oh + 6 * dy_plane, dy_oh + 7 * dy_plane,
            w_kh + 4 * w_cout_stride, w_kh + 5 * w_cout_stride,
            w_kh + 6 * w_cout_stride, w_kh + 7 * w_cout_stride
        );
    }

    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

// C_out=16: four fixed cout×4 groups, no cout loop.
static void stride1_bwd_dx_tile_c16_k6(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    static constexpr int K = Stride1Specialist<6>::K;
    (void)C_in;

    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, K, pad, conv_out_w)) {
        return;
    }

    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];

    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);

    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);

    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * 16 * dy_plane];
    const int64_t ow0 = doc.ow + pad + dy_pad_l;

    int64_t kh_lo = 0;
    int64_t kh_hi = K;
    bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);

    for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
        const float* __restrict w_kh = w_cin + kh * K;

        stride1_bwd_dx_accum_cout4_k6(
            v_dx, ow0,
            dy_oh + 0 * dy_plane, dy_oh + 1 * dy_plane,
            dy_oh + 2 * dy_plane, dy_oh + 3 * dy_plane,
            w_kh + 0 * w_cout_stride, w_kh + 1 * w_cout_stride,
            w_kh + 2 * w_cout_stride, w_kh + 3 * w_cout_stride
        );
        stride1_bwd_dx_accum_cout4_k6(
            v_dx, ow0,
            dy_oh + 4 * dy_plane, dy_oh + 5 * dy_plane,
            dy_oh + 6 * dy_plane, dy_oh + 7 * dy_plane,
            w_kh + 4 * w_cout_stride, w_kh + 5 * w_cout_stride,
            w_kh + 6 * w_cout_stride, w_kh + 7 * w_cout_stride
        );
        stride1_bwd_dx_accum_cout4_k6(
            v_dx, ow0,
            dy_oh + 8 * dy_plane, dy_oh + 9 * dy_plane,
            dy_oh + 10 * dy_plane, dy_oh + 11 * dy_plane,
            w_kh + 8 * w_cout_stride, w_kh + 9 * w_cout_stride,
            w_kh + 10 * w_cout_stride, w_kh + 11 * w_cout_stride
        );
        stride1_bwd_dx_accum_cout4_k6(
            v_dx, ow0,
            dy_oh + 12 * dy_plane, dy_oh + 13 * dy_plane,
            dy_oh + 14 * dy_plane, dy_oh + 15 * dy_plane,
            w_kh + 12 * w_cout_stride, w_kh + 13 * w_cout_stride,
            w_kh + 14 * w_cout_stride, w_kh + 15 * w_cout_stride
        );
    }

    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

void Stride1Specialist<6>::bwd_dx_tile(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t C_out, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    if (C_out == 8) {
        stride1_bwd_dx_tile_c8_k6(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
        return;
    }
    if (C_out == 16) {
        stride1_bwd_dx_tile_c16_k6(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
        return;
    }

    (void)C_in;

    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, K, pad, conv_out_w)) {
        return;
    }

    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];

    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);

    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);

    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * C_out * dy_plane];
    const int64_t ow0 = doc.ow + pad + dy_pad_l;

    int64_t kh_lo = 0;
    int64_t kh_hi = K;
    bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);

    for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
        const float* __restrict w_kh = w_cin + kh * K;

        int64_t cout = 0;
        for (; cout + 3 < C_out; cout += 4) {
            stride1_bwd_dx_accum_cout4_k6(
                v_dx, ow0,
                dy_oh + (cout + 0) * dy_plane,
                dy_oh + (cout + 1) * dy_plane,
                dy_oh + (cout + 2) * dy_plane,
                dy_oh + (cout + 3) * dy_plane,
                w_kh + (cout + 0) * w_cout_stride,
                w_kh + (cout + 1) * w_cout_stride,
                w_kh + (cout + 2) * w_cout_stride,
                w_kh + (cout + 3) * w_cout_stride
            );
        }
        for (; cout < C_out; ++cout) {
            const float* __restrict dy_row = dy_oh + cout * dy_plane;
            const float* __restrict wp = w_kh + cout * w_cout_stride;

            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 0), wp[0]);
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 1), wp[1]);
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 2), wp[2]);
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 3), wp[3]);
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 4), wp[4]);
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 5), wp[5]);
        }
    }

    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

// --- Stride1Specialist<1> backward implementations ---

#define STRIDE1_BWD_DX_ACCUM_K1(v_dx, ow0, dy0, dy1, dy2, dy3, wp0, wp1, wp2, wp3) \
    do { \
        (v_dx) = fmadd_dx_cout4( \
            (v_dx), \
            _mm256_loadu_ps((dy0) + (ow0) - 0), _mm256_loadu_ps((dy1) + (ow0) - 0), \
            _mm256_loadu_ps((dy2) + (ow0) - 0), _mm256_loadu_ps((dy3) + (ow0) - 0), \
            (wp0)[0], (wp1)[0], (wp2)[0], (wp3)[0] \
        ); \
    } while (0)

static void stride1_bwd_dx_tile_c8_k1(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    static constexpr int K = Stride1Specialist<1>::K;
    (void)C_in;
    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, K, pad, conv_out_w)) {
        return;
    }
    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];
    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);
    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);
    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * 8 * dy_plane];
    const int64_t ow0 = doc.ow + pad + dy_pad_l;
    int64_t kh_lo = 0, kh_hi = K;
    bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);
    for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
        const float* __restrict w_kh = w_cin + kh * K;
        STRIDE1_BWD_DX_ACCUM_K1(
            v_dx, ow0,
            dy_oh + 0 * dy_plane, dy_oh + 1 * dy_plane,
            dy_oh + 2 * dy_plane, dy_oh + 3 * dy_plane,
            w_kh + 0 * w_cout_stride, w_kh + 1 * w_cout_stride,
            w_kh + 2 * w_cout_stride, w_kh + 3 * w_cout_stride
        );
        STRIDE1_BWD_DX_ACCUM_K1(
            v_dx, ow0,
            dy_oh + 4 * dy_plane, dy_oh + 5 * dy_plane,
            dy_oh + 6 * dy_plane, dy_oh + 7 * dy_plane,
            w_kh + 4 * w_cout_stride, w_kh + 5 * w_cout_stride,
            w_kh + 6 * w_cout_stride, w_kh + 7 * w_cout_stride
        );
    }
    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

static void stride1_bwd_dx_tile_c16_k1(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    static constexpr int K = Stride1Specialist<1>::K;
    (void)C_in;
    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, K, pad, conv_out_w)) {
        return;
    }
    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];
    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);
    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);
    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * 16 * dy_plane];
    const int64_t ow0 = doc.ow + pad + dy_pad_l;
    int64_t kh_lo = 0, kh_hi = K;
    bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);
    for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
        const float* __restrict w_kh = w_cin + kh * K;
        for (int64_t g = 0; g < 16; g += 4) {
            STRIDE1_BWD_DX_ACCUM_K1(
                v_dx, ow0,
                dy_oh + (g + 0) * dy_plane, dy_oh + (g + 1) * dy_plane,
                dy_oh + (g + 2) * dy_plane, dy_oh + (g + 3) * dy_plane,
                w_kh + (g + 0) * w_cout_stride, w_kh + (g + 1) * w_cout_stride,
                w_kh + (g + 2) * w_cout_stride, w_kh + (g + 3) * w_cout_stride
            );
        }
    }
    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

void Stride1Specialist<1>::bwd_dx_tile(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t C_out, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    if (C_out == 8) {
        stride1_bwd_dx_tile_c8_k1(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
        return;
    }
    if (C_out == 16) {
        stride1_bwd_dx_tile_c16_k1(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
        return;
    }
    (void)C_in;
    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, K, pad, conv_out_w)) {
        return;
    }
    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];
    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);
    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);
    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * C_out * dy_plane];
    const int64_t ow0 = doc.ow + pad + dy_pad_l;
    int64_t kh_lo = 0, kh_hi = K;
    bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);
    for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
        const float* __restrict w_kh = w_cin + kh * K;
        int64_t cout = 0;
        for (; cout + 3 < C_out; cout += 4) {
            STRIDE1_BWD_DX_ACCUM_K1(
                v_dx, ow0,
                dy_oh + (cout + 0) * dy_plane,
                dy_oh + (cout + 1) * dy_plane,
                dy_oh + (cout + 2) * dy_plane,
                dy_oh + (cout + 3) * dy_plane,
                w_kh + (cout + 0) * w_cout_stride,
                w_kh + (cout + 1) * w_cout_stride,
                w_kh + (cout + 2) * w_cout_stride,
                w_kh + (cout + 3) * w_cout_stride
            );
        }
        for (; cout < C_out; ++cout) {
            const float* __restrict dy_row = dy_oh + cout * dy_plane;
            const float* __restrict wp = w_kh + cout * w_cout_stride;
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 0), wp[0]);
        }
    }
    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

void Stride1Specialist<1>::dw_nci(
    int64_t n, int64_t cout, int64_t cin,
    float* __restrict dw_slice,
    const float* __restrict dy_pad_buf,
    const float* __restrict x_pad_buf,
    int64_t C_in, int64_t C_out,
    int64_t dy_pad_l, int64_t dy_row_stride,
    int64_t x_pad_l, int64_t x_row_stride,
    int64_t H, int64_t pad,
    int64_t conv_out_h, int64_t conv_out_w
) {
    (void)C_in;
    (void)C_out;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const int64_t x_plane  = H * x_row_stride;
    const float* __restrict dy_nc =
        &dy_pad_buf[(n * C_out + cout) * dy_plane];
    const float* __restrict x_nc =
        &x_pad_buf[(n * C_in + cin) * x_plane];
    const int64_t ow_tiles = (conv_out_w + FWD_TILE_OW - 1) / FWD_TILE_OW;
    for (int64_t kh = 0; kh < K; ++kh) {
        __m256 v_acc0 = _mm256_setzero_ps();
        for (int64_t oh = 0; oh < conv_out_h; ++oh) {
            const int64_t ih = oh - pad + kh;
            if (ih < 0 || ih >= H) continue;
            const float* __restrict dy_row = &dy_nc[oh * dy_row_stride];
            const float* __restrict x_row  = &x_nc[ih * x_row_stride];
            for (int64_t t = 0; t < ow_tiles; ++t) {
                const int64_t ow = t * FWD_TILE_OW;
                const __m256 dy8 = _mm256_loadu_ps(dy_row + dy_pad_l + ow);
                const int64_t iw0 = x_pad_l + ow - pad;
                v_acc0 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 0), v_acc0);
            }
        }
        dw_slice[kh * K + 0] += _mm256_reduce_add_ps(v_acc0);
    }
}

#undef STRIDE1_BWD_DX_ACCUM_K1

// --- Stride1Specialist<2> backward implementations ---

#define STRIDE1_BWD_DX_ACCUM_K2(v_dx, ow0, dy0, dy1, dy2, dy3, wp0, wp1, wp2, wp3) \
    do { \
        (v_dx) = fmadd_dx_cout4( \
            (v_dx), \
            _mm256_loadu_ps((dy0) + (ow0) - 0), _mm256_loadu_ps((dy1) + (ow0) - 0), \
            _mm256_loadu_ps((dy2) + (ow0) - 0), _mm256_loadu_ps((dy3) + (ow0) - 0), \
            (wp0)[0], (wp1)[0], (wp2)[0], (wp3)[0] \
        ); \
        (v_dx) = fmadd_dx_cout4( \
            (v_dx), \
            _mm256_loadu_ps((dy0) + (ow0) - 1), _mm256_loadu_ps((dy1) + (ow0) - 1), \
            _mm256_loadu_ps((dy2) + (ow0) - 1), _mm256_loadu_ps((dy3) + (ow0) - 1), \
            (wp0)[1], (wp1)[1], (wp2)[1], (wp3)[1] \
        ); \
    } while (0)

static void stride1_bwd_dx_tile_c8_k2(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    static constexpr int K = Stride1Specialist<2>::K;
    (void)C_in;
    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, K, pad, conv_out_w)) {
        return;
    }
    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];
    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);
    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);
    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * 8 * dy_plane];
    const int64_t ow0 = doc.ow + pad + dy_pad_l;
    int64_t kh_lo = 0, kh_hi = K;
    bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);
    for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
        const float* __restrict w_kh = w_cin + kh * K;
        STRIDE1_BWD_DX_ACCUM_K2(
            v_dx, ow0,
            dy_oh + 0 * dy_plane, dy_oh + 1 * dy_plane,
            dy_oh + 2 * dy_plane, dy_oh + 3 * dy_plane,
            w_kh + 0 * w_cout_stride, w_kh + 1 * w_cout_stride,
            w_kh + 2 * w_cout_stride, w_kh + 3 * w_cout_stride
        );
        STRIDE1_BWD_DX_ACCUM_K2(
            v_dx, ow0,
            dy_oh + 4 * dy_plane, dy_oh + 5 * dy_plane,
            dy_oh + 6 * dy_plane, dy_oh + 7 * dy_plane,
            w_kh + 4 * w_cout_stride, w_kh + 5 * w_cout_stride,
            w_kh + 6 * w_cout_stride, w_kh + 7 * w_cout_stride
        );
    }
    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

static void stride1_bwd_dx_tile_c16_k2(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    static constexpr int K = Stride1Specialist<2>::K;
    (void)C_in;
    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, K, pad, conv_out_w)) {
        return;
    }
    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];
    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);
    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);
    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * 16 * dy_plane];
    const int64_t ow0 = doc.ow + pad + dy_pad_l;
    int64_t kh_lo = 0, kh_hi = K;
    bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);
    for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
        const float* __restrict w_kh = w_cin + kh * K;
        for (int64_t g = 0; g < 16; g += 4) {
            STRIDE1_BWD_DX_ACCUM_K2(
                v_dx, ow0,
                dy_oh + (g + 0) * dy_plane, dy_oh + (g + 1) * dy_plane,
                dy_oh + (g + 2) * dy_plane, dy_oh + (g + 3) * dy_plane,
                w_kh + (g + 0) * w_cout_stride, w_kh + (g + 1) * w_cout_stride,
                w_kh + (g + 2) * w_cout_stride, w_kh + (g + 3) * w_cout_stride
            );
        }
    }
    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

void Stride1Specialist<2>::bwd_dx_tile(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t C_out, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    if (C_out == 8) {
        stride1_bwd_dx_tile_c8_k2(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
        return;
    }
    if (C_out == 16) {
        stride1_bwd_dx_tile_c16_k2(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
        return;
    }
    (void)C_in;
    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, K, pad, conv_out_w)) {
        return;
    }
    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];
    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);
    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);
    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * C_out * dy_plane];
    const int64_t ow0 = doc.ow + pad + dy_pad_l;
    int64_t kh_lo = 0, kh_hi = K;
    bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);
    for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
        const float* __restrict w_kh = w_cin + kh * K;
        int64_t cout = 0;
        for (; cout + 3 < C_out; cout += 4) {
            STRIDE1_BWD_DX_ACCUM_K2(
                v_dx, ow0,
                dy_oh + (cout + 0) * dy_plane,
                dy_oh + (cout + 1) * dy_plane,
                dy_oh + (cout + 2) * dy_plane,
                dy_oh + (cout + 3) * dy_plane,
                w_kh + (cout + 0) * w_cout_stride,
                w_kh + (cout + 1) * w_cout_stride,
                w_kh + (cout + 2) * w_cout_stride,
                w_kh + (cout + 3) * w_cout_stride
            );
        }
        for (; cout < C_out; ++cout) {
            const float* __restrict dy_row = dy_oh + cout * dy_plane;
            const float* __restrict wp = w_kh + cout * w_cout_stride;
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 0), wp[0]);
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 1), wp[1]);
        }
    }
    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

void Stride1Specialist<2>::dw_nci(
    int64_t n, int64_t cout, int64_t cin,
    float* __restrict dw_slice,
    const float* __restrict dy_pad_buf,
    const float* __restrict x_pad_buf,
    int64_t C_in, int64_t C_out,
    int64_t dy_pad_l, int64_t dy_row_stride,
    int64_t x_pad_l, int64_t x_row_stride,
    int64_t H, int64_t pad,
    int64_t conv_out_h, int64_t conv_out_w
) {
    (void)C_in;
    (void)C_out;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const int64_t x_plane  = H * x_row_stride;
    const float* __restrict dy_nc =
        &dy_pad_buf[(n * C_out + cout) * dy_plane];
    const float* __restrict x_nc =
        &x_pad_buf[(n * C_in + cin) * x_plane];
    const int64_t ow_tiles = (conv_out_w + FWD_TILE_OW - 1) / FWD_TILE_OW;
    for (int64_t kh = 0; kh < K; ++kh) {
        __m256 v_acc0 = _mm256_setzero_ps();
        __m256 v_acc1 = _mm256_setzero_ps();
        for (int64_t oh = 0; oh < conv_out_h; ++oh) {
            const int64_t ih = oh - pad + kh;
            if (ih < 0 || ih >= H) continue;
            const float* __restrict dy_row = &dy_nc[oh * dy_row_stride];
            const float* __restrict x_row  = &x_nc[ih * x_row_stride];
            for (int64_t t = 0; t < ow_tiles; ++t) {
                const int64_t ow = t * FWD_TILE_OW;
                const __m256 dy8 = _mm256_loadu_ps(dy_row + dy_pad_l + ow);
                const int64_t iw0 = x_pad_l + ow - pad;
                v_acc0 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 0), v_acc0);
                v_acc1 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 1), v_acc1);
            }
        }
        const int64_t base = kh * K;
        dw_slice[base + 0] += _mm256_reduce_add_ps(v_acc0);
        dw_slice[base + 1] += _mm256_reduce_add_ps(v_acc1);
    }
}

#undef STRIDE1_BWD_DX_ACCUM_K2

#define STRIDE1_BWD_DX_ACCUM_K3(v_dx, ow0, dy0, dy1, dy2, dy3, wp0, wp1, wp2, wp3) \
    do { \
        (v_dx) = fmadd_dx_cout4( \
            (v_dx), \
            _mm256_loadu_ps((dy0) + (ow0) - 0), _mm256_loadu_ps((dy1) + (ow0) - 0), \
            _mm256_loadu_ps((dy2) + (ow0) - 0), _mm256_loadu_ps((dy3) + (ow0) - 0), \
            (wp0)[0], (wp1)[0], (wp2)[0], (wp3)[0] \
        ); \
        (v_dx) = fmadd_dx_cout4( \
            (v_dx), \
            _mm256_loadu_ps((dy0) + (ow0) - 1), _mm256_loadu_ps((dy1) + (ow0) - 1), \
            _mm256_loadu_ps((dy2) + (ow0) - 1), _mm256_loadu_ps((dy3) + (ow0) - 1), \
            (wp0)[1], (wp1)[1], (wp2)[1], (wp3)[1] \
        ); \
        (v_dx) = fmadd_dx_cout4( \
            (v_dx), \
            _mm256_loadu_ps((dy0) + (ow0) - 2), _mm256_loadu_ps((dy1) + (ow0) - 2), \
            _mm256_loadu_ps((dy2) + (ow0) - 2), _mm256_loadu_ps((dy3) + (ow0) - 2), \
            (wp0)[2], (wp1)[2], (wp2)[2], (wp3)[2] \
        ); \
    } while (0)

#define STRIDE1_BWD_DX_ACCUM_K4(v_dx, ow0, dy0, dy1, dy2, dy3, wp0, wp1, wp2, wp3) \
    do { \
        (v_dx) = fmadd_dx_cout4( \
            (v_dx), \
            _mm256_loadu_ps((dy0) + (ow0) - 0), _mm256_loadu_ps((dy1) + (ow0) - 0), \
            _mm256_loadu_ps((dy2) + (ow0) - 0), _mm256_loadu_ps((dy3) + (ow0) - 0), \
            (wp0)[0], (wp1)[0], (wp2)[0], (wp3)[0] \
        ); \
        (v_dx) = fmadd_dx_cout4( \
            (v_dx), \
            _mm256_loadu_ps((dy0) + (ow0) - 1), _mm256_loadu_ps((dy1) + (ow0) - 1), \
            _mm256_loadu_ps((dy2) + (ow0) - 1), _mm256_loadu_ps((dy3) + (ow0) - 1), \
            (wp0)[1], (wp1)[1], (wp2)[1], (wp3)[1] \
        ); \
        (v_dx) = fmadd_dx_cout4( \
            (v_dx), \
            _mm256_loadu_ps((dy0) + (ow0) - 2), _mm256_loadu_ps((dy1) + (ow0) - 2), \
            _mm256_loadu_ps((dy2) + (ow0) - 2), _mm256_loadu_ps((dy3) + (ow0) - 2), \
            (wp0)[2], (wp1)[2], (wp2)[2], (wp3)[2] \
        ); \
        (v_dx) = fmadd_dx_cout4( \
            (v_dx), \
            _mm256_loadu_ps((dy0) + (ow0) - 3), _mm256_loadu_ps((dy1) + (ow0) - 3), \
            _mm256_loadu_ps((dy2) + (ow0) - 3), _mm256_loadu_ps((dy3) + (ow0) - 3), \
            (wp0)[3], (wp1)[3], (wp2)[3], (wp3)[3] \
        ); \
    } while (0)

static void stride1_bwd_dx_tile_c8_k3(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    static constexpr int K = Stride1Specialist<3>::K;
    (void)C_in;
    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, K, pad, conv_out_w)) {
        return;
    }
    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];
    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);
    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);
    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * 8 * dy_plane];
    const int64_t ow0 = doc.ow + pad + dy_pad_l;
    int64_t kh_lo = 0, kh_hi = K;
    bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);
    for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
        const float* __restrict w_kh = w_cin + kh * K;
        STRIDE1_BWD_DX_ACCUM_K3(
            v_dx, ow0,
            dy_oh + 0 * dy_plane, dy_oh + 1 * dy_plane,
            dy_oh + 2 * dy_plane, dy_oh + 3 * dy_plane,
            w_kh + 0 * w_cout_stride, w_kh + 1 * w_cout_stride,
            w_kh + 2 * w_cout_stride, w_kh + 3 * w_cout_stride
        );
        STRIDE1_BWD_DX_ACCUM_K3(
            v_dx, ow0,
            dy_oh + 4 * dy_plane, dy_oh + 5 * dy_plane,
            dy_oh + 6 * dy_plane, dy_oh + 7 * dy_plane,
            w_kh + 4 * w_cout_stride, w_kh + 5 * w_cout_stride,
            w_kh + 6 * w_cout_stride, w_kh + 7 * w_cout_stride
        );
    }
    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

static void stride1_bwd_dx_tile_c16_k3(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    static constexpr int K = Stride1Specialist<3>::K;
    (void)C_in;
    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, K, pad, conv_out_w)) {
        return;
    }
    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];
    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);
    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);
    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * 16 * dy_plane];
    const int64_t ow0 = doc.ow + pad + dy_pad_l;
    int64_t kh_lo = 0, kh_hi = K;
    bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);
    for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
        const float* __restrict w_kh = w_cin + kh * K;
        for (int64_t g = 0; g < 16; g += 4) {
            STRIDE1_BWD_DX_ACCUM_K3(
                v_dx, ow0,
                dy_oh + (g + 0) * dy_plane, dy_oh + (g + 1) * dy_plane,
                dy_oh + (g + 2) * dy_plane, dy_oh + (g + 3) * dy_plane,
                w_kh + (g + 0) * w_cout_stride, w_kh + (g + 1) * w_cout_stride,
                w_kh + (g + 2) * w_cout_stride, w_kh + (g + 3) * w_cout_stride
            );
        }
    }
    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

void Stride1Specialist<3>::bwd_dx_tile(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t C_out, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    if (C_out == 8) {
        stride1_bwd_dx_tile_c8_k3(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
        return;
    }
    if (C_out == 16) {
        stride1_bwd_dx_tile_c16_k3(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
        return;
    }
    (void)C_in;
    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, K, pad, conv_out_w)) {
        return;
    }
    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];
    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);
    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);
    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * C_out * dy_plane];
    const int64_t ow0 = doc.ow + pad + dy_pad_l;
    int64_t kh_lo = 0, kh_hi = K;
    bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);
    for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
        const float* __restrict w_kh = w_cin + kh * K;
        int64_t cout = 0;
        for (; cout + 3 < C_out; cout += 4) {
            STRIDE1_BWD_DX_ACCUM_K3(
                v_dx, ow0,
                dy_oh + (cout + 0) * dy_plane,
                dy_oh + (cout + 1) * dy_plane,
                dy_oh + (cout + 2) * dy_plane,
                dy_oh + (cout + 3) * dy_plane,
                w_kh + (cout + 0) * w_cout_stride,
                w_kh + (cout + 1) * w_cout_stride,
                w_kh + (cout + 2) * w_cout_stride,
                w_kh + (cout + 3) * w_cout_stride
            );
        }
        for (; cout < C_out; ++cout) {
            const float* __restrict dy_row = dy_oh + cout * dy_plane;
            const float* __restrict wp = w_kh + cout * w_cout_stride;
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 0), wp[0]);
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 1), wp[1]);
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 2), wp[2]);
        }
    }
    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

void Stride1Specialist<3>::dw_nci(
    int64_t n, int64_t cout, int64_t cin,
    float* __restrict dw_slice,
    const float* __restrict dy_pad_buf,
    const float* __restrict x_pad_buf,
    int64_t C_in, int64_t C_out,
    int64_t dy_pad_l, int64_t dy_row_stride,
    int64_t x_pad_l, int64_t x_row_stride,
    int64_t H, int64_t pad,
    int64_t conv_out_h, int64_t conv_out_w
) {
    (void)C_in;
    (void)C_out;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const int64_t x_plane  = H * x_row_stride;
    const float* __restrict dy_nc =
        &dy_pad_buf[(n * C_out + cout) * dy_plane];
    const float* __restrict x_nc =
        &x_pad_buf[(n * C_in + cin) * x_plane];
    const int64_t ow_tiles = (conv_out_w + FWD_TILE_OW - 1) / FWD_TILE_OW;
    for (int64_t kh = 0; kh < K; ++kh) {
        __m256 v_acc0 = _mm256_setzero_ps();
        __m256 v_acc1 = _mm256_setzero_ps();
        __m256 v_acc2 = _mm256_setzero_ps();
        for (int64_t oh = 0; oh < conv_out_h; ++oh) {
            const int64_t ih = oh - pad + kh;
            if (ih < 0 || ih >= H) continue;
            const float* __restrict dy_row = &dy_nc[oh * dy_row_stride];
            const float* __restrict x_row  = &x_nc[ih * x_row_stride];
            for (int64_t t = 0; t < ow_tiles; ++t) {
                const int64_t ow = t * FWD_TILE_OW;
                const __m256 dy8 = _mm256_loadu_ps(dy_row + dy_pad_l + ow);
                const int64_t iw0 = x_pad_l + ow - pad;
                v_acc0 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 0), v_acc0);
                v_acc1 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 1), v_acc1);
                v_acc2 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 2), v_acc2);
            }
        }
        const int64_t base = kh * K;
        dw_slice[base + 0] += _mm256_reduce_add_ps(v_acc0);
        dw_slice[base + 1] += _mm256_reduce_add_ps(v_acc1);
        dw_slice[base + 2] += _mm256_reduce_add_ps(v_acc2);
    }
}

static void stride1_bwd_dx_tile_c8_k4(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    static constexpr int K = Stride1Specialist<4>::K;
    (void)C_in;
    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, K, pad, conv_out_w)) {
        return;
    }
    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];
    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);
    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);
    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * 8 * dy_plane];
    const int64_t ow0 = doc.ow + pad + dy_pad_l;
    int64_t kh_lo = 0, kh_hi = K;
    bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);
    for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
        const float* __restrict w_kh = w_cin + kh * K;
        STRIDE1_BWD_DX_ACCUM_K4(
            v_dx, ow0,
            dy_oh + 0 * dy_plane, dy_oh + 1 * dy_plane,
            dy_oh + 2 * dy_plane, dy_oh + 3 * dy_plane,
            w_kh + 0 * w_cout_stride, w_kh + 1 * w_cout_stride,
            w_kh + 2 * w_cout_stride, w_kh + 3 * w_cout_stride
        );
        STRIDE1_BWD_DX_ACCUM_K4(
            v_dx, ow0,
            dy_oh + 4 * dy_plane, dy_oh + 5 * dy_plane,
            dy_oh + 6 * dy_plane, dy_oh + 7 * dy_plane,
            w_kh + 4 * w_cout_stride, w_kh + 5 * w_cout_stride,
            w_kh + 6 * w_cout_stride, w_kh + 7 * w_cout_stride
        );
    }
    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

static void stride1_bwd_dx_tile_c16_k4(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    static constexpr int K = Stride1Specialist<4>::K;
    (void)C_in;
    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, K, pad, conv_out_w)) {
        return;
    }
    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];
    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);
    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);
    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * 16 * dy_plane];
    const int64_t ow0 = doc.ow + pad + dy_pad_l;
    int64_t kh_lo = 0, kh_hi = K;
    bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);
    for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
        const float* __restrict w_kh = w_cin + kh * K;
        for (int64_t g = 0; g < 16; g += 4) {
            STRIDE1_BWD_DX_ACCUM_K4(
                v_dx, ow0,
                dy_oh + (g + 0) * dy_plane, dy_oh + (g + 1) * dy_plane,
                dy_oh + (g + 2) * dy_plane, dy_oh + (g + 3) * dy_plane,
                w_kh + (g + 0) * w_cout_stride, w_kh + (g + 1) * w_cout_stride,
                w_kh + (g + 2) * w_cout_stride, w_kh + (g + 3) * w_cout_stride
            );
        }
    }
    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

void Stride1Specialist<4>::bwd_dx_tile(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t C_out, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    if (C_out == 8) {
        stride1_bwd_dx_tile_c8_k4(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
        return;
    }
    if (C_out == 16) {
        stride1_bwd_dx_tile_c16_k4(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
        return;
    }
    (void)C_in;
    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, K, pad, conv_out_w)) {
        return;
    }
    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];
    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);
    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);
    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * C_out * dy_plane];
    const int64_t ow0 = doc.ow + pad + dy_pad_l;
    int64_t kh_lo = 0, kh_hi = K;
    bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);
    for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
        const float* __restrict w_kh = w_cin + kh * K;
        int64_t cout = 0;
        for (; cout + 3 < C_out; cout += 4) {
            STRIDE1_BWD_DX_ACCUM_K4(
                v_dx, ow0,
                dy_oh + (cout + 0) * dy_plane,
                dy_oh + (cout + 1) * dy_plane,
                dy_oh + (cout + 2) * dy_plane,
                dy_oh + (cout + 3) * dy_plane,
                w_kh + (cout + 0) * w_cout_stride,
                w_kh + (cout + 1) * w_cout_stride,
                w_kh + (cout + 2) * w_cout_stride,
                w_kh + (cout + 3) * w_cout_stride
            );
        }
        for (; cout < C_out; ++cout) {
            const float* __restrict dy_row = dy_oh + cout * dy_plane;
            const float* __restrict wp = w_kh + cout * w_cout_stride;
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 0), wp[0]);
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 1), wp[1]);
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 2), wp[2]);
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 3), wp[3]);
        }
    }
    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

void Stride1Specialist<4>::dw_nci(
    int64_t n, int64_t cout, int64_t cin,
    float* __restrict dw_slice,
    const float* __restrict dy_pad_buf,
    const float* __restrict x_pad_buf,
    int64_t C_in, int64_t C_out,
    int64_t dy_pad_l, int64_t dy_row_stride,
    int64_t x_pad_l, int64_t x_row_stride,
    int64_t H, int64_t pad,
    int64_t conv_out_h, int64_t conv_out_w
) {
    (void)C_in;
    (void)C_out;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const int64_t x_plane  = H * x_row_stride;
    const float* __restrict dy_nc =
        &dy_pad_buf[(n * C_out + cout) * dy_plane];
    const float* __restrict x_nc =
        &x_pad_buf[(n * C_in + cin) * x_plane];
    const int64_t ow_tiles = (conv_out_w + FWD_TILE_OW - 1) / FWD_TILE_OW;
    for (int64_t kh = 0; kh < K; ++kh) {
        __m256 v_acc0 = _mm256_setzero_ps();
        __m256 v_acc1 = _mm256_setzero_ps();
        __m256 v_acc2 = _mm256_setzero_ps();
        __m256 v_acc3 = _mm256_setzero_ps();
        for (int64_t oh = 0; oh < conv_out_h; ++oh) {
            const int64_t ih = oh - pad + kh;
            if (ih < 0 || ih >= H) continue;
            const float* __restrict dy_row = &dy_nc[oh * dy_row_stride];
            const float* __restrict x_row  = &x_nc[ih * x_row_stride];
            for (int64_t t = 0; t < ow_tiles; ++t) {
                const int64_t ow = t * FWD_TILE_OW;
                const __m256 dy8 = _mm256_loadu_ps(dy_row + dy_pad_l + ow);
                const int64_t iw0 = x_pad_l + ow - pad;
                v_acc0 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 0), v_acc0);
                v_acc1 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 1), v_acc1);
                v_acc2 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 2), v_acc2);
                v_acc3 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 3), v_acc3);
            }
        }
        const int64_t base = kh * K;
        dw_slice[base + 0] += _mm256_reduce_add_ps(v_acc0);
        dw_slice[base + 1] += _mm256_reduce_add_ps(v_acc1);
        dw_slice[base + 2] += _mm256_reduce_add_ps(v_acc2);
        dw_slice[base + 3] += _mm256_reduce_add_ps(v_acc3);
    }
}

#undef STRIDE1_BWD_DX_ACCUM_K3
#undef STRIDE1_BWD_DX_ACCUM_K4

// --- Stride1Specialist<5> backward implementations (mirror k6, K=5 taps) ---

static __forceinline void stride1_bwd_dx_accum_cout4_k5(
    __m256& v_dx,
    int64_t ow0,
    const float* __restrict dy0,
    const float* __restrict dy1,
    const float* __restrict dy2,
    const float* __restrict dy3,
    const float* __restrict wp0,
    const float* __restrict wp1,
    const float* __restrict wp2,
    const float* __restrict wp3
) {
    v_dx = fmadd_dx_cout4(
        v_dx,
        _mm256_loadu_ps(dy0 + ow0 - 0), _mm256_loadu_ps(dy1 + ow0 - 0),
        _mm256_loadu_ps(dy2 + ow0 - 0), _mm256_loadu_ps(dy3 + ow0 - 0),
        wp0[0], wp1[0], wp2[0], wp3[0]
    );
    v_dx = fmadd_dx_cout4(
        v_dx,
        _mm256_loadu_ps(dy0 + ow0 - 1), _mm256_loadu_ps(dy1 + ow0 - 1),
        _mm256_loadu_ps(dy2 + ow0 - 1), _mm256_loadu_ps(dy3 + ow0 - 1),
        wp0[1], wp1[1], wp2[1], wp3[1]
    );
    v_dx = fmadd_dx_cout4(
        v_dx,
        _mm256_loadu_ps(dy0 + ow0 - 2), _mm256_loadu_ps(dy1 + ow0 - 2),
        _mm256_loadu_ps(dy2 + ow0 - 2), _mm256_loadu_ps(dy3 + ow0 - 2),
        wp0[2], wp1[2], wp2[2], wp3[2]
    );
    v_dx = fmadd_dx_cout4(
        v_dx,
        _mm256_loadu_ps(dy0 + ow0 - 3), _mm256_loadu_ps(dy1 + ow0 - 3),
        _mm256_loadu_ps(dy2 + ow0 - 3), _mm256_loadu_ps(dy3 + ow0 - 3),
        wp0[3], wp1[3], wp2[3], wp3[3]
    );
    v_dx = fmadd_dx_cout4(
        v_dx,
        _mm256_loadu_ps(dy0 + ow0 - 4), _mm256_loadu_ps(dy1 + ow0 - 4),
        _mm256_loadu_ps(dy2 + ow0 - 4), _mm256_loadu_ps(dy3 + ow0 - 4),
        wp0[4], wp1[4], wp2[4], wp3[4]
    );
}

static void stride1_bwd_dx_tile_c8_k5(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    static constexpr int K = Stride1Specialist<5>::K;
    (void)C_in;

    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, K, pad, conv_out_w)) {
        return;
    }

    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];

    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);

    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);

    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * 8 * dy_plane];
    const int64_t ow0 = doc.ow + pad + dy_pad_l;

    int64_t kh_lo = 0;
    int64_t kh_hi = K;
    bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);

    for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
        const float* __restrict w_kh = w_cin + kh * K;

        stride1_bwd_dx_accum_cout4_k5(
            v_dx, ow0,
            dy_oh + 0 * dy_plane, dy_oh + 1 * dy_plane,
            dy_oh + 2 * dy_plane, dy_oh + 3 * dy_plane,
            w_kh + 0 * w_cout_stride, w_kh + 1 * w_cout_stride,
            w_kh + 2 * w_cout_stride, w_kh + 3 * w_cout_stride
        );
        stride1_bwd_dx_accum_cout4_k5(
            v_dx, ow0,
            dy_oh + 4 * dy_plane, dy_oh + 5 * dy_plane,
            dy_oh + 6 * dy_plane, dy_oh + 7 * dy_plane,
            w_kh + 4 * w_cout_stride, w_kh + 5 * w_cout_stride,
            w_kh + 6 * w_cout_stride, w_kh + 7 * w_cout_stride
        );
    }

    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

static void stride1_bwd_dx_tile_c16_k5(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    static constexpr int K = Stride1Specialist<5>::K;
    (void)C_in;

    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, K, pad, conv_out_w)) {
        return;
    }

    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];

    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);

    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);

    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * 16 * dy_plane];
    const int64_t ow0 = doc.ow + pad + dy_pad_l;

    int64_t kh_lo = 0;
    int64_t kh_hi = K;
    bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);

    for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
        const float* __restrict w_kh = w_cin + kh * K;

        stride1_bwd_dx_accum_cout4_k5(
            v_dx, ow0,
            dy_oh + 0 * dy_plane, dy_oh + 1 * dy_plane,
            dy_oh + 2 * dy_plane, dy_oh + 3 * dy_plane,
            w_kh + 0 * w_cout_stride, w_kh + 1 * w_cout_stride,
            w_kh + 2 * w_cout_stride, w_kh + 3 * w_cout_stride
        );
        stride1_bwd_dx_accum_cout4_k5(
            v_dx, ow0,
            dy_oh + 4 * dy_plane, dy_oh + 5 * dy_plane,
            dy_oh + 6 * dy_plane, dy_oh + 7 * dy_plane,
            w_kh + 4 * w_cout_stride, w_kh + 5 * w_cout_stride,
            w_kh + 6 * w_cout_stride, w_kh + 7 * w_cout_stride
        );
        stride1_bwd_dx_accum_cout4_k5(
            v_dx, ow0,
            dy_oh + 8 * dy_plane, dy_oh + 9 * dy_plane,
            dy_oh + 10 * dy_plane, dy_oh + 11 * dy_plane,
            w_kh + 8 * w_cout_stride, w_kh + 9 * w_cout_stride,
            w_kh + 10 * w_cout_stride, w_kh + 11 * w_cout_stride
        );
        stride1_bwd_dx_accum_cout4_k5(
            v_dx, ow0,
            dy_oh + 12 * dy_plane, dy_oh + 13 * dy_plane,
            dy_oh + 14 * dy_plane, dy_oh + 15 * dy_plane,
            w_kh + 12 * w_cout_stride, w_kh + 13 * w_cout_stride,
            w_kh + 14 * w_cout_stride, w_kh + 15 * w_cout_stride
        );
    }

    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

void Stride1Specialist<5>::bwd_dx_tile(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t C_out, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    if (C_out == 8) {
        stride1_bwd_dx_tile_c8_k5(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
        return;
    }
    if (C_out == 16) {
        stride1_bwd_dx_tile_c16_k5(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
        return;
    }

    (void)C_in;

    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, K, pad, conv_out_w)) {
        return;
    }

    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];

    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);

    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);

    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * C_out * dy_plane];
    const int64_t ow0 = doc.ow + pad + dy_pad_l;

    int64_t kh_lo = 0;
    int64_t kh_hi = K;
    bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);

    for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
        const float* __restrict w_kh = w_cin + kh * K;

        int64_t cout = 0;
        for (; cout + 3 < C_out; cout += 4) {
            stride1_bwd_dx_accum_cout4_k5(
                v_dx, ow0,
                dy_oh + (cout + 0) * dy_plane,
                dy_oh + (cout + 1) * dy_plane,
                dy_oh + (cout + 2) * dy_plane,
                dy_oh + (cout + 3) * dy_plane,
                w_kh + (cout + 0) * w_cout_stride,
                w_kh + (cout + 1) * w_cout_stride,
                w_kh + (cout + 2) * w_cout_stride,
                w_kh + (cout + 3) * w_cout_stride
            );
        }
        for (; cout < C_out; ++cout) {
            const float* __restrict dy_row = dy_oh + cout * dy_plane;
            const float* __restrict wp = w_kh + cout * w_cout_stride;

            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 0), wp[0]);
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 1), wp[1]);
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 2), wp[2]);
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 3), wp[3]);
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 4), wp[4]);
        }
    }

    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

// Accumulate dX for one input tile: dx += sum_cout sum_kh,kw dY[cout,oh,ow] * W[cout,cin,tap]
// Padded dY rows (dy_pad_buf): loadu every kw. Else: loadu interior / maskload+gather edges.
static void process_bwd_dx_tile_stride1(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict d_conv_buf,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t C_out, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t k_h, int64_t k_w, int64_t pad,
    int64_t spatial_in, int64_t conv_spatial, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w, int64_t conv_out_w_stride
) {
    (void)H;
    (void)W_in;

    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, k_w, pad, conv_out_w)) {
        return;
    }

    if (stride1_try_bwd_dx_specialist(
            k_h, k_w, doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w)) {
        return;
    }

    float* __restrict dx_row = &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];

    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);

    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);

    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const bool use_dy_pad = (dy_pad_buf != nullptr);
    const int64_t dy_plane = conv_out_h * (use_dy_pad ? dy_row_stride : conv_out_w_stride);
    const __m256i v_cow = _mm256_set1_epi32((int)conv_out_w);
    const bool direct_dy = use_dy_pad || bwd_dx_tile_is_interior(doc, k_w, pad, conv_out_w);
    const float* __restrict dy_base = use_dy_pad ? dy_pad_buf : d_conv_buf;

    for (int64_t kh = 0; kh < k_h; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        if (oh < 0 || oh >= conv_out_h) continue;

        const float* __restrict dy_oh =
            &dy_base[doc.n * C_out * dy_plane + oh * (use_dy_pad ? dy_row_stride : conv_out_w_stride)];
        const float* __restrict w_kh = w_cin + kh * k_w;

        int64_t cout = 0;
        if (direct_dy) {
            for (; cout + 3 < C_out; cout += 4) {
                const float* __restrict dy0 = dy_oh + (cout + 0) * dy_plane;
                const float* __restrict dy1 = dy_oh + (cout + 1) * dy_plane;
                const float* __restrict dy2 = dy_oh + (cout + 2) * dy_plane;
                const float* __restrict dy3 = dy_oh + (cout + 3) * dy_plane;
                const float* __restrict wp0 = w_kh + (cout + 0) * w_cout_stride;
                const float* __restrict wp1 = w_kh + (cout + 1) * w_cout_stride;
                const float* __restrict wp2 = w_kh + (cout + 2) * w_cout_stride;
                const float* __restrict wp3 = w_kh + (cout + 3) * w_cout_stride;

                for (int64_t kw = 0; kw < k_w; ++kw) {
                    const int64_t ow_base = doc.ow + pad - kw;
                    const int64_t dy_off = ow_base + (use_dy_pad ? dy_pad_l : 0);
                    v_dx = fmadd_dx_cout4(
                        v_dx,
                        _mm256_loadu_ps(dy0 + dy_off),
                        _mm256_loadu_ps(dy1 + dy_off),
                        _mm256_loadu_ps(dy2 + dy_off),
                        _mm256_loadu_ps(dy3 + dy_off),
                        wp0[kw], wp1[kw], wp2[kw], wp3[kw]
                    );
                }
            }
            for (; cout < C_out; ++cout) {
                const float* __restrict dy_row = dy_oh + cout * dy_plane;
                const float* __restrict wp = w_kh + cout * w_cout_stride;
                for (int64_t kw = 0; kw < k_w; ++kw) {
                    const int64_t ow_base = doc.ow + pad - kw;
                    const int64_t dy_off = ow_base + (use_dy_pad ? dy_pad_l : 0);
                    v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + dy_off), wp[kw]);
                }
            }
        } else {
            for (; cout + 3 < C_out; cout += 4) {
                const float* __restrict dy0 = dy_oh + (cout + 0) * dy_plane;
                const float* __restrict dy1 = dy_oh + (cout + 1) * dy_plane;
                const float* __restrict dy2 = dy_oh + (cout + 2) * dy_plane;
                const float* __restrict dy3 = dy_oh + (cout + 3) * dy_plane;
                const float* __restrict wp0 = w_kh + (cout + 0) * w_cout_stride;
                const float* __restrict wp1 = w_kh + (cout + 1) * w_cout_stride;
                const float* __restrict wp2 = w_kh + (cout + 2) * w_cout_stride;
                const float* __restrict wp3 = w_kh + (cout + 3) * w_cout_stride;

                for (int64_t kw = 0; kw < k_w; ++kw) {
                    const int64_t ow_base = doc.ow + pad - kw;
                    v_dx = fmadd_dx_cout4(
                        v_dx,
                        load_dy_window_8(dy0, ow_base, conv_out_w, full_dx, dx_mask, v_idx, v_cow),
                        load_dy_window_8(dy1, ow_base, conv_out_w, full_dx, dx_mask, v_idx, v_cow),
                        load_dy_window_8(dy2, ow_base, conv_out_w, full_dx, dx_mask, v_idx, v_cow),
                        load_dy_window_8(dy3, ow_base, conv_out_w, full_dx, dx_mask, v_idx, v_cow),
                        wp0[kw], wp1[kw], wp2[kw], wp3[kw]
                    );
                }
            }
            for (; cout < C_out; ++cout) {
                const float* __restrict dy_row = dy_oh + cout * dy_plane;
                const float* __restrict wp = w_kh + cout * w_cout_stride;
                for (int64_t kw = 0; kw < k_w; ++kw) {
                    const int64_t ow_base = doc.ow + pad - kw;
                    v_dx = fmadd_dx_cout1(
                        v_dx,
                        load_dy_window_8(dy_row, ow_base, conv_out_w, full_dx, dx_mask, v_idx, v_cow),
                        wp[kw]
                    );
                }
            }
        }
    }

    if (full_dx) {
        _mm256_storeu_ps(dx_row, v_dx);
    } else {
        _mm256_maskstore_ps(dx_row, dx_mask, v_dx);
    }
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

// Stride-1 dW specialist: x_pad + dy_pad => all strips interior; K tap accs per kh.
void Stride1Specialist<6>::dw_nci(
    int64_t n, int64_t cout, int64_t cin,
    float* __restrict dw_slice,
    const float* __restrict dy_pad_buf,
    const float* __restrict x_pad_buf,
    int64_t C_in, int64_t C_out,
    int64_t dy_pad_l, int64_t dy_row_stride,
    int64_t x_pad_l, int64_t x_row_stride,
    int64_t H, int64_t pad,
    int64_t conv_out_h, int64_t conv_out_w
) {
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const int64_t x_plane  = H * x_row_stride;
    const float* __restrict dy_nc =
        &dy_pad_buf[(n * C_out + cout) * dy_plane];
    const float* __restrict x_nc =
        &x_pad_buf[(n * C_in + cin) * x_plane];

    const int64_t ow_tiles = (conv_out_w + FWD_TILE_OW - 1) / FWD_TILE_OW;

    for (int64_t kh = 0; kh < K; ++kh) {
        __m256 v_acc0 = _mm256_setzero_ps();
        __m256 v_acc1 = _mm256_setzero_ps();
        __m256 v_acc2 = _mm256_setzero_ps();
        __m256 v_acc3 = _mm256_setzero_ps();
        __m256 v_acc4 = _mm256_setzero_ps();
        __m256 v_acc5 = _mm256_setzero_ps();

        for (int64_t oh = 0; oh < conv_out_h; ++oh) {
            const int64_t ih = oh - pad + kh;
            if (ih < 0 || ih >= H) {
                continue;
            }

            const float* __restrict dy_row = &dy_nc[oh * dy_row_stride];
            const float* __restrict x_row  = &x_nc[ih * x_row_stride];

            for (int64_t t = 0; t < ow_tiles; ++t) {
                const int64_t ow = t * FWD_TILE_OW;
                const __m256 dy8 = _mm256_loadu_ps(dy_row + dy_pad_l + ow);
                const int64_t iw0 = x_pad_l + ow - pad;

                v_acc0 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 0), v_acc0);
                v_acc1 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 1), v_acc1);
                v_acc2 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 2), v_acc2);
                v_acc3 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 3), v_acc3);
                v_acc4 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 4), v_acc4);
                v_acc5 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 5), v_acc5);
            }
        }

        const int64_t base = kh * K;
        dw_slice[base + 0] += _mm256_reduce_add_ps(v_acc0);
        dw_slice[base + 1] += _mm256_reduce_add_ps(v_acc1);
        dw_slice[base + 2] += _mm256_reduce_add_ps(v_acc2);
        dw_slice[base + 3] += _mm256_reduce_add_ps(v_acc3);
        dw_slice[base + 4] += _mm256_reduce_add_ps(v_acc4);
        dw_slice[base + 5] += _mm256_reduce_add_ps(v_acc5);
    }
}

void Stride1Specialist<5>::dw_nci(
    int64_t n, int64_t cout, int64_t cin,
    float* __restrict dw_slice,
    const float* __restrict dy_pad_buf,
    const float* __restrict x_pad_buf,
    int64_t C_in, int64_t C_out,
    int64_t dy_pad_l, int64_t dy_row_stride,
    int64_t x_pad_l, int64_t x_row_stride,
    int64_t H, int64_t pad,
    int64_t conv_out_h, int64_t conv_out_w
) {
    (void)C_in;
    (void)C_out;

    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const int64_t x_plane  = H * x_row_stride;
    const float* __restrict dy_nc =
        &dy_pad_buf[(n * C_out + cout) * dy_plane];
    const float* __restrict x_nc =
        &x_pad_buf[(n * C_in + cin) * x_plane];

    const int64_t ow_tiles = (conv_out_w + FWD_TILE_OW - 1) / FWD_TILE_OW;

    for (int64_t kh = 0; kh < K; ++kh) {
        __m256 v_acc0 = _mm256_setzero_ps();
        __m256 v_acc1 = _mm256_setzero_ps();
        __m256 v_acc2 = _mm256_setzero_ps();
        __m256 v_acc3 = _mm256_setzero_ps();
        __m256 v_acc4 = _mm256_setzero_ps();

        for (int64_t oh = 0; oh < conv_out_h; ++oh) {
            const int64_t ih = oh - pad + kh;
            if (ih < 0 || ih >= H) {
                continue;
            }

            const float* __restrict dy_row = &dy_nc[oh * dy_row_stride];
            const float* __restrict x_row  = &x_nc[ih * x_row_stride];

            for (int64_t t = 0; t < ow_tiles; ++t) {
                const int64_t ow = t * FWD_TILE_OW;
                const __m256 dy8 = _mm256_loadu_ps(dy_row + dy_pad_l + ow);
                const int64_t iw0 = x_pad_l + ow - pad;

                v_acc0 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 0), v_acc0);
                v_acc1 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 1), v_acc1);
                v_acc2 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 2), v_acc2);
                v_acc3 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 3), v_acc3);
                v_acc4 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 4), v_acc4);
            }
        }

        const int64_t base = kh * K;
        dw_slice[base + 0] += _mm256_reduce_add_ps(v_acc0);
        dw_slice[base + 1] += _mm256_reduce_add_ps(v_acc1);
        dw_slice[base + 2] += _mm256_reduce_add_ps(v_acc2);
        dw_slice[base + 3] += _mm256_reduce_add_ps(v_acc3);
        dw_slice[base + 4] += _mm256_reduce_add_ps(v_acc4);
    }
}

// --- Stride1Specialist<7> backward implementations ---

static __forceinline void stride1_bwd_dx_accum_cout4_k7(
    __m256& v_dx,
    int64_t ow0,
    const float* __restrict dy0,
    const float* __restrict dy1,
    const float* __restrict dy2,
    const float* __restrict dy3,
    const float* __restrict wp0,
    const float* __restrict wp1,
    const float* __restrict wp2,
    const float* __restrict wp3
) {
    v_dx = fmadd_dx_cout4(
        v_dx,
        _mm256_loadu_ps(dy0 + ow0 - 0), _mm256_loadu_ps(dy1 + ow0 - 0),
        _mm256_loadu_ps(dy2 + ow0 - 0), _mm256_loadu_ps(dy3 + ow0 - 0),
        wp0[0], wp1[0], wp2[0], wp3[0]
    );
    v_dx = fmadd_dx_cout4(
        v_dx,
        _mm256_loadu_ps(dy0 + ow0 - 1), _mm256_loadu_ps(dy1 + ow0 - 1),
        _mm256_loadu_ps(dy2 + ow0 - 1), _mm256_loadu_ps(dy3 + ow0 - 1),
        wp0[1], wp1[1], wp2[1], wp3[1]
    );
    v_dx = fmadd_dx_cout4(
        v_dx,
        _mm256_loadu_ps(dy0 + ow0 - 2), _mm256_loadu_ps(dy1 + ow0 - 2),
        _mm256_loadu_ps(dy2 + ow0 - 2), _mm256_loadu_ps(dy3 + ow0 - 2),
        wp0[2], wp1[2], wp2[2], wp3[2]
    );
    v_dx = fmadd_dx_cout4(
        v_dx,
        _mm256_loadu_ps(dy0 + ow0 - 3), _mm256_loadu_ps(dy1 + ow0 - 3),
        _mm256_loadu_ps(dy2 + ow0 - 3), _mm256_loadu_ps(dy3 + ow0 - 3),
        wp0[3], wp1[3], wp2[3], wp3[3]
    );
    v_dx = fmadd_dx_cout4(
        v_dx,
        _mm256_loadu_ps(dy0 + ow0 - 4), _mm256_loadu_ps(dy1 + ow0 - 4),
        _mm256_loadu_ps(dy2 + ow0 - 4), _mm256_loadu_ps(dy3 + ow0 - 4),
        wp0[4], wp1[4], wp2[4], wp3[4]
    );
    v_dx = fmadd_dx_cout4(
        v_dx,
        _mm256_loadu_ps(dy0 + ow0 - 5), _mm256_loadu_ps(dy1 + ow0 - 5),
        _mm256_loadu_ps(dy2 + ow0 - 5), _mm256_loadu_ps(dy3 + ow0 - 5),
        wp0[5], wp1[5], wp2[5], wp3[5]
    );
    v_dx = fmadd_dx_cout4(
        v_dx,
        _mm256_loadu_ps(dy0 + ow0 - 6), _mm256_loadu_ps(dy1 + ow0 - 6),
        _mm256_loadu_ps(dy2 + ow0 - 6), _mm256_loadu_ps(dy3 + ow0 - 6),
        wp0[6], wp1[6], wp2[6], wp3[6]
    );
}

static void stride1_bwd_dx_tile_c8_k7(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    static constexpr int K = Stride1Specialist<7>::K;
    (void)C_in;
    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, K, pad, conv_out_w)) {
        return;
    }
    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];
    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);
    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);
    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * 8 * dy_plane];
    const int64_t ow0 = doc.ow + pad + dy_pad_l;
    int64_t kh_lo = 0, kh_hi = K;
    bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);
    for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
        const float* __restrict w_kh = w_cin + kh * K;
        stride1_bwd_dx_accum_cout4_k7(
            v_dx, ow0,
            dy_oh + 0 * dy_plane, dy_oh + 1 * dy_plane,
            dy_oh + 2 * dy_plane, dy_oh + 3 * dy_plane,
            w_kh + 0 * w_cout_stride, w_kh + 1 * w_cout_stride,
            w_kh + 2 * w_cout_stride, w_kh + 3 * w_cout_stride
        );
        stride1_bwd_dx_accum_cout4_k7(
            v_dx, ow0,
            dy_oh + 4 * dy_plane, dy_oh + 5 * dy_plane,
            dy_oh + 6 * dy_plane, dy_oh + 7 * dy_plane,
            w_kh + 4 * w_cout_stride, w_kh + 5 * w_cout_stride,
            w_kh + 6 * w_cout_stride, w_kh + 7 * w_cout_stride
        );
    }
    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

static void stride1_bwd_dx_tile_c16_k7(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    static constexpr int K = Stride1Specialist<7>::K;
    (void)C_in;
    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, K, pad, conv_out_w)) {
        return;
    }
    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];
    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);
    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);
    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * 16 * dy_plane];
    const int64_t ow0 = doc.ow + pad + dy_pad_l;
    int64_t kh_lo = 0, kh_hi = K;
    bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);
    for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
        const float* __restrict w_kh = w_cin + kh * K;
        for (int64_t g = 0; g < 16; g += 4) {
            stride1_bwd_dx_accum_cout4_k7(
                v_dx, ow0,
                dy_oh + (g + 0) * dy_plane, dy_oh + (g + 1) * dy_plane,
                dy_oh + (g + 2) * dy_plane, dy_oh + (g + 3) * dy_plane,
                w_kh + (g + 0) * w_cout_stride, w_kh + (g + 1) * w_cout_stride,
                w_kh + (g + 2) * w_cout_stride, w_kh + (g + 3) * w_cout_stride
            );
        }
    }
    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

void Stride1Specialist<7>::bwd_dx_tile(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t C_out, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    if (C_out == 8) {
        stride1_bwd_dx_tile_c8_k7(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
        return;
    }
    if (C_out == 16) {
        stride1_bwd_dx_tile_c16_k7(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
        return;
    }
    (void)C_in;
    if (bwd_dx_mock_edges_enabled() &&
        !bwd_dx_tile_is_interior(doc, K, pad, conv_out_w)) {
        return;
    }
    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];
    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);
    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);
    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * C_out * dy_plane];
    const int64_t ow0 = doc.ow + pad + dy_pad_l;
    int64_t kh_lo = 0, kh_hi = K;
    bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);
    for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
        const int64_t oh = doc.oh + pad - kh;
        const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
        const float* __restrict w_kh = w_cin + kh * K;
        int64_t cout = 0;
        for (; cout + 3 < C_out; cout += 4) {
            stride1_bwd_dx_accum_cout4_k7(
                v_dx, ow0,
                dy_oh + (cout + 0) * dy_plane,
                dy_oh + (cout + 1) * dy_plane,
                dy_oh + (cout + 2) * dy_plane,
                dy_oh + (cout + 3) * dy_plane,
                w_kh + (cout + 0) * w_cout_stride,
                w_kh + (cout + 1) * w_cout_stride,
                w_kh + (cout + 2) * w_cout_stride,
                w_kh + (cout + 3) * w_cout_stride
            );
        }
        for (; cout < C_out; ++cout) {
            const float* __restrict dy_row = dy_oh + cout * dy_plane;
            const float* __restrict wp = w_kh + cout * w_cout_stride;
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 0), wp[0]);
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 1), wp[1]);
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 2), wp[2]);
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 3), wp[3]);
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 4), wp[4]);
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 5), wp[5]);
            v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - 6), wp[6]);
        }
    }
    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

void Stride1Specialist<7>::dw_nci(
    int64_t n, int64_t cout, int64_t cin,
    float* __restrict dw_slice,
    const float* __restrict dy_pad_buf,
    const float* __restrict x_pad_buf,
    int64_t C_in, int64_t C_out,
    int64_t dy_pad_l, int64_t dy_row_stride,
    int64_t x_pad_l, int64_t x_row_stride,
    int64_t H, int64_t pad,
    int64_t conv_out_h, int64_t conv_out_w
) {
    (void)C_in;
    (void)C_out;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const int64_t x_plane  = H * x_row_stride;
    const float* __restrict dy_nc =
        &dy_pad_buf[(n * C_out + cout) * dy_plane];
    const float* __restrict x_nc =
        &x_pad_buf[(n * C_in + cin) * x_plane];
    const int64_t ow_tiles = (conv_out_w + FWD_TILE_OW - 1) / FWD_TILE_OW;
    for (int64_t kh = 0; kh < K; ++kh) {
        __m256 v_acc0 = _mm256_setzero_ps();
        __m256 v_acc1 = _mm256_setzero_ps();
        __m256 v_acc2 = _mm256_setzero_ps();
        __m256 v_acc3 = _mm256_setzero_ps();
        __m256 v_acc4 = _mm256_setzero_ps();
        __m256 v_acc5 = _mm256_setzero_ps();
        __m256 v_acc6 = _mm256_setzero_ps();
        for (int64_t oh = 0; oh < conv_out_h; ++oh) {
            const int64_t ih = oh - pad + kh;
            if (ih < 0 || ih >= H) continue;
            const float* __restrict dy_row = &dy_nc[oh * dy_row_stride];
            const float* __restrict x_row  = &x_nc[ih * x_row_stride];
            for (int64_t t = 0; t < ow_tiles; ++t) {
                const int64_t ow = t * FWD_TILE_OW;
                const __m256 dy8 = _mm256_loadu_ps(dy_row + dy_pad_l + ow);
                const int64_t iw0 = x_pad_l + ow - pad;
                v_acc0 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 0), v_acc0);
                v_acc1 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 1), v_acc1);
                v_acc2 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 2), v_acc2);
                v_acc3 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 3), v_acc3);
                v_acc4 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 4), v_acc4);
                v_acc5 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 5), v_acc5);
                v_acc6 = _mm256_fmadd_ps(dy8, _mm256_loadu_ps(x_row + iw0 + 6), v_acc6);
            }
        }
        const int64_t base = kh * K;
        dw_slice[base + 0] += _mm256_reduce_add_ps(v_acc0);
        dw_slice[base + 1] += _mm256_reduce_add_ps(v_acc1);
        dw_slice[base + 2] += _mm256_reduce_add_ps(v_acc2);
        dw_slice[base + 3] += _mm256_reduce_add_ps(v_acc3);
        dw_slice[base + 4] += _mm256_reduce_add_ps(v_acc4);
        dw_slice[base + 5] += _mm256_reduce_add_ps(v_acc5);
        dw_slice[base + 6] += _mm256_reduce_add_ps(v_acc6);
    }
}

// Spatial-first dW for one (n, cout, cin): accumulate over all oh x ow strips, then reduce.
static void process_dw_nci_stride1(
    int64_t n, int64_t cout, int64_t cin,
    float* __restrict dw_slice,
    const float* __restrict d_conv_buf,
    const float* __restrict dy_pad_buf,
    const float* __restrict x,
    const float* __restrict x_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    int64_t x_pad_l, int64_t x_row_stride,
    int64_t C_in, int64_t C_out, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t k_h, int64_t k_w, int64_t pad,
    int64_t spatial_in, int64_t conv_spatial,
    int64_t conv_out_h, int64_t conv_out_w, int64_t conv_out_w_stride
) {
    if (stride1_try_dw_specialist(
            k_h, k_w, n, cout, cin, dw_slice,
            dy_pad_buf, x_pad_buf,
            dy_pad_l, dy_row_stride, x_pad_l, x_row_stride,
            C_in, C_out, H, pad, conv_out_h, conv_out_w)) {
        return;
    }

    (void)x_pad_buf;
    (void)x_pad_l;
    (void)x_row_stride;

    const bool use_dy_pad = (dy_pad_buf != nullptr);
    const int64_t dy_plane = conv_out_h * (use_dy_pad ? dy_row_stride : conv_out_w_stride);
    const float* __restrict dy_nc =
        use_dy_pad ? &dy_pad_buf[(n * C_out + cout) * dy_plane]
                   : &d_conv_buf[(n * C_out + cout) * conv_spatial];
    const float* __restrict x_plane = &x[(n * C_in + cin) * spatial_in];

    BwdDwOwStripInfo strips[32];
    int n_strips = 0;
    bwd_dw_build_strip_info(
        strips, n_strips, conv_out_w, W_in, k_w, pad
    );

    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i v_cow = _mm256_set1_epi32((int)conv_out_w);

    for (int64_t kh = 0; kh < k_h; ++kh) {
        int64_t kw = 0;

        for (; kw + 3 < k_w; kw += 4) {
            __m256 v_acc0 = _mm256_setzero_ps();
            __m256 v_acc1 = _mm256_setzero_ps();
            __m256 v_acc2 = _mm256_setzero_ps();
            __m256 v_acc3 = _mm256_setzero_ps();

            for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                const int64_t ih = oh - pad + kh;
                if (ih < 0 || ih >= H) {
                    continue;
                }

                const float* __restrict dy_row =
                    &dy_nc[oh * (use_dy_pad ? dy_row_stride : conv_out_w_stride)];
                const float* __restrict x_row = &x_plane[ih * W_in_stride];

                for (int s = 0; s < n_strips; ++s) {
                    const BwdDwOwStripInfo& st = strips[s];
                    const __m256i lane_mask = bwd_dw_lane_mask(st.ow_count);

                    __m256 dy8;
                    if (use_dy_pad) {
                        dy8 = _mm256_loadu_ps(dy_row + dy_pad_l + st.ow);
                    } else if (st.strip_interior) {
                        dy8 = _mm256_loadu_ps(dy_row + st.ow);
                    } else {
                        dy8 = load_dy_window_8(
                            dy_row, st.ow, conv_out_w, st.full_ow, lane_mask, v_idx, v_cow
                        );
                    }

                    if (st.strip_interior) {
                        bwd_dw_fmadd_kw4(
                            dy8, x_row, st.ow, pad, kw, v_acc0, v_acc1, v_acc2, v_acc3
                        );
                    } else {
                        v_acc0 = _mm256_fmadd_ps(
                            dy8,
                            load_x_window_8(
                                x_row, st.ow - pad + kw + 0, W_in, st.full_ow, lane_mask, v_idx
                            ),
                            v_acc0
                        );
                        v_acc1 = _mm256_fmadd_ps(
                            dy8,
                            load_x_window_8(
                                x_row, st.ow - pad + kw + 1, W_in, st.full_ow, lane_mask, v_idx
                            ),
                            v_acc1
                        );
                        v_acc2 = _mm256_fmadd_ps(
                            dy8,
                            load_x_window_8(
                                x_row, st.ow - pad + kw + 2, W_in, st.full_ow, lane_mask, v_idx
                            ),
                            v_acc2
                        );
                        v_acc3 = _mm256_fmadd_ps(
                            dy8,
                            load_x_window_8(
                                x_row, st.ow - pad + kw + 3, W_in, st.full_ow, lane_mask, v_idx
                            ),
                            v_acc3
                        );
                    }
                }
            }

            const int64_t base = kh * k_w + kw;
            dw_slice[base + 0] += _mm256_reduce_add_ps(v_acc0);
            dw_slice[base + 1] += _mm256_reduce_add_ps(v_acc1);
            dw_slice[base + 2] += _mm256_reduce_add_ps(v_acc2);
            dw_slice[base + 3] += _mm256_reduce_add_ps(v_acc3);
        }

        for (; kw < k_w; ++kw) {
            __m256 v_acc0 = _mm256_setzero_ps();

            for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                const int64_t ih = oh - pad + kh;
                if (ih < 0 || ih >= H) {
                    continue;
                }

                const float* __restrict dy_row =
                    &dy_nc[oh * (use_dy_pad ? dy_row_stride : conv_out_w_stride)];
                const float* __restrict x_row = &x_plane[ih * W_in_stride];

                for (int s = 0; s < n_strips; ++s) {
                    const BwdDwOwStripInfo& st = strips[s];
                    const __m256i lane_mask = bwd_dw_lane_mask(st.ow_count);

                    __m256 dy8;
                    if (use_dy_pad) {
                        dy8 = _mm256_loadu_ps(dy_row + dy_pad_l + st.ow);
                    } else if (st.strip_interior) {
                        dy8 = _mm256_loadu_ps(dy_row + st.ow);
                    } else {
                        dy8 = load_dy_window_8(
                            dy_row, st.ow, conv_out_w, st.full_ow, lane_mask, v_idx, v_cow
                        );
                    }

                    __m256 x8;
                    if (st.strip_interior) {
                        x8 = _mm256_loadu_ps(x_row + st.ow - pad + kw);
                    } else {
                        x8 = load_x_window_8(
                            x_row, st.ow - pad + kw, W_in, st.full_ow, lane_mask, v_idx
                        );
                    }
                    v_acc0 = _mm256_fmadd_ps(dy8, x8, v_acc0);
                }
            }

            dw_slice[kh * k_w + kw] += _mm256_reduce_add_ps(v_acc0);
        }
    }
}

// Interleave dX tiles and dW (n,cout,cin) tasks in one queue (Bresenham-style).
static inline void decode_stream_work_item(
    int64_t wid, int64_t dx_count, int64_t dw_count,
    bool& is_dx, int64_t& local_id
) {
    if (dw_count == 0) {
        is_dx = true;
        local_id = wid;
        return;
    }
    if (dx_count == 0) {
        is_dx = false;
        local_id = wid;
        return;
    }

    const int64_t total = dx_count + dw_count;
    const int64_t dx_upto      = ((wid + 1) * dx_count) / total;
    const int64_t dx_upto_prev = (wid * dx_count) / total;
    is_dx = (dx_upto > dx_upto_prev);
    local_id = is_dx ? (dx_upto - 1) : (wid - dx_upto);
}

static void process_dw_nci_task(
    int64_t n, int64_t cout, int64_t cin,
    float* __restrict dw_slice,
    const float* __restrict d_conv_buf,
    const float* __restrict dy_pad_buf,
    const float* __restrict x,
    const float* __restrict x_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    int64_t x_pad_l, int64_t x_row_stride,
    int64_t C_in, int64_t C_out, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t k_h, int64_t k_w, int64_t stride, int64_t pad,
    int64_t spatial_in, int64_t conv_spatial, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w, int64_t conv_out_w_stride
) {
    (void)k_spatial;
    if (stride == 1) {
        process_dw_nci_stride1(
            n, cout, cin, dw_slice,
            d_conv_buf, dy_pad_buf, x, x_pad_buf,
            dy_pad_l, dy_row_stride,
            x_pad_l, x_row_stride,
            C_in, C_out, H, W_in, W_in_stride,
            k_h, k_w, pad,
            spatial_in, conv_spatial,
            conv_out_h, conv_out_w, conv_out_w_stride
        );
        return;
    }

    (void)C_in;
    (void)C_out;
    (void)dy_pad_buf;
    (void)dy_pad_l;
    (void)dy_row_stride;
    (void)x_pad_buf;
    (void)x_pad_l;
    (void)x_row_stride;

    const float* __restrict xp_n = &x[(n * C_in + cin) * spatial_in];

    for (int64_t kh = 0; kh < k_h; ++kh) {
        for (int64_t kw = 0; kw < k_w; ++kw) {
            const int64_t tap_idx = kh * k_w + kw;
            float tap_sum = 0.0f;
            const int64_t iw_base = -pad + kw;

            for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                const int64_t ih = oh * stride - pad + kh;
                if (ih < 0 || ih >= H) continue;

                const float* __restrict dr_row = &d_conv_buf[(n * C_out + cout) * conv_spatial
                    + oh * conv_out_w_stride];
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

// ========================================================================
// Stride-2 k=6 specialists (isolated; stride-1 and pad=1/2 paths unchanged)
// ========================================================================
static float* acquire_bwd_x_pad_buf(size_t need_floats);
static float* acquire_bwd_dy_pad_buf(size_t need_floats);

static constexpr int64_t STRIDE2_CONV = 2;

static inline int64_t stride2_specialist_k(int64_t k_h, int64_t k_w) {
    if (k_h != k_w) {
        return 0;
    }
    switch (k_h) {
        case 1: return 1;
        case 2: return 2;
        case 3: return 3;
        case 4: return 4;
        case 5: return 5;
        case 6: return 6;
        case 7: return 7;
        default: return 0;
    }
}

static inline int64_t stride2_x_row_stride(
    int64_t k_w, int64_t pad, int64_t W_in, int64_t conv_out_w
) {
    const int64_t pad_l = bwd_x_pad_l(pad);
    const int64_t max_iw = (conv_out_w - 1) * STRIDE2_CONV - pad + (k_w - 1);
    const int64_t need = pad_l + max_iw + 1;
    const int64_t min_stride = pad_l + W_in;
    return need > min_stride ? need : min_stride;
}

static inline __m256 gather_stride2_x8(const float* __restrict row, int64_t iw_base) {
    const __m256i lane_off = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i idx = _mm256_add_epi32(_mm256_set1_epi32((int)iw_base), lane_off);
    return _mm256_i32gather_ps(row, idx, 4);
}

static inline __m256 stride2_gather_x8_loadu(const float* __restrict row, int64_t iw) {
    const __m128 a0 = _mm_loadu_ps(row + iw);
    const __m128 a1 = _mm_loadu_ps(row + iw + 4);
    const __m128 b0 = _mm_loadu_ps(row + iw + 8);
    const __m128 b1 = _mm_loadu_ps(row + iw + 12);
    return _mm256_set_m128(
        _mm_shuffle_ps(b0, b1, _MM_SHUFFLE(2, 0, 2, 0)),
        _mm_shuffle_ps(a0, a1, _MM_SHUFFLE(2, 0, 2, 0))
    );
}

static inline __m256 stride2_spread_dy_even_ps(__m128 d) {
    const __m256 wide = _mm256_insertf128_ps(_mm256_setzero_ps(), d, 0);
    const __m256 dup = _mm256_moveldup_ps(wide);
    const __m256 mask = _mm256_castsi256_ps(_mm256_setr_epi64x(-1, 0, -1, 0));
    return _mm256_and_ps(dup, mask);
}

static inline __m256 stride2_spread_dy_odd_ps(__m128 d) {
    const __m256 ev = stride2_spread_dy_even_ps(d);
    return _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(ev), 4));
}

static inline __m256 stride2_spread_dy_even(const float* __restrict dy_row, int64_t ow0) {
    return stride2_spread_dy_even_ps(_mm_loadu_ps(dy_row + ow0));
}

static inline __m256 stride2_spread_dy_odd(const float* __restrict dy_row, int64_t ow0) {
    return stride2_spread_dy_odd_ps(_mm_loadu_ps(dy_row + ow0));
}

template<int K>
static inline void stride2_bwd_dx_k_kh_bounds(
    int64_t ih, int64_t pad, int64_t conv_out_h, int64_t& kh_lo, int64_t& kh_hi
) {
    bwd_dx_k_kh_bounds<K>(ih, pad, conv_out_h, kh_lo, kh_hi);
    const int64_t want_parity = (ih + pad) & 1;
    if ((kh_lo & 1) != want_parity) {
        ++kh_lo;
    }
    if (kh_hi > kh_lo && ((kh_hi - 1) & 1) != want_parity) {
        --kh_hi;
    }
}

template<int K>
static inline bool stride2_bwd_dx_tile_is_interior(
    const ConvBwdDxTileDoc& doc, int64_t pad, int64_t conv_out_w
) {
    if (doc.ow_count != FWD_TILE_OW) {
        return false;
    }
    for (int64_t kw = 0; kw < K; ++kw) {
        const int64_t t = doc.ow + pad - kw;
        if (t < 0) {
            return false;
        }
        if (!(t & 1)) {
            if (((t >> 1) + 4) > conv_out_w) {
                return false;
            }
        } else if ((((t + 1) >> 1) + 4) > conv_out_w) {
            return false;
        }
    }
    return true;
}

static inline void stride2_dw_nci_accum_tile_k5(
    __m256 dy8,
    const float* __restrict x_row,
    int64_t iw_base,
    __m256& acc0, __m256& acc1, __m256& acc2,
    __m256& acc3, __m256& acc4
) {
    const __m128 p0 = _mm_loadu_ps(x_row + iw_base + 0);
    const __m128 p1 = _mm_loadu_ps(x_row + iw_base + 4);
    const __m128 p2 = _mm_loadu_ps(x_row + iw_base + 8);
    const __m128 p3 = _mm_loadu_ps(x_row + iw_base + 12);
    const __m128 p4 = _mm_loadu_ps(x_row + iw_base + 16);
    const int sh_e = _MM_SHUFFLE(2, 0, 2, 0);
    const int sh_o = _MM_SHUFFLE(3, 1, 3, 1);

    acc0 = _mm256_fmadd_ps(
        dy8, _mm256_set_m128(_mm_shuffle_ps(p2, p3, sh_e), _mm_shuffle_ps(p0, p1, sh_e)), acc0
    );
    acc1 = _mm256_fmadd_ps(
        dy8, _mm256_set_m128(_mm_shuffle_ps(p2, p3, sh_o), _mm_shuffle_ps(p0, p1, sh_o)), acc1
    );
    acc2 = _mm256_fmadd_ps(
        dy8, _mm256_set_m128(_mm_shuffle_ps(p3, p4, sh_e), _mm_shuffle_ps(p1, p2, sh_e)), acc2
    );
    acc3 = _mm256_fmadd_ps(
        dy8, _mm256_set_m128(_mm_shuffle_ps(p3, p4, sh_o), _mm_shuffle_ps(p1, p2, sh_o)), acc3
    );
    acc4 = _mm256_fmadd_ps(
        dy8, _mm256_set_m128(_mm_shuffle_ps(p3, p4, sh_e), _mm_shuffle_ps(p1, p2, sh_e)), acc4
    );
}

static inline void stride2_dw_nci_accum_tile_k6(
    __m256 dy8,
    const float* __restrict x_row,
    int64_t iw_base,
    __m256& acc0, __m256& acc1, __m256& acc2,
    __m256& acc3, __m256& acc4, __m256& acc5
) {
    const __m128 p0 = _mm_loadu_ps(x_row + iw_base + 0);
    const __m128 p1 = _mm_loadu_ps(x_row + iw_base + 4);
    const __m128 p2 = _mm_loadu_ps(x_row + iw_base + 8);
    const __m128 p3 = _mm_loadu_ps(x_row + iw_base + 12);
    const __m128 p4 = _mm_loadu_ps(x_row + iw_base + 16);
    const __m128 p5 = _mm_loadu_ps(x_row + iw_base + 20);
    const int sh_e = _MM_SHUFFLE(2, 0, 2, 0);
    const int sh_o = _MM_SHUFFLE(3, 1, 3, 1);

    acc0 = _mm256_fmadd_ps(
        dy8, _mm256_set_m128(_mm_shuffle_ps(p2, p3, sh_e), _mm_shuffle_ps(p0, p1, sh_e)), acc0
    );
    acc1 = _mm256_fmadd_ps(
        dy8, _mm256_set_m128(_mm_shuffle_ps(p2, p3, sh_o), _mm_shuffle_ps(p0, p1, sh_o)), acc1
    );
    acc2 = _mm256_fmadd_ps(
        dy8, _mm256_set_m128(_mm_shuffle_ps(p3, p4, sh_e), _mm_shuffle_ps(p1, p2, sh_e)), acc2
    );
    acc3 = _mm256_fmadd_ps(
        dy8, _mm256_set_m128(_mm_shuffle_ps(p3, p4, sh_o), _mm_shuffle_ps(p1, p2, sh_o)), acc3
    );
    acc4 = _mm256_fmadd_ps(
        dy8, _mm256_set_m128(_mm_shuffle_ps(p4, p5, sh_e), _mm_shuffle_ps(p2, p3, sh_e)), acc4
    );
    acc5 = _mm256_fmadd_ps(
        dy8, _mm256_set_m128(_mm_shuffle_ps(p4, p5, sh_o), _mm_shuffle_ps(p2, p3, sh_o)), acc5
    );
}

static inline void stride2_dw_nci_fmadd_all_kw(
    __m256 dy8,
    const float* __restrict x_row,
    int64_t iw_base,
    int64_t K,
    __m256& acc0, __m256& acc1, __m256& acc2,
    __m256& acc3, __m256& acc4, __m256& acc5,
    __m256& acc6
) {
    if (K > 0) {
        acc0 = _mm256_fmadd_ps(dy8, stride2_gather_x8_loadu(x_row, iw_base + 0), acc0);
    }
    if (K > 1) {
        acc1 = _mm256_fmadd_ps(dy8, stride2_gather_x8_loadu(x_row, iw_base + 1), acc1);
    }
    if (K > 2) {
        acc2 = _mm256_fmadd_ps(dy8, stride2_gather_x8_loadu(x_row, iw_base + 2), acc2);
    }
    if (K > 3) {
        acc3 = _mm256_fmadd_ps(dy8, stride2_gather_x8_loadu(x_row, iw_base + 3), acc3);
    }
    if (K > 4) {
        acc4 = _mm256_fmadd_ps(dy8, stride2_gather_x8_loadu(x_row, iw_base + 4), acc4);
    }
    if (K > 5) {
        acc5 = _mm256_fmadd_ps(dy8, stride2_gather_x8_loadu(x_row, iw_base + 5), acc5);
    }
    if (K > 6) {
        acc6 = _mm256_fmadd_ps(dy8, stride2_gather_x8_loadu(x_row, iw_base + 6), acc6);
    }
}

template<int K>
static inline bool stride2_dw_tile_interior(
    int64_t ow, int64_t conv_out_w, int64_t iw_base, int64_t x_row_stride
) {
    if (ow + FWD_TILE_OW > conv_out_w) {
        return false;
    }
    if (iw_base < 0) {
        return false;
    }
    return (iw_base + K + 14) < x_row_stride;
}

static inline __m256 stride2_gather_dy_masked(
    const float* __restrict dy_row,
    int64_t dy_pad_l,
    int64_t doc_ow,
    int64_t pad,
    int64_t kw,
    int64_t conv_out_w
) {
    const __m256i v_lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i t = _mm256_add_epi32(
        _mm256_add_epi32(_mm256_set1_epi32((int)doc_ow), v_lane),
        _mm256_set1_epi32((int)(pad - kw))
    );
    const __m256i ge0 = _mm256_cmpgt_epi32(t, _mm256_set1_epi32(-1));
    const __m256i even = _mm256_cmpeq_epi32(
        _mm256_and_si256(t, _mm256_set1_epi32(1)),
        _mm256_setzero_si256()
    );
    const __m256i ow_i = _mm256_srli_epi32(t, 1);
    const __m256i lt_out = _mm256_cmpgt_epi32(_mm256_set1_epi32((int)conv_out_w), ow_i);
    const __m256i valid = _mm256_and_si256(_mm256_and_si256(ge0, even), lt_out);
    const __m256i safe_ow = _mm256_min_epi32(
        _mm256_max_epi32(ow_i, _mm256_setzero_si256()),
        _mm256_set1_epi32((int)(conv_out_w - 1))
    );
    const __m256i gidx = _mm256_add_epi32(safe_ow, _mm256_set1_epi32((int)dy_pad_l));
    __m256 dy_g = _mm256_i32gather_ps(dy_row, gidx, 4);
    const __m256i vm = _mm256_slli_epi32(valid, 31);
    return _mm256_blendv_ps(_mm256_setzero_ps(), dy_g, _mm256_castsi256_ps(vm));
}

static __forceinline void stride2_bwd_dx_accum_cout4(
    __m256& v_dx,
    __m256 dy0_g, __m256 dy1_g, __m256 dy2_g, __m256 dy3_g,
    float w0, float w1, float w2, float w3
) {
    v_dx = fmadd_dx_cout4(v_dx, dy0_g, dy1_g, dy2_g, dy3_g, w0, w1, w2, w3);
}

static __forceinline void stride2_bwd_dx_accum_cout4_kw(
    __m256& v_dx,
    int64_t doc_ow, int64_t pad, int64_t dy_pad_l, int64_t kw,
    int64_t conv_out_w,
    const float* __restrict dy0,
    const float* __restrict dy1,
    const float* __restrict dy2,
    const float* __restrict dy3,
    const float* __restrict wp0,
    const float* __restrict wp1,
    const float* __restrict wp2,
    const float* __restrict wp3
) {
    stride2_bwd_dx_accum_cout4(
        v_dx,
        stride2_gather_dy_masked(dy0, dy_pad_l, doc_ow, pad, kw, conv_out_w),
        stride2_gather_dy_masked(dy1, dy_pad_l, doc_ow, pad, kw, conv_out_w),
        stride2_gather_dy_masked(dy2, dy_pad_l, doc_ow, pad, kw, conv_out_w),
        stride2_gather_dy_masked(dy3, dy_pad_l, doc_ow, pad, kw, conv_out_w),
        wp0[kw], wp1[kw], wp2[kw], wp3[kw]
    );
}

template<int K>
static __forceinline void stride2_bwd_dx_accum_cout4_interior(
    __m256& v_dx,
    int64_t doc_ow, int64_t pad, int64_t dy_pad_l, int64_t conv_out_w,
    const float* __restrict dy0,
    const float* __restrict dy1,
    const float* __restrict dy2,
    const float* __restrict dy3,
    const float* __restrict wp0,
    const float* __restrict wp1,
    const float* __restrict wp2,
    const float* __restrict wp3
) {
    for (int64_t kw = 0; kw < K; ++kw) {
        stride2_bwd_dx_accum_cout4_kw(
            v_dx, doc_ow, pad, dy_pad_l, kw, conv_out_w,
            dy0, dy1, dy2, dy3, wp0, wp1, wp2, wp3
        );
    }
}

template<int K>
static void stride2_bwd_dx_tile_c8(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    (void)C_in;

    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];

    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);

    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);

    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * 8 * dy_plane];

    if (stride2_bwd_dx_tile_is_interior<K>(doc, pad, conv_out_w)) {
        int64_t kh_lo = 0;
        int64_t kh_hi = K;
        stride2_bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);

        for (int64_t kh = kh_lo; kh < kh_hi; kh += 2) {
            const int64_t oh = (doc.oh + pad - kh) >> 1;
            const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
            const float* __restrict w_kh = w_cin + kh * K;

            stride2_bwd_dx_accum_cout4_interior<K>(
                v_dx, doc.ow, pad, dy_pad_l, conv_out_w,
                dy_oh + 0 * dy_plane, dy_oh + 1 * dy_plane,
                dy_oh + 2 * dy_plane, dy_oh + 3 * dy_plane,
                w_kh + 0 * w_cout_stride, w_kh + 1 * w_cout_stride,
                w_kh + 2 * w_cout_stride, w_kh + 3 * w_cout_stride
            );
            stride2_bwd_dx_accum_cout4_interior<K>(
                v_dx, doc.ow, pad, dy_pad_l, conv_out_w,
                dy_oh + 4 * dy_plane, dy_oh + 5 * dy_plane,
                dy_oh + 6 * dy_plane, dy_oh + 7 * dy_plane,
                w_kh + 4 * w_cout_stride, w_kh + 5 * w_cout_stride,
                w_kh + 6 * w_cout_stride, w_kh + 7 * w_cout_stride
            );
        }
    } else {
        for (int64_t kh = 0; kh < K; ++kh) {
            const int64_t oh_rem = doc.oh + pad - kh;
            if (oh_rem < 0 || (oh_rem & 1)) {
                continue;
            }
            const int64_t oh = oh_rem >> 1;
            if (oh >= conv_out_h) {
                continue;
            }

            const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
            const float* __restrict w_kh = w_cin + kh * K;

            for (int64_t kw = 0; kw < K; ++kw) {
                const float* __restrict wp0 = w_kh + 0 * w_cout_stride;
                const float* __restrict wp1 = w_kh + 1 * w_cout_stride;
                const float* __restrict wp2 = w_kh + 2 * w_cout_stride;
                const float* __restrict wp3 = w_kh + 3 * w_cout_stride;
                const float* __restrict wp4 = w_kh + 4 * w_cout_stride;
                const float* __restrict wp5 = w_kh + 5 * w_cout_stride;
                const float* __restrict wp6 = w_kh + 6 * w_cout_stride;
                const float* __restrict wp7 = w_kh + 7 * w_cout_stride;

                stride2_bwd_dx_accum_cout4(
                    v_dx,
                    stride2_gather_dy_masked(
                        dy_oh + 0 * dy_plane, dy_pad_l, doc.ow, pad, kw, conv_out_w
                    ),
                    stride2_gather_dy_masked(
                        dy_oh + 1 * dy_plane, dy_pad_l, doc.ow, pad, kw, conv_out_w
                    ),
                    stride2_gather_dy_masked(
                        dy_oh + 2 * dy_plane, dy_pad_l, doc.ow, pad, kw, conv_out_w
                    ),
                    stride2_gather_dy_masked(
                        dy_oh + 3 * dy_plane, dy_pad_l, doc.ow, pad, kw, conv_out_w
                    ),
                    wp0[kw], wp1[kw], wp2[kw], wp3[kw]
                );
                stride2_bwd_dx_accum_cout4(
                    v_dx,
                    stride2_gather_dy_masked(
                        dy_oh + 4 * dy_plane, dy_pad_l, doc.ow, pad, kw, conv_out_w
                    ),
                    stride2_gather_dy_masked(
                        dy_oh + 5 * dy_plane, dy_pad_l, doc.ow, pad, kw, conv_out_w
                    ),
                    stride2_gather_dy_masked(
                        dy_oh + 6 * dy_plane, dy_pad_l, doc.ow, pad, kw, conv_out_w
                    ),
                    stride2_gather_dy_masked(
                        dy_oh + 7 * dy_plane, dy_pad_l, doc.ow, pad, kw, conv_out_w
                    ),
                    wp4[kw], wp5[kw], wp6[kw], wp7[kw]
                );
            }
        }
    }

    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

template<int K>
static void stride2_bwd_dx_tile_c16(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    (void)C_in;

    float* __restrict dx_row =
        &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];

    const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
    const bool full_dx = (doc.ow_count == FWD_TILE_OW);

    __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);

    const float* __restrict w_cin = &W[doc.cin * k_spatial];
    const int64_t w_cout_stride = C_in * k_spatial;
    const int64_t dy_plane = conv_out_h * dy_row_stride;
    const float* __restrict dy_n = &dy_pad_buf[doc.n * 16 * dy_plane];

    if (stride2_bwd_dx_tile_is_interior<K>(doc, pad, conv_out_w)) {
        int64_t kh_lo = 0;
        int64_t kh_hi = K;
        stride2_bwd_dx_k_kh_bounds<K>(doc.oh, pad, conv_out_h, kh_lo, kh_hi);

        for (int64_t kh = kh_lo; kh < kh_hi; kh += 2) {
            const int64_t oh = (doc.oh + pad - kh) >> 1;
            const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
            const float* __restrict w_kh = w_cin + kh * K;

            for (int64_t cg = 0; cg < 16; cg += 4) {
                stride2_bwd_dx_accum_cout4_interior<K>(
                    v_dx, doc.ow, pad, dy_pad_l, conv_out_w,
                    dy_oh + (cg + 0) * dy_plane, dy_oh + (cg + 1) * dy_plane,
                    dy_oh + (cg + 2) * dy_plane, dy_oh + (cg + 3) * dy_plane,
                    w_kh + (cg + 0) * w_cout_stride, w_kh + (cg + 1) * w_cout_stride,
                    w_kh + (cg + 2) * w_cout_stride, w_kh + (cg + 3) * w_cout_stride
                );
            }
        }
    } else {
        for (int64_t kh = 0; kh < K; ++kh) {
            const int64_t oh_rem = doc.oh + pad - kh;
            if (oh_rem < 0 || (oh_rem & 1)) {
                continue;
            }
            const int64_t oh = oh_rem >> 1;
            if (oh >= conv_out_h) {
                continue;
            }

            const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
            const float* __restrict w_kh = w_cin + kh * K;

            for (int64_t kw = 0; kw < K; ++kw) {
                for (int64_t cg = 0; cg < 16; cg += 4) {
                    const float* __restrict wp0 = w_kh + (cg + 0) * w_cout_stride;
                    const float* __restrict wp1 = w_kh + (cg + 1) * w_cout_stride;
                    const float* __restrict wp2 = w_kh + (cg + 2) * w_cout_stride;
                    const float* __restrict wp3 = w_kh + (cg + 3) * w_cout_stride;

                    stride2_bwd_dx_accum_cout4(
                        v_dx,
                        stride2_gather_dy_masked(
                            dy_oh + (cg + 0) * dy_plane, dy_pad_l, doc.ow, pad, kw, conv_out_w
                        ),
                        stride2_gather_dy_masked(
                            dy_oh + (cg + 1) * dy_plane, dy_pad_l, doc.ow, pad, kw, conv_out_w
                        ),
                        stride2_gather_dy_masked(
                            dy_oh + (cg + 2) * dy_plane, dy_pad_l, doc.ow, pad, kw, conv_out_w
                        ),
                        stride2_gather_dy_masked(
                            dy_oh + (cg + 3) * dy_plane, dy_pad_l, doc.ow, pad, kw, conv_out_w
                        ),
                        wp0[kw], wp1[kw], wp2[kw], wp3[kw]
                    );
                }
            }
        }
    }

    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

template<int KernelK>
struct Stride2Specialist {
    static constexpr int K = KernelK;

    static void fwd_tile(
        const ConvFwdTileDoc& doc,
        const float* __restrict x_pad_buf,
        int64_t x_pad_l, int64_t x_row_stride,
        const float* __restrict W,
        float* __restrict out,
        int64_t C_in, int64_t C_out, int64_t H,
        int64_t pad, int64_t k_spatial, int64_t spatial_out, int64_t out_w_stride
    ) {
        const int64_t ih_base = doc.oh * STRIDE2_CONV - pad;
        const int64_t c_rem   = doc.cout_count;
        const int64_t x_plane = H * x_row_stride;

        float* __restrict out_r0 = &out[(doc.n * C_out + doc.cout0 + 0) * spatial_out + doc.oh * out_w_stride + doc.ow];
        float* __restrict out_r1 = (c_rem > 1) ? &out[(doc.n * C_out + doc.cout0 + 1) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;
        float* __restrict out_r2 = (c_rem > 2) ? &out[(doc.n * C_out + doc.cout0 + 2) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;
        float* __restrict out_r3 = (c_rem > 3) ? &out[(doc.n * C_out + doc.cout0 + 3) * spatial_out + doc.oh * out_w_stride + doc.ow] : nullptr;

        __m256 vo0 = _mm256_loadu_ps(out_r0);
        __m256 vo1 = (c_rem > 1) ? _mm256_loadu_ps(out_r1) : _mm256_setzero_ps();
        __m256 vo2 = (c_rem > 2) ? _mm256_loadu_ps(out_r2) : _mm256_setzero_ps();
        __m256 vo3 = (c_rem > 3) ? _mm256_loadu_ps(out_r3) : _mm256_setzero_ps();

        const bool full_ow = (doc.ow_count == FWD_TILE_OW);
        const __m256i out_mask = bwd_dw_lane_mask(doc.ow_count);

        const float* __restrict xp_base = &x_pad_buf[doc.n * C_in * x_plane];
        const int64_t iw_base0 = x_pad_l + doc.ow * STRIDE2_CONV - pad;
        const bool x_interior = full_ow && (iw_base0 >= 0) && ((iw_base0 + K + 13) < x_row_stride);

        for (int64_t cin = 0; cin < C_in; ++cin) {
            const float* __restrict xp  = xp_base + cin * x_plane;
            const float* __restrict wp0 = &W[((doc.cout0 + 0) * C_in + cin) * k_spatial];
            const float* __restrict wp1 = (c_rem > 1) ? &W[((doc.cout0 + 1) * C_in + cin) * k_spatial] : nullptr;
            const float* __restrict wp2 = (c_rem > 2) ? &W[((doc.cout0 + 2) * C_in + cin) * k_spatial] : nullptr;
            const float* __restrict wp3 = (c_rem > 3) ? &W[((doc.cout0 + 3) * C_in + cin) * k_spatial] : nullptr;

            for (int64_t kh = 0; kh < K; ++kh) {
                const int64_t ih = ih_base + kh;
                if (ih < 0 || ih >= H) {
                    continue;
                }

                const float* __restrict in_row = xp + ih * x_row_stride;
                const float* __restrict w0 = wp0 + kh * K;
                const float* __restrict w1 = (c_rem > 1) ? wp1 + kh * K : nullptr;
                const float* __restrict w2 = (c_rem > 2) ? wp2 + kh * K : nullptr;
                const float* __restrict w3 = (c_rem > 3) ? wp3 + kh * K : nullptr;

                for (int64_t kw = 0; kw < K; ++kw) {
                    const __m256 vx = x_interior
                        ? stride2_gather_x8_loadu(in_row, iw_base0 + kw)
                        : gather_stride2_x8(in_row, iw_base0 + kw);
                    const __m256 vw0 = _mm256_set1_ps(w0[kw]);
                    vo0 = _mm256_fmadd_ps(vx, vw0, vo0);
                    if (c_rem > 1) vo1 = _mm256_fmadd_ps(vx, _mm256_set1_ps(w1[kw]), vo1);
                    if (c_rem > 2) vo2 = _mm256_fmadd_ps(vx, _mm256_set1_ps(w2[kw]), vo2);
                    if (c_rem > 3) vo3 = _mm256_fmadd_ps(vx, _mm256_set1_ps(w3[kw]), vo3);
                }
            }
        }

        if (full_ow) {
            _mm256_storeu_ps(out_r0, vo0);
            if (c_rem > 1) _mm256_storeu_ps(out_r1, vo1);
            if (c_rem > 2) _mm256_storeu_ps(out_r2, vo2);
            if (c_rem > 3) _mm256_storeu_ps(out_r3, vo3);
        } else {
            _mm256_maskstore_ps(out_r0, out_mask, vo0);
            if (c_rem > 1) _mm256_maskstore_ps(out_r1, out_mask, vo1);
            if (c_rem > 2) _mm256_maskstore_ps(out_r2, out_mask, vo2);
            if (c_rem > 3) _mm256_maskstore_ps(out_r3, out_mask, vo3);
        }
    }

    static void bwd_dx_tile(
        const ConvBwdDxTileDoc& doc,
        const float* __restrict dy_pad_buf,
        int64_t dy_pad_l, int64_t dy_row_stride,
        const float* __restrict W,
        float* __restrict dx,
        int64_t C_in, int64_t C_out, int64_t W_in_stride,
        int64_t pad, int64_t spatial_in, int64_t k_spatial,
        int64_t conv_out_h, int64_t conv_out_w
    ) {
        if (C_out == 8) {
            stride2_bwd_dx_tile_c8<K>(
                doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
                C_in, W_in_stride, pad, spatial_in, k_spatial,
                conv_out_h, conv_out_w
            );
            return;
        }
        if (C_out == 16) {
            stride2_bwd_dx_tile_c16<K>(
                doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
                C_in, W_in_stride, pad, spatial_in, k_spatial,
                conv_out_h, conv_out_w
            );
            return;
        }

        float* __restrict dx_row =
            &dx[(doc.n * C_in + doc.cin) * spatial_in + doc.oh * W_in_stride + doc.ow];

        const __m256i v_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        const __m256i dx_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(doc.ow_count), v_idx);
        const bool full_dx = (doc.ow_count == FWD_TILE_OW);

        __m256 v_dx = full_dx ? _mm256_loadu_ps(dx_row) : _mm256_maskload_ps(dx_row, dx_mask);

        const float* __restrict w_cin = &W[doc.cin * k_spatial];
        const int64_t w_cout_stride = C_in * k_spatial;
        const int64_t dy_plane = conv_out_h * dy_row_stride;
        const float* __restrict dy_n = &dy_pad_buf[doc.n * C_out * dy_plane];

        for (int64_t kh = 0; kh < K; ++kh) {
            const int64_t oh_rem = doc.oh + pad - kh;
            if (oh_rem < 0 || (oh_rem & 1)) {
                continue;
            }
            const int64_t oh = oh_rem >> 1;
            if (oh >= conv_out_h) {
                continue;
            }

            const float* __restrict dy_oh = dy_n + oh * dy_row_stride;
            const float* __restrict w_kh = w_cin + kh * K;

            for (int64_t kw = 0; kw < K; ++kw) {
                int64_t cout = 0;
                for (; cout + 3 < C_out; cout += 4) {
                    const float* __restrict wp0 = w_kh + (cout + 0) * w_cout_stride;
                    const float* __restrict wp1 = w_kh + (cout + 1) * w_cout_stride;
                    const float* __restrict wp2 = w_kh + (cout + 2) * w_cout_stride;
                    const float* __restrict wp3 = w_kh + (cout + 3) * w_cout_stride;

                    stride2_bwd_dx_accum_cout4(
                        v_dx,
                        stride2_gather_dy_masked(
                            dy_oh + (cout + 0) * dy_plane, dy_pad_l, doc.ow, pad, kw, conv_out_w
                        ),
                        stride2_gather_dy_masked(
                            dy_oh + (cout + 1) * dy_plane, dy_pad_l, doc.ow, pad, kw, conv_out_w
                        ),
                        stride2_gather_dy_masked(
                            dy_oh + (cout + 2) * dy_plane, dy_pad_l, doc.ow, pad, kw, conv_out_w
                        ),
                        stride2_gather_dy_masked(
                            dy_oh + (cout + 3) * dy_plane, dy_pad_l, doc.ow, pad, kw, conv_out_w
                        ),
                        wp0[kw], wp1[kw], wp2[kw], wp3[kw]
                    );
                }
                for (; cout < C_out; ++cout) {
                    const float* __restrict dy_row = dy_oh + cout * dy_plane;
                    const float w = w_kh[cout * w_cout_stride + kw];
                    const __m256 dy_g = stride2_gather_dy_masked(
                        dy_row, dy_pad_l, doc.ow, pad, kw, conv_out_w
                    );
                    v_dx = _mm256_fmadd_ps(dy_g, _mm256_set1_ps(w), v_dx);
                }
            }
        }

        bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
    }

    static void dw_nci(
        int64_t n, int64_t cout, int64_t cin,
        float* __restrict dw_slice,
        const float* __restrict dy_pad_buf,
        const float* __restrict x_pad_buf,
        int64_t C_in, int64_t C_out,
        int64_t dy_pad_l, int64_t dy_row_stride,
        int64_t x_pad_l, int64_t x_row_stride,
        int64_t H, int64_t pad,
        int64_t conv_out_h, int64_t conv_out_w
    ) {
        (void)C_in;
        (void)C_out;

        const int64_t dy_plane = conv_out_h * dy_row_stride;
        const int64_t x_plane  = H * x_row_stride;
        const float* __restrict dy_nc =
            &dy_pad_buf[(n * C_out + cout) * dy_plane];
        const float* __restrict x_nc =
            &x_pad_buf[(n * C_in + cin) * x_plane];

        const int64_t ow_tiles = (conv_out_w + FWD_TILE_OW - 1) / FWD_TILE_OW;

        for (int64_t kh = 0; kh < K; ++kh) {
            __m256 v_acc0 = _mm256_setzero_ps();
            __m256 v_acc1 = _mm256_setzero_ps();
            __m256 v_acc2 = _mm256_setzero_ps();
            __m256 v_acc3 = _mm256_setzero_ps();
            __m256 v_acc4 = _mm256_setzero_ps();
            __m256 v_acc5 = _mm256_setzero_ps();
            __m256 v_acc6 = _mm256_setzero_ps();

            for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                const int64_t ih = oh * STRIDE2_CONV - pad + kh;
                if (ih < 0 || ih >= H) {
                    continue;
                }

                const float* __restrict dy_row = &dy_nc[oh * dy_row_stride];
                const float* __restrict x_row  = &x_nc[ih * x_row_stride];

                for (int64_t t = 0; t < ow_tiles; ++t) {
                    const int64_t ow = t * FWD_TILE_OW;
                    const __m256 dy8 = _mm256_loadu_ps(dy_row + dy_pad_l + ow);
                    const int64_t iw_base = x_pad_l + ow * STRIDE2_CONV - pad;

                    stride2_dw_nci_fmadd_all_kw(
                        dy8, x_row, iw_base, K,
                        v_acc0, v_acc1, v_acc2, v_acc3, v_acc4, v_acc5, v_acc6
                    );
                }
            }

            const int64_t base = kh * K;
            dw_slice[base + 0] += _mm256_reduce_add_ps(v_acc0);
            if (K > 1) dw_slice[base + 1] += _mm256_reduce_add_ps(v_acc1);
            if (K > 2) dw_slice[base + 2] += _mm256_reduce_add_ps(v_acc2);
            if (K > 3) dw_slice[base + 3] += _mm256_reduce_add_ps(v_acc3);
            if (K > 4) dw_slice[base + 4] += _mm256_reduce_add_ps(v_acc4);
            if (K > 5) dw_slice[base + 5] += _mm256_reduce_add_ps(v_acc5);
            if (K > 6) dw_slice[base + 6] += _mm256_reduce_add_ps(v_acc6);
        }
    }
};

template<int KernelK>
static void process_bwd_dx_tile_stride2(
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t C_out, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    if (!dy_pad_buf) {
        return;
    }
    Stride2Specialist<KernelK>::bwd_dx_tile(
        doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
        C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
        conv_out_h, conv_out_w
    );
}

static void process_bwd_dx_tile_stride2_dispatch(
    int64_t specialist_k,
    const ConvBwdDxTileDoc& doc,
    const float* __restrict dy_pad_buf,
    int64_t dy_pad_l, int64_t dy_row_stride,
    const float* __restrict W,
    float* __restrict dx,
    int64_t C_in, int64_t C_out, int64_t W_in_stride,
    int64_t pad, int64_t spatial_in, int64_t k_spatial,
    int64_t conv_out_h, int64_t conv_out_w
) {
    if (specialist_k == 1) {
        process_bwd_dx_tile_stride2<1>(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
    } else if (specialist_k == 2) {
        process_bwd_dx_tile_stride2<2>(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
    } else if (specialist_k == 3) {
        process_bwd_dx_tile_stride2<3>(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
    } else if (specialist_k == 4) {
        process_bwd_dx_tile_stride2<4>(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
    } else if (specialist_k == 5) {
        process_bwd_dx_tile_stride2<5>(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
    } else if (specialist_k == 6) {
        process_bwd_dx_tile_stride2<6>(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
    } else if (specialist_k == 7) {
        process_bwd_dx_tile_stride2<7>(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
    }
}

template<int KernelK>
static void conv2d_forward_stride2_phase2(
    const float* x, const float* W, float* out,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t pad, int64_t out_h, int64_t out_w,
    int64_t spatial_in, int64_t spatial_out, int64_t k_spatial, int64_t out_w_stride,
    bool fwd_stats
) {
    const int64_t cout_blks = (C_out + FWD_TILE_COUT - 1) / FWD_TILE_COUT;
    const int64_t ow_tiles  = (out_w + FWD_TILE_OW - 1) / FWD_TILE_OW;
    const int64_t tile_count = N * cout_blks * out_h * ow_tiles;
    const int fwd_nthreads = omp_get_max_threads();
    int64_t fwd_tiles_per_thread[QUEUE_STATS_MAX_THREADS] = {};

    float* x_pad_buf = nullptr;
    int64_t x_pad_l = bwd_x_pad_l(pad);
    int64_t x_row_stride = stride2_x_row_stride(Stride2Specialist<KernelK>::K, pad, W_in, out_w);
    const size_t x_pad_floats = (size_t)(N * C_in * H * x_row_stride);
    x_pad_buf = acquire_bwd_x_pad_buf(x_pad_floats);
    if (x_pad_buf) {
        build_x_pad_buf(
            x, x_pad_buf,
            N, C_in, H, W_in,
            W_in_stride, x_pad_l, x_row_stride
        );
    }

    #pragma omp parallel for schedule(dynamic, 8)
    for (int64_t tid = 0; tid < tile_count; ++tid) {
        if (fwd_stats) {
            const int t = omp_get_thread_num();
            if (t >= 0 && t < QUEUE_STATS_MAX_THREADS) {
                ++fwd_tiles_per_thread[t];
            }
        }
        const ConvFwdTileDoc doc = decode_fwd_tile_doc(
            tid, N, C_out, out_h, out_w, 0, out_w
        );
        if (x_pad_buf) {
            Stride2Specialist<KernelK>::fwd_tile(
                doc, x_pad_buf, x_pad_l, x_row_stride,
                W, out, C_in, C_out, H, pad, k_spatial, spatial_out, out_w_stride
            );
        }
    }

    if (fwd_stats) {
        log_fwd_queue_runtime(
            fwd_tiles_per_thread, fwd_nthreads, tile_count, 8
        );
    }
}

static void conv2d_forward_stride2_dispatch(
    int64_t specialist_k,
    const float* x, const float* W, float* out,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t pad, int64_t out_h, int64_t out_w,
    int64_t spatial_in, int64_t spatial_out, int64_t k_spatial, int64_t out_w_stride,
    bool fwd_stats
) {
    if (specialist_k == 1) {
        conv2d_forward_stride2_phase2<1>(
            x, W, out,
            N, C_in, H, W_in, W_in_stride, C_out, pad, out_h, out_w,
            spatial_in, spatial_out, k_spatial, out_w_stride, fwd_stats
        );
    } else if (specialist_k == 2) {
        conv2d_forward_stride2_phase2<2>(
            x, W, out,
            N, C_in, H, W_in, W_in_stride, C_out, pad, out_h, out_w,
            spatial_in, spatial_out, k_spatial, out_w_stride, fwd_stats
        );
    } else if (specialist_k == 3) {
        conv2d_forward_stride2_phase2<3>(
            x, W, out,
            N, C_in, H, W_in, W_in_stride, C_out, pad, out_h, out_w,
            spatial_in, spatial_out, k_spatial, out_w_stride, fwd_stats
        );
    } else if (specialist_k == 4) {
        conv2d_forward_stride2_phase2<4>(
            x, W, out,
            N, C_in, H, W_in, W_in_stride, C_out, pad, out_h, out_w,
            spatial_in, spatial_out, k_spatial, out_w_stride, fwd_stats
        );
    } else if (specialist_k == 5) {
        conv2d_forward_stride2_phase2<5>(
            x, W, out,
            N, C_in, H, W_in, W_in_stride, C_out, pad, out_h, out_w,
            spatial_in, spatial_out, k_spatial, out_w_stride, fwd_stats
        );
    } else if (specialist_k == 6) {
        conv2d_forward_stride2_phase2<6>(
            x, W, out,
            N, C_in, H, W_in, W_in_stride, C_out, pad, out_h, out_w,
            spatial_in, spatial_out, k_spatial, out_w_stride, fwd_stats
        );
    } else if (specialist_k == 7) {
        conv2d_forward_stride2_phase2<7>(
            x, W, out,
            N, C_in, H, W_in, W_in_stride, C_out, pad, out_h, out_w,
            spatial_in, spatial_out, k_spatial, out_w_stride, fwd_stats
        );
    }
}

static void stride2_dw_nci_dispatch(
    int64_t specialist_k,
    int64_t n, int64_t cout, int64_t cin,
    float* __restrict dw_slice,
    const float* __restrict dy_pad_buf,
    const float* __restrict x_pad_buf,
    int64_t C_in, int64_t C_out,
    int64_t dy_pad_l, int64_t dy_row_stride,
    int64_t x_pad_l, int64_t x_row_stride,
    int64_t H, int64_t pad,
    int64_t conv_out_h, int64_t conv_out_w
) {
    if (specialist_k == 1) {
        Stride2Specialist<1>::dw_nci(
            n, cout, cin, dw_slice,
            dy_pad_buf, x_pad_buf,
            C_in, C_out,
            dy_pad_l, dy_row_stride,
            x_pad_l, x_row_stride,
            H, pad,
            conv_out_h, conv_out_w
        );
    } else if (specialist_k == 2) {
        Stride2Specialist<2>::dw_nci(
            n, cout, cin, dw_slice,
            dy_pad_buf, x_pad_buf,
            C_in, C_out,
            dy_pad_l, dy_row_stride,
            x_pad_l, x_row_stride,
            H, pad,
            conv_out_h, conv_out_w
        );
    } else if (specialist_k == 3) {
        Stride2Specialist<3>::dw_nci(
            n, cout, cin, dw_slice,
            dy_pad_buf, x_pad_buf,
            C_in, C_out,
            dy_pad_l, dy_row_stride,
            x_pad_l, x_row_stride,
            H, pad,
            conv_out_h, conv_out_w
        );
    } else if (specialist_k == 4) {
        Stride2Specialist<4>::dw_nci(
            n, cout, cin, dw_slice,
            dy_pad_buf, x_pad_buf,
            C_in, C_out,
            dy_pad_l, dy_row_stride,
            x_pad_l, x_row_stride,
            H, pad,
            conv_out_h, conv_out_w
        );
    } else if (specialist_k == 5) {
        Stride2Specialist<5>::dw_nci(
            n, cout, cin, dw_slice,
            dy_pad_buf, x_pad_buf,
            C_in, C_out,
            dy_pad_l, dy_row_stride,
            x_pad_l, x_row_stride,
            H, pad,
            conv_out_h, conv_out_w
        );
    } else if (specialist_k == 6) {
        Stride2Specialist<6>::dw_nci(
            n, cout, cin, dw_slice,
            dy_pad_buf, x_pad_buf,
            C_in, C_out,
            dy_pad_l, dy_row_stride,
            x_pad_l, x_row_stride,
            H, pad,
            conv_out_h, conv_out_w
        );
    } else if (specialist_k == 7) {
        Stride2Specialist<7>::dw_nci(
            n, cout, cin, dw_slice,
            dy_pad_buf, x_pad_buf,
            C_in, C_out,
            dy_pad_l, dy_row_stride,
            x_pad_l, x_row_stride,
            H, pad,
            conv_out_h, conv_out_w
        );
    }
}

// ========================================================================
// Forward Pass: Bias Init + Tiled Batch Dispatch + Optional ReLU
// OMP: dynamic,8 on every parallel for in this file (native fallback hot path).
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

    const bool fwd_stats = fwd_queue_stats_enabled();

    // Phase 1: bias broadcast
    #pragma omp parallel for collapse(2) schedule(dynamic, 8)
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
        const int fwd_nthreads = omp_get_max_threads();
        int64_t fwd_tiles_per_thread[QUEUE_STATS_MAX_THREADS] = {};

        float* x_pad_buf = nullptr;
        int64_t x_pad_l = 0;
        int64_t x_row_stride = 0;
        if (stride1_fwd_builds_x_pad(k_h, k_w)) {
            x_pad_l = bwd_x_pad_l(pad);
            x_row_stride = bwd_x_row_stride(k_w, pad, W_in, out_w);
            const size_t x_pad_floats = (size_t)(N * C_in * H * x_row_stride);
            x_pad_buf = acquire_bwd_x_pad_buf(x_pad_floats);
            if (x_pad_buf) {
                build_x_pad_buf(
                    x, x_pad_buf,
                    N, C_in, H, W_in,
                    W_in_stride, x_pad_l, x_row_stride
                );
            }
        }

        #pragma omp parallel for schedule(dynamic, 8)
        for (int64_t tid = 0; tid < tile_count; ++tid) {
            if (fwd_stats) {
                const int t = omp_get_thread_num();
                if (t >= 0 && t < QUEUE_STATS_MAX_THREADS) {
                    ++fwd_tiles_per_thread[t];
                }
            }
            const ConvFwdTileDoc doc = decode_fwd_tile_doc(
                tid, N, C_out, out_h, out_w, ow_safe_start, ow_safe_end
            );
            process_fwd_tile_stride1(
                doc, x, W, out,
                C_in, C_out, H, W_in, W_in_stride,
                k_h, k_w, pad,
                spatial_in, spatial_out, k_spatial, out_w_stride,
                x_pad_buf, x_pad_l, x_row_stride
            );
        }

        if (fwd_stats) {
            log_fwd_queue_runtime(
                fwd_tiles_per_thread, fwd_nthreads, tile_count, 8
            );
        }
    } else if (stride == 2 && stride2_specialist_k(k_h, k_w) != 0) {
        conv2d_forward_stride2_dispatch(
            stride2_specialist_k(k_h, k_w),
            x, W, out,
            N, C_in, H, W_in, W_in_stride, C_out, pad, out_h, out_w,
            spatial_in, spatial_out, k_spatial, out_w_stride, fwd_stats
        );
    } else {
        #pragma omp parallel for collapse(2) schedule(dynamic, 8)
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
        #pragma omp parallel for collapse(2) schedule(dynamic, 8)
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
// Backward Pass: spatial-tile dW (mirrors dX), interleaved omp dispatch when both run.
// OMP: dynamic,8 for all dX/dW tile and N×C work queues (uneven); never static here.
// ========================================================================
static thread_local float* tls_dy_pad_buf = nullptr;
static thread_local size_t tls_dy_pad_cap_floats = 0;
static thread_local float* tls_x_pad_buf = nullptr;
static thread_local size_t tls_x_pad_cap_floats = 0;

static float* acquire_bwd_dy_pad_buf(size_t need_floats) {
    if (need_floats == 0) return nullptr;
    if (need_floats > tls_dy_pad_cap_floats) {
        std::free(tls_dy_pad_buf);
        tls_dy_pad_buf = (float*)std::malloc(need_floats * sizeof(float));
        tls_dy_pad_cap_floats = tls_dy_pad_buf ? need_floats : 0;
    }
    return tls_dy_pad_buf;
}

static float* acquire_bwd_x_pad_buf(size_t need_floats) {
    if (need_floats == 0) return nullptr;
    if (need_floats > tls_x_pad_cap_floats) {
        std::free(tls_x_pad_buf);
        tls_x_pad_buf = (float*)std::malloc(need_floats * sizeof(float));
        tls_x_pad_cap_floats = tls_x_pad_buf ? need_floats : 0;
    }
    return tls_x_pad_buf;
}

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
        if (bwd_dx_mock_edges_enabled()) {
            log_bwd_dx_mock_edges_once();
        }
    }
    if (do_dw) {
        std::memset(dW, 0, (size_t)(C_out * C_in * k_spatial) * sizeof(float));
    }

    const int64_t dw_count = C_out * C_in * k_spatial;

    const int64_t iw_tiles = (W_in + FWD_TILE_OW - 1) / FWD_TILE_OW;
    const int64_t dx_tile_count = N * C_in * H * iw_tiles;
    const int64_t dw_task_count = N * C_out * C_in;
    const bool interleaved_dx_dw = (do_dx && do_dw && stride == 1);
    const int64_t total_bwd_work = interleaved_dx_dw
        ? (dx_tile_count + dw_task_count) : 0;
    const bool queue_stats = bwd_queue_stats_enabled();
    const int bwd_nthreads = omp_get_max_threads();
    BwdQueueThreadStats thread_stats[QUEUE_STATS_MAX_THREADS] = {};
    BwdOverlapProof overlap{};
    BwdOverlapProof* overlap_ptr = queue_stats ? &overlap : nullptr;

    if (queue_stats && interleaved_dx_dw) {
        log_bwd_queue_plan(
            N, C_in, C_out, H, W_in,
            dx_tile_count, dw_task_count, total_bwd_work, 8, bwd_nthreads
        );
    }

    float* dy_pad_buf = nullptr;
    int64_t dy_pad_l = 0;
    int64_t dy_row_stride = 0;
    float* x_pad_buf = nullptr;
    int64_t x_pad_l = 0;
    int64_t x_row_stride = 0;
    if (stride == 1 && (do_dx || do_dw)) {
        dy_pad_l = bwd_dy_pad_l(k_w, pad);
        dy_row_stride = bwd_dy_row_stride(k_w, pad, W_in, conv_out_w);
        const size_t dy_pad_floats =
            (size_t)(N * C_out * conv_out_h * dy_row_stride);
        dy_pad_buf = acquire_bwd_dy_pad_buf(dy_pad_floats);
        if (dy_pad_buf) {
            build_dy_pad_buf(
                d_conv_buf, dy_pad_buf,
                N, C_out, conv_out_h, conv_out_w,
                conv_out_w_stride, dy_pad_l, dy_row_stride
            );
        }
    }
    if (stride == 1 && do_dw && x) {
        x_pad_l = bwd_x_pad_l(pad);
        x_row_stride = bwd_x_row_stride(k_w, pad, W_in, conv_out_w);
        const size_t x_pad_floats = (size_t)(N * C_in * H * x_row_stride);
        x_pad_buf = acquire_bwd_x_pad_buf(x_pad_floats);
        if (x_pad_buf) {
            build_x_pad_buf(
                x, x_pad_buf,
                N, C_in, H, W_in,
                W_in_stride, x_pad_l, x_row_stride
            );
        }
    }
    if (stride == 2 && stride2_specialist_k(k_h, k_w) != 0 && (do_dx || do_dw)) {
        dy_pad_l = bwd_dy_pad_l(k_w, pad);
        dy_row_stride = bwd_dy_row_stride(k_w, pad, W_in, conv_out_w);
        const size_t dy_pad_floats =
            (size_t)(N * C_out * conv_out_h * dy_row_stride);
        dy_pad_buf = acquire_bwd_dy_pad_buf(dy_pad_floats);
        if (dy_pad_buf) {
            build_dy_pad_buf(
                d_conv_buf, dy_pad_buf,
                N, C_out, conv_out_h, conv_out_w,
                conv_out_w_stride, dy_pad_l, dy_row_stride
            );
        }
    }
    if (stride == 2 && stride2_specialist_k(k_h, k_w) != 0 && do_dw && x) {
        x_pad_l = bwd_x_pad_l(pad);
        x_row_stride = stride2_x_row_stride(k_w, pad, W_in, conv_out_w);
        const size_t x_pad_floats = (size_t)(N * C_in * H * x_row_stride);
        x_pad_buf = acquire_bwd_x_pad_buf(x_pad_floats);
        if (x_pad_buf) {
            build_x_pad_buf(
                x, x_pad_buf,
                N, C_in, H, W_in,
                W_in_stride, x_pad_l, x_row_stride
            );
        }
    }

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

        if (interleaved_dx_dw && tls_priv_dW) {
            #pragma omp for schedule(dynamic, 8)
            for (int64_t wid = 0; wid < total_bwd_work; ++wid) {
                bool is_dx = false;
                int64_t local_id = 0;
                decode_stream_work_item(
                    wid, dx_tile_count, dw_task_count, is_dx, local_id
                );

                if (is_dx) {
                    bwd_overlap_note_dx_start(overlap_ptr);
                    const ConvBwdDxTileDoc doc =
                        decode_bwd_dx_tile_doc(local_id, N, C_in, H, W_in);
                    const uint64_t t0 = queue_stats ? bwd_rdtsc() : 0;
                    process_bwd_dx_tile_stride1(
                        doc, d_conv_buf, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
                        C_in, C_out, H, W_in, W_in_stride,
                        k_h, k_w, pad,
                        spatial_in, conv_spatial, k_spatial,
                        conv_out_h, conv_out_w, conv_out_w_stride
                    );
                    bwd_overlap_note_dx_end(overlap_ptr);
                    if (queue_stats) {
                        const uint64_t dt = bwd_rdtsc() - t0;
                        const int t = omp_get_thread_num();
                        if (t >= 0 && t < QUEUE_STATS_MAX_THREADS) {
                            ++thread_stats[t].dx;
                            thread_stats[t].dx_cycles += dt;
                            if (dt < thread_stats[t].dx_min_cycles) {
                                thread_stats[t].dx_min_cycles = dt;
                            }
                            if (dt > thread_stats[t].dx_max_cycles) {
                                thread_stats[t].dx_max_cycles = dt;
                            }
                        }
                    }
                } else {
                    int64_t n, cout, cin;
                    decode_dw_nci_task(local_id, N, C_out, C_in, n, cout, cin);
                    float* __restrict dw_slice =
                        &tls_priv_dW[(cout * C_in + cin) * k_spatial];
                    bwd_overlap_note_dw_start(overlap_ptr);
                    const uint64_t t0 = queue_stats ? bwd_rdtsc() : 0;
                    process_dw_nci_stride1(
                        n, cout, cin, dw_slice,
                        d_conv_buf, dy_pad_buf, x, x_pad_buf,
                        dy_pad_l, dy_row_stride,
                        x_pad_l, x_row_stride,
                        C_in, C_out, H, W_in, W_in_stride,
                        k_h, k_w, pad,
                        spatial_in, conv_spatial,
                        conv_out_h, conv_out_w, conv_out_w_stride
                    );
                    bwd_overlap_note_dw_end(overlap_ptr);
                    if (queue_stats) {
                        const uint64_t dt = bwd_rdtsc() - t0;
                        const int t = omp_get_thread_num();
                        if (t >= 0 && t < QUEUE_STATS_MAX_THREADS) {
                            ++thread_stats[t].dw;
                            thread_stats[t].dw_cycles += dt;
                            if (dt < thread_stats[t].dw_min_cycles) {
                                thread_stats[t].dw_min_cycles = dt;
                            }
                            if (dt > thread_stats[t].dw_max_cycles) {
                                thread_stats[t].dw_max_cycles = dt;
                            }
                        }
                    }
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
                    #pragma omp for schedule(dynamic, 8)
                    for (int64_t tid = 0; tid < dx_tile_count; ++tid) {
                        const ConvBwdDxTileDoc doc = decode_bwd_dx_tile_doc(tid, N, C_in, H, W_in);
                        process_bwd_dx_tile_stride1(
                            doc, d_conv_buf, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
                            C_in, C_out, H, W_in, W_in_stride,
                            k_h, k_w, pad,
                            spatial_in, conv_spatial, k_spatial,
                            conv_out_h, conv_out_w, conv_out_w_stride
                        );
                    }
                } else if (stride == 2 && stride2_specialist_k(k_h, k_w) != 0) {
                    #pragma omp for schedule(dynamic, 8)
                    for (int64_t tid = 0; tid < dx_tile_count; ++tid) {
                        const ConvBwdDxTileDoc doc = decode_bwd_dx_tile_doc(tid, N, C_in, H, W_in);
                        process_bwd_dx_tile_stride2_dispatch(
                            stride2_specialist_k(k_h, k_w),
                            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
                            C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
                            conv_out_h, conv_out_w
                        );
                    }
                } else {
                    #pragma omp for collapse(2) schedule(dynamic, 8)
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
                #pragma omp for schedule(dynamic, 8)
                for (int64_t task_id = 0; task_id < dw_task_count; ++task_id) {
                    int64_t n, cout, cin;
                    decode_dw_nci_task(task_id, N, C_out, C_in, n, cout, cin);
                    float* __restrict dw_slice =
                        &tls_priv_dW[(cout * C_in + cin) * k_spatial];
                    if (stride == 1) {
                        process_dw_nci_stride1(
                            n, cout, cin, dw_slice,
                            d_conv_buf, dy_pad_buf, x, x_pad_buf,
                            dy_pad_l, dy_row_stride,
                            x_pad_l, x_row_stride,
                            C_in, C_out, H, W_in, W_in_stride,
                            k_h, k_w, pad,
                            spatial_in, conv_spatial,
                            conv_out_h, conv_out_w, conv_out_w_stride
                        );
                    } else if (stride == 2 && stride2_specialist_k(k_h, k_w) != 0) {
                        stride2_dw_nci_dispatch(
                            stride2_specialist_k(k_h, k_w),
                            n, cout, cin, dw_slice,
                            dy_pad_buf, x_pad_buf,
                            C_in, C_out,
                            dy_pad_l, dy_row_stride,
                            x_pad_l, x_row_stride,
                            H, pad,
                            conv_out_h, conv_out_w
                        );
                    } else {
                        process_dw_nci_task(
                            n, cout, cin, dw_slice,
                            d_conv_buf, dy_pad_buf, x, x_pad_buf,
                            dy_pad_l, dy_row_stride,
                            x_pad_l, x_row_stride,
                            C_in, C_out, H, W_in, W_in_stride,
                            k_h, k_w, stride, pad,
                            spatial_in, conv_spatial, k_spatial,
                            conv_out_h, conv_out_w, conv_out_w_stride
                        );
                    }
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

    if (queue_stats && interleaved_dx_dw) {
        log_bwd_queue_runtime(
            thread_stats, bwd_nthreads, dx_tile_count, dw_task_count
        );
        log_bwd_overlap_proof(&overlap);
    }

    if (do_dw) {
        for (int64_t i = 0; i < dw_count; ++i) {
            dW[i] *= inv_m;
        }
    }
}