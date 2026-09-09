#include <immintrin.h>
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#ifndef __forceinline
#ifdef ML_ENGINE_NO_FORCEINLINE
#define __forceinline __attribute__((noinline))
#else
#define __forceinline inline __attribute__((always_inline))
#endif
#endif
#endif
#include "export.h"
#include "omp_config.h"
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>
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

// Diagnostic: ML_ENGINE_FWD_ZONE_TIMING=1 → split each OW tile into
// left/center/right subranges and time them separately (changes the timed
// path: partial tiles instead of one FWD_TILE_OW call). Off = production path.
static bool fwd_zone_timing_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("ML_ENGINE_FWD_ZONE_TIMING");
        cached = (env && env[0] == '1' && env[1] == '\0') ? 1 : 0;
    }
    return cached != 0;
}

// Diagnostic: ML_ENGINE_FWD_TRACE=1 → print layouts + op steps (no behavior change).
static bool fwd_trace_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("ML_ENGINE_FWD_TRACE");
        cached = (env && env[0] == '1' && env[1] == '\0') ? 1 : 0;
    }
    return cached != 0;
}

static void fwd_trace_ptr(const char* label, const float* p, size_t n_sample) {
    if (!p) {
        std::printf("[FWD_TRACE]   %s: null\n", label);
        return;
    }
    std::printf("[FWD_TRACE]   %s: ptr=%p", label, (const void*)p);
    if (n_sample > 0) {
        std::printf("  first[%zu]=", n_sample);
        for (size_t i = 0; i < n_sample; ++i) {
            std::printf("%s%.5g", i ? "," : "", (double)p[i]);
        }
    }
    std::printf("\n");
}

enum : int { FWD_ZONE_LEFT = 0, FWD_ZONE_CENTER = 1, FWD_ZONE_RIGHT = 2, FWD_ZONE_HALO = 3 };

static void fwd_ow_zone_bounds(
    int64_t out_w, int64_t pad, int64_t* mid_lo, int64_t* mid_hi
) {
    *mid_lo = pad;
    *mid_hi = (out_w > pad) ? (out_w - pad) : pad;
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
    int64_t tid, int64_t N, int64_t C_out, int64_t out_h, int64_t compute_ow,
    int64_t ow_safe_start, int64_t ow_safe_end
) {
    const int64_t cout_blks = (C_out + FWD_TILE_COUT - 1) / FWD_TILE_COUT;
    const int64_t ow_tiles  = (compute_ow + FWD_TILE_OW - 1) / FWD_TILE_OW;

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
    doc.ow_count   = (int8_t)std::min((int64_t)FWD_TILE_OW, compute_ow - ow);
    doc.middle_zone = (int8_t)(
        doc.ow_count == FWD_TILE_OW &&
        ow >= ow_safe_start &&
        (ow + FWD_TILE_OW) <= ow_safe_end
    );
    return doc;
}

// Tile over round_up(logical_ow, 8) when out_w_stride has room (SIMD halo).
// Avoids rem OW tiles so Stride1Specialist<K> always sees full ow_count==8.
static inline int64_t fwd_compute_ow(int64_t out_w, int64_t out_w_stride) {
    const int64_t rounded =
        (out_w + FWD_TILE_OW - 1) & ~(FWD_TILE_OW - 1);
    if (rounded <= out_w_stride) {
        return rounded;
    }
    const int64_t aligned_stride =
        out_w_stride - (out_w_stride % FWD_TILE_OW);
    return aligned_stride > out_w ? aligned_stride : out_w;
}

// --- Stride-1 padded-buffer kernel specialists (template<int K>) ---
// Primary template defined after alignr helpers. Runtime dispatch via
// stride1_specialist_k() / stride1_try_*_specialist().

struct ConvBwdDxTileDoc;

template<int K>
struct Stride1Specialist;

static inline int64_t stride1_specialist_k(int64_t k_h, int64_t k_w) {
    if (k_h != k_w) {
        return 0;
    }
    // Primary Stride1Specialist<K> covers every square K in this range.
    if (k_h >= 1 && k_h <= 11) {
        return k_h;
    }
    return 0;
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

// Sliding kw windows from two AVX loads + alignr (identical math to
// loadu(row+base+kw) for kw=0..6). Used by Stride1Specialist fwd/dw for K=2..7
// so every K shares the same crawl; K=1 stays a single load.
static __forceinline void stride1_kw_windows_alignr(
    const float* __restrict row, int64_t base,
    __m256& vx0, __m256& vx1, __m256& vx2, __m256& vx3,
    __m256& vx4, __m256& vx5, __m256& vx6
) {
    const __m256 xa   = _mm256_loadu_ps(row + base);
    const __m256 xb   = _mm256_loadu_ps(row + base + FWD_TILE_OW);
    const __m256 xmid = _mm256_permute2f128_ps(xa, xb, 0x21);
    const __m256i ia   = _mm256_castps_si256(xa);
    const __m256i ib   = _mm256_castps_si256(xb);
    const __m256i imid = _mm256_castps_si256(xmid);
    vx0 = xa;
    vx1 = _mm256_castsi256_ps(_mm256_alignr_epi8(imid, ia, 4));
    vx2 = _mm256_castsi256_ps(_mm256_alignr_epi8(imid, ia, 8));
    vx3 = _mm256_castsi256_ps(_mm256_alignr_epi8(imid, ia, 12));
    vx4 = xmid;
    vx5 = _mm256_castsi256_ps(_mm256_alignr_epi8(ib, imid, 4));
    vx6 = _mm256_castsi256_ps(_mm256_alignr_epi8(ib, imid, 8));
}

// kw=0..8 from the same two AVX loads (vx8 == xb). Used by Stride1Specialist K=9.
static __forceinline void stride1_kw_windows_alignr9(
    const float* __restrict row, int64_t base,
    __m256& vx0, __m256& vx1, __m256& vx2, __m256& vx3,
    __m256& vx4, __m256& vx5, __m256& vx6, __m256& vx7, __m256& vx8
) {
    const __m256 xa   = _mm256_loadu_ps(row + base);
    const __m256 xb   = _mm256_loadu_ps(row + base + FWD_TILE_OW);
    const __m256 xmid = _mm256_permute2f128_ps(xa, xb, 0x21);
    const __m256i ia   = _mm256_castps_si256(xa);
    const __m256i ib   = _mm256_castps_si256(xb);
    const __m256i imid = _mm256_castps_si256(xmid);
    vx0 = xa;
    vx1 = _mm256_castsi256_ps(_mm256_alignr_epi8(imid, ia, 4));
    vx2 = _mm256_castsi256_ps(_mm256_alignr_epi8(imid, ia, 8));
    vx3 = _mm256_castsi256_ps(_mm256_alignr_epi8(imid, ia, 12));
    vx4 = xmid;
    vx5 = _mm256_castsi256_ps(_mm256_alignr_epi8(ib, imid, 4));
    vx6 = _mm256_castsi256_ps(_mm256_alignr_epi8(ib, imid, 8));
    vx7 = _mm256_castsi256_ps(_mm256_alignr_epi8(ib, imid, 12));
    vx8 = xb;
}

// kw=0..10: three AVX loads (base+0..23). Used by Stride1Specialist K=11.
static __forceinline void stride1_kw_windows_alignr11(
    const float* __restrict row, int64_t base,
    __m256& vx0, __m256& vx1, __m256& vx2, __m256& vx3,
    __m256& vx4, __m256& vx5, __m256& vx6, __m256& vx7,
    __m256& vx8, __m256& vx9, __m256& vx10
) {
    const __m256 xa   = _mm256_loadu_ps(row + base);
    const __m256 xb   = _mm256_loadu_ps(row + base + FWD_TILE_OW);
    const __m256 xc   = _mm256_loadu_ps(row + base + 2 * FWD_TILE_OW);
    const __m256 xmid_ab = _mm256_permute2f128_ps(xa, xb, 0x21);
    const __m256 xmid_bc = _mm256_permute2f128_ps(xb, xc, 0x21);
    const __m256i ia   = _mm256_castps_si256(xa);
    const __m256i ib   = _mm256_castps_si256(xb);
    const __m256i imid_ab = _mm256_castps_si256(xmid_ab);
    const __m256i imid_bc = _mm256_castps_si256(xmid_bc);
    vx0 = xa;
    vx1 = _mm256_castsi256_ps(_mm256_alignr_epi8(imid_ab, ia, 4));
    vx2 = _mm256_castsi256_ps(_mm256_alignr_epi8(imid_ab, ia, 8));
    vx3 = _mm256_castsi256_ps(_mm256_alignr_epi8(imid_ab, ia, 12));
    vx4 = xmid_ab;
    vx5 = _mm256_castsi256_ps(_mm256_alignr_epi8(ib, imid_ab, 4));
    vx6 = _mm256_castsi256_ps(_mm256_alignr_epi8(ib, imid_ab, 8));
    vx7 = _mm256_castsi256_ps(_mm256_alignr_epi8(ib, imid_ab, 12));
    vx8 = xb;
    vx9  = _mm256_castsi256_ps(_mm256_alignr_epi8(imid_bc, ib, 4));
    vx10 = _mm256_castsi256_ps(_mm256_alignr_epi8(imid_bc, ib, 8));
}

// Max square K handled by the primary Stride1Specialist template.
static constexpr int STRIDE1_SPECIALIST_K_MAX = 11;

// Fill vx[0..K) from the matching alignr helper (K=1: single loadu).
template<int K>
static __forceinline void stride1_load_kw_windows(
    const float* __restrict row, int64_t base, __m256 vx[STRIDE1_SPECIALIST_K_MAX]
) {
    static_assert(K >= 1 && K <= STRIDE1_SPECIALIST_K_MAX, "Stride1 K out of range");
    if constexpr (K == 1) {
        vx[0] = _mm256_loadu_ps(row + base);
    } else if constexpr (K <= 7) {
        stride1_kw_windows_alignr(
            row, base, vx[0], vx[1], vx[2], vx[3], vx[4], vx[5], vx[6]
        );
    } else if constexpr (K <= 9) {
        stride1_kw_windows_alignr9(
            row, base, vx[0], vx[1], vx[2], vx[3], vx[4], vx[5], vx[6], vx[7], vx[8]
        );
    } else {
        stride1_kw_windows_alignr11(
            row, base,
            vx[0], vx[1], vx[2], vx[3], vx[4], vx[5], vx[6], vx[7], vx[8], vx[9], vx[10]
        );
    }
}

// Primary Stride1 specialist (same idea as Stride2Specialist<KernelK>): one body,
// K chosen at compile time. Explicit per-K clones removed to cut blast radius.
template<int K>
struct Stride1Specialist {
    static_assert(K >= 1 && K <= STRIDE1_SPECIALIST_K_MAX, "Stride1 K out of range");

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
                __m256 vx[STRIDE1_SPECIALIST_K_MAX];
                stride1_load_kw_windows<K>(in_row, iw0, vx);

                const float* __restrict w0 = wp0 + kh * K;
#pragma GCC unroll 16
                for (int kw = 0; kw < K; ++kw) {
                    vo0 = _mm256_fmadd_ps(vx[kw], _mm256_set1_ps(w0[kw]), vo0);
                }
                if (c_rem > 1) {
                    const float* __restrict w1 = wp1 + kh * K;
#pragma GCC unroll 16
                    for (int kw = 0; kw < K; ++kw) {
                        vo1 = _mm256_fmadd_ps(vx[kw], _mm256_set1_ps(w1[kw]), vo1);
                    }
                }
                if (c_rem > 2) {
                    const float* __restrict w2 = wp2 + kh * K;
#pragma GCC unroll 16
                    for (int kw = 0; kw < K; ++kw) {
                        vo2 = _mm256_fmadd_ps(vx[kw], _mm256_set1_ps(w2[kw]), vo2);
                    }
                }
                if (c_rem > 3) {
                    const float* __restrict w3 = wp3 + kh * K;
#pragma GCC unroll 16
                    for (int kw = 0; kw < K; ++kw) {
                        vo3 = _mm256_fmadd_ps(vx[kw], _mm256_set1_ps(w3[kw]), vo3);
                    }
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

// Sentinel removed: explicit Stride1Specialist clones deleted below.

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
    const int64_t kspec = stride1_specialist_k(k_h, k_w);
    if (kspec == 0) {
        return false;
    }
    switch (kspec) {
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
        case 8:
            Stride1Specialist<8>::fwd_tile(
                doc, x_pad_buf, x_pad_l, x_row_stride,
                W, out, C_in, C_out, H, pad, k_spatial, spatial_out, out_w_stride
            );
            return true;
        case 9:
            Stride1Specialist<9>::fwd_tile(
                doc, x_pad_buf, x_pad_l, x_row_stride,
                W, out, C_in, C_out, H, pad, k_spatial, spatial_out, out_w_stride
            );
            return true;
        case 10:
            Stride1Specialist<10>::fwd_tile(
                doc, x_pad_buf, x_pad_l, x_row_stride,
                W, out, C_in, C_out, H, pad, k_spatial, spatial_out, out_w_stride
            );
            return true;
        case 11:
            Stride1Specialist<11>::fwd_tile(
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
        case 8:
            Stride1Specialist<8>::bwd_dx_tile(
                doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
                C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
                conv_out_h, conv_out_w
            );
            return true;
        case 9:
            Stride1Specialist<9>::bwd_dx_tile(
                doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
                C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
                conv_out_h, conv_out_w
            );
            return true;
        case 10:
            Stride1Specialist<10>::bwd_dx_tile(
                doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
                C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
                conv_out_h, conv_out_w
            );
            return true;
        case 11:
            Stride1Specialist<11>::bwd_dx_tile(
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
        case 8:
            Stride1Specialist<8>::dw_nci(
                n, cout, cin, dw_slice,
                dy_pad_buf, x_pad_buf,
                C_in, C_out,
                dy_pad_l, dy_row_stride,
                x_pad_l, x_row_stride,
                H, pad,
                conv_out_h, conv_out_w
            );
            return true;
        case 9:
            Stride1Specialist<9>::dw_nci(
                n, cout, cin, dw_slice,
                dy_pad_buf, x_pad_buf,
                C_in, C_out,
                dy_pad_l, dy_row_stride,
                x_pad_l, x_row_stride,
                H, pad,
                conv_out_h, conv_out_w
            );
            return true;
        case 10:
            Stride1Specialist<10>::dw_nci(
                n, cout, cin, dw_slice,
                dy_pad_buf, x_pad_buf,
                C_in, C_out,
                dy_pad_l, dy_row_stride,
                x_pad_l, x_row_stride,
                H, pad,
                conv_out_h, conv_out_w
            );
            return true;
        case 11:
            Stride1Specialist<11>::dw_nci(
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
    int64_t stride = need > min_stride ? need : min_stride;

    // Stride-1 dW derives kw windows from consecutive AVX loads starting at
    // tile base. Span must cover base..(base + (k_w-1) + FWD_TILE_OW - 1)
    // (two vectors for K<=9, three for K=11).
    const int64_t ow_last =
        ((conv_out_w + FWD_TILE_OW - 1) / FWD_TILE_OW - 1) * FWD_TILE_OW;
    const int64_t dw_span = pad_l + ow_last - pad + (k_w - 1) + FWD_TILE_OW;
    if (dw_span > stride) stride = dw_span;

    // Rounding to a whole vector keeps every row start 32B-aligned (the buffer
    // base is, and a tile base is pad_l + ow - pad == ow), which takes the tile
    // loads off split cache lines.
    return (stride + FWD_TILE_OW - 1) & ~(FWD_TILE_OW - 1);
}

// parallel=false when the Python main thread stages ahead of a busy OMP team
// (async overlap). parallel=true when async is off (main_stage_use_omp).
static void build_bwd_row_pad_buf(
    const float* __restrict src, float* __restrict dst,
    int64_t nplanes, int64_t nrows, int64_t row_w,
    int64_t src_row_stride, int64_t pad_l, int64_t row_stride,
    bool parallel = true
) {
    const int64_t src_plane = nrows * src_row_stride;
    const int64_t dst_plane = nrows * row_stride;
    const __m256 z = _mm256_setzero_ps();

    #pragma omp parallel for collapse(2) schedule(dynamic, 8) if(parallel)
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

// ---------------------------------------------------------------------------
// Contract async-overlap flag + main-thread staged x_pad
//
// async on  → pack serially on main (OMP team busy on the in-flight step)
// async off → pack with OMP (team free); sync path stages before native invoke
// ---------------------------------------------------------------------------
static std::atomic<int> g_contract_async_overlap{0};

extern "C" ML_ENGINE_EXPORT void set_contract_async_overlap(int32_t enabled) {
    g_contract_async_overlap.store(enabled ? 1 : 0, std::memory_order_release);
}

static inline bool main_stage_use_omp() {
    return g_contract_async_overlap.load(std::memory_order_acquire) == 0;
}

// Layer-0 padded input depends only on the batch tensor — stage before the
// step that consumes it. One entry per pipeline slot (no alias with in-flight).
constexpr int32_t STAGED_X_PAD_MAX_SLOTS = 4;

struct StagedXPad {
    const float* src = nullptr;
    int64_t N = 0;
    int64_t C_in = 0;
    int64_t H = 0;
    int64_t W_in = 0;
    int64_t src_row_stride = 0;
    int64_t pad_l = 0;
    int64_t row_stride = 0;
    float* buf = nullptr;
    size_t cap_floats = 0;
    bool valid = false;
};

static StagedXPad g_staged_x_pad[STAGED_X_PAD_MAX_SLOTS];

static float* staged_x_pad_alloc(StagedXPad& slot, size_t need_floats) {
    if (slot.buf && need_floats <= slot.cap_floats) {
        return slot.buf;
    }
    std::free(slot.buf);
    // 64B-aligned rows keep the sliding loads in the consumers off split lines.
    const size_t bytes = ((need_floats * sizeof(float)) + 63u) & ~(size_t)63u;
    slot.buf = (float*)std::aligned_alloc(64, bytes);
    slot.cap_floats = slot.buf ? (bytes / sizeof(float)) : 0;
    return slot.buf;
}

// Returns a staged buffer only on an exact match of source pointer + geometry.
static float* staged_x_pad_lookup(
    const float* src, int64_t N, int64_t C_in, int64_t H, int64_t W_in,
    int64_t src_row_stride, int64_t pad_l, int64_t row_stride
) {
    for (int32_t i = 0; i < STAGED_X_PAD_MAX_SLOTS; ++i) {
        const StagedXPad& s = g_staged_x_pad[i];
        if (!s.valid || !s.buf || s.src != src) continue;
        if (s.N != N || s.C_in != C_in || s.H != H || s.W_in != W_in) continue;
        if (s.src_row_stride != src_row_stride || s.pad_l != pad_l ||
            s.row_stride != row_stride) continue;
        return s.buf;
    }
    return nullptr;
}

// Called from Python on the main thread, before submitting the step that uses x.
extern "C" ML_ENGINE_EXPORT int32_t stage_conv_x_pad(
    int32_t slot_idx, const float* x,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t k_w, int64_t pad, int64_t conv_out_w
) {
    if (slot_idx < 0 || slot_idx >= STAGED_X_PAD_MAX_SLOTS) return -1;
    StagedXPad& slot = g_staged_x_pad[slot_idx];
    slot.valid = false;
    if (!x || N <= 0 || C_in <= 0 || H <= 0 || W_in <= 0) return -2;

    const int64_t pad_l = bwd_x_pad_l(pad);
    // Match fwd_compute_ow when out_w_stride >= round_up(ow,8) (normal SIMD buffers).
    const int64_t compute_ow =
        (conv_out_w + FWD_TILE_OW - 1) & ~(FWD_TILE_OW - 1);
    const int64_t row_stride = bwd_x_row_stride(k_w, pad, W_in, compute_ow);
    const size_t need = (size_t)(N * C_in * H * row_stride);
    float* buf = staged_x_pad_alloc(slot, need);
    if (!buf) return -3;

    if (main_stage_use_omp()) {
        ml_omp_before_parallel();
    }
    build_bwd_row_pad_buf(
        x, buf, N * C_in, H, W_in, W_in_stride, pad_l, row_stride,
        /*parallel=*/main_stage_use_omp()
    );

    slot.src = x;
    slot.N = N;
    slot.C_in = C_in;
    slot.H = H;
    slot.W_in = W_in;
    slot.src_row_stride = W_in_stride;
    slot.pad_l = pad_l;
    slot.row_stride = row_stride;
    slot.valid = true;
    return 0;
}

extern "C" ML_ENGINE_EXPORT void invalidate_conv_x_pad_stage(int32_t slot_idx) {
    if (slot_idx < 0) {
        for (int32_t i = 0; i < STAGED_X_PAD_MAX_SLOTS; ++i) {
            g_staged_x_pad[i].valid = false;
        }
    } else if (slot_idx < STAGED_X_PAD_MAX_SLOTS) {
        g_staged_x_pad[slot_idx].valid = false;
    }
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

// --- Primary Stride1Specialist<K> backward (one body for K=1..11) ---

template<int K>
static __forceinline void stride1_bwd_dx_accum_cout4(
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
#pragma GCC unroll 16
    for (int kw = 0; kw < K; ++kw) {
        v_dx = fmadd_dx_cout4(
            v_dx,
            _mm256_loadu_ps(dy0 + ow0 - kw), _mm256_loadu_ps(dy1 + ow0 - kw),
            _mm256_loadu_ps(dy2 + ow0 - kw), _mm256_loadu_ps(dy3 + ow0 - kw),
            wp0[kw], wp1[kw], wp2[kw], wp3[kw]
        );
    }
}

template<int K>
static void stride1_bwd_dx_tile_c8(
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
        stride1_bwd_dx_accum_cout4<K>(
            v_dx, ow0,
            dy_oh + 0 * dy_plane, dy_oh + 1 * dy_plane,
            dy_oh + 2 * dy_plane, dy_oh + 3 * dy_plane,
            w_kh + 0 * w_cout_stride, w_kh + 1 * w_cout_stride,
            w_kh + 2 * w_cout_stride, w_kh + 3 * w_cout_stride
        );
        stride1_bwd_dx_accum_cout4<K>(
            v_dx, ow0,
            dy_oh + 4 * dy_plane, dy_oh + 5 * dy_plane,
            dy_oh + 6 * dy_plane, dy_oh + 7 * dy_plane,
            w_kh + 4 * w_cout_stride, w_kh + 5 * w_cout_stride,
            w_kh + 6 * w_cout_stride, w_kh + 7 * w_cout_stride
        );
    }
    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

template<int K>
static void stride1_bwd_dx_tile_c16(
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
            stride1_bwd_dx_accum_cout4<K>(
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

template<int K>
void Stride1Specialist<K>::bwd_dx_tile(
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
        stride1_bwd_dx_tile_c8<K>(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
        return;
    }
    if (C_out == 16) {
        stride1_bwd_dx_tile_c16<K>(
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
            stride1_bwd_dx_accum_cout4<K>(
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
#pragma GCC unroll 16
            for (int kw = 0; kw < K; ++kw) {
                v_dx = fmadd_dx_cout1(v_dx, _mm256_loadu_ps(dy_row + ow0 - kw), wp[kw]);
            }
        }
    }
    bwd_dx_store_tile(dx_row, v_dx, full_dx, dx_mask);
}

template<int K>
void Stride1Specialist<K>::dw_nci(
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
        __m256 v_acc[STRIDE1_SPECIALIST_K_MAX];
#pragma GCC unroll 16
        for (int kw = 0; kw < K; ++kw) {
            v_acc[kw] = _mm256_setzero_ps();
        }
        for (int64_t oh = 0; oh < conv_out_h; ++oh) {
            const int64_t ih = oh - pad + kh;
            if (ih < 0 || ih >= H) continue;
            const float* __restrict dy_row = &dy_nc[oh * dy_row_stride];
            const float* __restrict x_row  = &x_nc[ih * x_row_stride];
            for (int64_t t = 0; t < ow_tiles; ++t) {
                const int64_t ow = t * FWD_TILE_OW;
                const __m256 dy8 = _mm256_loadu_ps(dy_row + dy_pad_l + ow);
                const int64_t iw0 = x_pad_l + ow - pad;
                __m256 vx[STRIDE1_SPECIALIST_K_MAX];
                stride1_load_kw_windows<K>(x_row, iw0, vx);
#pragma GCC unroll 16
                for (int kw = 0; kw < K; ++kw) {
                    v_acc[kw] = _mm256_fmadd_ps(dy8, vx[kw], v_acc[kw]);
                }
            }
        }
        const int64_t base = kh * K;
#pragma GCC unroll 16
        for (int kw = 0; kw < K; ++kw) {
            dw_slice[base + kw] += _mm256_reduce_add_ps(v_acc[kw]);
        }
    }
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
    if (k_h >= 1 && k_h <= 11) {
        return k_h;
    }
    return 0;
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
    __m256& acc6, __m256& acc7, __m256& acc8,
    __m256& acc9, __m256& acc10
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
    if (K > 7) {
        acc7 = _mm256_fmadd_ps(dy8, stride2_gather_x8_loadu(x_row, iw_base + 7), acc7);
    }
    if (K > 8) {
        acc8 = _mm256_fmadd_ps(dy8, stride2_gather_x8_loadu(x_row, iw_base + 8), acc8);
    }
    if (K > 9) {
        acc9 = _mm256_fmadd_ps(dy8, stride2_gather_x8_loadu(x_row, iw_base + 9), acc9);
    }
    if (K > 10) {
        acc10 = _mm256_fmadd_ps(dy8, stride2_gather_x8_loadu(x_row, iw_base + 10), acc10);
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
            __m256 v_acc7 = _mm256_setzero_ps();
            __m256 v_acc8 = _mm256_setzero_ps();
            __m256 v_acc9 = _mm256_setzero_ps();
            __m256 v_acc10 = _mm256_setzero_ps();

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
                        v_acc0, v_acc1, v_acc2, v_acc3, v_acc4, v_acc5, v_acc6,
                        v_acc7, v_acc8, v_acc9, v_acc10
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
            if (K > 7) dw_slice[base + 7] += _mm256_reduce_add_ps(v_acc7);
            if (K > 8) dw_slice[base + 8] += _mm256_reduce_add_ps(v_acc8);
            if (K > 9) dw_slice[base + 9] += _mm256_reduce_add_ps(v_acc9);
            if (K > 10) dw_slice[base + 10] += _mm256_reduce_add_ps(v_acc10);
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
    } else if (specialist_k == 8) {
        process_bwd_dx_tile_stride2<8>(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
    } else if (specialist_k == 9) {
        process_bwd_dx_tile_stride2<9>(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
    } else if (specialist_k == 10) {
        process_bwd_dx_tile_stride2<10>(
            doc, dy_pad_buf, dy_pad_l, dy_row_stride, W, dx,
            C_in, C_out, W_in_stride, pad, spatial_in, k_spatial,
            conv_out_h, conv_out_w
        );
    } else if (specialist_k == 11) {
        process_bwd_dx_tile_stride2<11>(
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
    const int64_t compute_ow = fwd_compute_ow(out_w, out_w_stride);
    const int64_t cout_blks = (C_out + FWD_TILE_COUT - 1) / FWD_TILE_COUT;
    const int64_t ow_tiles  = (compute_ow + FWD_TILE_OW - 1) / FWD_TILE_OW;
    const int64_t tile_count = N * cout_blks * out_h * ow_tiles;
    const int fwd_nthreads = omp_get_max_threads();
    int64_t fwd_tiles_per_thread[QUEUE_STATS_MAX_THREADS] = {};

    float* x_pad_buf = nullptr;
    int64_t x_pad_l = bwd_x_pad_l(pad);
    int64_t x_row_stride = stride2_x_row_stride(Stride2Specialist<KernelK>::K, pad, W_in, compute_ow);
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
            tid, N, C_out, out_h, compute_ow, 0, out_w
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
    } else if (specialist_k == 8) {
        conv2d_forward_stride2_phase2<8>(
            x, W, out,
            N, C_in, H, W_in, W_in_stride, C_out, pad, out_h, out_w,
            spatial_in, spatial_out, k_spatial, out_w_stride, fwd_stats
        );
    } else if (specialist_k == 9) {
        conv2d_forward_stride2_phase2<9>(
            x, W, out,
            N, C_in, H, W_in, W_in_stride, C_out, pad, out_h, out_w,
            spatial_in, spatial_out, k_spatial, out_w_stride, fwd_stats
        );
    } else if (specialist_k == 10) {
        conv2d_forward_stride2_phase2<10>(
            x, W, out,
            N, C_in, H, W_in, W_in_stride, C_out, pad, out_h, out_w,
            spatial_in, spatial_out, k_spatial, out_w_stride, fwd_stats
        );
    } else if (specialist_k == 11) {
        conv2d_forward_stride2_phase2<11>(
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
    } else if (specialist_k == 8) {
        Stride2Specialist<8>::dw_nci(
            n, cout, cin, dw_slice,
            dy_pad_buf, x_pad_buf,
            C_in, C_out,
            dy_pad_l, dy_row_stride,
            x_pad_l, x_row_stride,
            H, pad,
            conv_out_h, conv_out_w
        );
    } else if (specialist_k == 9) {
        Stride2Specialist<9>::dw_nci(
            n, cout, cin, dw_slice,
            dy_pad_buf, x_pad_buf,
            C_in, C_out,
            dy_pad_l, dy_row_stride,
            x_pad_l, x_row_stride,
            H, pad,
            conv_out_h, conv_out_w
        );
    } else if (specialist_k == 10) {
        Stride2Specialist<10>::dw_nci(
            n, cout, cin, dw_slice,
            dy_pad_buf, x_pad_buf,
            C_in, C_out,
            dy_pad_l, dy_row_stride,
            x_pad_l, x_row_stride,
            H, pad,
            conv_out_h, conv_out_w
        );
    } else if (specialist_k == 11) {
        Stride2Specialist<11>::dw_nci(
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

    // OW-tile / compute-ow specialists (NCHW + x_pad).
    if (fwd_trace_enabled()) {
        std::printf(
            "[FWD_TRACE] dispatch: OW-tile specialists  "
            "N=%lld Cin=%lld Cout=%lld H=%lld Win=%lld K=%lldx%lld pad=%lld "
            "out_w_stride=%lld  SIMD=OW8 tiles FWD_TILE_OW=%lld\n",
            (long long)N, (long long)C_in, (long long)C_out,
            (long long)H, (long long)W_in, (long long)k_h, (long long)k_w,
            (long long)pad, (long long)out_w_stride, (long long)FWD_TILE_OW);
        std::printf(
            "[FWD_TRACE]   layouts stay NCHW + x_pad row buffer; "
            "not nChw8c/OIhw8i8o\n");
        std::fflush(stdout);
    }

    // Single-op NCHW API: OW-tile / compute-ow specialists.
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
        const int64_t compute_ow = fwd_compute_ow(out_w, out_w_stride);
        const int64_t cout_blks = (C_out + FWD_TILE_COUT - 1) / FWD_TILE_COUT;
        const int64_t ow_tiles  = (compute_ow + FWD_TILE_OW - 1) / FWD_TILE_OW;
        const int64_t tile_count = N * cout_blks * out_h * ow_tiles;
        // middle_zone still keyed off logical out_w (halo tiles are edge).
        const int64_t ow_safe_start = std::min(out_w, pad);
        const int64_t ow_safe_end   = std::max(ow_safe_start, out_w - pad);
        const int fwd_nthreads = omp_get_max_threads();
        int64_t fwd_tiles_per_thread[QUEUE_STATS_MAX_THREADS] = {};

        float* x_pad_buf = nullptr;
        int64_t x_pad_l = 0;
        int64_t x_row_stride = 0;
        if (stride1_fwd_builds_x_pad(k_h, k_w)) {
            x_pad_l = bwd_x_pad_l(pad);
            x_row_stride = bwd_x_row_stride(k_w, pad, W_in, compute_ow);
            x_pad_buf = staged_x_pad_lookup(
                x, N, C_in, H, W_in, W_in_stride, x_pad_l, x_row_stride
            );
            if (!x_pad_buf) {
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
        }

        const bool zone_time = fwd_zone_timing_enabled();
        std::atomic<uint64_t> zone_ns[4] = {};
        std::atomic<uint64_t> zone_tiles[4] = {};
        std::atomic<uint64_t> zone_ow_cols[4] = {};
        int64_t zone_mid_lo = 0, zone_mid_hi = 0;
        if (zone_time) {
            fwd_ow_zone_bounds(out_w, pad, &zone_mid_lo, &zone_mid_hi);
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
                tid, N, C_out, out_h, compute_ow, ow_safe_start, ow_safe_end
            );
            if (!zone_time) {
                process_fwd_tile_stride1(
                    doc, x, W, out,
                    C_in, C_out, H, W_in, W_in_stride,
                    k_h, k_w, pad,
                    spatial_in, spatial_out, k_spatial, out_w_stride,
                    x_pad_buf, x_pad_l, x_row_stride
                );
                continue;
            }
            // In-tile split: time pure L/C/R (and compute_ow halo past out_w).
            const int64_t tile_lo = doc.ow;
            const int64_t tile_hi = doc.ow + (int64_t)doc.ow_count;
            const int64_t ranges[4][2] = {
                {0, zone_mid_lo},
                {zone_mid_lo, zone_mid_hi},
                {zone_mid_hi, out_w},
                {out_w, tile_hi > out_w ? tile_hi : out_w},
            };
            for (int z = 0; z < 4; ++z) {
                const int64_t lo = std::max(tile_lo, ranges[z][0]);
                const int64_t hi = std::min(tile_hi, ranges[z][1]);
                if (hi <= lo) continue;
                ConvFwdTileDoc sub = doc;
                sub.ow = lo;
                sub.ow_count = (int8_t)(hi - lo);
                sub.middle_zone = (int8_t)(
                    sub.ow_count == FWD_TILE_OW &&
                    lo >= ow_safe_start &&
                    (lo + FWD_TILE_OW) <= ow_safe_end
                );
                const auto t0 = std::chrono::steady_clock::now();
                process_fwd_tile_stride1(
                    sub, x, W, out,
                    C_in, C_out, H, W_in, W_in_stride,
                    k_h, k_w, pad,
                    spatial_in, spatial_out, k_spatial, out_w_stride,
                    x_pad_buf, x_pad_l, x_row_stride
                );
                const auto t1 = std::chrono::steady_clock::now();
                const uint64_t ns = (uint64_t)std::chrono::duration_cast<
                    std::chrono::nanoseconds>(t1 - t0).count();
                zone_ns[z].fetch_add(ns, std::memory_order_relaxed);
                zone_tiles[z].fetch_add(1, std::memory_order_relaxed);
                zone_ow_cols[z].fetch_add((uint64_t)(hi - lo), std::memory_order_relaxed);
            }
        }

        if (zone_time) {
            static const char* names[4] = {"left", "center", "right", "halo"};
            std::printf(
                "[FWD_ZONE] split-in-tile Cin=%lld Cout=%lld H=%lld Win=%lld "
                "K=%lld pad=%lld out_w=%lld compute_ow=%lld mid=[%lld,%lld)\n",
                (long long)C_in, (long long)C_out, (long long)H, (long long)W_in,
                (long long)k_h, (long long)pad, (long long)out_w,
                (long long)compute_ow,
                (long long)zone_mid_lo, (long long)zone_mid_hi);
            uint64_t total_ns = 0;
            for (int z = 0; z < 4; ++z) total_ns += zone_ns[z].load();
            for (int z = 0; z < 4; ++z) {
                const uint64_t ns = zone_ns[z].load();
                const uint64_t nt = zone_tiles[z].load();
                const uint64_t nc = zone_ow_cols[z].load();
                const double ms = ns * 1e-6;
                const double pct = total_ns ? (100.0 * (double)ns / (double)total_ns) : 0.0;
                const double us_per_col = nc ? (ns * 1e-3 / (double)nc) : 0.0;
                std::printf(
                    "[FWD_ZONE]  %-6s  %7.3f ms  %5.1f%%  subtiles=%llu  "
                    "ow_cols=%llu  %.3f us/ow_col\n",
                    names[z], ms, pct,
                    (unsigned long long)nt, (unsigned long long)nc, us_per_col);
            }
            std::fflush(stdout);
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
        // 64B-aligned base: with an 8-float row stride this puts every padded
        // row start on a 32B boundary, matching the staged buffer.
        const size_t bytes = ((need_floats * sizeof(float)) + 63u) & ~(size_t)63u;
        tls_x_pad_buf = (float*)std::aligned_alloc(64, bytes);
        tls_x_pad_cap_floats = tls_x_pad_buf ? (bytes / sizeof(float)) : 0;
    }
    return tls_x_pad_buf;
}

// ---------------------------------------------------------------------------
// cin-blocked backward-dX: transposed-W staging (32B aligned)
//
// conv2d_backward_dx_cin_blocked_avx2 reads Wt in contiguous 8-wide (C_in)
// groups. Because the gate requires C_in % 8 == 0, every group is exactly
// 32 bytes, so once the buffer base is 32B-aligned every read in the kernel
// lands on a 32B boundary -- loadu was only there because std::vector /
// malloc don't guarantee that alignment. aligned_alloc(32, ...) here plus
// _mm256_load_ps at the call site removes those misaligned loads.
//
// W only changes when Adam applies it, which happens inside the native call
// that just completed -- by the time Python reaches try_submit_step for the
// next job, that Adam apply is done and W is final. Staging here lets the
// (otherwise idle) main thread build Wt instead of the calling thread doing
// it serially inside the OMP call.
// ---------------------------------------------------------------------------
static thread_local float* tls_dx_cin_blocked_wt_buf = nullptr;
static thread_local size_t tls_dx_cin_blocked_wt_cap = 0;

static float* acquire_dx_cin_blocked_wt_buf(size_t need_floats) {
    if (need_floats == 0) return nullptr;
    if (need_floats > tls_dx_cin_blocked_wt_cap) {
        std::free(tls_dx_cin_blocked_wt_buf);
        const size_t bytes = ((need_floats * sizeof(float)) + 31u) & ~(size_t)31u;
        tls_dx_cin_blocked_wt_buf = (float*)std::aligned_alloc(32, bytes);
        tls_dx_cin_blocked_wt_cap = tls_dx_cin_blocked_wt_buf ? (bytes / sizeof(float)) : 0;
    }
    return tls_dx_cin_blocked_wt_buf;
}

static void transpose_dx_cin_blocked_wt(
    const float* __restrict W, float* __restrict dst,
    int64_t C_out, int64_t C_in, int64_t K
) {
    for (int64_t cout = 0; cout < C_out; ++cout) {
        for (int64_t kh = 0; kh < K; ++kh) {
            for (int64_t kw = 0; kw < K; ++kw) {
                float* __restrict d = &dst[((cout * K + kh) * K + kw) * C_in];
                const float* __restrict s = &W[(cout * C_in) * (K * K) + kh * K + kw];
                for (int64_t cin = 0; cin < C_in; ++cin) {
                    d[cin] = s[cin * K * K];
                }
            }
        }
    }
}

constexpr int32_t STAGED_DX_WT_MAX_SLOTS = 4;

struct StagedDxWt {
    const float* w_src = nullptr;
    int64_t C_out = 0;
    int64_t C_in = 0;
    int64_t K = 0;
    float* buf = nullptr;
    size_t cap_floats = 0;
    bool valid = false;
};

static StagedDxWt g_staged_dx_wt[STAGED_DX_WT_MAX_SLOTS];

static float* staged_dx_wt_alloc(StagedDxWt& slot, size_t need_floats) {
    if (slot.buf && need_floats <= slot.cap_floats) {
        return slot.buf;
    }
    std::free(slot.buf);
    const size_t bytes = ((need_floats * sizeof(float)) + 31u) & ~(size_t)31u;
    slot.buf = (float*)std::aligned_alloc(32, bytes);
    slot.cap_floats = slot.buf ? (bytes / sizeof(float)) : 0;
    return slot.buf;
}

static float* staged_dx_wt_lookup(
    const float* w_src, int64_t C_out, int64_t C_in, int64_t K
) {
    for (int32_t i = 0; i < STAGED_DX_WT_MAX_SLOTS; ++i) {
        const StagedDxWt& s = g_staged_dx_wt[i];
        if (!s.valid || !s.buf || s.w_src != w_src) continue;
        if (s.C_out != C_out || s.C_in != C_in || s.K != K) continue;
        return s.buf;
    }
    return nullptr;
}

// Called from Python's main thread, right before submitting the job that
// will read this layer's W. At that point Adam has already applied W for
// the job that just completed, so its contents are final.
extern "C" ML_ENGINE_EXPORT int32_t stage_dx_cin_blocked_wt(
    int32_t slot_idx, const float* W, int64_t C_out, int64_t C_in, int64_t K
) {
    if (slot_idx < 0 || slot_idx >= STAGED_DX_WT_MAX_SLOTS) return -1;
    StagedDxWt& slot = g_staged_dx_wt[slot_idx];
    slot.valid = false;
    if (!W || C_out <= 0 || C_in <= 0 || K <= 0) return -2;

    const size_t need = (size_t)(C_out * K * K * C_in);
    float* buf = staged_dx_wt_alloc(slot, need);
    if (!buf) return -3;

    transpose_dx_cin_blocked_wt(W, buf, C_out, C_in, K);

    slot.w_src = W;
    slot.C_out = C_out;
    slot.C_in = C_in;
    slot.K = K;
    slot.valid = true;
    return 0;
}

extern "C" ML_ENGINE_EXPORT void invalidate_dx_cin_blocked_wt_stage(int32_t slot_idx) {
    if (slot_idx < 0) {
        for (int32_t i = 0; i < STAGED_DX_WT_MAX_SLOTS; ++i) {
            g_staged_dx_wt[i].valid = false;
        }
    } else if (slot_idx < STAGED_DX_WT_MAX_SLOTS) {
        g_staged_dx_wt[slot_idx].valid = false;
    }
}

// --- Option B experiment: cin-blocked backward-dX for stride-1 K=7 -----------
// Diagnostic-scoped, narrowly gated (stride==1 && k_h==k_w==7 && C_in%8==0).
// Verified against a naive scalar reference in a standalone microbenchmark
// (benchmark_diagnostics/scratch/option_b_microbench.cpp) and, before trusting
// this in the real engine, against the existing crawl path on real inputs via
// ML_ENGINE_FORCE_DX_CRAWL (see verify script). See summary discussion:
// cin is the free axis for backward-dX (cout is reduced), so W is
// pre-transposed to [cout][kh][kw][cin_block] and dY is read as a scalar
// broadcast — this removes the sliding-window vector crawl entirely rather
// than just reducing its instruction count (which Option A did, and which did
// not move throughput).
static bool force_dx_crawl_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("ML_ENGINE_FORCE_DX_CRAWL");
        cached = (env && env[0] == '1' && env[1] == '\0') ? 1 : 0;
    }
    return cached != 0;
}

template <int K>
static void conv2d_backward_dx_cin_blocked_avx2(
    const float* d_conv_buf, const float* W, float* dx,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t pad, int64_t conv_out_h, int64_t conv_out_w,
    int64_t conv_out_w_stride
) {
    const int64_t cin_blocks = C_in / 8;
    const int64_t conv_spatial = conv_out_h * conv_out_w_stride;

    // Pre-transpose W [C_out][C_in][K][K] -> Wt [C_out][K][K][C_in] once per
    // backward call (small: C_out*K*K*C_in floats; cheap vs. the O(N*H*W*
    // C_in*C_out*K*K) accumulation below). The main thread may have already
    // staged this from the finalized weights (see stage_dx_cin_blocked_wt);
    // on a miss it's rebuilt here into the same 32B-aligned buffer kind, so
    // the aligned loads below are safe either way.
    float* Wt = staged_dx_wt_lookup(W, C_out, C_in, K);
    if (!Wt) {
        Wt = acquire_dx_cin_blocked_wt_buf((size_t)(C_out * K * K * C_in));
        transpose_dx_cin_blocked_wt(W, Wt, C_out, C_in, K);
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t ih = 0; ih < H; ++ih) {
            for (int64_t cb = 0; cb < cin_blocks; ++cb) {
                for (int64_t iw = 0; iw < W_in; ++iw) {
                    __m256 acc = _mm256_setzero_ps();

                    // oh/ow run backwards over kh/kw, so the taps that land in
                    // range form one contiguous window per (ih, iw). Those
                    // bounds do not depend on cout or cb, so they are derived
                    // once here instead of re-testing every tap in the innermost
                    // loop.
                    int64_t kh_lo, kh_hi;
                    bwd_dx_k_kh_bounds<K>(ih, pad, conv_out_h, kh_lo, kh_hi);
                    int64_t kw_lo = iw + pad - conv_out_w + 1;
                    if (kw_lo < 0) kw_lo = 0;
                    int64_t kw_hi = iw + pad + 1;
                    if (kw_hi > K) kw_hi = K;

                    for (int64_t cout = 0; cout < C_out; ++cout) {
                        const float* __restrict dy_plane =
                            &d_conv_buf[(n * C_out + cout) * conv_spatial];
                        for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
                            const int64_t oh = ih - kh + pad;
                            const float* __restrict dy_row = &dy_plane[oh * conv_out_w_stride];
                            const float* __restrict wt_tap = &Wt[((cout * K + kh) * K) * C_in + cb * 8];
                            for (int64_t kw = kw_lo; kw < kw_hi; ++kw) {
                                const float dy = dy_row[iw - kw + pad];
                                // Safe as an aligned load: C_in % 8 == 0 (gate
                                // below) and Wt is 32B-aligned, so every
                                // kw*C_in-float offset lands on a 32B boundary.
                                const __m256 wv = _mm256_load_ps(wt_tap + kw * C_in);
                                acc = _mm256_fmadd_ps(_mm256_set1_ps(dy), wv, acc);
                            }
                        }
                    }
                    alignas(32) float lanes[8];
                    _mm256_storeu_ps(lanes, acc);
                    float* __restrict dx_base =
                        &dx[(n * C_in + cb * 8) * H * W_in_stride + ih * W_in_stride + iw];
                    const int64_t cin_plane = H * W_in_stride;
                    for (int64_t c = 0; c < 8; ++c) {
                        dx_base[c * cin_plane] = lanes[c];
                    }
                }
            }
        }
    }
}

// Dispatch, mirroring stride1_specialist_k()/stride1_try_*_specialist().
// Square K in [1, DX_BLOCKED_K_MAX] for Cin%8==0 layers (fallback when BRGEMM
// does not take the call). K>7 without a Stride1Specialist still land here.
static constexpr int64_t DX_BLOCKED_K_MAX = 11;

static inline bool try_cin_blocked_dx(
    int64_t k_h, int64_t k_w, int64_t stride, int64_t C_in,
    const float* d_conv_buf, const float* W, float* dx,
    int64_t N, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t pad, int64_t conv_out_h, int64_t conv_out_w,
    int64_t conv_out_w_stride
) {
    if (stride != 1 || k_h != k_w || (C_in % 8) != 0) {
        return false;
    }
    if (k_h < 1 || k_h > DX_BLOCKED_K_MAX) {
        return false;
    }
    switch (k_h) {
        case 1:
            conv2d_backward_dx_cin_blocked_avx2<1>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        case 2:
            conv2d_backward_dx_cin_blocked_avx2<2>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        case 3:
            conv2d_backward_dx_cin_blocked_avx2<3>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        case 4:
            conv2d_backward_dx_cin_blocked_avx2<4>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        case 5:
            conv2d_backward_dx_cin_blocked_avx2<5>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        case 6:
            conv2d_backward_dx_cin_blocked_avx2<6>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        case 7:
            conv2d_backward_dx_cin_blocked_avx2<7>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        case 8:
            conv2d_backward_dx_cin_blocked_avx2<8>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        case 9:
            conv2d_backward_dx_cin_blocked_avx2<9>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        case 10:
            conv2d_backward_dx_cin_blocked_avx2<10>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        case 11:
            conv2d_backward_dx_cin_blocked_avx2<11>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        default:
            return false;
    }
}
// --- end Option B experiment --------------------------------------------------

// --- Option D experiment: C_out-blocked backward-dX ---------------------------
// For layers where C_in fails Option B's C_in%8==0 gate (e.g. layer 0: C_in=3)
// but C_out is a clean AVX2 width (C_out%8==0), vectorize the reduction over
// output channels at a fixed spatial position instead of over input channels.
// Requires dY transposed to channel-last [N][H_out][W_out][C_out] and W
// transposed to [C_in][K][K][C_out] (both done once per call, amortized).
// Template is K-generic; dispatch covers square K in [1, DX_BLOCKED_K_MAX].
// Gated behind ML_ENGINE_FORCE_DX_CRAWL like Option B for A/B testing.
static bool dx_cout_blocked_disabled() {
    static bool val = [] {
        const char* v = std::getenv("ML_ENGINE_DISABLE_DX_COUT_BLOCKED");
        return v && v[0] == '1';
    }();
    return val;
}

static thread_local float* tls_dx_cout_blocked_dy_b_buf = nullptr;
static thread_local size_t tls_dx_cout_blocked_dy_b_cap = 0;
static thread_local float* tls_dx_cout_blocked_wt_buf = nullptr;
static thread_local size_t tls_dx_cout_blocked_wt_cap = 0;

static float* acquire_dx_cout_blocked_dy_b_buf(size_t need_floats) {
    if (need_floats == 0) return nullptr;
    if (need_floats > tls_dx_cout_blocked_dy_b_cap) {
        std::free(tls_dx_cout_blocked_dy_b_buf);
        tls_dx_cout_blocked_dy_b_buf = (float*)std::malloc(need_floats * sizeof(float));
        tls_dx_cout_blocked_dy_b_cap = tls_dx_cout_blocked_dy_b_buf ? need_floats : 0;
    }
    return tls_dx_cout_blocked_dy_b_buf;
}

static float* acquire_dx_cout_blocked_wt_buf(size_t need_floats) {
    if (need_floats == 0) return nullptr;
    if (need_floats > tls_dx_cout_blocked_wt_cap) {
        std::free(tls_dx_cout_blocked_wt_buf);
        tls_dx_cout_blocked_wt_buf = (float*)std::malloc(need_floats * sizeof(float));
        tls_dx_cout_blocked_wt_cap = tls_dx_cout_blocked_wt_buf ? need_floats : 0;
    }
    return tls_dx_cout_blocked_wt_buf;
}

template <int K>
static void conv2d_backward_dx_cout_blocked_avx2(
    const float* d_conv_buf, const float* W, float* dx,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t pad, int64_t conv_out_h, int64_t conv_out_w,
    int64_t conv_out_w_stride
) {
    // Only C_out==8 (a single 8-wide block) is implemented/validated; the
    // dispatcher below enforces this exactly.
    const int64_t conv_spatial = conv_out_h * conv_out_w_stride;

    // dY_b: [N][conv_out_h][conv_out_w][C_out] (channel-last; contiguous 8-wide
    // groups for aligned loads). Built once per call from d_conv_buf (NCHW).
    // Reused thread_local buffer (same pattern as acquire_bwd_dy_pad_buf below)
    // instead of a fresh std::vector/malloc every call.
    const size_t dY_b_floats = (size_t)(N * conv_out_h * conv_out_w * C_out);
    float* __restrict dY_b = acquire_dx_cout_blocked_dy_b_buf(dY_b_floats);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t cout = 0; cout < C_out; ++cout) {
            const float* __restrict src = &d_conv_buf[(n * C_out + cout) * conv_spatial];
            for (int64_t oh = 0; oh < conv_out_h; ++oh) {
                const float* __restrict src_row = src + oh * conv_out_w_stride;
                float* __restrict dst_row =
                    &dY_b[((n * conv_out_h + oh) * conv_out_w) * C_out + cout];
                for (int64_t ow = 0; ow < conv_out_w; ++ow) {
                    dst_row[ow * C_out] = src_row[ow];
                }
            }
        }
    }

    // Wt: [C_in][K][K][C_out] (channel-last on C_out; contiguous 8-wide groups).
    // Tiny (C_in*K*K*C_out floats); kept as a reused thread_local buffer too,
    // but left serial -- not worth parallelizing at this size.
    const size_t Wt_floats = (size_t)(C_in * K * K * C_out);
    float* __restrict Wt = acquire_dx_cout_blocked_wt_buf(Wt_floats);
    for (int64_t cin = 0; cin < C_in; ++cin) {
        for (int64_t kh = 0; kh < K; ++kh) {
            for (int64_t kw = 0; kw < K; ++kw) {
                float* __restrict dst = &Wt[((cin * K + kh) * K + kw) * C_out];
                for (int64_t cout = 0; cout < C_out; ++cout) {
                    dst[cout] = W[((cout * C_in + cin) * K + kh) * K + kw];
                }
            }
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t ih = 0; ih < H; ++ih) {
            // oh = ih - kh + pad must land in [0, conv_out_h): solve for kh once.
            const int64_t kh_lo = std::max<int64_t>(0, ih + pad - (conv_out_h - 1));
            const int64_t kh_hi = std::min<int64_t>(K - 1, ih + pad);
            for (int64_t cin = 0; cin < C_in; ++cin) {
                const float* __restrict w_cin = &Wt[cin * K * K * C_out];
                float* __restrict dx_row =
                    &dx[(n * C_in + cin) * H * W_in_stride + ih * W_in_stride];
                for (int64_t iw = 0; iw < W_in; ++iw) {
                    const int64_t kw_lo = std::max<int64_t>(0, iw + pad - (conv_out_w - 1));
                    const int64_t kw_hi = std::min<int64_t>(K - 1, iw + pad);
                    __m256 acc = _mm256_setzero_ps();
                    for (int64_t kh = kh_lo; kh <= kh_hi; ++kh) {
                        const int64_t oh = ih - kh + pad;
                        for (int64_t kw = kw_lo; kw <= kw_hi; ++kw) {
                            const int64_t ow = iw - kw + pad;
                            const float* __restrict dy =
                                &dY_b[((n * conv_out_h + oh) * conv_out_w + ow) * C_out];
                            const float* __restrict ww =
                                &w_cin[(kh * K + kw) * C_out];
                            acc = _mm256_fmadd_ps(
                                _mm256_loadu_ps(dy), _mm256_loadu_ps(ww), acc);
                        }
                    }
                    dx_row[iw] += _mm256_reduce_add_ps(acc);
                }
            }
        }
    }
}

// Dispatch: Option B miss (C_in%8!=0) + C_out==8 + square K in [1, DX_BLOCKED_K_MAX].
// This is the L0 (Cin=3) fast path; K=7-only was why K=6 crawled.
static inline bool try_cout_blocked_dx(
    int64_t k_h, int64_t k_w, int64_t stride, int64_t C_out,
    const float* d_conv_buf, const float* W, float* dx,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t pad, int64_t conv_out_h, int64_t conv_out_w,
    int64_t conv_out_w_stride
) {
    if (dx_cout_blocked_disabled()) {
        return false;
    }
    if (stride != 1 || k_h != k_w || C_out != 8) {
        return false; // only C_out==8 (single 8-wide block) is implemented
    }
    if (k_h < 1 || k_h > DX_BLOCKED_K_MAX) {
        return false;
    }
    switch (k_h) {
        case 1:
            conv2d_backward_dx_cout_blocked_avx2<1>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        case 2:
            conv2d_backward_dx_cout_blocked_avx2<2>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        case 3:
            conv2d_backward_dx_cout_blocked_avx2<3>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        case 4:
            conv2d_backward_dx_cout_blocked_avx2<4>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        case 5:
            conv2d_backward_dx_cout_blocked_avx2<5>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        case 6:
            conv2d_backward_dx_cout_blocked_avx2<6>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        case 7:
            conv2d_backward_dx_cout_blocked_avx2<7>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        case 8:
            conv2d_backward_dx_cout_blocked_avx2<8>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        case 9:
            conv2d_backward_dx_cout_blocked_avx2<9>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        case 10:
            conv2d_backward_dx_cout_blocked_avx2<10>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        case 11:
            conv2d_backward_dx_cout_blocked_avx2<11>(
                d_conv_buf, W, dx, N, C_in, H, W_in, W_in_stride,
                C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
            );
            return true;
        default:
            return false;
    }
}
// --- end Option D experiment ---------------------------------------------------

// --- BRGEMM-style dW (bwd_w loop shape, AVX2, no full im2col) ----------
// Matches brgemm convolution bwd_weights structure for our shapes:
//   1) pack src / diff_dst so channel blocks are contiguous
//   2) for each (kh,kw): C[ic_block,oc_block] += Σ_{n,oh} A(oh) @ B(oh)
//      where spatial width is the GEMM K and (n,oh) is the BRGEMM batch
// Gates: stride-1, square K in [1, 11], Cin%8==0, Cout%8==0 (L1-style layers).
// Layer-0 (Cin=3) keeps the existing specialist path.
// ---------------------------------------------------------------------------
static constexpr int64_t BRG_DW_IC = 8;
static constexpr int64_t BRG_DW_OC = 8;
static constexpr int32_t BRG_DW_X_STAGE_MAX = 8;
static constexpr int64_t BRG_K_MAX = DX_BLOCKED_K_MAX;

struct BrgDwXStage {
    const float* src = nullptr;
    int64_t N = 0;
    int64_t C_in = 0;
    int64_t H = 0;
    int64_t W_in = 0;
    int64_t src_row_stride = 0;
    int64_t x_pad_l = 0;
    int64_t W_ext = 0;
    float* buf = nullptr;
    size_t cap_floats = 0;
    bool valid = false;
};

static BrgDwXStage g_brg_dw_x_stage[BRG_DW_X_STAGE_MAX];

static thread_local float* tls_brg_dy_pack = nullptr;
static thread_local size_t tls_brg_dy_cap = 0;
static thread_local float* tls_brg_x_pack = nullptr;
static thread_local size_t tls_brg_x_cap = 0;
// Per-thread nChw8c slab for brgemm dX: [H][W_in][8]. Avoids planar
// scatter RMW inside (kh,kw,oc,ow); unpack to NCHW once per (n,ic_b).
static thread_local float* tls_brg_dx_blocked = nullptr;
static thread_local size_t tls_brg_dx_blocked_cap = 0;

static float* brg_dw_alloc(float*& slot, size_t& cap, size_t need) {
    if (need <= cap && slot) return slot;
    std::free(slot);
    const size_t bytes = ((need * sizeof(float)) + 63u) & ~(size_t)63u;
    slot = (float*)std::aligned_alloc(64, bytes);
    cap = slot ? (bytes / sizeof(float)) : 0;
    return slot;
}

static float* brg_dw_stage_alloc(BrgDwXStage& slot, size_t need) {
    if (slot.buf && need <= slot.cap_floats) return slot.buf;
    std::free(slot.buf);
    const size_t bytes = ((need * sizeof(float)) + 63u) & ~(size_t)63u;
    slot.buf = (float*)std::aligned_alloc(64, bytes);
    slot.cap_floats = slot.buf ? (bytes / sizeof(float)) : 0;
    return slot.buf;
}

static bool brg_dw_x_geom_ok(
    int64_t C_in, int64_t C_out, int64_t k_h, int64_t k_w, int64_t stride
) {
    // Square K only: dX Wt transpose is K×K. Pack/dW loops are already generic.
    return stride == 1
        && k_h == k_w
        && k_h >= 1 && k_h <= BRG_K_MAX
        && (C_in % BRG_DW_IC) == 0 && (C_out % BRG_DW_OC) == 0;
}

// Diagnostic-only: ML_ENGINE_BWD_PACK_TIMING=1 accumulates wall ns for brgemm
// bwd pack vs compute. Zero cost when unset. Dump via dump_bwd_pack_timing().
static bool bwd_pack_timing_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("ML_ENGINE_BWD_PACK_TIMING");
        cached = (env && env[0] == '1' && env[1] == '\0') ? 1 : 0;
    }
    return cached != 0;
}

static std::atomic<uint64_t> g_bwd_pack_calls{0};
static std::atomic<uint64_t> g_bwd_ns_dy_pack{0};
static std::atomic<uint64_t> g_bwd_ns_x_pack{0};
static std::atomic<uint64_t> g_bwd_ns_dw_compute{0};
static std::atomic<uint64_t> g_bwd_ns_dx_wt{0};
static std::atomic<uint64_t> g_bwd_ns_dx_compute{0};
static std::atomic<uint64_t> g_bwd_ns_fused_total{0};
static std::atomic<uint64_t> g_bwd_x_pack_hits{0};
static std::atomic<uint64_t> g_bwd_x_pack_misses{0};
static std::atomic<uint64_t> g_bwd_dy_pack_shared{0};
static std::atomic<uint64_t> g_bwd_dy_pack_local{0};

static inline uint64_t bwd_pack_now_ns() {
    using clock = std::chrono::steady_clock;
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        clock::now().time_since_epoch()).count();
}

extern "C" ML_ENGINE_EXPORT void dump_bwd_pack_timing(void) {
    const uint64_t calls = g_bwd_pack_calls.load(std::memory_order_relaxed);
    if (calls == 0) {
        std::fprintf(stderr, "[BWD_PACK_TIMING] no samples\n");
        std::fflush(stderr);
        return;
    }
    const double inv = 1e-6 / (double)calls; // ns → ms per call
    std::fprintf(
        stderr,
        "[BWD_PACK_TIMING] calls=%llu  dy_pack=%.3fms  x_pack=%.3fms "
        "(hit=%llu miss=%llu)  dw_compute=%.3fms  dx_wt=%.3fms  "
        "dx_compute=%.3fms  fused_total=%.3fms  "
        "dy_shared=%llu dy_local=%llu\n",
        (unsigned long long)calls,
        g_bwd_ns_dy_pack.load() * inv,
        g_bwd_ns_x_pack.load() * inv,
        (unsigned long long)g_bwd_x_pack_hits.load(),
        (unsigned long long)g_bwd_x_pack_misses.load(),
        g_bwd_ns_dw_compute.load() * inv,
        g_bwd_ns_dx_wt.load() * inv,
        g_bwd_ns_dx_compute.load() * inv,
        g_bwd_ns_fused_total.load() * inv,
        (unsigned long long)g_bwd_dy_pack_shared.load(),
        (unsigned long long)g_bwd_dy_pack_local.load()
    );
    std::fflush(stderr);
}

extern "C" ML_ENGINE_EXPORT void reset_bwd_pack_timing(void) {
    g_bwd_pack_calls.store(0);
    g_bwd_ns_dy_pack.store(0);
    g_bwd_ns_x_pack.store(0);
    g_bwd_ns_dw_compute.store(0);
    g_bwd_ns_dx_wt.store(0);
    g_bwd_ns_dx_compute.store(0);
    g_bwd_ns_fused_total.store(0);
    g_bwd_x_pack_hits.store(0);
    g_bwd_x_pack_misses.store(0);
    g_bwd_dy_pack_shared.store(0);
    g_bwd_dy_pack_local.store(0);
}

// x: NCHW -> [N][nb_ic][H][W_ext][8]. parallel=true uses OMP team; false = main/serial.
static void brg_pack_x_blocked(
    const float* __restrict x,
    float* __restrict out,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t x_pad_l, int64_t W_ext,
    bool parallel
) {
    const int64_t nb_ic = C_in / BRG_DW_IC;
    const int64_t plane = H * W_ext * BRG_DW_IC;
    if (parallel) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t ib = 0; ib < nb_ic; ++ib) {
                for (int64_t ih = 0; ih < H; ++ih) {
                    float* __restrict dst =
                        &out[(n * nb_ic + ib) * plane
                             + ih * W_ext * BRG_DW_IC];
                    std::memset(
                        dst, 0, (size_t)W_ext * BRG_DW_IC * sizeof(float));
                    for (int64_t iw = 0; iw < W_in; ++iw) {
                        float* __restrict d = &dst[(x_pad_l + iw) * BRG_DW_IC];
                        for (int64_t c = 0; c < BRG_DW_IC; ++c) {
                            const int64_t cin = ib * BRG_DW_IC + c;
                            d[c] = x[(n * C_in + cin) * H * W_in_stride
                                     + ih * W_in_stride + iw];
                        }
                    }
                }
            }
        }
    } else {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t ib = 0; ib < nb_ic; ++ib) {
                for (int64_t ih = 0; ih < H; ++ih) {
                    float* __restrict dst =
                        &out[(n * nb_ic + ib) * plane
                             + ih * W_ext * BRG_DW_IC];
                    std::memset(
                        dst, 0, (size_t)W_ext * BRG_DW_IC * sizeof(float));
                    for (int64_t iw = 0; iw < W_in; ++iw) {
                        float* __restrict d = &dst[(x_pad_l + iw) * BRG_DW_IC];
                        for (int64_t c = 0; c < BRG_DW_IC; ++c) {
                            const int64_t cin = ib * BRG_DW_IC + c;
                            d[c] = x[(n * C_in + cin) * H * W_in_stride
                                     + ih * W_in_stride + iw];
                        }
                    }
                }
            }
        }
    }
}

static float* brg_dw_x_stage_lookup(
    const float* src, int64_t N, int64_t C_in, int64_t H, int64_t W_in,
    int64_t src_row_stride, int64_t x_pad_l, int64_t W_ext
) {
    for (int32_t i = 0; i < BRG_DW_X_STAGE_MAX; ++i) {
        const BrgDwXStage& s = g_brg_dw_x_stage[i];
        if (!s.valid || !s.buf || s.src != src) continue;
        if (s.N != N || s.C_in != C_in || s.H != H || s.W_in != W_in) continue;
        if (s.src_row_stride != src_row_stride || s.x_pad_l != x_pad_l
            || s.W_ext != W_ext) {
            continue;
        }
        return s.buf;
    }
    return nullptr;
}

// Fill a durable slot (main-thread prepare, or publish after fwd).
// parallel follows async overlap: OMP when sync, serial when overlapping.
static int32_t brg_dw_x_stage_store(
    int32_t slot_idx, const float* x,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t k_w, int64_t pad
) {
    if (slot_idx < 0 || slot_idx >= BRG_DW_X_STAGE_MAX) return -1;
    if (!x || N <= 0 || (C_in % BRG_DW_IC) != 0) return -2;
    BrgDwXStage& slot = g_brg_dw_x_stage[slot_idx];
    slot.valid = false;
    const int64_t x_pad_l = pad;
    const int64_t W_ext = x_pad_l + W_in + (k_w - 1);
    const int64_t nb_ic = C_in / BRG_DW_IC;
    const size_t need = (size_t)N * (size_t)nb_ic * (size_t)H
        * (size_t)W_ext * (size_t)BRG_DW_IC;
    float* buf = brg_dw_stage_alloc(slot, need);
    if (!buf) return -3;
    if (main_stage_use_omp()) {
        ml_omp_before_parallel();
    }
    brg_pack_x_blocked(
        x, buf, N, C_in, H, W_in, W_in_stride, x_pad_l, W_ext,
        /*parallel=*/main_stage_use_omp()
    );
    slot.src = x;
    slot.N = N;
    slot.C_in = C_in;
    slot.H = H;
    slot.W_in = W_in;
    slot.src_row_stride = W_in_stride;
    slot.x_pad_l = x_pad_l;
    slot.W_ext = W_ext;
    slot.valid = true;
    return 0;
}

// Main thread: prepare_step overlap (layer-0 when Cin%8==0).
extern "C" ML_ENGINE_EXPORT int32_t stage_brgemm_dw_x_pack(
    int32_t slot_idx, const float* x,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t k_w, int64_t pad
) {
    return brg_dw_x_stage_store(
        slot_idx, x, N, C_in, H, W_in, W_in_stride, k_w, pad
    );
}

// After conv fwd: publish this layer's x pack for its upcoming dW (L1 path).
extern "C" ML_ENGINE_EXPORT int32_t publish_brgemm_dw_x_pack(
    const float* x,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t k_h, int64_t k_w, int64_t stride, int64_t pad
) {
    if (!brg_dw_x_geom_ok(C_in, C_out, k_h, k_w, stride)) return 1; // skip
    // Prefer an existing entry for this src, else first free, else slot 0.
    int32_t slot_i = -1;
    int32_t free_i = -1;
    for (int32_t i = 0; i < BRG_DW_X_STAGE_MAX; ++i) {
        if (g_brg_dw_x_stage[i].src == x) {
            slot_i = i;
            break;
        }
        if (free_i < 0 && !g_brg_dw_x_stage[i].valid) free_i = i;
    }
    if (slot_i < 0) slot_i = (free_i >= 0) ? free_i : 0;
    return brg_dw_x_stage_store(
        slot_i, x, N, C_in, H, W_in, W_in_stride, k_w, pad
    );
}

extern "C" ML_ENGINE_EXPORT void invalidate_brgemm_dw_x_pack(int32_t slot_idx) {
    if (slot_idx < 0) {
        for (int32_t i = 0; i < BRG_DW_X_STAGE_MAX; ++i) {
            g_brg_dw_x_stage[i].valid = false;
        }
    } else if (slot_idx < BRG_DW_X_STAGE_MAX) {
        g_brg_dw_x_stage[slot_idx].valid = false;
    }
}

// Mid-step pack handoff: async worker posts; Python main packs while OMP runs.
enum : int32_t {
    BRG_PACK_IDLE = 0,
    BRG_PACK_PENDING = 1,
};

struct BrgPackRequest {
    int32_t state = BRG_PACK_IDLE;
    const float* x = nullptr;
    int64_t N = 0;
    int64_t C_in = 0;
    int64_t H = 0;
    int64_t W_in = 0;
    int64_t W_in_stride = 0;
    int64_t C_out = 0;
    int64_t k_h = 0;
    int64_t k_w = 0;
    int64_t stride = 0;
    int64_t pad = 0;
};

static BrgPackRequest g_brg_pack_req;
static std::mutex g_brg_pack_mtx;

// Worker: ask main to pack this x for upcoming BRGEMM dW. Non-blocking.
// Returns 0 if queued/already staged, 1 if geometry skips, <0 on error.
extern "C" ML_ENGINE_EXPORT int32_t request_brgemm_dw_x_pack(
    const float* x,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t k_h, int64_t k_w, int64_t stride, int64_t pad
) {
    if (!brg_dw_x_geom_ok(C_in, C_out, k_h, k_w, stride) || !x || N <= 0) {
        return 1;
    }
    const int64_t x_pad_l = pad;
    const int64_t W_ext = x_pad_l + W_in + (k_w - 1);
    if (brg_dw_x_stage_lookup(
            x, N, C_in, H, W_in, W_in_stride, x_pad_l, W_ext
        )) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(g_brg_pack_mtx);
    if (g_brg_pack_req.state == BRG_PACK_PENDING) {
        if (g_brg_pack_req.x == x) return 0;
        return -2; // prior request still waiting on main
    }
    g_brg_pack_req.x = x;
    g_brg_pack_req.N = N;
    g_brg_pack_req.C_in = C_in;
    g_brg_pack_req.H = H;
    g_brg_pack_req.W_in = W_in;
    g_brg_pack_req.W_in_stride = W_in_stride;
    g_brg_pack_req.C_out = C_out;
    g_brg_pack_req.k_h = k_h;
    g_brg_pack_req.k_w = k_w;
    g_brg_pack_req.stride = stride;
    g_brg_pack_req.pad = pad;
    g_brg_pack_req.state = BRG_PACK_PENDING;
    return 0;
}

// Main/Python: drain one pending pack request (serial pack into durable stage).
// Returns 1 if packed, 0 if nothing pending, <0 on failure.
extern "C" ML_ENGINE_EXPORT int32_t service_brgemm_dw_x_pack_requests(void) {
    const float* x = nullptr;
    int64_t N = 0, C_in = 0, H = 0, W_in = 0, W_in_stride = 0;
    int64_t C_out = 0, k_h = 0, k_w = 0, stride = 0, pad = 0;
    {
        std::lock_guard<std::mutex> lock(g_brg_pack_mtx);
        if (g_brg_pack_req.state != BRG_PACK_PENDING) return 0;
        x = g_brg_pack_req.x;
        N = g_brg_pack_req.N;
        C_in = g_brg_pack_req.C_in;
        H = g_brg_pack_req.H;
        W_in = g_brg_pack_req.W_in;
        W_in_stride = g_brg_pack_req.W_in_stride;
        C_out = g_brg_pack_req.C_out;
        k_h = g_brg_pack_req.k_h;
        k_w = g_brg_pack_req.k_w;
        stride = g_brg_pack_req.stride;
        pad = g_brg_pack_req.pad;
    }
    const int32_t rc = publish_brgemm_dw_x_pack(
        x, N, C_in, H, W_in, W_in_stride, C_out, k_h, k_w, stride, pad
    );
    {
        std::lock_guard<std::mutex> lock(g_brg_pack_mtx);
        // Only clear if this is still the same pending request.
        if (g_brg_pack_req.state == BRG_PACK_PENDING && g_brg_pack_req.x == x) {
            g_brg_pack_req.state = BRG_PACK_IDLE;
        }
    }
    return (rc < 0) ? rc : 1;
}

extern "C" ML_ENGINE_EXPORT void reset_brgemm_dw_x_pack_request(void) {
    std::lock_guard<std::mutex> lock(g_brg_pack_mtx);
    g_brg_pack_req.state = BRG_PACK_IDLE;
}

// dy: NCHW planar cout -> [N][OH][nb_oc][OW][8]  (ow-major, then oc in block)
static void brg_pack_dy_blocked(
    const float* __restrict d_conv,
    float* __restrict out,
    int64_t N, int64_t C_out, int64_t OH, int64_t OW,
    int64_t conv_out_w_stride, int64_t conv_spatial
) {
    const int64_t nb_oc = C_out / BRG_DW_OC;
    #pragma omp parallel for collapse(3) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t oh = 0; oh < OH; ++oh) {
            for (int64_t ob = 0; ob < nb_oc; ++ob) {
                float* __restrict dst =
                    &out[(((n * OH + oh) * nb_oc + ob) * OW) * BRG_DW_OC];
                for (int64_t ow = 0; ow < OW; ++ow) {
                    for (int64_t c = 0; c < BRG_DW_OC; ++c) {
                        const int64_t cout = ob * BRG_DW_OC + c;
                        dst[ow * BRG_DW_OC + c] = d_conv[
                            (n * C_out + cout) * conv_spatial
                            + oh * conv_out_w_stride + ow];
                    }
                }
            }
        }
    }
}

// C[8][8] += A[8][OW] * B[OW][8]  (A: ic x ow, B: ow x oc)
static inline void brg_dw_gemm_ic8_oc8(
    const float* __restrict A, // ic-major: A[ci * OW + ow]
    const float* __restrict B, // ow-major: B[ow * 8 + co]
    int64_t OW,
    __m256* __restrict Crow // 8 accumulators, one per ic lane, each holds 8 oc
) {
    for (int64_t ow = 0; ow < OW; ++ow) {
        const __m256 bv = _mm256_loadu_ps(B + ow * BRG_DW_OC);
        for (int64_t ci = 0; ci < BRG_DW_IC; ++ci) {
            Crow[ci] = _mm256_fmadd_ps(
                _mm256_set1_ps(A[ci * OW + ow]), bv, Crow[ci]);
        }
    }
}

static bool try_brgemm_style_dw(
    const float* d_conv_buf, const float* x, float* dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t k_h, int64_t k_w, int64_t stride, int64_t pad,
    int64_t conv_out_w_stride, float inv_m, int64_t dw_prezeroed,
    float* dy_pack_opt
) {
    if (!brg_dw_x_geom_ok(C_in, C_out, k_h, k_w, stride)) return false;
    if (!dW || !x || !d_conv_buf || N <= 0) return false;

    const int64_t OH = (H + 2 * pad - k_h) / stride + 1;
    const int64_t OW = (W_in + 2 * pad - k_w) / stride + 1;
    if (OH <= 0 || OW <= 0 || OW > 64) return false;

    const int64_t conv_spatial = OH * conv_out_w_stride;
    const int64_t k_spatial = k_h * k_w;
    const int64_t nb_ic = C_in / BRG_DW_IC;
    const int64_t nb_oc = C_out / BRG_DW_OC;
    const int64_t x_pad_l = pad;
    const int64_t W_ext = x_pad_l + W_in + (k_w - 1);

    float* dy_pack = dy_pack_opt;
    if (!dy_pack) {
        const size_t dy_need = (size_t)N * (size_t)OH * (size_t)nb_oc
            * (size_t)OW * (size_t)BRG_DW_OC;
        dy_pack = brg_dw_alloc(tls_brg_dy_pack, tls_brg_dy_cap, dy_need);
        if (!dy_pack) return false;
        const uint64_t t0 = bwd_pack_timing_enabled() ? bwd_pack_now_ns() : 0;
        brg_pack_dy_blocked(
            d_conv_buf, dy_pack, N, C_out, OH, OW, conv_out_w_stride,
            conv_spatial);
        if (bwd_pack_timing_enabled()) {
            g_bwd_ns_dy_pack.fetch_add(bwd_pack_now_ns() - t0, std::memory_order_relaxed);
            g_bwd_dy_pack_local.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Prefer prepare/fwd-published pack (main-thread or post-fwd overlap).
    float* x_pack = brg_dw_x_stage_lookup(
        x, N, C_in, H, W_in, W_in_stride, x_pad_l, W_ext
    );
    if (!x_pack) {
        const size_t x_need = (size_t)N * (size_t)nb_ic * (size_t)H
            * (size_t)W_ext * (size_t)BRG_DW_IC;
        x_pack = brg_dw_alloc(tls_brg_x_pack, tls_brg_x_cap, x_need);
        if (!x_pack) return false;
        const uint64_t t0 = bwd_pack_timing_enabled() ? bwd_pack_now_ns() : 0;
        brg_pack_x_blocked(
            x, x_pack, N, C_in, H, W_in, W_in_stride, x_pad_l, W_ext,
            /*parallel=*/true
        );
        if (bwd_pack_timing_enabled()) {
            g_bwd_ns_x_pack.fetch_add(bwd_pack_now_ns() - t0, std::memory_order_relaxed);
            g_bwd_x_pack_misses.fetch_add(1, std::memory_order_relaxed);
        }
    } else if (bwd_pack_timing_enabled()) {
        g_bwd_x_pack_hits.fetch_add(1, std::memory_order_relaxed);
    }

    if (!dw_prezeroed) {
        std::memset(dW, 0, (size_t)(C_out * C_in * k_spatial) * sizeof(float));
    }

    // Dual-OC: one x broadcast FMAs into two OC8 tiles (nb_oc_blocking=2).
    // Parallel over (ic, oc_pair, kh) so L1 (nb_ic=1, nb_oc=2) keeps OMP work.
    const int64_t nb_oc_pairs = nb_oc / 2;
    const uint64_t t_comp0 = bwd_pack_timing_enabled() ? bwd_pack_now_ns() : 0;
    if (nb_oc_pairs > 0) {
        #pragma omp parallel for collapse(3) schedule(static)
        for (int64_t ic_b = 0; ic_b < nb_ic; ++ic_b) {
            for (int64_t oc_p = 0; oc_p < nb_oc_pairs; ++oc_p) {
                for (int64_t kh = 0; kh < k_h; ++kh) {
                    const int64_t oc0 = oc_p * 2;
                    const int64_t oc1 = oc0 + 1;
                    for (int64_t kw = 0; kw < k_w; ++kw) {
                        __m256 Crow0[BRG_DW_IC];
                        __m256 Crow1[BRG_DW_IC];
                        for (int64_t ci = 0; ci < BRG_DW_IC; ++ci) {
                            Crow0[ci] = _mm256_setzero_ps();
                            Crow1[ci] = _mm256_setzero_ps();
                        }

                        for (int64_t n = 0; n < N; ++n) {
                            for (int64_t oh = 0; oh < OH; ++oh) {
                                const int64_t ih = oh - pad + kh;
                                if (ih < 0 || ih >= H) continue;

                                const float* __restrict x_base =
                                    &x_pack[((n * nb_ic + ic_b) * H + ih) * W_ext
                                            * BRG_DW_IC];
                                const float* __restrict B0 =
                                    &dy_pack[(((n * OH + oh) * nb_oc + oc0) * OW)
                                             * BRG_DW_OC];
                                const float* __restrict B1 =
                                    &dy_pack[(((n * OH + oh) * nb_oc + oc1) * OW)
                                             * BRG_DW_OC];
                                for (int64_t ow = 0; ow < OW; ++ow) {
                                    const __m256 bv0 =
                                        _mm256_loadu_ps(B0 + ow * BRG_DW_OC);
                                    const __m256 bv1 =
                                        _mm256_loadu_ps(B1 + ow * BRG_DW_OC);
                                    const float* __restrict xp =
                                        &x_base[(ow + kw) * BRG_DW_IC];
                                    for (int64_t ci = 0; ci < BRG_DW_IC; ++ci) {
                                        const __m256 av =
                                            _mm256_set1_ps(xp[ci]);
                                        Crow0[ci] = _mm256_fmadd_ps(
                                            av, bv0, Crow0[ci]);
                                        Crow1[ci] = _mm256_fmadd_ps(
                                            av, bv1, Crow1[ci]);
                                    }
                                }
                            }
                        }

                        alignas(32) float Cstore0[BRG_DW_IC][BRG_DW_OC];
                        alignas(32) float Cstore1[BRG_DW_IC][BRG_DW_OC];
                        for (int64_t ci = 0; ci < BRG_DW_IC; ++ci) {
                            _mm256_store_ps(Cstore0[ci], Crow0[ci]);
                            _mm256_store_ps(Cstore1[ci], Crow1[ci]);
                        }
                        for (int64_t co = 0; co < BRG_DW_OC; ++co) {
                            for (int64_t ci = 0; ci < BRG_DW_IC; ++ci) {
                                const int64_t cin = ic_b * BRG_DW_IC + ci;
                                const int64_t cout0 = oc0 * BRG_DW_OC + co;
                                const int64_t cout1 = oc1 * BRG_DW_OC + co;
                                dW[((cout0 * C_in + cin) * k_spatial) + kh * k_w + kw]
                                    += Cstore0[ci][co] * inv_m;
                                dW[((cout1 * C_in + cin) * k_spatial) + kh * k_w + kw]
                                    += Cstore1[ci][co] * inv_m;
                            }
                        }
                    }
                }
            }
        }
    }

    // Odd OC tile rem (nb_oc == 1 on L0-like geoms).
    if ((nb_oc & 1) != 0) {
        const int64_t oc_b = nb_oc - 1;
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t ic_b = 0; ic_b < nb_ic; ++ic_b) {
            for (int64_t kh = 0; kh < k_h; ++kh) {
                for (int64_t kw = 0; kw < k_w; ++kw) {
                    __m256 Crow[BRG_DW_IC];
                    for (int64_t ci = 0; ci < BRG_DW_IC; ++ci) {
                        Crow[ci] = _mm256_setzero_ps();
                    }

                    for (int64_t n = 0; n < N; ++n) {
                        for (int64_t oh = 0; oh < OH; ++oh) {
                            const int64_t ih = oh - pad + kh;
                            if (ih < 0 || ih >= H) continue;

                            const float* __restrict x_base =
                                &x_pack[((n * nb_ic + ic_b) * H + ih) * W_ext
                                        * BRG_DW_IC];
                            const float* __restrict B =
                                &dy_pack[(((n * OH + oh) * nb_oc + oc_b) * OW)
                                         * BRG_DW_OC];
                            for (int64_t ow = 0; ow < OW; ++ow) {
                                const __m256 bv =
                                    _mm256_loadu_ps(B + ow * BRG_DW_OC);
                                const float* __restrict xp =
                                    &x_base[(ow + kw) * BRG_DW_IC];
                                for (int64_t ci = 0; ci < BRG_DW_IC; ++ci) {
                                    Crow[ci] = _mm256_fmadd_ps(
                                        _mm256_set1_ps(xp[ci]), bv, Crow[ci]);
                                }
                            }
                        }
                    }

                    alignas(32) float Cstore[BRG_DW_IC][BRG_DW_OC];
                    for (int64_t ci = 0; ci < BRG_DW_IC; ++ci) {
                        _mm256_store_ps(Cstore[ci], Crow[ci]);
                    }
                    for (int64_t co = 0; co < BRG_DW_OC; ++co) {
                        for (int64_t ci = 0; ci < BRG_DW_IC; ++ci) {
                            const int64_t cout = oc_b * BRG_DW_OC + co;
                            const int64_t cin = ic_b * BRG_DW_IC + ci;
                            dW[((cout * C_in + cin) * k_spatial) + kh * k_w + kw]
                                += Cstore[ci][co] * inv_m;
                        }
                    }
                }
            }
        }
    }
    if (bwd_pack_timing_enabled()) {
        g_bwd_ns_dw_compute.fetch_add(bwd_pack_now_ns() - t_comp0, std::memory_order_relaxed);
    }
    return true;
}

// BRGEMM-style dX: pack dy to oc-blocks; for each spatial dx pixel, reduce
// over (kh,kw,oc_b) with 8-wide dy × 8-wide W rows (same grads as cin-blocked).
static bool try_brgemm_style_dx(
    const float* d_conv_buf, const float* W, float* dx,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t k_h, int64_t k_w, int64_t stride, int64_t pad,
    int64_t conv_out_h, int64_t conv_out_w, int64_t conv_out_w_stride,
    float* dy_pack_opt
) {
    if (!brg_dw_x_geom_ok(C_in, C_out, k_h, k_w, stride)) return false;
    if (!dx || !W || !d_conv_buf || N <= 0) return false;

    const int64_t OH = conv_out_h;
    const int64_t OW = conv_out_w;
    if (OH <= 0 || OW <= 0 || OW > 64) return false;

    const int64_t conv_spatial = OH * conv_out_w_stride;
    const int64_t nb_ic = C_in / BRG_DW_IC;
    const int64_t nb_oc = C_out / BRG_DW_OC;
    const int64_t spatial_in = H * W_in_stride;

    float* dy_pack = dy_pack_opt;
    if (!dy_pack) {
        const size_t dy_need = (size_t)N * (size_t)OH * (size_t)nb_oc
            * (size_t)OW * (size_t)BRG_DW_OC;
        dy_pack = brg_dw_alloc(tls_brg_dy_pack, tls_brg_dy_cap, dy_need);
        if (!dy_pack) return false;
        const uint64_t t0 = bwd_pack_timing_enabled() ? bwd_pack_now_ns() : 0;
        brg_pack_dy_blocked(
            d_conv_buf, dy_pack, N, C_out, OH, OW, conv_out_w_stride,
            conv_spatial);
        if (bwd_pack_timing_enabled()) {
            g_bwd_ns_dy_pack.fetch_add(bwd_pack_now_ns() - t0, std::memory_order_relaxed);
            g_bwd_dy_pack_local.fetch_add(1, std::memory_order_relaxed);
        }
    }

    float* Wt = staged_dx_wt_lookup(W, C_out, C_in, k_h);
    if (!Wt) {
        Wt = acquire_dx_cin_blocked_wt_buf((size_t)(C_out * k_h * k_w * C_in));
        if (!Wt) return false;
        const uint64_t t0 = bwd_pack_timing_enabled() ? bwd_pack_now_ns() : 0;
        transpose_dx_cin_blocked_wt(W, Wt, C_out, C_in, k_h);
        if (bwd_pack_timing_enabled()) {
            g_bwd_ns_dx_wt.fetch_add(bwd_pack_now_ns() - t0, std::memory_order_relaxed);
        }
    }

    const uint64_t t_comp0 = bwd_pack_timing_enabled() ? bwd_pack_now_ns() : 0;
    // Hyp: planar += scatter (spatial_in stride) was the uProf hotspot.
    // Accumulate in blocked [H][W_in][8], unpack once to NCHW.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t ic_b = 0; ic_b < nb_ic; ++ic_b) {
            const size_t blocked_need = (size_t)H * (size_t)W_in * (size_t)BRG_DW_IC;
            float* blocked = brg_dw_alloc(
                tls_brg_dx_blocked, tls_brg_dx_blocked_cap, blocked_need);
            if (!blocked) {
                // Fall through impossible inside parallel — zero slab and skip.
                continue;
            }
            std::memset(blocked, 0, blocked_need * sizeof(float));

            for (int64_t kh = 0; kh < k_h; ++kh) {
                for (int64_t kw = 0; kw < k_w; ++kw) {
                    for (int64_t oc_b = 0; oc_b < nb_oc; ++oc_b) {
                        alignas(32) __m256 w_oc[BRG_DW_OC];
                        for (int64_t o = 0; o < BRG_DW_OC; ++o) {
                            const int64_t cout = oc_b * BRG_DW_OC + o;
                            w_oc[o] = _mm256_load_ps(
                                &Wt[(((cout * k_h + kh) * k_w + kw) * C_in)
                                    + ic_b * BRG_DW_IC]);
                        }
                        for (int64_t oh = 0; oh < OH; ++oh) {
                            const int64_t ih = oh - pad + kh;
                            if (ih < 0 || ih >= H) continue;
                            const float* __restrict dy_row =
                                &dy_pack[(((n * OH + oh) * nb_oc + oc_b) * OW)
                                         * BRG_DW_OC];
                            float* __restrict blk_row =
                                &blocked[(size_t)ih * (size_t)W_in * (size_t)BRG_DW_IC];
                            for (int64_t ow = 0; ow < OW; ++ow) {
                                const int64_t iw = ow - pad + kw;
                                if (iw < 0 || iw >= W_in) continue;
                                const float* __restrict dy =
                                    dy_row + ow * BRG_DW_OC;
                                __m256 acc = _mm256_setzero_ps();
                                for (int64_t o = 0; o < BRG_DW_OC; ++o) {
                                    acc = _mm256_fmadd_ps(
                                        _mm256_set1_ps(dy[o]), w_oc[o], acc);
                                }
                                float* __restrict dst =
                                    blk_row + (size_t)iw * (size_t)BRG_DW_IC;
                                const __m256 prev = _mm256_load_ps(dst);
                                _mm256_store_ps(dst, _mm256_add_ps(prev, acc));
                            }
                        }
                    }
                }
            }

            // Unpack blocked → planar NCHW for this (n, ic_b) slab.
            float* __restrict dx_slab =
                &dx[(n * C_in + ic_b * BRG_DW_IC) * spatial_in];
            for (int64_t ih = 0; ih < H; ++ih) {
                const float* __restrict blk_row =
                    &blocked[(size_t)ih * (size_t)W_in * (size_t)BRG_DW_IC];
                for (int64_t iw = 0; iw < W_in; ++iw) {
                    alignas(32) float lanes[8];
                    _mm256_store_ps(lanes, _mm256_load_ps(blk_row + (size_t)iw * 8));
                    float* __restrict p = &dx_slab[ih * W_in_stride + iw];
                    p[0] = lanes[0]; p += spatial_in;
                    p[0] = lanes[1]; p += spatial_in;
                    p[0] = lanes[2]; p += spatial_in;
                    p[0] = lanes[3]; p += spatial_in;
                    p[0] = lanes[4]; p += spatial_in;
                    p[0] = lanes[5]; p += spatial_in;
                    p[0] = lanes[6]; p += spatial_in;
                    p[0] = lanes[7];
                }
            }
        }
    }
    if (bwd_pack_timing_enabled()) {
        g_bwd_ns_dx_compute.fetch_add(bwd_pack_now_ns() - t_comp0, std::memory_order_relaxed);
    }
    return true;
}
// --- end BRGEMM-style dW/dX ----------------------------------------------------

void conv2d_backward_fallback_avx2(
    const float* d_conv_buf, const float* x, const float* W,
    float* dx, float* dW,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t k_h, int64_t k_w, int64_t stride, int64_t pad,
    int64_t conv_out_w_stride, float inv_m, int64_t dx_prezeroed, int64_t dw_prezeroed
) {
    const int64_t conv_out_h   = (H + 2 * pad - k_h) / stride + 1;
    const int64_t conv_out_w   = (W_in + 2 * pad - k_w) / stride + 1;
    const int64_t conv_spatial = conv_out_h * conv_out_w_stride;
    const int64_t spatial_in   = H * W_in_stride;
    const int64_t k_spatial    = k_h * k_w;

    bool do_dx = (dx && W);
    bool do_dw = (dW && x);

    if (do_dx) {
        if (!dx_prezeroed) {
            std::memset(dx, 0, (size_t)(N * C_in * spatial_in) * sizeof(float));
        }
        if (bwd_dx_mock_edges_enabled()) {
            log_bwd_dx_mock_edges_once();
        }
    }
    if (do_dw && !dw_prezeroed) {
        std::memset(dW, 0, (size_t)(C_out * C_in * k_spatial) * sizeof(float));
    }

    // Prefer BRGEMM-style dW/dX when geometry matches (before pad/nci spray).
    const uint64_t t_fused0 = bwd_pack_timing_enabled() ? bwd_pack_now_ns() : 0;
    float* shared_dy_pack = nullptr;
    if ((do_dw || do_dx) &&
        brg_dw_x_geom_ok(C_in, C_out, k_h, k_w, stride)) {
        const int64_t OH = conv_out_h;
        const int64_t OW = conv_out_w;
        if (OH > 0 && OW > 0 && OW <= 64) {
            const int64_t nb_oc = C_out / BRG_DW_OC;
            const size_t dy_need = (size_t)N * (size_t)OH * (size_t)nb_oc
                * (size_t)OW * (size_t)BRG_DW_OC;
            shared_dy_pack =
                brg_dw_alloc(tls_brg_dy_pack, tls_brg_dy_cap, dy_need);
            if (shared_dy_pack) {
                const uint64_t t0 = bwd_pack_timing_enabled() ? bwd_pack_now_ns() : 0;
                brg_pack_dy_blocked(
                    d_conv_buf, shared_dy_pack, N, C_out, OH, OW,
                    conv_out_w_stride, conv_spatial);
                if (bwd_pack_timing_enabled()) {
                    g_bwd_ns_dy_pack.fetch_add(
                        bwd_pack_now_ns() - t0, std::memory_order_relaxed);
                    g_bwd_dy_pack_shared.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }

    if (do_dw &&
        try_brgemm_style_dw(
            d_conv_buf, x, dW,
            N, C_in, H, W_in, W_in_stride,
            C_out, k_h, k_w, stride, pad,
            conv_out_w_stride, inv_m, /*dw_prezeroed=*/1,
            shared_dy_pack
        )) {
        do_dw = false;
    }

    if (do_dx && !force_dx_crawl_enabled() &&
        try_brgemm_style_dx(
            d_conv_buf, W, dx,
            N, C_in, H, W_in, W_in_stride,
            C_out, k_h, k_w, stride, pad,
            conv_out_h, conv_out_w, conv_out_w_stride,
            shared_dy_pack
        )) {
        do_dx = false;
    } else if (do_dx && !force_dx_crawl_enabled() &&
        try_cin_blocked_dx(
            k_h, k_w, stride, C_in,
            d_conv_buf, W, dx,
            N, H, W_in, W_in_stride,
            C_out, pad, conv_out_h, conv_out_w, conv_out_w_stride
        )) {
        do_dx = false; // handled above; skip the existing tile-queue dx path
    } else if (do_dx && !force_dx_crawl_enabled() &&
        try_cout_blocked_dx(
            k_h, k_w, stride, C_out,
            d_conv_buf, W, dx,
            N, C_in, H, W_in, W_in_stride,
            pad, conv_out_h, conv_out_w, conv_out_w_stride
        )) {
        do_dx = false; // handled above (Option D); skip the existing tile-queue dx path
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
        x_pad_buf = staged_x_pad_lookup(
            x, N, C_in, H, W_in, W_in_stride, x_pad_l, x_row_stride
        );
        if (!x_pad_buf) {
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
    if (bwd_pack_timing_enabled()) {
        g_bwd_ns_fused_total.fetch_add(
            bwd_pack_now_ns() - t_fused0, std::memory_order_relaxed);
        g_bwd_pack_calls.fetch_add(1, std::memory_order_relaxed);
    }
}