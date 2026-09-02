#include "im2col_telemetry.h"

#include <atomic>
#include <cstdio>

namespace {

std::atomic<uint64_t> g_im2col_calls{0};
std::atomic<uint64_t> g_im2col_tile_fast{0};
std::atomic<uint64_t> g_im2col_tile_padded{0};
std::atomic<uint64_t> g_col2im_calls{0};
std::atomic<uint64_t> g_col2im_tile_fast{0};
std::atomic<uint64_t> g_col2im_tile_xclip{0};
std::atomic<uint64_t> g_col2im_tile_yclip{0};
std::atomic<uint64_t> g_col2im_tile_corner{0};
std::atomic<uint64_t> g_col2im_memset_bytes{0};
std::atomic<uint64_t> g_col2im_memset_calls{0};
std::atomic<uint64_t> g_misalign_x_ptr{0};
std::atomic<uint64_t> g_misalign_col_ptr{0};
std::atomic<uint64_t> g_misalign_dx_ptr{0};
std::atomic<uint64_t> g_misalign_out_ptr{0};
std::atomic<uint64_t> g_gemm_fwd_calls{0};
std::atomic<uint64_t> g_gemm_bwd_w_calls{0};
std::atomic<uint64_t> g_gemm_bwd_x_calls{0};
std::atomic<uint64_t> g_fuse_dout_transpose_calls{0};

}  // namespace

namespace im2col_telemetry {

void record_ptr_alignment(const float* x, const float* col_or_out, const float* dx) {
    if (x && ptr_misaligned(x)) {
        g_misalign_x_ptr.fetch_add(1, std::memory_order_relaxed);
    }
    if (col_or_out && ptr_misaligned(col_or_out)) {
        g_misalign_col_ptr.fetch_add(1, std::memory_order_relaxed);
    }
    if (dx && ptr_misaligned(dx)) {
        g_misalign_dx_ptr.fetch_add(1, std::memory_order_relaxed);
    }
}

void record_gemm_fwd() {
    g_gemm_fwd_calls.fetch_add(1, std::memory_order_relaxed);
}

void record_gemm_bwd_w() {
    g_gemm_bwd_w_calls.fetch_add(1, std::memory_order_relaxed);
}

void record_gemm_bwd_x() {
    g_gemm_bwd_x_calls.fetch_add(1, std::memory_order_relaxed);
}

void on_fuse_dout_transpose() {
    g_fuse_dout_transpose_calls.fetch_add(1, std::memory_order_relaxed);
}

void add_im2col_tiles(uint64_t fast_tiles, uint64_t padded_tiles) {
    g_im2col_tile_fast.fetch_add(fast_tiles, std::memory_order_relaxed);
    g_im2col_tile_padded.fetch_add(padded_tiles, std::memory_order_relaxed);
}

void add_col2im_tiles(uint64_t fast_t, uint64_t xclip_t, uint64_t yclip_t, uint64_t corner_t) {
    g_col2im_tile_fast.fetch_add(fast_t, std::memory_order_relaxed);
    g_col2im_tile_xclip.fetch_add(xclip_t, std::memory_order_relaxed);
    g_col2im_tile_yclip.fetch_add(yclip_t, std::memory_order_relaxed);
    g_col2im_tile_corner.fetch_add(corner_t, std::memory_order_relaxed);
}

void on_im2col_call() {
    g_im2col_calls.fetch_add(1, std::memory_order_relaxed);
}

void on_col2im_call(uint64_t memset_bytes) {
    g_col2im_calls.fetch_add(1, std::memory_order_relaxed);
    g_col2im_memset_calls.fetch_add(1, std::memory_order_relaxed);
    g_col2im_memset_bytes.fetch_add(memset_bytes, std::memory_order_relaxed);
}

void record_out_misalign() {
    g_misalign_out_ptr.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace im2col_telemetry

extern "C" {

void reset_im2col_telemetry(void) {
    g_im2col_calls.store(0);
    g_im2col_tile_fast.store(0);
    g_im2col_tile_padded.store(0);
    g_col2im_calls.store(0);
    g_col2im_tile_fast.store(0);
    g_col2im_tile_xclip.store(0);
    g_col2im_tile_yclip.store(0);
    g_col2im_tile_corner.store(0);
    g_col2im_memset_bytes.store(0);
    g_col2im_memset_calls.store(0);
    g_misalign_x_ptr.store(0);
    g_misalign_col_ptr.store(0);
    g_misalign_dx_ptr.store(0);
    g_misalign_out_ptr.store(0);
    g_gemm_fwd_calls.store(0);
    g_gemm_bwd_w_calls.store(0);
    g_gemm_bwd_x_calls.store(0);
    g_fuse_dout_transpose_calls.store(0);
}

void get_im2col_telemetry(Im2ColTelemetry* out) {
    if (!out) {
        return;
    }
    out->im2col_calls = g_im2col_calls.load();
    out->im2col_tile_fast = g_im2col_tile_fast.load();
    out->im2col_tile_padded = g_im2col_tile_padded.load();
    out->col2im_calls = g_col2im_calls.load();
    out->col2im_tile_fast = g_col2im_tile_fast.load();
    out->col2im_tile_xclip = g_col2im_tile_xclip.load();
    out->col2im_tile_yclip = g_col2im_tile_yclip.load();
    out->col2im_tile_corner = g_col2im_tile_corner.load();
    out->col2im_memset_bytes = g_col2im_memset_bytes.load();
    out->col2im_memset_calls = g_col2im_memset_calls.load();
    out->misalign_x_ptr = g_misalign_x_ptr.load();
    out->misalign_col_ptr = g_misalign_col_ptr.load();
    out->misalign_dx_ptr = g_misalign_dx_ptr.load();
    out->misalign_out_ptr = g_misalign_out_ptr.load();
    out->gemm_fwd_calls = g_gemm_fwd_calls.load();
    out->gemm_bwd_w_calls = g_gemm_bwd_w_calls.load();
    out->gemm_bwd_x_calls = g_gemm_bwd_x_calls.load();
    out->fuse_dout_transpose_calls = g_fuse_dout_transpose_calls.load();
}

void log_im2col_telemetry(void) {
    Im2ColTelemetry t{};
    get_im2col_telemetry(&t);

    const uint64_t im2col_tiles = t.im2col_tile_fast + t.im2col_tile_padded;
    const uint64_t col2im_tiles = t.col2im_tile_fast + t.col2im_tile_xclip +
                                  t.col2im_tile_yclip + t.col2im_tile_corner;
    const uint64_t gemm_total = t.gemm_fwd_calls + t.gemm_bwd_w_calls + t.gemm_bwd_x_calls;

    std::fprintf(stderr, "\n=== im2col+gemm telemetry (cumulative) ===\n");
    std::fprintf(stderr, "im2col calls=%llu tiles=%llu (fast=%.1f%% padded=%.1f%%)\n",
        static_cast<unsigned long long>(t.im2col_calls),
        static_cast<unsigned long long>(im2col_tiles),
        im2col_tiles ? (100.0 * t.im2col_tile_fast / im2col_tiles) : 0.0,
        im2col_tiles ? (100.0 * t.im2col_tile_padded / im2col_tiles) : 0.0);
    std::fprintf(stderr, "col2im calls=%llu tiles=%llu (fast=%.1f%% xclip=%.1f%% yclip=%.1f%% corner=%.1f%%)\n",
        static_cast<unsigned long long>(t.col2im_calls),
        static_cast<unsigned long long>(col2im_tiles),
        col2im_tiles ? (100.0 * t.col2im_tile_fast / col2im_tiles) : 0.0,
        col2im_tiles ? (100.0 * t.col2im_tile_xclip / col2im_tiles) : 0.0,
        col2im_tiles ? (100.0 * t.col2im_tile_yclip / col2im_tiles) : 0.0,
        col2im_tiles ? (100.0 * t.col2im_tile_corner / col2im_tiles) : 0.0);
    std::fprintf(stderr, "col2im memset: calls=%llu total_bytes=%llu (%.2f MiB) avg=%.0f KiB/call\n",
        static_cast<unsigned long long>(t.col2im_memset_calls),
        static_cast<unsigned long long>(t.col2im_memset_bytes),
        t.col2im_memset_bytes / (1024.0 * 1024.0),
        t.col2im_memset_calls ? (t.col2im_memset_bytes / (1024.0 * t.col2im_memset_calls)) : 0.0);
    std::fprintf(stderr, "GEMM calls: fwd=%llu bwd_w=%llu bwd_x=%llu total=%llu",
        static_cast<unsigned long long>(t.gemm_fwd_calls),
        static_cast<unsigned long long>(t.gemm_bwd_w_calls),
        static_cast<unsigned long long>(t.gemm_bwd_x_calls),
        static_cast<unsigned long long>(gemm_total));
    if (gemm_total > 0) {
        std::fprintf(stderr, " (fwd=%.1f%% bwd_w=%.1f%% bwd_x=%.1f%%)",
            100.0 * t.gemm_fwd_calls / gemm_total,
            100.0 * t.gemm_bwd_w_calls / gemm_total,
            100.0 * t.gemm_bwd_x_calls / gemm_total);
    }
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "fuse_dout_transpose calls=%llu\n",
        static_cast<unsigned long long>(t.fuse_dout_transpose_calls));
    std::fprintf(stderr, "ptr misalign (32B): x=%llu col/out=%llu dx=%llu out_only=%llu",
        static_cast<unsigned long long>(t.misalign_x_ptr),
        static_cast<unsigned long long>(t.misalign_col_ptr),
        static_cast<unsigned long long>(t.misalign_dx_ptr),
        static_cast<unsigned long long>(t.misalign_out_ptr));
    if (t.misalign_x_ptr + t.misalign_col_ptr + t.misalign_dx_ptr + t.misalign_out_ptr > 0) {
        std::fprintf(stderr, "  [WARN: unaligned buffers can hurt AVX/memcpy]");
    }
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "hint: high col2im memset MiB + kernel MiUnlockWorkingSetShared in uProf\n");
    std::fprintf(stderr, "      often means full dx zeroing is faulting cold pages each bwd-x.\n");
    std::fprintf(stderr, "==========================================\n\n");
}

}  // extern "C"
