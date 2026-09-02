#pragma once
#include <cstdint>

#ifdef _WIN32
#define ML_IM2COL_EXPORT __declspec(dllexport)
#else
#define ML_IM2COL_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Cumulative counters for im2col+gemm diagnostics (read via log_im2col_telemetry).
struct Im2ColTelemetry {
    uint64_t im2col_calls;
    uint64_t im2col_tile_fast;
    uint64_t im2col_tile_padded;
    uint64_t col2im_calls;
    uint64_t col2im_tile_fast;
    uint64_t col2im_tile_xclip;
    uint64_t col2im_tile_yclip;
    uint64_t col2im_tile_corner;
    uint64_t col2im_memset_bytes;
    uint64_t col2im_memset_calls;
    uint64_t misalign_x_ptr;
    uint64_t misalign_col_ptr;
    uint64_t misalign_dx_ptr;
    uint64_t misalign_out_ptr;
    uint64_t gemm_fwd_calls;
    uint64_t gemm_bwd_w_calls;
    uint64_t gemm_bwd_x_calls;
    uint64_t fuse_dout_transpose_calls;
};

ML_IM2COL_EXPORT void reset_im2col_telemetry(void);
ML_IM2COL_EXPORT void get_im2col_telemetry(Im2ColTelemetry* out);
ML_IM2COL_EXPORT void log_im2col_telemetry(void);

#ifdef __cplusplus
}

namespace im2col_telemetry {

constexpr uintptr_t kAvxAlign = 32u;

inline bool ptr_misaligned(const void* p) {
    const auto addr = reinterpret_cast<uintptr_t>(p);
    return (addr & (kAvxAlign - 1u)) != 0u;
}

void record_ptr_alignment(const float* x, const float* col_or_out, const float* dx);
void record_out_misalign();
void record_gemm_fwd();
void record_gemm_bwd_w();
void record_gemm_bwd_x();
void on_fuse_dout_transpose();
void on_im2col_call();
void on_col2im_call(uint64_t memset_bytes);
void add_im2col_tiles(uint64_t fast_tiles, uint64_t padded_tiles);
void add_col2im_tiles(uint64_t fast_t, uint64_t xclip_t, uint64_t yclip_t, uint64_t corner_t);

}  // namespace im2col_telemetry
#endif
