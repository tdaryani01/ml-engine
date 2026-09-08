#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// libdnnl fwd inspired by Torch/mkldnn (cached PD, packed W, fused relu).
bool conv2d_forward_onednn(
    const float* x, const float* W, const float* bias, float* out,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t k_h, int64_t k_w, int64_t stride, int64_t pad,
    int64_t out_w_stride, int32_t fuse_relu);

#ifdef __cplusplus
}
#endif
