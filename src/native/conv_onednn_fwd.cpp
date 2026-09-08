// Forward via libdnnl, patterns inspired by Torch/mkldnn (not a clone):
// format_tag::any PD → jit, fused relu post-op, cached PD/primitives/scratch.

#include "export.h"
#include "conv_onednn_fwd.h"

#include <oneapi/dnnl/dnnl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

using namespace dnnl;

static bool fwd_trace_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("ML_ENGINE_FWD_TRACE");
        cached = (env && env[0] == '1' && env[1] == '\0') ? 1 : 0;
    }
    return cached != 0;
}

struct OnednnEngine {
    engine eng;
    stream s;
    OnednnEngine() : eng(engine::kind::cpu, 0), s(eng) {}
};

static OnednnEngine& get_eng() {
    static OnednnEngine e;
    return e;
}

struct FwdKey {
    int64_t N, C_in, H, W_in, C_out, k_h, k_w, stride, pad;
    int32_t fuse_relu;
    int32_t with_bias;

    bool operator==(const FwdKey& o) const {
        return N == o.N && C_in == o.C_in && H == o.H && W_in == o.W_in
            && C_out == o.C_out && k_h == o.k_h && k_w == o.k_w
            && stride == o.stride && pad == o.pad
            && fuse_relu == o.fuse_relu && with_bias == o.with_bias;
    }
};

struct FwdKeyHash {
    size_t operator()(const FwdKey& k) const {
        size_t h = 1469598103934665603ull;
        auto mix = [&](int64_t v) {
            h ^= (size_t)v;
            h *= 1099511628211ull;
        };
        mix(k.N); mix(k.C_in); mix(k.H); mix(k.W_in); mix(k.C_out);
        mix(k.k_h); mix(k.k_w); mix(k.stride); mix(k.pad);
        mix(k.fuse_relu); mix(k.with_bias);
        return h;
    }
};

struct FwdEntry {
    std::mutex mu;
    convolution_forward::primitive_desc conv_pd;
    convolution_forward conv_prim;
    memory::desc src_md_user;
    memory::desc wei_md_user;
    memory::desc dst_md_user;
    memory::desc bias_md_user;

    memory src_blocked;
    memory dst_blocked;
    memory wei_packed;
    memory bias_packed;

    reorder src_to_blocked;
    reorder wei_to_packed;
    reorder bias_to_packed;
    reorder dst_to_user;

    bool need_src_reorder = false;
    bool need_wei_reorder = false;
    bool need_bias_reorder = false;
    bool need_dst_reorder = false;
    bool has_bias = false;
    bool built = false;

    std::vector<float> src_contig;
    std::vector<float> dst_contig;
};

static std::mutex g_cache_mu;
static std::unordered_map<FwdKey, std::unique_ptr<FwdEntry>, FwdKeyHash> g_fwd_cache;

static void maybe_copy_strided_out(
    const float* __restrict contig, float* __restrict out,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t W_stride
) {
    if (W_stride == W) {
        std::memcpy(out, contig, (size_t)(N * C * H * W) * sizeof(float));
        return;
    }
    const int64_t spatial_dst = H * W_stride;
    const int64_t spatial_src = H * W;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const float* __restrict src =
                contig + (n * C + c) * spatial_src;
            float* __restrict dst = out + (n * C + c) * spatial_dst;
            for (int64_t h = 0; h < H; ++h) {
                std::memcpy(
                    dst + h * W_stride, src + h * W, (size_t)W * sizeof(float));
                for (int64_t w = W; w < W_stride; ++w) {
                    dst[h * W_stride + w] = 0.0f;
                }
            }
        }
    }
}

static void build_entry(
    FwdEntry& e, engine& eng, const FwdKey& key, bool with_bias
) {
    e.has_bias = with_bias;

    memory::dims src_dims = {key.N, key.C_in, key.H, key.W_in};
    memory::dims wei_dims = {key.C_out, key.C_in, key.k_h, key.k_w};
    const int64_t out_h =
        (key.H + 2 * key.pad - key.k_h) / key.stride + 1;
    const int64_t out_w =
        (key.W_in + 2 * key.pad - key.k_w) / key.stride + 1;
    memory::dims dst_dims = {key.N, key.C_out, out_h, out_w};
    memory::dims strides_dims = {key.stride, key.stride};
    memory::dims pads = {key.pad, key.pad};

    auto src_md_any = memory::desc(
        src_dims, memory::data_type::f32, memory::format_tag::any);
    auto wei_md_any = memory::desc(
        wei_dims, memory::data_type::f32, memory::format_tag::any);
    auto dst_md_any = memory::desc(
        dst_dims, memory::data_type::f32, memory::format_tag::any);

    primitive_attr attr;
    if (key.fuse_relu) {
        post_ops po;
        po.append_eltwise(algorithm::eltwise_relu, 0.f, 0.f);
        attr.set_post_ops(po);
    }

    const prop_kind pk = key.fuse_relu
        ? prop_kind::forward_inference
        : prop_kind::forward_training;

    if (with_bias) {
        auto bias_md_any = memory::desc(
            {key.C_out}, memory::data_type::f32, memory::format_tag::any);
        e.conv_pd = convolution_forward::primitive_desc(
            eng, pk, algorithm::convolution_direct,
            src_md_any, wei_md_any, bias_md_any, dst_md_any,
            strides_dims, pads, pads, attr);
    } else {
        e.conv_pd = convolution_forward::primitive_desc(
            eng, pk, algorithm::convolution_direct,
            src_md_any, wei_md_any, dst_md_any,
            strides_dims, pads, pads, attr);
    }
    e.conv_prim = convolution_forward(e.conv_pd);

    e.src_md_user = memory::desc(
        src_dims, memory::data_type::f32, memory::format_tag::nchw);
    e.wei_md_user = memory::desc(
        wei_dims, memory::data_type::f32, memory::format_tag::oihw);
    e.dst_md_user = memory::desc(
        dst_dims, memory::data_type::f32, memory::format_tag::nchw);

    e.need_src_reorder = (e.conv_pd.src_desc() != e.src_md_user);
    e.need_wei_reorder = (e.conv_pd.weights_desc() != e.wei_md_user);
    e.need_dst_reorder = (e.conv_pd.dst_desc() != e.dst_md_user);

    if (e.need_src_reorder) {
        e.src_blocked = memory(e.conv_pd.src_desc(), eng);
        e.src_to_blocked = reorder(
            reorder::primitive_desc(eng, e.src_md_user, eng, e.conv_pd.src_desc()));
    }
    if (e.need_wei_reorder) {
        e.wei_packed = memory(e.conv_pd.weights_desc(), eng);
        e.wei_to_packed = reorder(
            reorder::primitive_desc(
                eng, e.wei_md_user, eng, e.conv_pd.weights_desc()));
    }
    if (e.need_dst_reorder) {
        e.dst_blocked = memory(e.conv_pd.dst_desc(), eng);
        e.dst_to_user = reorder(
            reorder::primitive_desc(
                eng, e.conv_pd.dst_desc(), eng, e.dst_md_user));
    }

    if (with_bias) {
        e.bias_md_user = memory::desc(
            {key.C_out}, memory::data_type::f32, memory::format_tag::a);
        e.need_bias_reorder = (e.conv_pd.bias_desc() != e.bias_md_user);
        if (e.need_bias_reorder) {
            e.bias_packed = memory(e.conv_pd.bias_desc(), eng);
            e.bias_to_packed = reorder(
                reorder::primitive_desc(
                    eng, e.bias_md_user, eng, e.conv_pd.bias_desc()));
        }
    }
    e.built = true;
}

// Global lock only for map; entry lock for build + execute.
static FwdEntry& get_entry(const FwdKey& key) {
    std::lock_guard<std::mutex> lock(g_cache_mu);
    auto it = g_fwd_cache.find(key);
    if (it != g_fwd_cache.end()) return *it->second;
    auto up = std::make_unique<FwdEntry>();
    FwdEntry& ref = *up;
    g_fwd_cache.emplace(key, std::move(up));
    return ref;
}

}  // namespace

extern "C" {

bool conv2d_forward_onednn(
    const float* x, const float* W, const float* bias, float* out,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t k_h, int64_t k_w, int64_t stride, int64_t pad,
    int64_t out_w_stride, int32_t fuse_relu
) {
    if (!x || !W || !out) return false;
    if (N <= 0 || C_in <= 0 || C_out <= 0 || H <= 0 || W_in <= 0) return false;
    if (k_h <= 0 || k_w <= 0 || stride <= 0 || pad < 0) return false;
    if (W_in_stride < W_in || out_w_stride <= 0) return false;

    const int64_t out_h = (H + 2 * pad - k_h) / stride + 1;
    const int64_t out_w = (W_in + 2 * pad - k_w) / stride + 1;
    if (out_h <= 0 || out_w <= 0) return false;
    if (out_w_stride < out_w) return false;

    const bool src_dense = (W_in_stride == W_in);
    const bool dst_dense = (out_w_stride == out_w);
    const bool with_bias = (bias != nullptr);

    try {
        OnednnEngine& oe = get_eng();
        engine& eng = oe.eng;
        stream& s = oe.s;

        FwdKey key{
            N, C_in, H, W_in, C_out, k_h, k_w, stride, pad,
            fuse_relu ? 1 : 0, with_bias ? 1 : 0
        };

        FwdEntry& e = get_entry(key);
        std::lock_guard<std::mutex> elock(e.mu);
        if (!e.built) build_entry(e, eng, key, with_bias);

        const float* src_ptr = x;
        float* dst_ptr = out;

        if (!src_dense) {
            const size_t n = (size_t)N * (size_t)C_in * (size_t)H * (size_t)W_in;
            if (e.src_contig.size() < n) e.src_contig.resize(n);
            #pragma omp parallel for collapse(2) schedule(static)
            for (int64_t n_i = 0; n_i < N; ++n_i) {
                for (int64_t c = 0; c < C_in; ++c) {
                    for (int64_t h = 0; h < H; ++h) {
                        std::memcpy(
                            e.src_contig.data()
                                + ((n_i * C_in + c) * H + h) * W_in,
                            x + (n_i * C_in + c) * H * W_in_stride
                                + h * W_in_stride,
                            (size_t)W_in * sizeof(float));
                    }
                }
            }
            src_ptr = e.src_contig.data();
        }
        if (!dst_dense) {
            const size_t n =
                (size_t)N * (size_t)C_out * (size_t)out_h * (size_t)out_w;
            if (e.dst_contig.size() < n) e.dst_contig.resize(n);
            dst_ptr = e.dst_contig.data();
        }

        memory src_user(e.src_md_user, eng, (void*)src_ptr);
        memory wei_user(e.wei_md_user, eng, (void*)W);
        memory dst_user(e.dst_md_user, eng, (void*)dst_ptr);

        memory src_arg = src_user;
        memory wei_arg = wei_user;
        memory dst_arg = dst_user;
        memory bias_arg;

        if (e.need_src_reorder) {
            e.src_to_blocked.execute(s, src_user, e.src_blocked);
            src_arg = e.src_blocked;
        }
        if (e.need_wei_reorder) {
            e.wei_to_packed.execute(s, wei_user, e.wei_packed);
            wei_arg = e.wei_packed;
        }
        if (with_bias) {
            memory bias_user(e.bias_md_user, eng, (void*)bias);
            if (e.need_bias_reorder) {
                e.bias_to_packed.execute(s, bias_user, e.bias_packed);
                bias_arg = e.bias_packed;
            } else {
                bias_arg = bias_user;
            }
        }
        if (e.need_dst_reorder) dst_arg = e.dst_blocked;

        if (fwd_trace_enabled()) {
            std::printf(
                "[FWD_TRACE] === oneDNN ENTER === cached PD + "
                "packed W + %s\n",
                fuse_relu ? "fused relu post-op" : "no post-op");
            std::fflush(stdout);
        }

        std::unordered_map<int, memory> args;
        args.insert({DNNL_ARG_SRC, src_arg});
        args.insert({DNNL_ARG_WEIGHTS, wei_arg});
        args.insert({DNNL_ARG_DST, dst_arg});
        if (with_bias) args.insert({DNNL_ARG_BIAS, bias_arg});

        e.conv_prim.execute(s, args);
        if (e.need_dst_reorder) {
            e.dst_to_user.execute(s, e.dst_blocked, dst_user);
        }
        s.wait();

        if (!dst_dense) {
            maybe_copy_strided_out(
                dst_ptr, out, N, C_out, out_h, out_w, out_w_stride);
        }

        if (fwd_trace_enabled()) {
            std::printf("[FWD_TRACE] === oneDNN EXIT ===\n");
            std::fflush(stdout);
        }
        return true;
    } catch (const dnnl::error& err) {
        if (fwd_trace_enabled()) {
            std::printf(
                "[FWD_TRACE] oneDNN error: %s (%d)\n", err.what(), err.status);
            std::fflush(stdout);
        }
        return false;
    } catch (...) {
        return false;
    }
}

}  // extern "C"
