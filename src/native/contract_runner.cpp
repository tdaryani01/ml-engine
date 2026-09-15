// contract_runner.cpp — Phase F: execute compiled contract list in one native call.
#include "export.h"
#include "native_tenant.h"
#include "omp_config.h"
#include "mhsa_kernels.h"
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(ML_ENGINE_PROFILE_CONTRACT_THREADS) && defined(__linux__)
#include <ctime>
#include <sys/syscall.h>
#include <unistd.h>
#endif

static int64_t round_up_simd(int64_t w) { return (w + 7) & ~7; }

extern "C" {

int32_t direct_conv_block_forward_avx2(
    const float* x, const float* W, const float* bias,
    float* out_conv, float* out_pool, uint8_t* argmax_buf,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t k_h, int64_t k_w,
    int64_t conv_stride, int64_t conv_pad, int64_t conv_out_w_stride,
    int64_t pool_size, int64_t pool_stride);

int32_t direct_conv_block_backward_avx2(
    const float* dout_pool, const uint8_t* argmax_buf,
    const float* x, const float* W, const float* conv_act,
    float* d_conv_buf, float* dx_buf, float* dW_buf, float* db_buf,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t k_h, int64_t k_w,
    int64_t conv_stride, int64_t conv_pad, int64_t conv_out_w_stride,
    int64_t pool_size, int64_t pool_stride,
    int64_t pool_out_h, int64_t pool_out_w, float inv_m,
    int64_t d_conv_prezeroed, int64_t dx_prezeroed, int64_t dw_prezeroed);

// BRGEMM dW x-pack: worker posts; Python main packs (see conv_fallback).
int32_t request_brgemm_dw_x_pack(
    const float* x,
    int64_t N, int64_t C_in, int64_t H, int64_t W_in, int64_t W_in_stride,
    int64_t C_out, int64_t k_h, int64_t k_w, int64_t stride, int64_t pad);

void reset_brgemm_dw_x_pack_request(void);

}  // extern "C"

enum ContractOpcode : int32_t {
    OP_CONV2D_FWD = 1,
    OP_CONV2D_BWD = 2,
    OP_RELU_FWD = 3,
    OP_RELU_BWD = 4,
    OP_MAXPOOL_FWD = 5,
    OP_MAXPOOL_BWD = 6,
    OP_FLATTEN_FWD = 7,
    OP_FLATTEN_BWD = 8,
    OP_DENSE_FWD = 9,
    OP_DENSE_BWD = 10,
    OP_ADAM_APPLY = 11,
    OP_CONV_BLOCK_FWD = 20,
    OP_CONV_BLOCK_BWD = 21,
    OP_MHSA_BLOCK_FWD = 30,
    OP_MHSA_BLOCK_BWD = 31,
    OP_MHSA_ACTION_FWD = 32,
    OP_MHSA_ACTION_BWD = 33,
};

struct ContractOpRow {
    int32_t opcode;
    int32_t layer_idx;
    int32_t param_idx;
    int32_t flags;
    int32_t i0;
    int32_t i1;
    int32_t i2;
};

struct LayerBinding {
    float* W;
    float* b;
    float* W_next;
    float* b_next;
    float* dW;
    float* db;
    float* out_conv;
    float* out_pool;
    uint8_t* argmax;
    float* dx;
    float* d_conv;
    float* x_cache;
    float* conv_act_cache;
    float* ms_w;
    float* vs_w;
    float* ms_b;
    float* vs_b;
    float* ms_w_next;
    float* vs_w_next;
    float* ms_b_next;
    float* vs_b_next;
    int64_t w_count;
    int64_t b_count;
    int64_t C_in;
    int64_t C_out;
    int64_t H;
    int64_t W_in;
    int64_t W_stride;
    int64_t k_h;
    int64_t k_w;
    int64_t conv_stride;
    int64_t conv_pad;
    int64_t pool_size;
    int64_t pool_stride;
    int64_t pool_out_h;
    int64_t pool_out_w;
    int64_t conv_out_w_stride;
    // Set by the Python main thread when it has already zeroed d_conv for this
    // slot; the maxpool backward then scatters into it without memsetting.
    int64_t d_conv_prezeroed;
    // Same idea for dx: Python zeroes it during prepare_step (overlap window,
    // input-batch-shape-only, no data dependency) so backward skips its memset.
    int64_t dx_prezeroed;
    // Same idea for dW: Python already zeroes it in prepare_step (conv_grads
    // fill(0.0)) before this slot is ever submitted, so backward's own
    // memset is pure duplicate work.
    int64_t dw_prezeroed;
};

struct DenseBinding {
    float* W;
    float* b;
    float* W_next;
    float* b_next;
    float* dW;
    float* db;
    float* z;
    float* output;
    float* delta;
    float* input_cache;
    float* dx_flat;
    float* ms_w;
    float* vs_w;
    float* ms_b;
    float* vs_b;
    float* ms_w_next;
    float* vs_w_next;
    float* ms_b_next;
    float* vs_b_next;
    int64_t fan_in;
    int64_t fan_out;
};

struct AdamBinding {
    float beta1;
    float beta2;
    float eps;
    int32_t t;
};

struct ContractExecCtx {
    int64_t N;
    float lr;
    float lam_l2;
    float max_norm;
    int32_t skip_adam;
    const float* X;
    const float* y;
    float* act;
    int64_t flat_dim;
    int32_t num_layers;
    LayerBinding layers[8];
    int32_t num_dense;
    DenseBinding dense[8];
    AdamBinding adam;
    float* loss_out;
    // MHSA (composed; unused when has_mhsa==0)
    int32_t has_mhsa;
    MhsaBinding mhsa;
};

static void softmax_cross_entropy_loss(
    const float* probs, const float* y, int64_t N, int64_t C, float* loss_out
) {
    const float eps = 1e-15f;
    double total = 0.0;
    for (int64_t n = 0; n < N; ++n) {
        const float* p = probs + n * C;
        const float* t = y + n * C;
        for (int64_t c = 0; c < C; ++c) {
            float pc = p[c];
            if (pc < eps) pc = eps;
            if (pc > 1.0f - eps) pc = 1.0f - eps;
            total += (double)t[c] * std::log((double)pc);
        }
    }
    *loss_out = (float)(-total / (double)N);
}

static inline float hsum256_ps(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

// z[n][:] = b + x[n][:] * W, accumulated along fan_out (W rows are contiguous).
static void dense_linear_forward(
    const ContractExecCtx* ctx, DenseBinding* d, const float* x_in
) {
    const int64_t N = ctx->N;
    const int64_t fin = d->fan_in;
    const int64_t fout = d->fan_out;
    ml_omp_before_parallel();
    #pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        const float* __restrict x = x_in + n * fin;
        float* __restrict z = d->z + n * fout;
        int64_t j = 0;
        for (; j + 7 < fout; j += 8) {
            _mm256_storeu_ps(z + j, _mm256_loadu_ps(d->b + j));
        }
        for (; j < fout; ++j) {
            z[j] = d->b[j];
        }
        for (int64_t k = 0; k < fin; ++k) {
            const __m256 xv = _mm256_set1_ps(x[k]);
            const float* __restrict w_row = d->W + k * fout;
            j = 0;
            for (; j + 7 < fout; j += 8) {
                _mm256_storeu_ps(
                    z + j,
                    _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_row + j),
                                    _mm256_loadu_ps(z + j)));
            }
            for (; j < fout; ++j) {
                z[j] += x[k] * w_row[j];
            }
        }
    }
}

static void dense_softmax_forward(DenseBinding* d, int64_t N) {
    const int64_t fout = d->fan_out;
    for (int64_t n = 0; n < N; ++n) {
        const float* z = d->z + n * fout;
        float* out = d->output + n * fout;
        float max_z = z[0];
        for (int64_t j = 1; j < fout; ++j) {
            if (z[j] > max_z) max_z = z[j];
        }
        double sum_exp = 0.0;
        for (int64_t j = 0; j < fout; ++j) {
            out[j] = std::exp((double)z[j] - (double)max_z);
            sum_exp += (double)out[j];
        }
        const float inv = (float)(1.0 / sum_exp);
        for (int64_t j = 0; j < fout; ++j) {
            out[j] *= inv;
        }
    }
}

static void dense_relu_forward(DenseBinding* d, int64_t N) {
    const int64_t fout = d->fan_out;
    for (int64_t n = 0; n < N; ++n) {
        const float* z = d->z + n * fout;
        float* out = d->output + n * fout;
        for (int64_t j = 0; j < fout; ++j) {
            out[j] = z[j] > 0.0f ? z[j] : 0.0f;
        }
    }
}

static void dense_forward_layer(ContractExecCtx* ctx, int32_t di, bool is_last) {
    DenseBinding* d = &ctx->dense[di];
    d->input_cache = ctx->act;
    dense_linear_forward(ctx, d, ctx->act);
    if (is_last) {
        dense_softmax_forward(d, ctx->N);
        if (ctx->loss_out) {
            softmax_cross_entropy_loss(
                d->output, ctx->y, ctx->N, d->fan_out, ctx->loss_out);
        }
    } else {
        dense_relu_forward(d, ctx->N);
    }
    ctx->act = d->output;
}

static void dense_backward_layer(ContractExecCtx* ctx, int32_t di, bool is_last) {
    DenseBinding* d = &ctx->dense[di];
    const int64_t N = ctx->N;
    const int64_t fin = d->fan_in;
    const int64_t fout = d->fan_out;
    const float inv_m = 1.0f / (float)N;

    if (is_last) {
        for (int64_t n = 0; n < N; ++n) {
            const float* out = d->output + n * fout;
            const float* t = ctx->y + n * fout;
            float* delta = d->delta + n * fout;
            for (int64_t j = 0; j < fout; ++j) {
                delta[j] = out[j] - t[j];
            }
        }
    } else {
        DenseBinding* d_next = &ctx->dense[di + 1];
        const int64_t fout_next = d_next->fan_out;
        for (int64_t n = 0; n < N; ++n) {
            const float* delta_next = d_next->delta + n * fout_next;
            float* delta = d->delta + n * fout;
            for (int64_t j = 0; j < fout; ++j) {
                double sum = 0.0;
                for (int64_t k = 0; k < fout_next; ++k) {
                    sum += (double)delta_next[k] * (double)d_next->W[j * fout_next + k];
                }
                delta[j] = (float)sum;
            }
        }
        for (int64_t n = 0; n < N; ++n) {
            float* delta = d->delta + n * fout;
            const float* z = d->z + n * fout;
            for (int64_t j = 0; j < fout; ++j) {
                if (z[j] <= 0.0f) delta[j] = 0.0f;
            }
        }
    }

    // db is accumulated into below; the Python side already zeroes db/dW
    // for every dense layer during prepare_step's overlap window (both the
    // sync and async slot paths), so no memset is needed here.
    //
    // dW row k is owned by one thread, so it is written (not accumulated) and
    // needs no prior memset either.
    ml_omp_before_parallel();
    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < fin; ++k) {
        float* __restrict dw_row = d->dW + k * fout;
        int64_t j = 0;
        for (; j + 7 < fout; j += 8) {
            _mm256_storeu_ps(dw_row + j, _mm256_setzero_ps());
        }
        for (; j < fout; ++j) {
            dw_row[j] = 0.0f;
        }
        for (int64_t n = 0; n < N; ++n) {
            const float xk = d->input_cache[n * fin + k] * inv_m;
            const __m256 xv = _mm256_set1_ps(xk);
            const float* __restrict delta = d->delta + n * fout;
            j = 0;
            for (; j + 7 < fout; j += 8) {
                _mm256_storeu_ps(
                    dw_row + j,
                    _mm256_fmadd_ps(xv, _mm256_loadu_ps(delta + j),
                                    _mm256_loadu_ps(dw_row + j)));
            }
            for (; j < fout; ++j) {
                dw_row[j] += xk * delta[j];
            }
        }
    }

    for (int64_t n = 0; n < N; ++n) {
        const float* delta = d->delta + n * fout;
        for (int64_t j = 0; j < fout; ++j) {
            d->db[j] += delta[j] * inv_m;
        }
    }

    // dx[n][k] = dot(delta[n][:], W[k][:]) over fan_out.
    ml_omp_before_parallel();
    #pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        const float* __restrict delta = d->delta + n * fout;
        float* __restrict dx = d->dx_flat + n * fin;
        for (int64_t k = 0; k < fin; ++k) {
            const float* __restrict w_row = d->W + k * fout;
            __m256 acc = _mm256_setzero_ps();
            int64_t j = 0;
            for (; j + 7 < fout; j += 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(delta + j),
                                      _mm256_loadu_ps(w_row + j), acc);
            }
            float sum = hsum256_ps(acc);
            for (; j < fout; ++j) {
                sum += delta[j] * w_row[j];
            }
            dx[k] = sum;
        }
    }
}

static void adam_update_tensor(
    const float* param, const float* grad, const float* ms, const float* vs,
    float* param_next, float* ms_next, float* vs_next,
    int64_t count, const AdamBinding* a, float lr, float decay_factor
) {
    const float t = (float)a->t;
    const float bc1 = 1.0f - std::pow(a->beta1, t);
    const float bc2 = 1.0f - std::pow(a->beta2, t);
    const float step_scale = lr * (std::sqrt(bc2) / bc1);
    const float eps_c = a->eps * std::sqrt(bc2);
    const float one_minus_beta1 = 1.0f - a->beta1;
    const float one_minus_beta2 = 1.0f - a->beta2;
    if (!param_next) param_next = const_cast<float*>(param);
    if (!ms_next) ms_next = const_cast<float*>(ms);
    if (!vs_next) vs_next = const_cast<float*>(vs);
    for (int64_t i = 0; i < count; ++i) {
        const float next_m = a->beta1 * ms[i] + one_minus_beta1 * grad[i];
        const float next_v = a->beta2 * vs[i] + one_minus_beta2 * grad[i] * grad[i];
        float next_param = param[i];
        if (decay_factor > 0.0f) next_param -= decay_factor * param[i];
        next_param -= step_scale * next_m / (std::sqrt(next_v) + eps_c);
        ms_next[i] = next_m;
        vs_next[i] = next_v;
        param_next[i] = next_param;
    }
}

static void adam_apply_mhsa(ContractExecCtx* ctx, float decay_factor) {
    if (!ctx->has_mhsa) return;
    MhsaBinding* m = &ctx->mhsa;
    AdamBinding* a = &ctx->adam;
    const int64_t D = m->D;
    const int64_t Hff = m->ffn_hidden;
    const int64_t A = m->action_dim;
    if (D < 1 || Hff < 1 || A < 1 || m->num_layers < 1) return;

    for (int64_t li = 0; li < m->num_layers; ++li) {
        MhsaLayerBind* L = &m->layers[li];
        if (!L->ms_W_qkv || !L->vs_W_qkv) continue;
        adam_update_tensor(
            L->W_qkv, L->dW_qkv, L->ms_W_qkv, L->vs_W_qkv,
            L->W_qkv_next, L->ms_W_qkv_next, L->vs_W_qkv_next,
            D * 3 * D, a, ctx->lr, decay_factor);
        adam_update_tensor(
            L->b_qkv, L->db_qkv, L->ms_b_qkv, L->vs_b_qkv,
            L->b_qkv_next, L->ms_b_qkv_next, L->vs_b_qkv_next,
            3 * D, a, ctx->lr, 0.0f);
        adam_update_tensor(
            L->W_o, L->dW_o, L->ms_W_o, L->vs_W_o,
            L->W_o_next, L->ms_W_o_next, L->vs_W_o_next,
            D * D, a, ctx->lr, decay_factor);
        adam_update_tensor(
            L->b_o, L->db_o, L->ms_b_o, L->vs_b_o,
            L->b_o_next, L->ms_b_o_next, L->vs_b_o_next,
            D, a, ctx->lr, 0.0f);
        adam_update_tensor(
            L->W_ff1, L->dW_ff1, L->ms_W_ff1, L->vs_W_ff1,
            L->W_ff1_next, L->ms_W_ff1_next, L->vs_W_ff1_next,
            D * Hff, a, ctx->lr, decay_factor);
        adam_update_tensor(
            L->b_ff1, L->db_ff1, L->ms_b_ff1, L->vs_b_ff1,
            L->b_ff1_next, L->ms_b_ff1_next, L->vs_b_ff1_next,
            Hff, a, ctx->lr, 0.0f);
        adam_update_tensor(
            L->W_ff2, L->dW_ff2, L->ms_W_ff2, L->vs_W_ff2,
            L->W_ff2_next, L->ms_W_ff2_next, L->vs_W_ff2_next,
            Hff * D, a, ctx->lr, decay_factor);
        adam_update_tensor(
            L->b_ff2, L->db_ff2, L->ms_b_ff2, L->vs_b_ff2,
            L->b_ff2_next, L->ms_b_ff2_next, L->vs_b_ff2_next,
            D, a, ctx->lr, 0.0f);
        adam_update_tensor(
            L->ln1_gamma, L->d_ln1_gamma, L->ms_ln1_g, L->vs_ln1_g,
            L->ln1_gamma_next, L->ms_ln1_g_next, L->vs_ln1_g_next,
            D, a, ctx->lr, 0.0f);
        adam_update_tensor(
            L->ln1_beta, L->d_ln1_beta, L->ms_ln1_b, L->vs_ln1_b,
            L->ln1_beta_next, L->ms_ln1_b_next, L->vs_ln1_b_next,
            D, a, ctx->lr, 0.0f);
        adam_update_tensor(
            L->ln2_gamma, L->d_ln2_gamma, L->ms_ln2_g, L->vs_ln2_g,
            L->ln2_gamma_next, L->ms_ln2_g_next, L->vs_ln2_g_next,
            D, a, ctx->lr, 0.0f);
        adam_update_tensor(
            L->ln2_beta, L->d_ln2_beta, L->ms_ln2_b, L->vs_ln2_b,
            L->ln2_beta_next, L->ms_ln2_b_next, L->vs_ln2_b_next,
            D, a, ctx->lr, 0.0f);
    }

    if (m->ms_W_act && m->vs_W_act) {
        adam_update_tensor(
            m->W_act, m->dW_act, m->ms_W_act, m->vs_W_act,
            m->W_act_next, m->ms_W_act_next, m->vs_W_act_next,
            D * A, a, ctx->lr, decay_factor);
        adam_update_tensor(
            m->b_act, m->db_act, m->ms_b_act, m->vs_b_act,
            m->b_act_next, m->ms_b_act_next, m->vs_b_act_next,
            A, a, ctx->lr, 0.0f);
    }
    if (m->pos && m->d_pos && m->ms_pos && m->vs_pos && m->max_seq_len > 0) {
        adam_update_tensor(
            m->pos, m->d_pos, m->ms_pos, m->vs_pos,
            m->pos_next, m->ms_pos_next, m->vs_pos_next,
            m->max_seq_len * D, a, ctx->lr, decay_factor);
    }
    if (m->W_in && m->dW_in && m->ms_W_in && m->vs_W_in) {
        adam_update_tensor(
            m->W_in, m->dW_in, m->ms_W_in, m->vs_W_in,
            m->W_in_next, m->ms_W_in_next, m->vs_W_in_next,
            D * D, a, ctx->lr, decay_factor);
        adam_update_tensor(
            m->b_in, m->db_in, m->ms_b_in, m->vs_b_in,
            m->b_in_next, m->ms_b_in_next, m->vs_b_in_next,
            D, a, ctx->lr, 0.0f);
    }
}

static void adam_apply_all(ContractExecCtx* ctx) {
    AdamBinding* a = &ctx->adam;
    a->t += 1;
    const float decay_factor = (ctx->lam_l2 > 0.0f)
        ? ctx->lr * (ctx->lam_l2 / (float)ctx->N)
        : 0.0f;

    for (int32_t li = 0; li < ctx->num_layers; ++li) {
        LayerBinding* L = &ctx->layers[li];
        if (L->w_count <= 0 || !L->ms_w) continue;
        adam_update_tensor(
            L->W, L->dW, L->ms_w, L->vs_w,
            L->W_next, L->ms_w_next, L->vs_w_next,
            L->w_count, a, ctx->lr, decay_factor);
        adam_update_tensor(
            L->b, L->db, L->ms_b, L->vs_b,
            L->b_next, L->ms_b_next, L->vs_b_next,
            L->b_count, a, ctx->lr, 0.0f);
    }

    for (int32_t di = 0; di < ctx->num_dense; ++di) {
        DenseBinding* d = &ctx->dense[di];
        if (!d->ms_w) continue;
        adam_update_tensor(
            d->W, d->dW, d->ms_w, d->vs_w,
            d->W_next, d->ms_w_next, d->vs_w_next,
            d->fan_in * d->fan_out, a, ctx->lr, decay_factor);
        adam_update_tensor(
            d->b, d->db, d->ms_b, d->vs_b,
            d->b_next, d->ms_b_next, d->vs_b_next,
            d->fan_out, a, ctx->lr, 0.0f);
    }

    adam_apply_mhsa(ctx, decay_factor);
}

#if defined(ML_ENGINE_PROFILE_CONTRACT_THREADS) && defined(__linux__) && defined(_OPENMP)
namespace {

constexpr int32_t CONTRACT_PROFILE_MAX_THREADS = 128;

struct ContractThreadSample {
    pid_t tid = 0;
    double wall_start = 0.0;
    double cpu_start = 0.0;
};

struct ContractThreadAggregate {
    pid_t tid = 0;
    uint64_t calls = 0;
    double wall_seconds = 0.0;
    double cpu_seconds = 0.0;
    double min_wall_seconds = 0.0;
    double max_wall_seconds = 0.0;
    uint64_t tid_mismatches = 0;
};

ContractThreadSample g_contract_thread_samples[CONTRACT_PROFILE_MAX_THREADS];
ContractThreadAggregate g_contract_thread_totals[CONTRACT_PROFILE_MAX_THREADS];
int32_t g_contract_profile_team_size = 0;
bool g_contract_profile_reported = false;

// Master-thread per-opcode wall time inside one contract execution.
constexpr int32_t CONTRACT_PROFILE_MAX_OPCODE = 32;

struct ContractOpAggregate {
    uint64_t calls = 0;
    double wall_seconds = 0.0;
};

ContractOpAggregate g_contract_op_totals[CONTRACT_PROFILE_MAX_OPCODE];
uint64_t g_contract_profile_steps = 0;

const char* contract_opcode_name(int32_t opcode) {
    switch (opcode) {
        case OP_CONV2D_FWD: return "CONV2D_FWD";
        case OP_CONV2D_BWD: return "CONV2D_BWD";
        case OP_RELU_FWD: return "RELU_FWD";
        case OP_RELU_BWD: return "RELU_BWD";
        case OP_MAXPOOL_FWD: return "MAXPOOL_FWD";
        case OP_MAXPOOL_BWD: return "MAXPOOL_BWD";
        case OP_FLATTEN_FWD: return "FLATTEN_FWD";
        case OP_FLATTEN_BWD: return "FLATTEN_BWD";
        case OP_DENSE_FWD: return "DENSE_FWD";
        case OP_DENSE_BWD: return "DENSE_BWD";
        case OP_ADAM_APPLY: return "ADAM_APPLY";
        case OP_CONV_BLOCK_FWD: return "CONV_BLOCK_FWD";
        case OP_CONV_BLOCK_BWD: return "CONV_BLOCK_BWD";
        case OP_MHSA_BLOCK_FWD: return "MHSA_BLOCK_FWD";
        case OP_MHSA_BLOCK_BWD: return "MHSA_BLOCK_BWD";
        case OP_MHSA_ACTION_FWD: return "MHSA_ACTION_FWD";
        case OP_MHSA_ACTION_BWD: return "MHSA_ACTION_BWD";
        default: return "UNKNOWN";
    }
}

double contract_profile_clock(clockid_t clock_id) {
    timespec ts{};
    clock_gettime(clock_id, &ts);
    return static_cast<double>(ts.tv_sec) +
        static_cast<double>(ts.tv_nsec) * 1.0e-9;
}

pid_t contract_profile_tid() {
    return static_cast<pid_t>(syscall(SYS_gettid));
}

bool is_training_contract(const ContractOpRow* ops, int32_t op_count) {
    for (int32_t i = 0; i < op_count; ++i) {
        if (ops[i].opcode == OP_CONV_BLOCK_BWD ||
            ops[i].opcode == OP_DENSE_BWD ||
            ops[i].opcode == OP_MHSA_BLOCK_BWD ||
            ops[i].opcode == OP_MHSA_ACTION_BWD ||
            ops[i].opcode == OP_ADAM_APPLY) {
            return true;
        }
    }
    return false;
}

void contract_profile_begin() {
    #pragma omp parallel
    {
        const int32_t idx = omp_get_thread_num();
        if (idx < CONTRACT_PROFILE_MAX_THREADS) {
            ContractThreadSample& sample = g_contract_thread_samples[idx];
            sample.tid = contract_profile_tid();
            sample.wall_start = contract_profile_clock(CLOCK_MONOTONIC);
            sample.cpu_start = contract_profile_clock(CLOCK_THREAD_CPUTIME_ID);
        }
        #pragma omp single
        {
            g_contract_profile_team_size = std::min(
                omp_get_num_threads(), CONTRACT_PROFILE_MAX_THREADS);
        }
    }
}

void contract_profile_end() {
    #pragma omp parallel
    {
        const double cpu_end = contract_profile_clock(CLOCK_THREAD_CPUTIME_ID);
        const double wall_end = contract_profile_clock(CLOCK_MONOTONIC);
        const pid_t tid = contract_profile_tid();
        const int32_t idx = omp_get_thread_num();
        if (idx < CONTRACT_PROFILE_MAX_THREADS) {
            const ContractThreadSample& sample = g_contract_thread_samples[idx];
            ContractThreadAggregate& total = g_contract_thread_totals[idx];
            if (sample.tid != tid) {
                ++total.tid_mismatches;
            } else {
                const double wall = wall_end - sample.wall_start;
                const double cpu = cpu_end - sample.cpu_start;
                total.tid = tid;
                ++total.calls;
                total.wall_seconds += wall;
                total.cpu_seconds += cpu;
                if (total.calls == 1 || wall < total.min_wall_seconds) {
                    total.min_wall_seconds = wall;
                }
                total.max_wall_seconds = std::max(total.max_wall_seconds, wall);
            }
        }
    }
}

void contract_profile_report() {
    if (g_contract_profile_reported) {
        return;
    }
    g_contract_profile_reported = true;
    if (g_contract_profile_steps > 0) {
        const double steps = static_cast<double>(g_contract_profile_steps);
        std::fprintf(
            stderr,
            "[CONTRACT_OP_PROFILE] steps=%llu (master-thread wall per contract op)\n",
            static_cast<unsigned long long>(g_contract_profile_steps));
        std::fprintf(
            stderr,
            "[CONTRACT_OP_PROFILE] opcode name calls_per_step avg_us_per_step\n");
        double total_us = 0.0;
        for (int32_t opcode = 0; opcode < CONTRACT_PROFILE_MAX_OPCODE; ++opcode) {
            const ContractOpAggregate& op = g_contract_op_totals[opcode];
            if (op.calls == 0) {
                continue;
            }
            const double us_per_step = op.wall_seconds * 1.0e6 / steps;
            total_us += us_per_step;
            std::fprintf(
                stderr,
                "[CONTRACT_OP_PROFILE] %d %s %.2f %.3f\n",
                opcode,
                contract_opcode_name(opcode),
                static_cast<double>(op.calls) / steps,
                us_per_step);
        }
        std::fprintf(
            stderr,
            "[CONTRACT_OP_PROFILE] TOTAL_OPS_US_PER_STEP %.3f\n",
            total_us);
    }
    std::fprintf(
        stderr,
        "[CONTRACT_THREAD_PROFILE] omp_idx tid calls avg_wall_us avg_cpu_us "
        "avg_offcpu_us cpu_pct min_wall_us max_wall_us tid_mismatches\n");
    for (int32_t idx = 0; idx < g_contract_profile_team_size; ++idx) {
        const ContractThreadAggregate& total = g_contract_thread_totals[idx];
        if (total.calls == 0) {
            continue;
        }
        const double avg_wall = total.wall_seconds / static_cast<double>(total.calls);
        const double avg_cpu = total.cpu_seconds / static_cast<double>(total.calls);
        const double avg_offcpu = std::max(0.0, avg_wall - avg_cpu);
        const double cpu_pct = avg_wall > 0.0 ? 100.0 * avg_cpu / avg_wall : 0.0;
        std::fprintf(
            stderr,
            "[CONTRACT_THREAD_PROFILE] %d %d %llu %.3f %.3f %.3f %.2f %.3f %.3f %llu\n",
            idx,
            static_cast<int>(total.tid),
            static_cast<unsigned long long>(total.calls),
            avg_wall * 1.0e6,
            avg_cpu * 1.0e6,
            avg_offcpu * 1.0e6,
            cpu_pct,
            total.min_wall_seconds * 1.0e6,
            total.max_wall_seconds * 1.0e6,
            static_cast<unsigned long long>(total.tid_mismatches));
    }
    std::fflush(stderr);
}

class ContractThreadProfileScope {
public:
    explicit ContractThreadProfileScope(bool enabled) : enabled_(enabled) {
        if (enabled_) {
            contract_profile_begin();
        }
    }

    ~ContractThreadProfileScope() {
        if (enabled_) {
            contract_profile_end();
        }
    }

    bool enabled() const { return enabled_; }

private:
    bool enabled_;
};

}  // namespace
#endif

static int32_t run_contract_training_step_impl(
    const ContractOpRow* ops,
    int32_t op_count,
    ContractExecCtx* ctx
) {
    if (!ops || !ctx || op_count <= 0 || ctx->N <= 0) {
        return -1;
    }
    reset_brgemm_dw_x_pack_request();

#if defined(ML_ENGINE_PROFILE_CONTRACT_THREADS) && defined(__linux__) && defined(_OPENMP)
    ContractThreadProfileScope thread_profile(is_training_contract(ops, op_count));
#endif

    const float inv_m = 1.0f / (float)ctx->N;
    ctx->act = const_cast<float*>(ctx->X);

    for (int32_t i = 0; i < op_count; ++i) {
        const ContractOpRow* op = &ops[i];
#if defined(ML_ENGINE_PROFILE_CONTRACT_THREADS) && defined(__linux__) && defined(_OPENMP)
        const bool op_profile_enabled =
            thread_profile.enabled() &&
            op->opcode >= 0 && op->opcode < CONTRACT_PROFILE_MAX_OPCODE;
        const double op_profile_start =
            op_profile_enabled ? contract_profile_clock(CLOCK_MONOTONIC) : 0.0;
#endif
        switch (op->opcode) {
            case OP_CONV_BLOCK_FWD: {
                if (op->layer_idx < 0 || op->layer_idx >= ctx->num_layers) return -2;
                LayerBinding* L = &ctx->layers[op->layer_idx];
                const float* x_in = (op->layer_idx == 0) ? ctx->X : ctx->act;
                L->x_cache = const_cast<float*>(x_in);
                const int64_t conv_out_w = (L->W_in + 2 * L->conv_pad - L->k_w) / L->conv_stride + 1;
                // Trust Python bind: train uses SIMD-padded stride, eval/predict
                // uses dense stride matching ensure_conv_block_eval. Overwriting
                // with round_up_simd here made forward write past dense eval
                // buffers (heap corruption / async predict abort).
                if (L->conv_out_w_stride < conv_out_w) {
                    return -2;
                }
                const int64_t conv_out_h = (L->H + 2 * L->conv_pad - L->k_h) / L->conv_stride + 1;
                L->pool_out_h = (conv_out_h - L->pool_size) / L->pool_stride + 1;
                L->pool_out_w = (conv_out_w - L->pool_size) / L->pool_stride + 1;
                int32_t st = direct_conv_block_forward_avx2(
                    x_in, L->W, L->b, L->out_conv, L->out_pool, L->argmax,
                    ctx->N, L->C_in, L->H, L->W_in, L->W_stride, L->C_out,
                    L->k_h, L->k_w, L->conv_stride, L->conv_pad, L->conv_out_w_stride,
                    L->pool_size, L->pool_stride);
                if (st != 0) return st;
                L->conv_act_cache = L->out_conv;
                ctx->act = L->out_pool;
                ctx->flat_dim = L->C_out * L->pool_out_h * L->pool_out_w;
                if (op->layer_idx + 1 < ctx->num_layers) {
                    LayerBinding* Nxt = &ctx->layers[op->layer_idx + 1];
                    (void)request_brgemm_dw_x_pack(
                        ctx->act, ctx->N, Nxt->C_in, Nxt->H, Nxt->W_in,
                        Nxt->W_stride, Nxt->C_out, Nxt->k_h, Nxt->k_w,
                        Nxt->conv_stride, Nxt->conv_pad);
                }
                break;
            }
            case OP_FLATTEN_FWD:
                break;
            case OP_DENSE_FWD: {
                if (op->layer_idx < 0 || op->layer_idx >= ctx->num_dense) return -4;
                const bool is_last = (op->layer_idx == ctx->num_dense - 1);
                dense_forward_layer(ctx, op->layer_idx, is_last);
                break;
            }
            case OP_DENSE_BWD: {
                if (op->layer_idx < 0 || op->layer_idx >= ctx->num_dense) return -5;
                const bool is_last = (op->layer_idx == ctx->num_dense - 1);
                dense_backward_layer(ctx, op->layer_idx, is_last);
                ctx->act = ctx->dense[op->layer_idx].dx_flat;
                break;
            }
            case OP_FLATTEN_BWD:
                break;
            case OP_CONV_BLOCK_BWD: {
                if (op->layer_idx < 0 || op->layer_idx >= ctx->num_layers) return -3;
                LayerBinding* L = &ctx->layers[op->layer_idx];
                // Layer 0's dx is d(loss)/d(input image): no upstream consumer.
                // Passing null makes the backward skip the dx solve entirely.
                const bool need_dx = (op->layer_idx != 0);
                int32_t st = direct_conv_block_backward_avx2(
                    ctx->act, L->argmax, L->x_cache, L->W, L->conv_act_cache,
                    L->d_conv, need_dx ? L->dx : nullptr, L->dW, L->db,
                    ctx->N, L->C_in, L->H, L->W_in, L->W_stride, L->C_out,
                    L->k_h, L->k_w, L->conv_stride, L->conv_pad, L->conv_out_w_stride,
                    L->pool_size, L->pool_stride, L->pool_out_h, L->pool_out_w, inv_m,
                    L->d_conv_prezeroed, need_dx ? L->dx_prezeroed : 0, L->dw_prezeroed);
                if (st != 0) return st;
                if (need_dx) {
                    ctx->act = L->dx;
                }
                break;
            }
            case OP_ADAM_APPLY:
                if (!ctx->skip_adam) {
                    adam_apply_all(ctx);
                }
                break;
            case OP_MHSA_BLOCK_FWD: {
                if (!ctx->has_mhsa) return -20;
                int32_t st = mhsa_block_forward(ctx->X, &ctx->mhsa);
                if (st != 0) return st;
                ctx->act = ctx->mhsa.O;
                break;
            }
            case OP_MHSA_BLOCK_BWD: {
                if (!ctx->has_mhsa) return -21;
                int32_t st = mhsa_block_backward(ctx->X, &ctx->mhsa);
                if (st != 0) return st;
                if (ctx->mhsa.dX) ctx->act = ctx->mhsa.dX;
                break;
            }
            case OP_MHSA_ACTION_FWD: {
                if (!ctx->has_mhsa) return -22;
                int32_t st = mhsa_action_forward(&ctx->mhsa);
                if (st != 0) return st;
                ctx->act = ctx->mhsa.actions;
                break;
            }
            case OP_MHSA_ACTION_BWD: {
                if (!ctx->has_mhsa) return -23;
                int32_t st = mhsa_action_backward(&ctx->mhsa);
                if (st != 0) return st;
                break;
            }
            default:
                std::fprintf(stderr, "[CONTRACT] unsupported opcode %d\n", op->opcode);
                return -10;
        }
#if defined(ML_ENGINE_PROFILE_CONTRACT_THREADS) && defined(__linux__) && defined(_OPENMP)
        if (op_profile_enabled) {
            ContractOpAggregate& total = g_contract_op_totals[op->opcode];
            ++total.calls;
            total.wall_seconds +=
                contract_profile_clock(CLOCK_MONOTONIC) - op_profile_start;
        }
#endif
    }
#if defined(ML_ENGINE_PROFILE_CONTRACT_THREADS) && defined(__linux__) && defined(_OPENMP)
    if (thread_profile.enabled()) {
        ++g_contract_profile_steps;
    }
#endif
    return 0;
}

// ---------------------------------------------------------------------------
// F2.3 / F4: per-tenant async mailbox (one worker thread per native tenant)
// ---------------------------------------------------------------------------
namespace {

enum AsyncState : int32_t {
    ASYNC_IDLE = 0,
    ASYNC_RUNNING = 1,
    ASYNC_READY = 2,
};

struct AsyncMailbox {
    std::mutex mtx;
    std::condition_variable job_cv;
    std::condition_variable completion_cv;
    std::thread worker;
    bool worker_started = false;
    bool shutdown = false;
    bool has_job = false;
    const ContractOpRow* ops = nullptr;
    int32_t op_count = 0;
    ContractExecCtx* ctx = nullptr;
    int64_t submit_token = 0;
    std::atomic<int32_t> state{ASYNC_IDLE};
    int64_t ready_token = 0;
    int32_t ready_status = 0;
    int32_t tenant_id = 0;
};

AsyncMailbox g_async_mailboxes[NATIVE_MAX_TENANTS];

void trace_mailbox_invariant_locked(AsyncMailbox& mb, const char* where) {
    const int32_t state = mb.state.load(std::memory_order_relaxed);
    const bool bad_job_state = mb.has_job && state != ASYNC_RUNNING;
    const bool bad_idle_job = state == ASYNC_IDLE && mb.has_job;
    const bool bad_ready_job = state == ASYNC_READY && mb.has_job;
    const bool bad_running_worker = state == ASYNC_RUNNING && !mb.worker_started;
    if (bad_job_state || bad_idle_job || bad_ready_job || bad_running_worker) {
        std::fprintf(
            stderr,
            "[MAILBOX_DESYNC][native] tenant=%d where=%s state=%d has_job=%d "
            "worker_started=%d shutdown=%d submit_token=%lld ready_token=%lld\n",
            mb.tenant_id,
            where,
            state,
            mb.has_job ? 1 : 0,
            mb.worker_started ? 1 : 0,
            mb.shutdown ? 1 : 0,
            static_cast<long long>(mb.submit_token),
            static_cast<long long>(mb.ready_token)
        );
        std::fflush(stderr);
    }
}

void contract_async_worker_loop(int32_t tenant_id) {
    AsyncMailbox& mb = g_async_mailboxes[tenant_id];
    native_tenant_put(tenant_id);
    for (;;) {
        std::unique_lock<std::mutex> lock(mb.mtx);
        mb.job_cv.wait(lock, [&mb] {
            return mb.shutdown || mb.has_job;
        });
        if (mb.shutdown) {
            break;
        }

        const ContractOpRow* ops = mb.ops;
        const int32_t op_count = mb.op_count;
        ContractExecCtx* ctx = mb.ctx;
        const int64_t token = mb.submit_token;
        mb.has_job = false;
        trace_mailbox_invariant_locked(mb, "worker_take");
        lock.unlock();

        native_tenant_put(tenant_id);
        const int32_t status = run_contract_training_step_impl(ops, op_count, ctx);

        {
            std::lock_guard<std::mutex> ready_lock(mb.mtx);
            mb.ready_token = token;
            mb.ready_status = status;
            mb.state.store(ASYNC_READY, std::memory_order_release);
            trace_mailbox_invariant_locked(mb, "worker_ready");
        }
        mb.completion_cv.notify_one();
    }
}

void ensure_async_worker_started(AsyncMailbox& mb) {
    if (mb.worker_started) {
        return;
    }
    mb.worker_started = true;
    mb.tenant_id = native_tenant_get();
    const int32_t tid = mb.tenant_id;
    mb.worker = std::thread(contract_async_worker_loop, tid);
}

void shutdown_async_mailbox(AsyncMailbox& mb) {
    {
        std::lock_guard<std::mutex> lock(mb.mtx);
        mb.shutdown = true;
        trace_mailbox_invariant_locked(mb, "shutdown");
    }
    mb.job_cv.notify_all();
    mb.completion_cv.notify_all();
    if (mb.worker.joinable()) {
        mb.worker.join();
    }
    mb.worker_started = false;
    mb.shutdown = false;
    mb.has_job = false;
    mb.state.store(ASYNC_IDLE, std::memory_order_release);
}

}  // namespace

extern "C" {

ML_ENGINE_EXPORT int32_t run_contract_training_step(
    const ContractOpRow* ops,
    int32_t op_count,
    ContractExecCtx* ctx
) {
    return run_contract_training_step_impl(ops, op_count, ctx);
}

// Deprecated compatibility export. Completion is reaped by the submitting
// Python thread; the native worker must never call into Python.
typedef void (*ContractCompletionFn)(int64_t step_token, int32_t status);
ML_ENGINE_EXPORT void contract_register_completion_callback(
    ContractCompletionFn
) {
}

// Returns 0 on accept, -2 if busy/running, -3 if completion must be reaped first.
ML_ENGINE_EXPORT int32_t submit_contract_training_step(
    const ContractOpRow* ops,
    int32_t op_count,
    ContractExecCtx* ctx,
    int64_t step_token
) {
    if (!ops || !ctx || op_count <= 0 || ctx->N <= 0) {
        return -1;
    }

    AsyncMailbox& mb = g_async_mailboxes[native_tenant_get()];
    const int32_t state = mb.state.load(std::memory_order_acquire);
    if (state == ASYNC_READY) {
        return -3;
    }
    if (state == ASYNC_RUNNING) {
        return -2;
    }

    ensure_async_worker_started(mb);

    {
        std::lock_guard<std::mutex> lock(mb.mtx);
        if (mb.has_job || mb.state.load(std::memory_order_relaxed) != ASYNC_IDLE) {
            return -2;
        }
        mb.ops = ops;
        mb.op_count = op_count;
        mb.ctx = ctx;
        mb.submit_token = step_token;
        mb.has_job = true;
        mb.state.store(ASYNC_RUNNING, std::memory_order_release);
        trace_mailbox_invariant_locked(mb, "submit");
    }
    mb.job_cv.notify_one();
    return 0;
}

// Returns 1 if a completion was reaped, 0 if not ready, -1 on bad args.
ML_ENGINE_EXPORT int32_t try_reap_contract_completion(
    int64_t* out_step_token,
    int32_t* out_status
) {
    if (!out_step_token || !out_status) {
        return -1;
    }
    AsyncMailbox& mb = g_async_mailboxes[native_tenant_get()];
    std::lock_guard<std::mutex> lock(mb.mtx);
    if (mb.state.load(std::memory_order_acquire) != ASYNC_READY) {
        return 0;
    }
    *out_step_token = mb.ready_token;
    *out_status = mb.ready_status;
    mb.state.store(ASYNC_IDLE, std::memory_order_release);
    trace_mailbox_invariant_locked(mb, "try_reap");
    return 1;
}

// Block without spinning until READY, then reap it. timeout_ms < 0 waits
// indefinitely; timeout_ms == 0 is non-blocking.
ML_ENGINE_EXPORT int32_t wait_contract_completion(
    int64_t* out_step_token,
    int32_t* out_status,
    int64_t timeout_ms
) {
    if (!out_step_token || !out_status) {
        return -1;
    }

    AsyncMailbox& mb = g_async_mailboxes[native_tenant_get()];
    std::unique_lock<std::mutex> lock(mb.mtx);
    const auto ready_or_shutdown = [&mb] {
        return mb.shutdown ||
            mb.state.load(std::memory_order_acquire) == ASYNC_READY;
    };
    bool signaled = true;
    if (timeout_ms < 0) {
        mb.completion_cv.wait(lock, ready_or_shutdown);
    } else {
        signaled = mb.completion_cv.wait_for(
            lock, std::chrono::milliseconds(timeout_ms), ready_or_shutdown);
    }
    if (!signaled ||
        mb.state.load(std::memory_order_acquire) != ASYNC_READY) {
        return 0;
    }

    *out_step_token = mb.ready_token;
    *out_status = mb.ready_status;
    mb.state.store(ASYNC_IDLE, std::memory_order_release);
    trace_mailbox_invariant_locked(mb, "wait_reap");
    return 1;
}

// Diagnostic-only snapshot used by the Python-side cross-mailbox invariant
// checker whenever native async submit is enabled.
ML_ENGINE_EXPORT int32_t contract_async_debug_snapshot(
    int32_t* out_state,
    int32_t* out_has_job,
    int32_t* out_worker_started,
    int32_t* out_shutdown,
    int64_t* out_submit_token,
    int64_t* out_ready_token
) {
    if (!out_state || !out_has_job || !out_worker_started || !out_shutdown ||
        !out_submit_token || !out_ready_token) {
        return -1;
    }
    AsyncMailbox& mb = g_async_mailboxes[native_tenant_get()];
    std::lock_guard<std::mutex> lock(mb.mtx);
    *out_state = mb.state.load(std::memory_order_relaxed);
    *out_has_job = mb.has_job ? 1 : 0;
    *out_worker_started = mb.worker_started ? 1 : 0;
    *out_shutdown = mb.shutdown ? 1 : 0;
    *out_submit_token = mb.submit_token;
    *out_ready_token = mb.ready_token;
    trace_mailbox_invariant_locked(mb, "python_snapshot");
    return 0;
}

ML_ENGINE_EXPORT int32_t contract_async_in_flight() {
    AsyncMailbox& mb = g_async_mailboxes[native_tenant_get()];
    const int32_t state = mb.state.load(std::memory_order_acquire);
    return (state == ASYNC_RUNNING) ? 1 : 0;
}

ML_ENGINE_EXPORT void contract_async_shutdown_tenant(int32_t tenant_id) {
    if (tenant_id < 0 || tenant_id >= NATIVE_MAX_TENANTS) {
        return;
    }
    shutdown_async_mailbox(g_async_mailboxes[tenant_id]);
}

ML_ENGINE_EXPORT void contract_async_shutdown() {
    for (int32_t t = 0; t < NATIVE_MAX_TENANTS; ++t) {
        shutdown_async_mailbox(g_async_mailboxes[t]);
    }
#if defined(ML_ENGINE_PROFILE_CONTRACT_THREADS) && defined(__linux__) && defined(_OPENMP)
    contract_profile_report();
#endif
}

}  // extern "C"
