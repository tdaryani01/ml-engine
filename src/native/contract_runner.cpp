// contract_runner.cpp — Phase F: execute compiled contract list in one native call.
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

static int64_t round_up_simd(int64_t w) { return (w + 3) & ~3; }

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
    int64_t pool_out_h, int64_t pool_out_w, float inv_m);

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
};

struct DenseBinding {
    float* W;
    float* b;
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

static void dense_linear_forward(
    const ContractExecCtx* ctx, DenseBinding* d, const float* x_in
) {
    const int64_t N = ctx->N;
    const int64_t fin = d->fan_in;
    const int64_t fout = d->fan_out;
    for (int64_t n = 0; n < N; ++n) {
        const float* x = x_in + n * fin;
        float* z = d->z + n * fout;
        for (int64_t j = 0; j < fout; ++j) {
            double sum = (double)d->b[j];
            for (int64_t k = 0; k < fin; ++k) {
                sum += (double)x[k] * (double)d->W[k * fout + j];
            }
            z[j] = (float)sum;
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

    std::memset(d->dW, 0, (size_t)(fin * fout) * sizeof(float));
    std::memset(d->db, 0, (size_t)fout * sizeof(float));

    for (int64_t n = 0; n < N; ++n) {
        const float* x = d->input_cache + n * fin;
        const float* delta = d->delta + n * fout;
        for (int64_t k = 0; k < fin; ++k) {
            float* dw_row = d->dW + k * fout;
            const float xk = x[k];
            for (int64_t j = 0; j < fout; ++j) {
                dw_row[j] += xk * delta[j] * inv_m;
            }
        }
        for (int64_t j = 0; j < fout; ++j) {
            d->db[j] += delta[j] * inv_m;
        }
    }

    for (int64_t n = 0; n < N; ++n) {
        const float* delta = d->delta + n * fout;
        float* dx = d->dx_flat + n * fin;
        for (int64_t k = 0; k < fin; ++k) {
            double sum = 0.0;
            for (int64_t j = 0; j < fout; ++j) {
                sum += (double)delta[j] * (double)d->W[k * fout + j];
            }
            dx[k] = (float)sum;
        }
    }
}

static void adam_update_tensor(
    float* param, const float* grad, float* ms, float* vs,
    int64_t count, const AdamBinding* a, float lr, float decay_factor
) {
    const float t = (float)a->t;
    const float bc1 = 1.0f - std::pow(a->beta1, t);
    const float bc2 = 1.0f - std::pow(a->beta2, t);
    const float step_scale = lr * (std::sqrt(bc2) / bc1);
    const float eps_c = a->eps * std::sqrt(bc2);
    const float one_minus_beta1 = 1.0f - a->beta1;
    const float one_minus_beta2 = 1.0f - a->beta2;
    for (int64_t i = 0; i < count; ++i) {
        ms[i] = a->beta1 * ms[i] + one_minus_beta1 * grad[i];
        vs[i] = a->beta2 * vs[i] + one_minus_beta2 * grad[i] * grad[i];
        if (decay_factor > 0.0f) {
            param[i] -= decay_factor * param[i];
        }
        param[i] -= step_scale * ms[i] / (std::sqrt(vs[i]) + eps_c);
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
        adam_update_tensor(L->W, L->dW, L->ms_w, L->vs_w, L->w_count, a, ctx->lr, decay_factor);
        adam_update_tensor(L->b, L->db, L->ms_b, L->vs_b, L->b_count, a, ctx->lr, 0.0f);
    }

    for (int32_t di = 0; di < ctx->num_dense; ++di) {
        DenseBinding* d = &ctx->dense[di];
        if (!d->ms_w) continue;
        adam_update_tensor(
            d->W, d->dW, d->ms_w, d->vs_w, d->fan_in * d->fan_out, a, ctx->lr, decay_factor);
        adam_update_tensor(d->b, d->db, d->ms_b, d->vs_b, d->fan_out, a, ctx->lr, 0.0f);
    }
}

static int32_t run_contract_training_step_impl(
    const ContractOpRow* ops,
    int32_t op_count,
    ContractExecCtx* ctx
) {
    if (!ops || !ctx || op_count <= 0 || ctx->N <= 0) {
        return -1;
    }

    const float inv_m = 1.0f / (float)ctx->N;
    ctx->act = const_cast<float*>(ctx->X);

    for (int32_t i = 0; i < op_count; ++i) {
        const ContractOpRow* op = &ops[i];
        switch (op->opcode) {
            case OP_CONV_BLOCK_FWD: {
                if (op->layer_idx < 0 || op->layer_idx >= ctx->num_layers) return -2;
                LayerBinding* L = &ctx->layers[op->layer_idx];
                const float* x_in = (op->layer_idx == 0) ? ctx->X : ctx->act;
                L->x_cache = const_cast<float*>(x_in);
                const int64_t conv_out_w = (L->W_in + 2 * L->conv_pad - L->k_w) / L->conv_stride + 1;
                L->conv_out_w_stride = round_up_simd(conv_out_w);
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
                int32_t st = direct_conv_block_backward_avx2(
                    ctx->act, L->argmax, L->x_cache, L->W, L->conv_act_cache,
                    L->d_conv, L->dx, L->dW, L->db,
                    ctx->N, L->C_in, L->H, L->W_in, L->W_stride, L->C_out,
                    L->k_h, L->k_w, L->conv_stride, L->conv_pad, L->conv_out_w_stride,
                    L->pool_size, L->pool_stride, L->pool_out_h, L->pool_out_w, inv_m);
                if (st != 0) return st;
                ctx->act = L->dx;
                break;
            }
            case OP_ADAM_APPLY:
                if (!ctx->skip_adam) {
                    adam_apply_all(ctx);
                }
                break;
            default:
                std::fprintf(stderr, "[CONTRACT] unsupported opcode %d\n", op->opcode);
                return -10;
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// F2.3 / F4: non-blocking submit + completion ring (one native worker thread)
// ---------------------------------------------------------------------------
namespace {

enum AsyncState : int32_t {
    ASYNC_IDLE = 0,
    ASYNC_RUNNING = 1,
    ASYNC_READY = 2,
};

std::mutex g_async_mtx;
std::condition_variable g_async_cv;
std::thread g_async_worker;
bool g_async_worker_started = false;
bool g_async_shutdown = false;
bool g_async_has_job = false;

const ContractOpRow* g_async_ops = nullptr;
int32_t g_async_op_count = 0;
ContractExecCtx* g_async_ctx = nullptr;
int64_t g_async_submit_token = 0;

std::atomic<int32_t> g_async_state{ASYNC_IDLE};
int64_t g_async_ready_token = 0;
int32_t g_async_ready_status = 0;

typedef void (*ContractCompletionFn)(int64_t step_token, int32_t status);
ContractCompletionFn g_completion_cb = nullptr;

void contract_async_worker_loop() {
    for (;;) {
        std::unique_lock<std::mutex> lock(g_async_mtx);
        g_async_cv.wait(lock, [] {
            return g_async_shutdown || g_async_has_job;
        });
        if (g_async_shutdown) {
            break;
        }

        const ContractOpRow* ops = g_async_ops;
        const int32_t op_count = g_async_op_count;
        ContractExecCtx* ctx = g_async_ctx;
        const int64_t token = g_async_submit_token;
        g_async_has_job = false;
        lock.unlock();

        const int32_t status = run_contract_training_step_impl(ops, op_count, ctx);

        if (g_completion_cb) {
            g_completion_cb(token, status);
            g_async_state.store(ASYNC_IDLE, std::memory_order_release);
        } else {
            g_async_ready_token = token;
            g_async_ready_status = status;
            g_async_state.store(ASYNC_READY, std::memory_order_release);
        }
    }
}

void ensure_async_worker_started() {
    if (g_async_worker_started) {
        return;
    }
    g_async_worker_started = true;
    g_async_worker = std::thread(contract_async_worker_loop);
}

}  // namespace

extern "C" {

__declspec(dllexport) int32_t run_contract_training_step(
    const ContractOpRow* ops,
    int32_t op_count,
    ContractExecCtx* ctx
) {
    return run_contract_training_step_impl(ops, op_count, ctx);
}

// Optional: native worker invokes this when a submitted contract finishes (post-process in Python).
__declspec(dllexport) void contract_register_completion_callback(
    ContractCompletionFn cb
) {
    g_completion_cb = cb;
}

// Returns 0 on accept, -2 if busy/running, -3 if completion must be reaped first.
__declspec(dllexport) int32_t submit_contract_training_step(
    const ContractOpRow* ops,
    int32_t op_count,
    ContractExecCtx* ctx,
    int64_t step_token
) {
    if (!ops || !ctx || op_count <= 0 || ctx->N <= 0) {
        return -1;
    }

    const int32_t state = g_async_state.load(std::memory_order_acquire);
    if (state == ASYNC_READY) {
        return -3;
    }
    if (state == ASYNC_RUNNING) {
        return -2;
    }

    ensure_async_worker_started();

    {
        std::lock_guard<std::mutex> lock(g_async_mtx);
        if (g_async_has_job || g_async_state.load(std::memory_order_relaxed) != ASYNC_IDLE) {
            return -2;
        }
        g_async_ops = ops;
        g_async_op_count = op_count;
        g_async_ctx = ctx;
        g_async_submit_token = step_token;
        g_async_has_job = true;
        g_async_state.store(ASYNC_RUNNING, std::memory_order_release);
    }
    g_async_cv.notify_one();
    return 0;
}

// Returns 1 if a completion was reaped, 0 if not ready, -1 on bad args.
__declspec(dllexport) int32_t try_reap_contract_completion(
    int64_t* out_step_token,
    int32_t* out_status
) {
    if (!out_step_token || !out_status) {
        return -1;
    }
    if (g_async_state.load(std::memory_order_acquire) != ASYNC_READY) {
        return 0;
    }
    *out_step_token = g_async_ready_token;
    *out_status = g_async_ready_status;
    g_async_state.store(ASYNC_IDLE, std::memory_order_release);
    return 1;
}

__declspec(dllexport) int32_t contract_async_in_flight() {
    const int32_t state = g_async_state.load(std::memory_order_acquire);
    return (state == ASYNC_RUNNING) ? 1 : 0;
}

__declspec(dllexport) void contract_async_shutdown() {
    {
        std::lock_guard<std::mutex> lock(g_async_mtx);
        g_async_shutdown = true;
    }
    g_async_cv.notify_all();
    if (g_async_worker.joinable()) {
        g_async_worker.join();
    }
}

}  // extern "C"
