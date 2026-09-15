// mhsa_kernels.cpp — Multi-layer causal MHSA; BLAS for projections + attention GEMMs.
#include "mhsa_kernels.h"
#include "blas_dynamic.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace {

constexpr float kEps = 1e-5f;

inline float gelu(float x) {
    const float k = 0.7978845608f;
    const float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x3)));
}

inline float gelu_bwd(float x, float dy) {
    const float k = 0.7978845608f;
    const float c = 0.044715f;
    const float x2 = x * x;
    const float x3 = x2 * x;
    const float u = k * (x + c * x3);
    const float tanh_u = std::tanh(u);
    const float sech2 = 1.0f - tanh_u * tanh_u;
    const float du = k * (1.0f + 3.0f * c * x2);
    return dy * (0.5f * (1.0f + tanh_u) + 0.5f * x * sech2 * du);
}

void layernorm_rows(
    const float* x, float* y, const float* gamma, const float* beta,
    int64_t rows, int64_t D
) {
    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = x + r * D;
        float* yr = y + r * D;
        float mean = 0.0f;
        for (int64_t d = 0; d < D; ++d) mean += xr[d];
        mean /= static_cast<float>(D);
        float var = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
            const float t = xr[d] - mean;
            var += t * t;
        }
        var /= static_cast<float>(D);
        const float inv = 1.0f / std::sqrt(var + kEps);
        for (int64_t d = 0; d < D; ++d) {
            yr[d] = (xr[d] - mean) * inv * gamma[d] + beta[d];
        }
    }
}

void layernorm_rows_bwd(
    const float* x, const float* /*y*/, const float* gamma,
    const float* dy, float* dx,
    float* dgamma, float* dbeta,
    int64_t rows, int64_t D
) {
    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = x + r * D;
        const float* dyr = dy + r * D;
        float* dxr = dx + r * D;

        float mean = 0.0f;
        for (int64_t d = 0; d < D; ++d) mean += xr[d];
        mean /= static_cast<float>(D);
        float var = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
            const float t = xr[d] - mean;
            var += t * t;
        }
        var /= static_cast<float>(D);
        const float inv = 1.0f / std::sqrt(var + kEps);

        float sum_dy = 0.0f;
        float sum_dy_xhat = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
            const float xhat = (xr[d] - mean) * inv;
            dgamma[d] += dyr[d] * xhat;
            dbeta[d] += dyr[d];
            const float dxhat = dyr[d] * gamma[d];
            sum_dy += dxhat;
            sum_dy_xhat += dxhat * xhat;
        }
        const float inv_D = 1.0f / static_cast<float>(D);
        for (int64_t d = 0; d < D; ++d) {
            const float xhat = (xr[d] - mean) * inv;
            const float dxhat = dyr[d] * gamma[d];
            dxr[d] = inv * (dxhat - inv_D * sum_dy - xhat * inv_D * sum_dy_xhat);
        }
    }
}

void add_bias_rows(float* y, const float* b, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; ++r) {
        float* yr = y + r * cols;
        for (int64_t c = 0; c < cols; ++c) yr[c] += b[c];
    }
}

void residual_add(float* y, const float* x, int64_t n) {
    for (int64_t i = 0; i < n; ++i) y[i] += x[i];
}

void accumulate_bias_grad(const float* dy, float* db, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; ++r) {
        const float* yr = dy + r * cols;
        for (int64_t c = 0; c < cols; ++c) db[c] += yr[c];
    }
}

// Pack one head's slice from interleaved qkv [B*T, 3D] into contiguous [T, Dh].
void pack_qkv_head(
    const float* qkv, float* dst, int64_t b, int64_t h, int64_t part,
    int64_t T, int64_t D, int64_t Dh
) {
    const int64_t off = part * D + h * Dh;
    for (int64_t t = 0; t < T; ++t) {
        std::memcpy(
            dst + t * Dh,
            qkv + ((b * T + t) * 3 * D) + off,
            static_cast<size_t>(Dh) * sizeof(float));
    }
}

void pack_dout_head(
    const float* dout, float* dst, int64_t b, int64_t h,
    int64_t T, int64_t D, int64_t Dh
) {
    for (int64_t t = 0; t < T; ++t) {
        std::memcpy(
            dst + t * Dh,
            dout + (b * T + t) * D + h * Dh,
            static_cast<size_t>(Dh) * sizeof(float));
    }
}

void scatter_head_add(
    float* qkv_grad, const float* src, int64_t b, int64_t h, int64_t part,
    int64_t T, int64_t D, int64_t Dh
) {
    const int64_t off = part * D + h * Dh;
    for (int64_t t = 0; t < T; ++t) {
        float* dst = qkv_grad + ((b * T + t) * 3 * D) + off;
        const float* s = src + t * Dh;
        for (int64_t d = 0; d < Dh; ++d) dst[d] += s[d];
    }
}

void scatter_attn_out_head(
    float* attn_out, const float* src, int64_t b, int64_t h,
    int64_t T, int64_t D, int64_t Dh
) {
    for (int64_t t = 0; t < T; ++t) {
        std::memcpy(
            attn_out + (b * T + t) * D + h * Dh,
            src + t * Dh,
            static_cast<size_t>(Dh) * sizeof(float));
    }
}

void causal_softmax_rows(float* S, int64_t T) {
    for (int64_t i = 0; i < T; ++i) {
        float* row = S + i * T;
        float max_s = -1e30f;
        for (int64_t j = 0; j <= i; ++j) {
            if (row[j] > max_s) max_s = row[j];
        }
        for (int64_t j = i + 1; j < T; ++j) row[j] = -1e30f;
        float sum_exp = 0.0f;
        for (int64_t j = 0; j <= i; ++j) {
            const float e = std::exp(row[j] - max_s);
            row[j] = e;
            sum_exp += e;
        }
        const float inv = 1.0f / sum_exp;
        for (int64_t j = 0; j <= i; ++j) row[j] *= inv;
        for (int64_t j = i + 1; j < T; ++j) row[j] = 0.0f;
    }
}

// In-place: dP -> dS for causal softmax; P is probs in scores row.
void causal_softmax_bwd_rows(const float* P, float* dP, int64_t T) {
    for (int64_t i = 0; i < T; ++i) {
        const float* p_row = P + i * T;
        float* dp_row = dP + i * T;
        float sum_p_dp = 0.0f;
        for (int64_t j = 0; j <= i; ++j) sum_p_dp += p_row[j] * dp_row[j];
        for (int64_t j = 0; j <= i; ++j) {
            dp_row[j] = p_row[j] * (dp_row[j] - sum_p_dp);
        }
        for (int64_t j = i + 1; j < T; ++j) dp_row[j] = 0.0f;
    }
}

int32_t require_blas_ready() {
    if (!blas_runtime_ready()) {
        init_openblas_runtime(nullptr);
    }
    if (!blas_runtime_ready()) {
        std::fprintf(stderr, "[MHSA] OpenBLAS/sgemm not ready\n");
        return -101;
    }
    return 0;
}

bool layer_ok_fwd(const MhsaLayerBind* L) {
    return L && L->W_qkv && L->b_qkv && L->W_o && L->b_o && L->W_ff1 && L->b_ff1 &&
           L->W_ff2 && L->b_ff2 && L->ln1_gamma && L->ln1_beta && L->ln2_gamma &&
           L->ln2_beta && L->qkv && L->scores && L->attn_out && L->ln1_out && L->h1 &&
           L->ln2_out && L->ffn_pre && L->ffn_h && L->O && L->q_pack && L->k_pack &&
           L->v_pack;
}

bool layer_ok_bwd(const MhsaLayerBind* L) {
    return layer_ok_fwd(L) && L->dW_qkv && L->db_qkv && L->dW_o && L->db_o &&
           L->dW_ff1 && L->db_ff1 && L->dW_ff2 && L->db_ff2 && L->d_ln1_gamma &&
           L->d_ln1_beta && L->d_ln2_gamma && L->d_ln2_beta && L->d_scores;
}

bool bind_ok_fwd(const MhsaBinding* m) {
    if (!m || m->num_layers < 1 || m->num_layers > MHSA_MAX_LAYERS || !m->scratch) {
        return false;
    }
    for (int64_t li = 0; li < m->num_layers; ++li) {
        if (!layer_ok_fwd(&m->layers[li])) return false;
    }
    return true;
}

bool bind_ok_bwd(const MhsaBinding* m) {
    if (!bind_ok_fwd(m) || !m->dO || !m->d_qkv || !m->d_stream) return false;
    for (int64_t li = 0; li < m->num_layers; ++li) {
        if (!layer_ok_bwd(&m->layers[li])) return false;
    }
    return true;
}

// One Pre-LN block: X -> L->O
int32_t layer_forward(
    const float* X, MhsaLayerBind* L, float* scratch,
    int64_t B, int64_t T, int64_t D, int64_t H, int64_t Dh, int64_t Hff
) {
    const int64_t rows = B * T;
    layernorm_rows(X, L->ln1_out, L->ln1_gamma, L->ln1_beta, rows, D);
    blas_gemm_forward(L->ln1_out, L->W_qkv, L->qkv, rows, 3 * D, D);
    add_bias_rows(L->qkv, L->b_qkv, rows, 3 * D);

    const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
    std::memset(L->attn_out, 0, static_cast<size_t>(rows * D) * sizeof(float));

    // Per (b,h): S = scale * Q @ K^T (causal softmax) ; O = P @ V via BLAS GEMM.
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            float* S = L->scores + ((b * H + h) * T * T);
            pack_qkv_head(L->qkv, L->q_pack, b, h, /*Q*/ 0, T, D, Dh);
            pack_qkv_head(L->qkv, L->k_pack, b, h, /*K*/ 1, T, D, Dh);
            // S[T,T] = Q[T,Dh] @ K[T,Dh]^T
            blas_gemm_input_grad_rm(L->q_pack, L->k_pack, S, T, Dh, T);
            for (int64_t i = 0; i < T * T; ++i) S[i] *= scale;
            causal_softmax_rows(S, T);

            pack_qkv_head(L->qkv, L->v_pack, b, h, /*V*/ 2, T, D, Dh);
            // out[T,Dh] = P[T,T] @ V[T,Dh]  (reuse q_pack as out)
            blas_gemm_forward(S, L->v_pack, L->q_pack, T, Dh, T);
            scatter_attn_out_head(L->attn_out, L->q_pack, b, h, T, D, Dh);
        }
    }

    blas_gemm_forward(L->attn_out, L->W_o, scratch, rows, D, D);
    add_bias_rows(scratch, L->b_o, rows, D);
    std::memcpy(L->h1, scratch, static_cast<size_t>(rows * D) * sizeof(float));
    residual_add(L->h1, X, rows * D);

    layernorm_rows(L->h1, L->ln2_out, L->ln2_gamma, L->ln2_beta, rows, D);
    blas_gemm_forward(L->ln2_out, L->W_ff1, L->ffn_pre, rows, Hff, D);
    add_bias_rows(L->ffn_pre, L->b_ff1, rows, Hff);
    for (int64_t i = 0; i < rows * Hff; ++i) L->ffn_h[i] = gelu(L->ffn_pre[i]);
    blas_gemm_forward(L->ffn_h, L->W_ff2, scratch, rows, D, Hff);
    add_bias_rows(scratch, L->b_ff2, rows, D);
    std::memcpy(L->O, L->h1, static_cast<size_t>(rows * D) * sizeof(float));
    residual_add(L->O, scratch, rows * D);
    return 0;
}

// d_out -> d_in (written to d_in_out). May clobber L->O and L->ffn_h.
int32_t layer_backward(
    const float* X, MhsaLayerBind* L, const float* d_out, float* d_in_out,
    float* scratch, float* d_qkv, float* tmp_D,
    int64_t B, int64_t T, int64_t D, int64_t H, int64_t Dh, int64_t Hff
) {
    const int64_t rows = B * T;
    const size_t bytes_D = static_cast<size_t>(rows * D) * sizeof(float);

    float* d_ffn_out = scratch;
    std::memcpy(d_ffn_out, d_out, bytes_D);

    blas_gemm_weight_grad_rm(L->ffn_h, d_ffn_out, L->dW_ff2, rows, D, Hff, 1.0f);
    accumulate_bias_grad(d_ffn_out, L->db_ff2, rows, D);

    float* d_ffn_h = L->ffn_h;
    blas_gemm_input_grad_rm(d_ffn_out, L->W_ff2, d_ffn_h, rows, D, Hff);
    for (int64_t i = 0; i < rows * Hff; ++i) {
        d_ffn_h[i] = gelu_bwd(L->ffn_pre[i], d_ffn_h[i]);
    }

    blas_gemm_weight_grad_rm(L->ln2_out, d_ffn_h, L->dW_ff1, rows, Hff, D, 1.0f);
    accumulate_bias_grad(d_ffn_h, L->db_ff1, rows, Hff);

    float* d_ln2_out = tmp_D;
    blas_gemm_input_grad_rm(d_ffn_h, L->W_ff1, d_ln2_out, rows, Hff, D);

    float* dh1 = scratch;
    layernorm_rows_bwd(
        L->h1, L->ln2_out, L->ln2_gamma, d_ln2_out, dh1,
        L->d_ln2_gamma, L->d_ln2_beta, rows, D);
    residual_add(dh1, d_out, rows * D);

    blas_gemm_weight_grad_rm(L->attn_out, dh1, L->dW_o, rows, D, D, 1.0f);
    accumulate_bias_grad(dh1, L->db_o, rows, D);

    float* d_attn_out = tmp_D;
    blas_gemm_input_grad_rm(dh1, L->W_o, d_attn_out, rows, D, D);

    // Residual path into d_in
    std::memcpy(d_in_out, dh1, bytes_D);

    std::memset(d_qkv, 0, static_cast<size_t>(rows * 3 * D) * sizeof(float));
    const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            const float* P = L->scores + ((b * H + h) * T * T);
            float* dS = L->d_scores;

            pack_qkv_head(L->qkv, L->v_pack, b, h, /*V*/ 2, T, D, Dh);
            pack_dout_head(d_attn_out, L->q_pack, b, h, T, D, Dh);  // dO in q_pack

            // dV = P^T @ dO
            blas_gemm_weight_grad_rm(P, L->q_pack, L->k_pack, T, Dh, T, 1.0f);
            scatter_head_add(d_qkv, L->k_pack, b, h, /*V*/ 2, T, D, Dh);

            // dP = dO @ V^T
            blas_gemm_input_grad_rm(L->q_pack, L->v_pack, dS, T, Dh, T);
            causal_softmax_bwd_rows(P, dS, T);
            for (int64_t i = 0; i < T * T; ++i) dS[i] *= scale;

            pack_qkv_head(L->qkv, L->q_pack, b, h, /*Q*/ 0, T, D, Dh);
            pack_qkv_head(L->qkv, L->k_pack, b, h, /*K*/ 1, T, D, Dh);

            // dQ = dS @ K ; dK = dS^T @ Q
            blas_gemm_forward(dS, L->k_pack, L->v_pack, T, Dh, T);
            scatter_head_add(d_qkv, L->v_pack, b, h, /*Q*/ 0, T, D, Dh);
            blas_gemm_weight_grad_rm(dS, L->q_pack, L->v_pack, T, Dh, T, 1.0f);
            scatter_head_add(d_qkv, L->v_pack, b, h, /*K*/ 1, T, D, Dh);
        }
    }

    blas_gemm_weight_grad_rm(L->ln1_out, d_qkv, L->dW_qkv, rows, 3 * D, D, 1.0f);
    accumulate_bias_grad(d_qkv, L->db_qkv, rows, 3 * D);

    float* d_ln1_out = tmp_D;
    blas_gemm_input_grad_rm(d_qkv, L->W_qkv, d_ln1_out, rows, 3 * D, D);

    float* dX_ln = L->attn_out;
    layernorm_rows_bwd(
        X, L->ln1_out, L->ln1_gamma, d_ln1_out, dX_ln,
        L->d_ln1_gamma, L->d_ln1_beta, rows, D);
    residual_add(d_in_out, dX_ln, rows * D);
    return 0;
}

}  // namespace

extern "C" {

int32_t mhsa_block_forward(const float* X, MhsaBinding* m) {
    if (!X || !bind_ok_fwd(m)) return -2;
    const int32_t br = require_blas_ready();
    if (br != 0) return br;

    const int64_t B = m->B;
    const int64_t T = m->T;
    const int64_t D = m->D;
    const int64_t H = m->H;
    const int64_t Dh = m->d_head;
    const int64_t Hff = m->ffn_hidden;
    if (B < 1 || T < 1 || D < 1 || H < 1 || Dh * H != D || Hff < 1) return -3;

    const int64_t rows = B * T;
    const size_t bytes_D = static_cast<size_t>(rows * D) * sizeof(float);
    const bool use_proj = (m->W_in != nullptr && m->b_in != nullptr);
    const bool use_pos = (m->pos != nullptr && m->max_seq_len >= T);
    if ((use_proj || use_pos) && !m->X_emb) return -2;

    const float* layer0_X = X;
    if (use_proj) {
        blas_gemm_forward(X, m->W_in, m->X_emb, rows, D, D);
        add_bias_rows(m->X_emb, m->b_in, rows, D);
        layer0_X = m->X_emb;
    } else if (use_pos) {
        std::memcpy(m->X_emb, X, bytes_D);
        layer0_X = m->X_emb;
    }
    if (use_pos) {
        float* emb = m->X_emb;
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t t = 0; t < T; ++t) {
                float* row = emb + (b * T + t) * D;
                const float* p = m->pos + t * D;
                for (int64_t d = 0; d < D; ++d) row[d] += p[d];
            }
        }
    }

    const float* cur = layer0_X;
    for (int64_t li = 0; li < m->num_layers; ++li) {
        const int32_t st = layer_forward(
            cur, &m->layers[li], m->scratch, B, T, D, H, Dh, Hff);
        if (st != 0) return st;
        cur = m->layers[li].O;
    }
    m->O = m->layers[m->num_layers - 1].O;
    return 0;
}

int32_t mhsa_action_forward(MhsaBinding* m) {
    if (!m || !m->O || !m->W_act || !m->actions || !m->scratch) return -2;
    const int32_t br = require_blas_ready();
    if (br != 0) return br;

    const int64_t B = m->B;
    const int64_t T = m->T;
    const int64_t D = m->D;
    const int64_t A = m->action_dim;
    if (B < 1 || T < 1 || D < 1 || A < 1) return -3;

    for (int64_t b = 0; b < B; ++b) {
        const float* src = m->O + (b * T + (T - 1)) * D;
        std::memcpy(m->scratch + b * D, src, static_cast<size_t>(D) * sizeof(float));
    }
    blas_gemm_forward(m->scratch, m->W_act, m->actions, B, A, D);
    add_bias_rows(m->actions, m->b_act, B, A);
    for (int64_t i = 0; i < B * A; ++i) m->actions[i] = std::tanh(m->actions[i]);
    return 0;
}

int32_t mhsa_action_backward(MhsaBinding* m) {
    if (!m || !m->O || !m->W_act || !m->actions || !m->y || !m->scratch ||
        !m->dW_act || !m->db_act || !m->dO || !m->d_qkv) {
        return -2;
    }
    const int32_t br = require_blas_ready();
    if (br != 0) return br;

    const int64_t B = m->B;
    const int64_t T = m->T;
    const int64_t D = m->D;
    const int64_t A = m->action_dim;
    if (B < 1 || T < 1 || D < 1 || A < 1) return -3;

    const float inv = 2.0f / static_cast<float>(B * A);
    float loss = 0.0f;
    float* dz = m->d_qkv;
    for (int64_t i = 0; i < B * A; ++i) {
        const float diff = m->actions[i] - m->y[i];
        loss += diff * diff;
        dz[i] = inv * diff * (1.0f - m->actions[i] * m->actions[i]);
    }
    if (m->loss_out) m->loss_out[0] = loss / static_cast<float>(B * A);

    for (int64_t b = 0; b < B; ++b) {
        const float* src = m->O + (b * T + (T - 1)) * D;
        std::memcpy(m->scratch + b * D, src, static_cast<size_t>(D) * sizeof(float));
    }
    blas_gemm_weight_grad_rm(m->scratch, dz, m->dW_act, B, A, D, 1.0f);
    accumulate_bias_grad(dz, m->db_act, B, A);

    float* d_last = m->scratch;
    blas_gemm_input_grad_rm(dz, m->W_act, d_last, B, A, D);

    std::memset(m->dO, 0, static_cast<size_t>(B * T * D) * sizeof(float));
    for (int64_t b = 0; b < B; ++b) {
        std::memcpy(
            m->dO + (b * T + (T - 1)) * D,
            d_last + b * D,
            static_cast<size_t>(D) * sizeof(float));
    }
    return 0;
}

int32_t mhsa_block_backward(const float* X, MhsaBinding* m) {
    if (!X || !bind_ok_bwd(m)) return -2;
    const int32_t br = require_blas_ready();
    if (br != 0) return br;

    const int64_t B = m->B;
    const int64_t T = m->T;
    const int64_t D = m->D;
    const int64_t H = m->H;
    const int64_t Dh = m->d_head;
    const int64_t Hff = m->ffn_hidden;
    const int64_t rows = B * T;
    if (B < 1 || T < 1 || D < 1 || H < 1 || Dh * H != D || Hff < 1) return -3;

    const bool use_proj = (m->W_in != nullptr && m->b_in != nullptr);
    const bool use_pos = (m->pos != nullptr && m->max_seq_len >= T);
    if ((use_proj || use_pos) && !m->X_emb) return -2;
    if (use_proj && (!m->dW_in || !m->db_in)) return -2;

    // Layer 0 saw X_emb when proj and/or pos ran; otherwise raw X.
    const float* X_l0 = (use_proj || use_pos) ? m->X_emb : X;

    // d_cur starts as action dO; after each layer becomes that layer's dX.
    float* d_cur = m->dO;
    float* d_next = m->d_stream;
    for (int64_t li = m->num_layers - 1; li >= 0; --li) {
        const float* X_l = (li == 0) ? X_l0 : m->layers[li - 1].O;
        float* tmp_D = m->layers[li].O;  // safe: this layer's O not needed as X anymore
        const int32_t st = layer_backward(
            X_l, &m->layers[li], d_cur, d_next,
            m->scratch, m->d_qkv, tmp_D,
            B, T, D, H, Dh, Hff);
        if (st != 0) return st;
        float* tmp = d_cur;
        d_cur = d_next;
        d_next = tmp;
    }

    // Grad w.r.t. learned positions (added after optional input proj).
    if (use_pos && m->d_pos) {
        for (int64_t t = 0; t < T; ++t) {
            float* dp = m->d_pos + t * D;
            for (int64_t b = 0; b < B; ++b) {
                const float* dx = d_cur + (b * T + t) * D;
                for (int64_t d = 0; d < D; ++d) dp[d] += dx[d];
            }
        }
    }

    if (use_proj) {
        // dW_in / db_in from X_raw and d_emb; dX = d_emb @ W_in^T.
        blas_gemm_weight_grad_rm(X, d_cur, m->dW_in, rows, D, D, 1.0f);
        accumulate_bias_grad(d_cur, m->db_in, rows, D);
        if (m->dX) {
            blas_gemm_input_grad_rm(d_cur, m->W_in, m->dX, rows, D, D);
        }
    } else if (m->dX) {
        std::memcpy(m->dX, d_cur, static_cast<size_t>(rows * D) * sizeof(float));
    }
    return 0;
}

}  // extern "C"
