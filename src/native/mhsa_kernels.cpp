// mhsa_kernels.cpp — Naive causal MHSA forward/backward; BLAS for projections.
#include "mhsa_kernels.h"
#include "blas_dynamic.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace {

constexpr float kEps = 1e-5f;

inline float gelu(float x) {
    const float k = 0.7978845608f;  // sqrt(2/pi)
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

// dy → dx, accumulates dgamma/dbeta. Uses y as LN output (to recover xhat).
void layernorm_rows_bwd(
    const float* x, const float* y, const float* gamma,
    const float* dy, float* dx,
    float* dgamma, float* dbeta,
    int64_t rows, int64_t D
) {
    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = x + r * D;
        const float* yr = y + r * D;
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

        // xhat from y: (y - beta) / gamma, with safe gamma
        float sum_dy = 0.0f;
        float sum_dy_xhat = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
            const float xhat = (xr[d] - mean) * inv;
            const float g = gamma[d];
            dgamma[d] += dyr[d] * xhat;
            dbeta[d] += dyr[d];
            const float dxhat = dyr[d] * g;
            sum_dy += dxhat;
            sum_dy_xhat += dxhat * xhat;
            (void)yr;
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
        for (int64_t c = 0; c < cols; ++c) {
            yr[c] += b[c];
        }
    }
}

void residual_add(float* y, const float* x, int64_t n) {
    for (int64_t i = 0; i < n; ++i) y[i] += x[i];
}

void accumulate_bias_grad(const float* dy, float* db, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; ++r) {
        const float* yr = dy + r * cols;
        for (int64_t c = 0; c < cols; ++c) {
            db[c] += yr[c];
        }
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

bool bind_ok_fwd(const MhsaBinding* m) {
    return m && m->W_qkv && m->O && m->qkv && m->scores && m->attn_out &&
           m->ln1_out && m->h1 && m->ln2_out && m->ffn_pre && m->ffn_h && m->scratch &&
           m->ln1_gamma && m->ln1_beta && m->ln2_gamma && m->ln2_beta &&
           m->W_o && m->W_ff1 && m->W_ff2 && m->b_qkv && m->b_o && m->b_ff1 && m->b_ff2;
}

bool bind_ok_bwd(const MhsaBinding* m) {
    return bind_ok_fwd(m) && m->dO && m->d_qkv &&
           m->dW_qkv && m->db_qkv && m->dW_o && m->db_o &&
           m->dW_ff1 && m->db_ff1 && m->dW_ff2 && m->db_ff2 &&
           m->d_ln1_gamma && m->d_ln1_beta && m->d_ln2_gamma && m->d_ln2_beta;
}

}  // namespace

extern "C" {

int32_t mhsa_block_forward(const float* X, MhsaBinding* m) {
    if (!X || !bind_ok_fwd(m)) {
        return -2;
    }
    const int32_t br = require_blas_ready();
    if (br != 0) return br;

    const int64_t B = m->B;
    const int64_t T = m->T;
    const int64_t D = m->D;
    const int64_t H = m->H;
    const int64_t Dh = m->d_head;
    const int64_t Hff = m->ffn_hidden;
    const int64_t rows = B * T;
    if (B < 1 || T < 1 || D < 1 || H < 1 || Dh * H != D || Hff < 1) {
        return -3;
    }

    // Pre-LN:
    //   h1 = X + Wo(Attn(LN1(X)))
    //   O  = h1 + FFN(LN2(h1))
    layernorm_rows(X, m->ln1_out, m->ln1_gamma, m->ln1_beta, rows, D);

    blas_gemm_forward(m->ln1_out, m->W_qkv, m->qkv, rows, 3 * D, D);
    add_bias_rows(m->qkv, m->b_qkv, rows, 3 * D);

    const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
    std::memset(m->attn_out, 0, static_cast<size_t>(rows * D) * sizeof(float));

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t i = 0; i < T; ++i) {
                float* score_row =
                    m->scores + (((b * H + h) * T + i) * T);
                float max_s = -1e30f;
                for (int64_t j = 0; j < T; ++j) {
                    if (j > i) {
                        score_row[j] = -1e30f;
                        continue;
                    }
                    const float* q =
                        m->qkv + ((b * T + i) * 3 * D) + (0 * D) + (h * Dh);
                    const float* k =
                        m->qkv + ((b * T + j) * 3 * D) + (1 * D) + (h * Dh);
                    float dot = 0.0f;
                    for (int64_t d = 0; d < Dh; ++d) {
                        dot += q[d] * k[d];
                    }
                    const float s = dot * scale;
                    score_row[j] = s;
                    if (s > max_s) max_s = s;
                }
                float sum_exp = 0.0f;
                for (int64_t j = 0; j <= i; ++j) {
                    const float e = std::exp(score_row[j] - max_s);
                    score_row[j] = e;
                    sum_exp += e;
                }
                const float inv = 1.0f / sum_exp;
                for (int64_t j = 0; j <= i; ++j) {
                    score_row[j] *= inv;
                }
                for (int64_t j = i + 1; j < T; ++j) {
                    score_row[j] = 0.0f;
                }

                float* out = m->attn_out + (b * T + i) * D + (h * Dh);
                for (int64_t d = 0; d < Dh; ++d) {
                    float acc = 0.0f;
                    for (int64_t j = 0; j <= i; ++j) {
                        const float* v =
                            m->qkv + ((b * T + j) * 3 * D) + (2 * D) + (h * Dh);
                        acc += score_row[j] * v[d];
                    }
                    out[d] = acc;
                }
            }
        }
    }

    blas_gemm_forward(m->attn_out, m->W_o, m->scratch, rows, D, D);
    add_bias_rows(m->scratch, m->b_o, rows, D);
    std::memcpy(m->h1, m->scratch, static_cast<size_t>(rows * D) * sizeof(float));
    residual_add(m->h1, X, rows * D);

    layernorm_rows(m->h1, m->ln2_out, m->ln2_gamma, m->ln2_beta, rows, D);
    blas_gemm_forward(m->ln2_out, m->W_ff1, m->ffn_pre, rows, Hff, D);
    add_bias_rows(m->ffn_pre, m->b_ff1, rows, Hff);
    for (int64_t i = 0; i < rows * Hff; ++i) {
        m->ffn_h[i] = gelu(m->ffn_pre[i]);
    }
    blas_gemm_forward(m->ffn_h, m->W_ff2, m->scratch, rows, D, Hff);
    add_bias_rows(m->scratch, m->b_ff2, rows, D);
    std::memcpy(m->O, m->h1, static_cast<size_t>(rows * D) * sizeof(float));
    residual_add(m->O, m->scratch, rows * D);
    return 0;
}

int32_t mhsa_action_forward(MhsaBinding* m) {
    if (!m || !m->O || !m->W_act || !m->actions || !m->scratch) {
        return -2;
    }
    const int32_t br = require_blas_ready();
    if (br != 0) return br;

    const int64_t B = m->B;
    const int64_t T = m->T;
    const int64_t D = m->D;
    const int64_t A = m->action_dim;
    if (B < 1 || T < 1 || D < 1 || A < 1) {
        return -3;
    }

    for (int64_t b = 0; b < B; ++b) {
        const float* src = m->O + (b * T + (T - 1)) * D;
        float* dst = m->scratch + b * D;
        std::memcpy(dst, src, static_cast<size_t>(D) * sizeof(float));
    }
    blas_gemm_forward(m->scratch, m->W_act, m->actions, B, A, D);
    add_bias_rows(m->actions, m->b_act, B, A);
    for (int64_t i = 0; i < B * A; ++i) {
        m->actions[i] = std::tanh(m->actions[i]);
    }
    return 0;
}

int32_t mhsa_action_backward(MhsaBinding* m) {
    if (!m || !m->O || !m->W_act || !m->actions || !m->y || !m->scratch ||
        !m->dW_act || !m->db_act || !m->dO) {
        return -2;
    }
    const int32_t br = require_blas_ready();
    if (br != 0) return br;

    const int64_t B = m->B;
    const int64_t T = m->T;
    const int64_t D = m->D;
    const int64_t A = m->action_dim;
    if (B < 1 || T < 1 || D < 1 || A < 1) {
        return -3;
    }

    // loss = mean((tanh(z) - y)^2); d_a = 2/(B*A) * (a - y); dz = da * (1-a^2)
    const float inv = 2.0f / static_cast<float>(B * A);
    float loss = 0.0f;
    // Reuse first B*A of d_qkv or allocate via scratch for dz: use actions buffer sibling —
    // write dz into beginning of d_qkv if present, else stack via scratch after last-token pack.
    float* dz = m->d_qkv;  // [B, A] fits in leading slice of [B*T, 3D]
    if (!dz) {
        return -2;
    }
    for (int64_t i = 0; i < B * A; ++i) {
        const float diff = m->actions[i] - m->y[i];
        loss += diff * diff;
        const float da = inv * diff;
        dz[i] = da * (1.0f - m->actions[i] * m->actions[i]);
    }
    if (m->loss_out) {
        m->loss_out[0] = loss / static_cast<float>(B * A);
    }

    // Last-token pack into scratch [B, D]
    for (int64_t b = 0; b < B; ++b) {
        const float* src = m->O + (b * T + (T - 1)) * D;
        float* dst = m->scratch + b * D;
        std::memcpy(dst, src, static_cast<size_t>(D) * sizeof(float));
    }

    // dW_act = X^T @ dz  (dz already mean-scaled)
    blas_gemm_weight_grad_rm(m->scratch, dz, m->dW_act, B, A, D, 1.0f);
    accumulate_bias_grad(dz, m->db_act, B, A);

    // d_last = dz @ W_act^T
    float* d_last = m->scratch;  // reuse [B, D] — overwrite packed X after param grad
    blas_gemm_input_grad_rm(dz, m->W_act, d_last, B, A, D);

    std::memset(m->dO, 0, static_cast<size_t>(B * T * D) * sizeof(float));
    for (int64_t b = 0; b < B; ++b) {
        float* dst = m->dO + (b * T + (T - 1)) * D;
        const float* src = d_last + b * D;
        std::memcpy(dst, src, static_cast<size_t>(D) * sizeof(float));
    }
    return 0;
}

int32_t mhsa_block_backward(const float* X, MhsaBinding* m) {
    if (!X || !bind_ok_bwd(m)) {
        return -2;
    }
    const int32_t br = require_blas_ready();
    if (br != 0) return br;

    const int64_t B = m->B;
    const int64_t T = m->T;
    const int64_t D = m->D;
    const int64_t H = m->H;
    const int64_t Dh = m->d_head;
    const int64_t Hff = m->ffn_hidden;
    const int64_t rows = B * T;
    if (B < 1 || T < 1 || D < 1 || H < 1 || Dh * H != D || Hff < 1) {
        return -3;
    }

    // --- FFN bwd: O = h1 + (ffn_h @ W_ff2 + b); ffn_h = gelu(ffn_pre) ---
    // d_ffn_out = dO; dh1 += dO
    float* d_ffn_out = m->scratch;  // [rows, D]
    std::memcpy(d_ffn_out, m->dO, static_cast<size_t>(rows * D) * sizeof(float));

    blas_gemm_weight_grad_rm(m->ffn_h, d_ffn_out, m->dW_ff2, rows, D, Hff, 1.0f);
    accumulate_bias_grad(d_ffn_out, m->db_ff2, rows, D);

    // d_ffn_h reuses ffn_h (post-gelu no longer needed after dW_ff2)
    float* d_ffn_h = m->ffn_h;
    blas_gemm_input_grad_rm(d_ffn_out, m->W_ff2, d_ffn_h, rows, D, Hff);

    // gelu bwd into d_ffn_pre (overwrite d_ffn_h in place using ffn_pre values)
    for (int64_t i = 0; i < rows * Hff; ++i) {
        d_ffn_h[i] = gelu_bwd(m->ffn_pre[i], d_ffn_h[i]);
    }

    blas_gemm_weight_grad_rm(m->ln2_out, d_ffn_h, m->dW_ff1, rows, Hff, D, 1.0f);
    accumulate_bias_grad(d_ffn_h, m->db_ff1, rows, Hff);

    // d_ln2_out = d_ffn_h @ W_ff1^T → use O as temp (O not needed after)
    float* d_ln2_out = m->O;
    blas_gemm_input_grad_rm(d_ffn_h, m->W_ff1, d_ln2_out, rows, Hff, D);

    // LN2 bwd: dh1_from_ln + residual dO
    float* dh1 = m->scratch;
    layernorm_rows_bwd(
        m->h1, m->ln2_out, m->ln2_gamma, d_ln2_out, dh1,
        m->d_ln2_gamma, m->d_ln2_beta, rows, D);
    residual_add(dh1, m->dO, rows * D);

    // --- Attn out projection: h1 = X + (attn_out @ W_o + b) ---
    // d_attn_proj = dh1
    float* d_attn_proj = dh1;
    blas_gemm_weight_grad_rm(m->attn_out, d_attn_proj, m->dW_o, rows, D, D, 1.0f);
    accumulate_bias_grad(d_attn_proj, m->db_o, rows, D);

    float* d_attn_out = m->O;  // reuse
    blas_gemm_input_grad_rm(d_attn_proj, m->W_o, d_attn_out, rows, D, D);

    // dX from residual
    if (m->dX) {
        std::memcpy(m->dX, dh1, static_cast<size_t>(rows * D) * sizeof(float));
    }

    // --- Causal attention bwd into d_qkv ---
    std::memset(m->d_qkv, 0, static_cast<size_t>(rows * 3 * D) * sizeof(float));
    const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t i = 0; i < T; ++i) {
                const float* score_row =
                    m->scores + (((b * H + h) * T + i) * T);
                const float* dout =
                    d_attn_out + (b * T + i) * D + (h * Dh);

                // dV[j] += P[i,j] * dout[i]
                for (int64_t j = 0; j <= i; ++j) {
                    float* dv =
                        m->d_qkv + ((b * T + j) * 3 * D) + (2 * D) + (h * Dh);
                    const float p = score_row[j];
                    for (int64_t d = 0; d < Dh; ++d) {
                        dv[d] += p * dout[d];
                    }
                }

                // dP[j] = sum_d dout[d] * V[j,d]
                // Softmax bwd: ds = P * (dP - sum(P*dP))
                float sum_p_dp = 0.0f;
                float* dp_row = m->ffn_h;  // temp T-vector (ffn_h no longer needed)
                for (int64_t j = 0; j <= i; ++j) {
                    const float* v =
                        m->qkv + ((b * T + j) * 3 * D) + (2 * D) + (h * Dh);
                    float s = 0.0f;
                    for (int64_t d = 0; d < Dh; ++d) {
                        s += dout[d] * v[d];
                    }
                    dp_row[j] = s;
                    sum_p_dp += score_row[j] * s;
                }
                for (int64_t j = 0; j <= i; ++j) {
                    const float ds = score_row[j] * (dp_row[j] - sum_p_dp);
                    const float d_dot = ds * scale;
                    float* dq =
                        m->d_qkv + ((b * T + i) * 3 * D) + (0 * D) + (h * Dh);
                    float* dk =
                        m->d_qkv + ((b * T + j) * 3 * D) + (1 * D) + (h * Dh);
                    const float* q =
                        m->qkv + ((b * T + i) * 3 * D) + (0 * D) + (h * Dh);
                    const float* k =
                        m->qkv + ((b * T + j) * 3 * D) + (1 * D) + (h * Dh);
                    for (int64_t d = 0; d < Dh; ++d) {
                        dq[d] += d_dot * k[d];
                        dk[d] += d_dot * q[d];
                    }
                }
            }
        }
    }

    // QKV projection bwd
    blas_gemm_weight_grad_rm(m->ln1_out, m->d_qkv, m->dW_qkv, rows, 3 * D, D, 1.0f);
    accumulate_bias_grad(m->d_qkv, m->db_qkv, rows, 3 * D);

    float* d_ln1_out = m->O;
    blas_gemm_input_grad_rm(m->d_qkv, m->W_qkv, d_ln1_out, rows, 3 * D, D);

    float* dX_ln = m->attn_out;  // attn_out done
    layernorm_rows_bwd(
        X, m->ln1_out, m->ln1_gamma, d_ln1_out, dX_ln,
        m->d_ln1_gamma, m->d_ln1_beta, rows, D);

    if (m->dX) {
        residual_add(m->dX, dX_ln, rows * D);
    }
    return 0;
}

}  // extern "C"
