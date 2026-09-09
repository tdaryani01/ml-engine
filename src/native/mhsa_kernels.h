// mhsa_kernels.h — Causal MHSA bind + naive fwd/bwd kernels.
#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Composed bind (alongside LayerBinding / DenseBinding — not a CNN subclass).
struct MhsaBinding {
    int64_t B;
    int64_t T;
    int64_t D;
    int64_t H;
    int64_t d_head;
    int64_t action_dim;
    int64_t ffn_hidden;

    float* W_qkv;  // [D, 3D]
    float* b_qkv;  // [3D]
    float* W_o;    // [D, D]
    float* b_o;    // [D]
    float* W_ff1;  // [D, ffn_hidden]
    float* b_ff1;  // [ffn_hidden]
    float* W_ff2;  // [ffn_hidden, D]
    float* b_ff2;  // [D]
    float* W_act;  // [D, action_dim]
    float* b_act;  // [action_dim]

    float* ln1_gamma;  // [D]
    float* ln1_beta;
    float* ln2_gamma;
    float* ln2_beta;

    // Workspace (caller-owned, float32, contiguous)
    float* qkv;       // [B*T, 3D]
    float* scores;    // [B*H*T*T] softmax probs (causal)
    float* attn_out;  // [B*T, D] head concat before Wo
    float* ln1_out;   // [B*T, D]
    float* h1;        // [B*T, D] post-attn residual
    float* ln2_out;   // [B*T, D]
    float* ffn_pre;   // [B*T, ffn_hidden] pre-GELU
    float* ffn_h;     // [B*T, ffn_hidden] post-GELU
    float* O;         // [B*T, D] residual stream out
    float* scratch;   // [B*T, D] temp
    float* actions;   // [B, action_dim]

    // Gradients (caller-owned; zeroed by caller each step)
    float* dW_qkv;
    float* db_qkv;
    float* dW_o;
    float* db_o;
    float* dW_ff1;
    float* db_ff1;
    float* dW_ff2;
    float* db_ff2;
    float* dW_act;
    float* db_act;
    float* d_ln1_gamma;
    float* d_ln1_beta;
    float* d_ln2_gamma;
    float* d_ln2_beta;
    float* dO;     // [B*T, D]
    float* dX;     // [B*T, D] optional (may be null)
    float* d_qkv;  // [B*T, 3D] temp for attn bwd

    // Targets / loss (action head)
    float* y;         // [B, action_dim]
    float* loss_out;  // optional scalar
};

// X: [B*T, D] row-major. Writes O and saved intermediates in `m`.
int32_t mhsa_block_forward(const float* X, MhsaBinding* m);

// Reads m->O last token → m->actions.
int32_t mhsa_action_forward(MhsaBinding* m);

// MSE on tanh actions → dO (last token). Uses m->y; writes m->loss_out if set.
int32_t mhsa_action_backward(MhsaBinding* m);

// dO → param grads (+ dX if non-null). Requires intermediates from forward.
int32_t mhsa_block_backward(const float* X, MhsaBinding* m);

#ifdef __cplusplus
}
#endif
