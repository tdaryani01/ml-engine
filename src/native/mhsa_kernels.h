// mhsa_kernels.h — Causal MHSA bind + naive multi-layer fwd/bwd.
#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

enum { MHSA_MAX_LAYERS = 8 };

// Per Pre-LN block (MHSA + FFN).
struct MhsaLayerBind {
    float* W_qkv;  // [D, 3D]
    float* b_qkv;
    float* W_o;  // [D, D]
    float* b_o;
    float* W_ff1;  // [D, ffn_hidden]
    float* b_ff1;
    float* W_ff2;  // [ffn_hidden, D]
    float* b_ff2;
    float* ln1_gamma;  // [D]
    float* ln1_beta;
    float* ln2_gamma;
    float* ln2_beta;

    float* dW_qkv;
    float* db_qkv;
    float* dW_o;
    float* db_o;
    float* dW_ff1;
    float* db_ff1;
    float* dW_ff2;
    float* db_ff2;
    float* d_ln1_gamma;
    float* d_ln1_beta;
    float* d_ln2_gamma;
    float* d_ln2_beta;

    // Adam moments (null → skip ADAM_APPLY for this layer)
    float* ms_W_qkv;
    float* vs_W_qkv;
    float* ms_b_qkv;
    float* vs_b_qkv;
    float* ms_W_o;
    float* vs_W_o;
    float* ms_b_o;
    float* vs_b_o;
    float* ms_W_ff1;
    float* vs_W_ff1;
    float* ms_b_ff1;
    float* vs_b_ff1;
    float* ms_W_ff2;
    float* vs_W_ff2;
    float* ms_b_ff2;
    float* vs_b_ff2;
    float* ms_ln1_g;
    float* vs_ln1_g;
    float* ms_ln1_b;
    float* vs_ln1_b;
    float* ms_ln2_g;
    float* vs_ln2_g;
    float* ms_ln2_b;
    float* vs_ln2_b;

    // Per-layer workspace (caller-owned)
    float* qkv;       // [B*T, 3D]
    float* scores;    // [B*H*T*T]
    float* attn_out;  // [B*T, D]
    float* ln1_out;   // [B*T, D]
    float* h1;        // [B*T, D]
    float* ln2_out;   // [B*T, D]
    float* ffn_pre;   // [B*T, ffn_hidden]
    float* ffn_h;     // [B*T, ffn_hidden]
    float* O;         // [B*T, D] block output (next layer input)
};

struct MhsaBinding {
    int64_t B;
    int64_t T;
    int64_t D;
    int64_t H;
    int64_t d_head;
    int64_t action_dim;
    int64_t ffn_hidden;
    int64_t num_layers;  // 1..MHSA_MAX_LAYERS

    MhsaLayerBind layers[MHSA_MAX_LAYERS];

    float* W_act;  // [D, action_dim]
    float* b_act;
    float* dW_act;
    float* db_act;
    float* ms_W_act;
    float* vs_W_act;
    float* ms_b_act;
    float* vs_b_act;

    // Learned absolute positions [max_seq_len, D] (null → disabled)
    float* pos;
    float* d_pos;
    float* ms_pos;
    float* vs_pos;
    int64_t max_seq_len;

    float* scratch;   // [B*T, D]
    float* actions;   // [B, action_dim]
    float* dO;        // [B*T, D] incoming to top block / between layers
    float* dX;        // [B*T, D] optional grad w.r.t. model input
    float* d_qkv;     // [B*T, 3D] attn bwd temp
    float* d_stream;  // [B*T, D] interlayer grad carrier

    float* y;         // [B, action_dim]
    float* loss_out;  // optional scalar

    // Final residual stream (alias of layers[num_layers-1].O after fwd)
    float* O;
};

// X: [B*T, D]. Runs num_layers Pre-LN blocks; sets m->O to last block out.
int32_t mhsa_block_forward(const float* X, MhsaBinding* m);

int32_t mhsa_action_forward(MhsaBinding* m);
int32_t mhsa_action_backward(MhsaBinding* m);

// Backprop through layers[num_layers-1] .. layers[0].
int32_t mhsa_block_backward(const float* X, MhsaBinding* m);

#ifdef __cplusplus
}
#endif
