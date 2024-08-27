#pragma once
#include "utils.cuh"
#include "attention.cuh"
#include "linear.cuh"
#include "mlp.cuh"

typedef struct
{
    int hidden_dim;
    int mlp_hidden_dim;
    int n_heads;
    int block_size;
    // Input tensors
    float *x;       // (B, N, hidden_dim)
    const float *c; // Conditioning vector (B, hidden_dim)
    // LayerNorm parameters
    float *ln1_w; // (hidden_dim,)
    float *ln1_b; // (hidden_dim,)
    float *ln2_w; // (hidden_dim,)
    float *ln2_b; // (hidden_dim,)
    // Projection parameters for Q, K, V
    LinearParams proj_qkv; // (3 * hidden_dim, hidden_dim)
    // MLP parameters
    LinearParams mlp_fc1; // (mlp_hidden_dim, hidden_dim)
    LinearParams mlp_fc2; // (hidden_dim, mlp_hidden_dim)
    // Intermediate activations
    float *ln1_out;      // (B, N, hidden_dim)
    float *ln1_mean;     // (B * N,)
    float *ln1_rstd;     // (B * N,)
    float *ln1_scratch;  // Scratch space for layernorm backward
    float *qkv;          // (B, N, 3 * hidden_dim)
    float *qkvr;         // (B, N, 3 * hidden_dim)
    float *preatt;       // (B, n_heads, N, N)
    float *att;          // (B, n_heads, N, N)
    float *attn_out;     // (B, N, hidden_dim)
    float *ln2_out;      // (B, N, hidden_dim)
    float *ln2_mean;     // (B * N,)
    float *ln2_rstd;     // (B * N,)
    float *ln2_scratch;  // Scratch space for layernorm backward
    float *mlp_out;      // (B, N, hidden_dim)
    float *mlp_act_buf1; // Buffer for MLP activations (B, N, mlp_hidden_dim)
    float *mlp_act_buf2; // Buffer for MLP activations (B, N, mlp_hidden_dim)
    float *attn_scratch; // Scratch space for attention backward (B, N, hidden_dim)
    // Gradients
    float *dx;               // Gradient of the input tensor (B, N, hidden_dim)
    LinearParams d_proj_qkv; // Gradient of the projection parameters
    LinearParams d_mlp_fc1;  // Gradient of the MLP FC1 parameters
    LinearParams d_mlp_fc2;  // Gradient of the MLP FC2 parameters
    float *d_ln1_out;        // Gradient of the LayerNorm1 output
    float *d_qkv;            // Gradient of the QKV tensor
    float *d_qkvr;           // Gradient of the QKVR tensor
    float *d_preatt;         // Gradient of the pre-attention scores
    float *d_att;            // Gradient of the attention weights
    float *d_attn_out;       // Gradient of the attention output
} DiTBlock;

// Forward pass of the DiT block
void dit_block_forward(
    DiTBlock *block,
    const float *x,
    const float *c,
    cublasHandle_t cublas_handle,
    int B,
    int N,
    int hidden_dim,
    int block_size);

// Backward pass of the DiT block
void dit_block_backward(
    DiTBlock *block,
    const float *dout,
    const float *c,
    cublasHandle_t cublas_handle,
    int B,
    int N,
    int hidden_dim,
    int block_size);

// Initialize the DiT block
void dit_block_init(
    DiTBlock *block,
    int hidden_dim,
    int mlp_hidden_dim,
    int B,
    int N,
    int block_size);

// Free the memory allocated for the DiT block
void dit_block_free(DiTBlock *block);