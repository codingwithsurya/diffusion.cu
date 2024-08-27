#include "dit_block.cuh"
#include "attention.cuh"
#include "layernorm.cuh"
#include "linear.cuh"
#include "mlp.cuh"
#include "utils.cuh"
#include "silu.cuh"
#include <assert.h>

// ----------------------------------------------------------------------------
// DiT block

// x: (B, N, hidden_dim), c: (B, hidden_dim)
// attn_out: (B, N, hidden_dim)
void dit_block_forward(
    DiTBlock *block,
    const float *x,
    const float *c,
    cublasHandle_t cublas_handle,
    int B,
    int N,
    int hidden_dim,
    int block_size)
{
    // 1. Apply LayerNorm to the input
    layernorm_forward<float, float, float, float>(
        block->ln1_out,
        block->ln1_mean,
        block->ln1_rstd,
        x,
        block->ln1_w,
        block->ln1_b,
        c, c + B * hidden_dim,
        B * N,
        hidden_dim,
        block_size,
        0);

    // 2. Project the input to Q, K, V for attention
    matmul_forward2(
        cublas_handle,
        block->qkv,
        block->ln1_out,
        block->proj_qkv.w,
        block->proj_qkv.b,
        B * N,
        hidden_dim,
        3 * hidden_dim,
        block_size);

    // 3. Apply attention
    attention_forward1(
        cublas_handle,
        block->attn_out,
        block->qkvr,
        block->preatt,
        block->att,
        block->qkv, // Re-use qkv as a scratch buffer
        B,
        N,
        hidden_dim,
        block->n_heads,
        block_size);

    // 4. Apply LayerNorm after attention
    layernorm_forward<float, float, float, float>(
        block->ln2_out,
        block->ln2_mean,
        block->ln2_rstd,
        block->attn_out,
        block->ln2_w,
        block->ln2_b,
        c, c + B * hidden_dim,
        B * N,
        hidden_dim,
        block_size,
        0);

    // 5. Apply MLP
    mlp_forward(
        cublas_handle,
        block->mlp_out,
        block->ln2_out,
        block->mlp_fc1.w,
        block->mlp_fc2.w,
        B,
        N,
        hidden_dim,
        block->mlp_hidden_dim,
        block_size);
}

void dit_block_backward(
    DiTBlock *block,
    const float *dout,
    const float *c,
    cublasHandle_t cublas_handle,
    int B,
    int N,
    int hidden_dim,
    int block_size)
{
    // 5. Backward through MLP
    mlp_backward(
        cublas_handle,
        dout,
        block->ln2_out,
        block->mlp_fc1.w,
        block->mlp_fc2.w,
        block->d_mlp_fc1.w,
        block->d_mlp_fc2.w,
        block->mlp_act_buf1,
        block->mlp_act_buf2,
        B,
        N,
        hidden_dim,
        block->mlp_hidden_dim,
        block_size);

    // 4. Backward through LayerNorm after attention
    layernorm_backward<float, float, float>(
        block->d_attn_out,
        block->ln2_w,
        block->ln2_b,
        block->ln2_scratch,
        dout,
        block->attn_out,
        block->ln2_w,
        block->ln2_mean,
        block->ln2_rstd,
        c, c + B * hidden_dim,
        B * N,
        hidden_dim,
        block_size,
        0);
    add_inplace_forward(block->d_attn_out, block->mlp_act_buf1, B * N * hidden_dim, block_size);

    // 3. Backward through attention
    attention_backward(
        cublas_handle,
        block->d_qkv,
        block->d_qkvr,
        block->d_preatt,
        block->d_att,
        block->attn_scratch,
        block->d_attn_out,
        block->qkvr,
        block->att,
        B,
        N,
        hidden_dim,
        block->n_heads);

    // 2. Backward through the projection to Q, K, V
    matmul_backward1(
        cublas_handle,
        block->d_ln1_out,
        block->proj_qkv.w,
        block->proj_qkv.b,
        block->d_qkv,
        block->ln1_out,
        block->proj_qkv.w,
        B * N,
        hidden_dim,
        3 * hidden_dim);

    // 1. Backward through LayerNorm
    layernorm_backward<float, float, float>(
        block->dx,
        block->ln1_w,
        block->ln1_b,
        block->ln1_scratch,
        block->d_ln1_out,
        x,
        block->ln1_w,
        block->ln1_mean,
        block->ln1_rstd,
        c, c + B * hidden_dim,
        B * N,
        hidden_dim,
        block_size,
        0);
}

// ----------------------------------------------------------------------------
// memory management

void dit_block_init(
    DiTBlock *block,
    int hidden_dim,
    int mlp_hidden_dim,
    int B,
    int N,
    int block_size)
{
    block->hidden_dim = hidden_dim;
    block->mlp_hidden_dim = mlp_hidden_dim;
    block->n_heads = hidden_dim / 64; // Assuming head size is always 64
    block->block_size = block_size;

    // Allocate memory for weights and biases
    cudaCheck(cudaMalloc(&(block->ln1_w), hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->ln1_b), hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->proj_qkv.w), 3 * hidden_dim * hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->proj_qkv.b), 3 * hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->ln2_w), hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->ln2_b), hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->mlp_fc1.w), hidden_dim * mlp_hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->mlp_fc2.w), mlp_hidden_dim * hidden_dim * sizeof(float)));

    // Allocate memory for gradients
    cudaCheck(cudaMalloc(&(block->d_mlp_fc1.w), hidden_dim * mlp_hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->d_mlp_fc2.w), mlp_hidden_dim * hidden_dim * sizeof(float)));

    // Allocate memory for intermediate activations
    cudaCheck(cudaMalloc(&(block->ln1_out), B * N * hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->ln1_mean), B * N * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->ln1_rstd), B * N * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->ln1_scratch), (1024 / 32) * cuda_num_SMs * (2 * hidden_dim + 32) * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->qkv), B * N * 3 * hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->qkvr), B * N * 3 * hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->preatt), B * block->n_heads * N * N * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->att), B * block->n_heads * N * N * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->attn_out), B * N * hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->ln2_out), B * N * hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->ln2_mean), B * N * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->ln2_rstd), B * N * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->ln2_scratch), (1024 / 32) * cuda_num_SMs * (2 * hidden_dim + 32) * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->mlp_out), B * N * hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->mlp_act_buf1), B * N * mlp_hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->mlp_act_buf2), B * N * mlp_hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->attn_scratch), B * N * hidden_dim * sizeof(float)));

    // Allocate memory for gradients of intermediate activations
    cudaCheck(cudaMalloc(&(block->d_ln1_out), B * N * hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->d_qkv), B * N * 3 * hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->d_qkvr), B * N * 3 * hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->d_preatt), B * block->n_heads * N * N * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->d_att), B * block->n_heads * N * N * sizeof(float)));
    cudaCheck(cudaMalloc(&(block->d_attn_out), B * N * hidden_dim * sizeof(float)));

    // Set input pointers for backward pass
    block->proj_qkv.inp = block->ln1_out;
    block->mlp_fc1.inp = block->ln2_out;
    block->mlp_fc2.inp = block->mlp_act_buf1;
}

void dit_block_free(DiTBlock *block)
{
    // Free weights and biases
    cudaCheck(cudaFree(block->ln1_w));
    cudaCheck(cudaFree(block->ln1_b));
    cudaCheck(cudaFree(block->proj_qkv.w));
    cudaCheck(cudaFree(block->proj_qkv.b));
    cudaCheck(cudaFree(block->ln2_w));
    cudaCheck(cudaFree(block->ln2_b));
    cudaCheck(cudaFree(block->mlp_fc1.w));
    cudaCheck(cudaFree(block->mlp_fc2.w));

    // Free gradients
    cudaCheck(cudaFree(block->d_mlp_fc1.w));
    cudaCheck(cudaFree(block->d_mlp_fc2.w));

    // Free intermediate activations
    cudaCheck(cudaFree(block->ln1_out));
    cudaCheck(cudaFree(block->ln1_mean));
    cudaCheck(cudaFree(block->ln1_rstd));
    cudaCheck(cudaFree(block->ln1_scratch));
    cudaCheck(cudaFree(block->qkv));
    cudaCheck(cudaFree(block->qkvr));
    cudaCheck(cudaFree(block->preatt));
    cudaCheck(cudaFree(block->att));
    cudaCheck(cudaFree(block->attn_out));
    cudaCheck(cudaFree(block->ln2_out));
    cudaCheck(cudaFree(block->ln2_mean));
    cudaCheck(cudaFree(block->ln2_rstd));
    cudaCheck(cudaFree(block->ln2_scratch));
    cudaCheck(cudaFree(block->mlp_out));
    cudaCheck(cudaFree(block->mlp_act_buf1));
    cudaCheck(cudaFree(block->mlp_act_buf2));
    cudaCheck(cudaFree(block->attn_scratch));

    // Free gradients of intermediate activations
    cudaCheck(cudaFree(block->d_ln1_out));
    cudaCheck(cudaFree(block->d_qkv));
    cudaCheck(cudaFree(block->d_qkvr));
    cudaCheck(cudaFree(block->d_preatt));
    cudaCheck(cudaFree(block->d_att));
    cudaCheck(cudaFree(block->d_attn_out));
}