#include "timestep_embedder.cuh"
#include "utils.cuh"
#include "linear.cuh"
#include <assert.h>

// ----------------------------------------------------------------------------
// Timestep Embedder

// t_embedder->out is of shape (B, dim), where B is the batch size
void timestep_embedder_forward(
    TimestepEmbedder *t_embedder,
    const float *timesteps,
    cublasHandle_t cublas_handle)
{
    int B = t_embedder->B;
    int dim = t_embedder->dim;
    int hidden_size = t_embedder->hidden_size;
    float *t_emb_buf = t_embedder->t_emb_buf;
    get_timestep_embeddings(&(t_embedder->time_emb_state), timesteps, t_emb_buf);
    matmul_forward2(
        cublas_handle,
        t_embedder->out,
        t_emb_buf,
        t_embedder->fc1.w,
        t_embedder->fc1.b,
        B,
        dim,
        hidden_size,
        t_embedder->block_size);
    // apply silu
    silu_forward(t_embedder->out, t_embedder->out, B * hidden_size, t_embedder->block_size);

    matmul_forward2(
        cublas_handle,
        t_embedder->out,
        t_embedder->out,
        t_embedder->fc2.w,
        t_embedder->fc2.b,
        B,
        hidden_size,
        hidden_size,
        t_embedder->block_size);
}

void timestep_embedder_backward(
    TimestepEmbedder *t_embedder,
    const float *dout,
    cublasHandle_t cublas_handle)
{
    int B = t_embedder->B;
    int dim = t_embedder->dim;
    int hidden_size = t_embedder->hidden_size;
    float *t_emb_buf = t_embedder->t_emb_buf;

    // backward through fc2
    matmul_backward1(
        cublas_handle,
        t_emb_buf,
        t_embedder->fc2.w,
        t_embedder->fc2.b,
        dout,
        t_embedder->fc2.inp,
        t_embedder->fc2.w,
        B,
        hidden_size,
        hidden_size);

    // backward through silu
    silu_backward(t_emb_buf, t_embedder->fc2.inp, t_emb_buf, B * hidden_size, t_embedder->block_size);

    // backward through fc1
    matmul_backward1(
        cublas_handle,
        t_emb_buf,
        t_embedder->fc1.w,
        t_embedder->fc1.b,
        t_emb_buf,
        t_embedder->time_emb_state.embeddings,
        t_embedder->fc1.w,
        B,
        dim,
        hidden_size);

    // No need to backpropagate through the timestep embeddings, as they are not trainable parameters.
}

// ----------------------------------------------------------------------------
// memory management

void timestep_embedder_init(TimestepEmbedder *t_embedder, int dim, int hidden_size, int B, int max_period, int block_size)
{
    // t_embedder->out shape is (B, hidden_size), same as in DiT python code
    t_embedder->B = B;
    t_embedder->dim = dim;
    t_embedder->hidden_size = hidden_size;
    t_embedder->block_size = block_size;

    // init the timestep embedding
    init_timestep_embedding(&(t_embedder->time_emb_state), dim, B, max_period);

    // malloc timestep emb buffer
    cudaCheck(cudaMalloc(&(t_embedder->t_emb_buf), B * dim * sizeof(float)));

    // allocate space for the weights and biases
    cudaCheck(cudaMalloc(&(t_embedder->fc1.w), dim * hidden_size * sizeof(float)));
    cudaCheck(cudaMalloc(&(t_embedder->fc1.b), hidden_size * sizeof(float)));
    cudaCheck(cudaMalloc(&(t_embedder->fc2.w), hidden_size * hidden_size * sizeof(float)));
    cudaCheck(cudaMalloc(&(t_embedder->fc2.b), hidden_size * sizeof(float)));

    // set the input pointer for fc2
    t_embedder->fc2.inp = t_embedder->t_emb_buf;

    // allocate memory for the output
    cudaCheck(cudaMalloc(&(t_embedder->out), B * hidden_size * sizeof(float)));
}

void timestep_embedder_free(TimestepEmbedder *t_embedder)
{
    cudaCheck(cudaFree(t_embedder->t_emb_buf));
    cudaCheck(cudaFree(t_embedder->fc1.w));
    cudaCheck(cudaFree(t_embedder->fc1.b));
    cudaCheck(cudaFree(t_embedder->fc2.w));
    cudaCheck(cudaFree(t_embedder->fc2.b));
    cudaCheck(cudaFree(t_embedder->out));
    free_timestep_embedding(&(t_embedder->time_emb_state));
}