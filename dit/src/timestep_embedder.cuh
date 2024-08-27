#pragma once
#include "utils.cuh"
#include "linear.cuh"

typedef struct
{
    int B;
    int dim;
    int hidden_size;
    int block_size;
    // using LinearParams instead of just raw pointers
    // because we want to keep track of the input pointer
    // for the backward pass
    LinearParams fc1;
    LinearParams fc2;
    TimestepEmbedding time_emb_state;
    float *t_emb_buf; // buffer to store timestep embeddings of shape (B, dim)
    float *out;       // output of the embedder, of shape (B, hidden_size)
} TimestepEmbedder;

void timestep_embedder_forward(
    TimestepEmbedder *t_embedder,
    const float *timesteps,
    cublasHandle_t cublas_handle);

void timestep_embedder_backward(
    TimestepEmbedder *t_embedder,
    const float *dout,
    cublasHandle_t cublas_handle);

void timestep_embedder_init(TimestepEmbedder *t_embedder, int dim, int hidden_size, int B, int max_period, int block_size);

void timestep_embedder_free(TimestepEmbedder *t_embedder);