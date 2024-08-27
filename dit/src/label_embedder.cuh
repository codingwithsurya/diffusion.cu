#pragma once
#include "utils.cuh"

typedef struct
{
    int num_classes;
    int hidden_size;
    int B;
    int block_size;
    float *embedding_table; // shape (num_classes + 1, hidden_size)
    float *out;             // output of shape (B, hidden_size)
    curandState *curand_states;
} LabelEmbedder;

void label_embedder_forward(
    LabelEmbedder *l_embedder,
    const int *labels,
    float dropout_prob,
    int block_size);

void label_embedder_backward(
    LabelEmbedder *l_embedder,
    const float *dout,
    int block_size);

void label_embedder_init(LabelEmbedder *l_embedder, int num_classes, int hidden_size, int B, int block_size, unsigned long long seed);

void label_embedder_free(LabelEmbedder *l_embedder);