#pragma once
#include "utils.cuh"
#include "patch_embed.cuh"
#include "timestep_embedder.cuh"
#include "label_embedder.cuh"
#include "dit_block.cuh"
#include "final_layer.cuh"

typedef struct
{
    int B;
    int C;
    int H;
    int W;
    int patch_size;
    int hidden_size;
    int dim;   // dim of timestep embeddings
    int depth; // number of dit blocks
    int num_classes;
    int block_size;
    float dropout_prob;
    int max_period;
} DiTConfig;

typedef struct
{
    DiTConfig config;
    // Input data
    float *patch_embeddings; // (B, N, hidden_dim)
    float *t_plus_y;         // (B, hidden_dim)
    float *dt_plus_dy;       // (B, hidden_dim)
    float *out;              // output of shape (B, C, H, W)

    // Model components
    PatchEmbedder patch_embedder;
    TimestepEmbedder t_embedder;
    LabelEmbedder l_embedder;
    DiTBlock *blocks;
    FinalLayer final_layer;
} DiT;

// Forward pass of the DiT model
void dit_forward(
    DiT *dit,                     // DiT model
    const float *x,               // Input image: (B, C, H, W)
    const float *t,               // Timesteps: (B,)
    const int *y,                 // Class labels: (B,)
    cublasHandle_t cublas_handle, // cuBLAS handle
    float cfg_scale = 1.0f,       // Classifier-free guidance scale
    bool use_cfg = false,         // Whether to use classifier-free guidance
    float *t_out = nullptr        // Output for timing measurements (optional)
);

// Backward pass of the DiT model
void dit_backward(
    DiT *dit,                     // DiT model
    const float *dout,            // Gradient of the output
    const float *x,               // Input image: (B, C, H, W)
    const float *t,               // Timesteps: (B,)
    const int *y,                 // Class labels: (B,)
    cublasHandle_t cublas_handle, // cuBLAS handle
    float cfg_scale = 1.0f,       // Classifier-free guidance scale
    bool use_cfg = false,         // Whether to use classifier-free guidance
    float *t_out = nullptr        // Output for timing measurements (optional)
);

// Initialize the DiT model
void dit_init(DiT *dit, const DiTConfig config);

// Free the memory allocated for the DiT model
void dit_free(DiT *dit);