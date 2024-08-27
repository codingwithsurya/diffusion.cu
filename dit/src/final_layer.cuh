#pragma once

#include "utils.cuh"
#include "linear.cuh"

typedef struct
{
    int patch_size;
    int in_channels;
    int out_channels;
    int B;
    int H;
    int W;
    int N; // number of patches
    int C; // number of channels in the input to the final layer
    int hidden_size;
    int block_size;
    LinearParams fc;   // Linear layer for adaLN modulation
    LinearParams proj; // Linear layer for final projection
    float *ln_w;       // LayerNorm weights
    float *ln_b;       // LayerNorm biases
    // Intermediate activations
    float *shift_scale; // Output of the `fc` layer (shift and scale for adaLN)
    float *ln_out;      // Output of the LayerNorm
    float *ln_mean;     // Mean of the LayerNorm input
    float *ln_rstd;     // Reciprocal standard deviation of the LayerNorm input
    float *scratch;     // Scratchpad memory for layernorm backward
    // Output and gradients
    float *out; // Output tensor
    float *dx;  // Gradient of the input tensor
    float *dc;  // Gradient of the conditioning vector `c`
} FinalLayer;

void final_layer_forward(
    FinalLayer *final_layer,
    const float *x,
    const float *c,
    cublasHandle_t cublas_handle);

void final_layer_backward(
    FinalLayer *final_layer,
    const float *dout,
    const float *x,
    const float *c,
    cublasHandle_t cublas_handle);

void final_layer_init(
    FinalLayer *final_layer,
    int patch_size,
    int in_channels,
    int out_channels,
    int B,
    int H,
    int W,
    int hidden_size,
    int block_size);

void final_layer_free(FinalLayer *final_layer);