#pragma once

// Includes CUDA runtime and common definitions
#include "common.h"

// Defines the forward pass of the SiLU activation function
void silu_forward(
    const float *x, // Input tensor
    float *out,     // Output tensor
    int N,          // Total number of elements in the tensors
    int block_size  // Block size for CUDA kernel launch
);

// Defines the backward pass of the SiLU activation function
void silu_backward(
    const float *dout, // Gradient of the output tensor
    const float *x,    // Input tensor (used for computing the gradient)
    float *dx,         // Gradient of the input tensor
    int N,             // Total number of elements in the tensors
    int block_size     // Block size for CUDA kernel launch
);