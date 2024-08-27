#pragma once

#include <cublas_v2.h>

// Forward pass for a linear layer using cuBLAS
void matmul_forward2(
    cublasHandle_t cublas_handle, // cuBLAS handle
    float *out,                   // Output tensor
    const float *inp,             // Input tensor
    const float *weight,          // Weight matrix
    const float *bias,            // Bias vector (can be NULL)
    int N, int C, int OC,         // Dimensions: Batch size, Input channels, Output channels
    const int block_size          // Block size for CUDA kernel (used for bias addition if bias is not NULL)
);

// Backward pass for a linear layer using cuBLAS
void matmul_backward1(
    cublasHandle_t cublas_handle, // cuBLAS handle
    float *dinp,                  // Gradient of the input tensor
    float *dweight,               // Gradient of the weight matrix
    float *dbias,                 // Gradient of the bias vector (can be NULL)
    float *dout,                  // Gradient of the output tensor
    float *inp,                   // Input tensor
    float *weight,                // Weight matrix
    int N, int C, int OC          // Dimensions: Batch size, Input channels, Output channels
);

// Structure to hold the parameters (weights and bias) of a linear layer
typedef struct
{
    float *w; // Weight matrix of shape (OC, C)
    float *b; // Bias vector of shape (OC)
} LinearParams;

// Structure to hold the input and output tensors of a linear layer
typedef struct
{
    float *inp; // Input tensor of shape (N, C)
    float *out; // Output tensor of shape (N, OC)
} LinearActs;

// Function to set pointers to the weight and bias within the parameter memory block
void linear_set_param_ptrs(
    LinearParams *params, // Linear layer parameters structure
    float *params_memory, // Pointer to the start of the parameter memory block
    int C, int OC         // Dimensions: Input channels, Output channels
);

// Function to calculate the number of parameters in a linear layer
inline size_t linear_count_params(int C, int OC);