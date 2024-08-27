#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"
#include "silu.cuh"

// ----------------------------------------------------------------------------
// GPU kernels

// Kernel for the forward pass of SiLU activation
__global__ void silu_forward_kernel(
    const float *x, // Input tensor
    float *out,     // Output tensor
    int N           // Total number of elements
)
{
    // Calculate the global thread index
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Bounds check to ensure we don't access memory out of bounds
    if (idx >= N)
    {
        return;
    }

    // Apply SiLU activation: x / (1 + exp(-x))
    float x_val = x[idx];
    out[idx] = x_val / (1.0f + expf(-x_val));
}

// Kernel for the backward pass of SiLU activation
__global__ void silu_backward_kernel(
    const float *dout, // Gradient of the output tensor
    const float *x,    // Input tensor (used for computing the gradient)
    float *dx,         // Gradient of the input tensor
    int N              // Total number of elements
)
{
    // Calculate the global thread index
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Bounds check to ensure we don't access memory out of bounds
    if (idx >= N)
    {
        return;
    }

    // Calculate the gradient of SiLU
    float out_val = dout[idx];
    float x_val = x[idx];
    float expx = expf(-x_val);
    float grad_silu = (1.0f + x_val * expx / (1.0f + expx)) / (1.0f + expx);

    // Apply the chain rule: multiply the output gradient by the gradient of SiLU
    dx[idx] = out_val * grad_silu;
}

// ----------------------------------------------------------------------------
// CUDA kernel launcher

// Function to launch the SiLU forward kernel
void silu_forward(
    const float *x, // Input tensor
    float *out,     // Output tensor
    int N,          // Total number of elements
    int block_size  // Block size for CUDA kernel launch
)
{
    // Calculate the number of blocks needed based on the total number of elements and block size
    int n_blk = ceil_div(N, block_size);

    // Launch the kernel with the calculated grid and block dimensions
    silu_forward_kernel<<<n_blk, block_size>>>(x, out, N);
}

// Function to launch the SiLU backward kernel
void silu_backward(
    const float *dout, // Gradient of the output tensor
    const float *x,    // Input tensor (used for computing the gradient)
    float *dx,         // Gradient of the input tensor
    int N,             // Total number of elements
    int block_size     // Block size for CUDA kernel launch
)
{
    // Calculate the number of blocks needed based on the total number of elements and block size
    int n_blk = ceil_div(N, block_size);

    // Launch the kernel with the calculated grid and block dimensions
    silu_backward_kernel<<<n_blk, block_size>>>(dout, x, dx, N);
}

// ----------------------------------------------------------------------------

#ifndef LINKING
// Main function for testing purposes. Only compiled when not linking.
int main(int argc, char **argv)
{
    srand(0); // Seed the random number generator for deterministic results

    // Define the tensor dimensions
    int B = 1;
    int C = 32;
    int H = 32;
    int W = 32;
    int N = B * C * H * W;

    // Set up the CUDA device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);

    // Allocate host memory for input, output, input gradient, and output gradient
    float *x = (float *)malloc(N * sizeof(float));
    float *out = (float *)malloc(N * sizeof(float));
    float *dout = (float *)malloc(N * sizeof(float));
    float *dx = (float *)malloc(N * sizeof(float));

    // Read data from a binary file for testing
    FILE *file = fopenCheck("silu.bin", "rb");
    if (!file)
    {
        perror("Failed to load data");
        return -1;
    }

    // Read the input, output, and output gradient from the file
    freadCheck(x, sizeof(float), N, file);
    freadCheck(out, sizeof(float), N, file);
    freadCheck(dout, sizeof(float), N, file);
    freadCheck(dx, sizeof(float), N, file);
    fcloseCheck(file);

    // Allocate device memory for input, output, input gradient, and output gradient
    float *d_x, *d_out, *d_dout, *d_dx;
    cudaCheck(cudaMalloc(&d_x, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dx, N * sizeof(float)));

    // Copy the input and output gradient to the device
    cudaCheck(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, dout, N * sizeof(float), cudaMemcpyHostToDevice));

    // Define an array of block sizes to test
    int block_sizes[] = {128, 256, 512, 1024};

    // Test the forward pass for each block size
    printf("Checking forward pass\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        printf("Checking block size %d\n", block_size);

        // Call the SiLU forward function
        silu_forward(d_x, d_out, N, block_size);

        // Validate the result against the reference output
        validate_result(d_out, out, "out", N);
    }
    printf("Forward pass: all results match\n\n");

    // Test the backward pass for each block size
    printf("Checking backward pass\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        printf("Checking block size %d\n", block_size);

        // Call the SiLU backward function
        silu_backward(d_dout, d_x, d_dx, N, block_size);

        // Validate the result against the reference input gradient
        validate_result(d_dx, dx, "dx", N);
    }
    printf("Backward pass: all results match\n\n");

    // Benchmark the forward pass for each block size
    printf("All results match. Starting benchmarks.\n\n");
    printf("Forward pass benchmarks:\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];

        // Run the benchmark for 100 iterations
        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, silu_forward,
                                              d_x, d_out, N, block_size);

        // Print the benchmark results
        printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
    }

    // Benchmark the backward pass for each block size
    printf("\nBackward pass benchmarks:\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];

        // Run the benchmark for 100 iterations
        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, silu_backward,
                                              d_dout, d_x, d_dx,
                                              N, block_size);

        // Print the benchmark results
        printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
    }

    // Free host memory
    free(x);
    free(out);
    free(dout);
    free(dx);

    // Free device memory
    cudaCheck(cudaFree(d_x));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_dx));

    return 0;
}
#endif