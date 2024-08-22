#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "add.cuh"
#include "common.h"

// This file contains the CUDA implementations of the add functions

// FP32 kernels
__global__ void add_forward_kernel_fp32(
    const float* a, const float* b,
    float* out,
    int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        out[idx] = a[idx] + b[idx];
    }
}

// FP16 kernels
__global__ void add_forward_kernel_fp16(
    const half* a, const half* b,
    half* out,
    int N,
    float loss_scale
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        out[idx] = __hadd(__hmul(a[idx], __float2half(loss_scale)), __hmul(b[idx], __float2half(loss_scale)));
    }
}

void add_forward(
    const void* a, const void* b,
    void* out,
    int N, int block_size,
    bool use_fp16,
    float loss_scale
) {
    int n_blk = ceil_div(N, block_size);
    if (use_fp16) {
        add_forward_kernel_fp16<<<n_blk, block_size>>>(
            static_cast<const half*>(a),
            static_cast<const half*>(b),
            static_cast<half*>(out),
            N,
            loss_scale
        );
    } else {
        add_forward_kernel_fp32<<<n_blk, block_size>>>(
            static_cast<const float*>(a),
            static_cast<const float*>(b),
            static_cast<float*>(out),
            N
        );
    }
}

// add A and B in place, and store result in B
__global__ void add_inplace_forward_kernel_fp32(
    const float* a, float* b,
    int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        b[idx] += a[idx];
    }
}

__global__ void add_inplace_forward_kernel_fp16(
    const half* a, half* b,
    int N,
    float loss_scale
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        b[idx] = __hadd(b[idx], __hmul(a[idx], __float2half(loss_scale)));
    }
}

void add_inplace_forward(
    const void* a, void* b,
    int N, int block_size,
    bool use_fp16,
    float loss_scale
) {
    int n_blk = ceil_div(N, block_size);
    if (use_fp16) {
        add_inplace_forward_kernel_fp16<<<n_blk, block_size>>>(
            static_cast<const half*>(a),
            static_cast<half*>(b),
            N,
            loss_scale
        );
    } else {
        add_inplace_forward_kernel_fp32<<<n_blk, block_size>>>(
            static_cast<const float*>(a),
            static_cast<float*>(b),
            N
        );
    }
}