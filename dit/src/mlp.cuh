#pragma once

#include <cublas_v2.h>

void mlp_forward(
    cublasHandle_t cublas_handle,
    float *out, const float *inp,
    float *fc1_w, float *fc2_w,
    int B, int T, int C, int hidden_dim, int block_size);

void mlp_backward(
    cublasHandle_t cublas_handle,
    const float *dout, const float *inp,
    float *fc1_w, float *fc2_w,
    float *dfc1_w, float *dfc2_w,
    float *act_buf1, float *act_buf2,
    int B, int T, int C, int hidden_dim, int block_size);